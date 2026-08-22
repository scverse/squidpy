from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.metrics import fowlkes_mallows_score

from squidpy.gr import cluster_auto_k, cluster_stability
from squidpy.gr._autok import (
    DEFAULT_INIT_PARAMS,
    ClusterAutoKResult,
    _score_block,
    expand_n_clusters,
    mirror_stability,
    sweep_auto_k,
    to_uns,
)
from squidpy.gr._autok import cluster_stability as _cluster_stability


def scored(result: ClusterAutoKResult) -> list[int]:
    table = result["table"]
    return table.index[table["stability_mean"].notna()].tolist()


def make_blobs(n_per_blob: int = 30, n_features: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(centre, 1.0, (n_per_blob, n_features)) for centre in (0.0, 8.0, -8.0)])


# resolving the requested K values


def test_expand_n_clusters_adds_halo_to_a_tuple():
    assert expand_n_clusters((2, 10)) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_expand_n_clusters_clamps_the_halo_at_one():
    assert expand_n_clusters((1, 3)) == [1, 2, 3, 4]


def test_expand_n_clusters_takes_a_sequence_literally():
    assert expand_n_clusters([2, 3, 4, 5]) == [2, 3, 4, 5]
    assert expand_n_clusters([2, 5, 9]) == [2, 5, 9]


@pytest.mark.parametrize(
    ("n_clusters", "match"),
    [
        ((2, 10, 3), r"must be \(min, max\)"),
        ((10, 2), r"min <= max"),
        ([0, 1, 2], r"at least 1"),
        ([2, 2, 3], r"must not contain duplicates"),
        ([2, 3], r"at least 3 K values"),
    ],
)
def test_expand_n_clusters_rejects_bad_requests(n_clusters, match):
    with pytest.raises(ValueError, match=match):
        expand_n_clusters(n_clusters)


# folding one-directional similarities into a per-K score


def test_mirror_stability_keeps_only_interior_k():
    blocks = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # 4 K values -> 3 adjacent pairs
    mirrored = mirror_stability(blocks)
    assert mirrored.tolist() == [[0.2, 0.5, 0.1, 0.4], [0.3, 0.6, 0.2, 0.5]]


# the sweep


def test_sweep_requires_more_than_one_run():
    with pytest.raises(ValueError, match=r"at least 2 runs"):
        sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=1)


def test_sweep_is_reproducible_from_the_seed():
    X, ks = make_blobs(), [1, 2, 3, 4]
    first = sweep_auto_k(X, ks, max_runs=3, seed=0)
    second = sweep_auto_k(X, ks, max_runs=3, seed=0)
    assert first["best_k"] == second["best_k"]
    assert np.array_equal(first["stability"], second["stability"])
    assert_frame_equal(first["table"], second["table"])


def test_sweep_handles_non_contiguous_k_values():
    # regression: adjacent entries, not k+1
    result = sweep_auto_k(make_blobs(), [2, 5, 9], max_runs=2, seed=0)
    assert list(result["table"].index) == [2, 5, 9]
    assert scored(result) == [5]
    assert result["best_k"] == 5


def test_sweep_scores_only_interior_but_fits_the_halo():
    result = sweep_auto_k(make_blobs(), [1, 2, 3, 4, 5], max_runs=2, seed=0)
    assert scored(result) == [2, 3, 4]
    assert result["stability"].shape[0] == 3
    assert result["best_k"] in scored(result)
    # nll is defined for every fitted K, including the halo
    assert result["table"]["nll"].notna().all()
    assert sorted(result["labels"]) == [1, 2, 3, 4, 5]


def test_sweep_convergence():
    X, ks = make_blobs(), [1, 2, 3, 4]
    converged = sweep_auto_k(X, ks, max_runs=10, convergence_tol=np.inf, seed=0)
    assert converged["converged"]
    assert converged["n_runs"] < 10

    exhausted = sweep_auto_k(X, ks, max_runs=3, convergence_tol=0.0, seed=0)
    assert not exhausted["converged"]
    assert exhausted["n_runs"] == 3


def test_sweep_rejects_params_it_controls_itself():
    for owned in ("n_components", "random_state"):
        with pytest.raises(ValueError, match=rf"'{owned}' cannot be set through 'model_params'"):
            sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=2, model_params={owned: 3})


def test_sweep_does_not_mutate_the_callers_model_params():
    # upstream pops `random_state` out of the caller's dict; we must not
    model_params = {"max_iter": 10}
    sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=2, model_params=model_params, seed=0)
    assert model_params == {"max_iter": 10}


def test_sweep_init_params_is_pinned_but_overridable(monkeypatch):
    from squidpy.gr import _autok

    seen: list[str] = []
    original = _autok.GaussianMixture

    def spy(*args, **kwargs):
        seen.append(kwargs["init_params"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_autok, "GaussianMixture", spy)

    sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=2, seed=0)
    assert set(seen) == {DEFAULT_INIT_PARAMS}

    seen.clear()
    sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=2, model_params={"init_params": "kmeans"}, seed=0)
    assert set(seen) == {"kmeans"}


def test_sweep_reg_covar_hint():
    # only two distinct points, so with the regularisation switched off every component
    # lands on a singular covariance
    X = np.repeat(np.array([[0.0, 0.0], [1.0, 1.0]]), 15, axis=0)
    with pytest.raises(ValueError, match=r"model_params=\{'reg_covar'"):
        sweep_auto_k(X, [2, 3, 4], max_runs=2, model_params={"reg_covar": 0.0}, seed=0)


def test_sweep_reg_covar_hint_not_for_other_errors():
    X = np.zeros((2, 2))  # fewer samples than components
    with pytest.raises(ValueError, match=r"the mixture fit at K=\d+ failed") as excinfo:
        sweep_auto_k(X, [2, 3, 4], max_runs=2, seed=0)
    assert "reg_covar" not in str(excinfo.value)


def test_sweep_auto_k_keeps_labels_narrow():
    result = sweep_auto_k(make_blobs(), [2, 3, 4], max_runs=2, seed=0)
    assert {labels.dtype for labels in result["labels"].values()} == {np.dtype(np.uint32)}


# the per-K table


def test_table_halo_unscored():
    result = sweep_auto_k(make_blobs(), [1, 2, 3, 4, 5], max_runs=2, seed=0)
    frame = result["table"]

    assert list(frame.index) == [1, 2, 3, 4, 5]
    assert frame.index.name == "k"
    assert list(frame.columns) == ["stability_mean", "stability_std", "nll"]

    halo = frame.loc[[1, 5]]
    assert halo["stability_mean"].isna().all()
    assert halo["stability_std"].isna().all()
    assert not halo["nll"].isna().any(), "the halo is fitted, so it has an nll"

    assert scored(result) == [2, 3, 4], "the halo is never scored, so it can never be best"


def test_best_k_is_the_most_stable_scored_k():
    result = sweep_auto_k(make_blobs(), [1, 2, 3, 4, 5], max_runs=3, seed=0)
    assert result["best_k"] == scored(result)[int(np.argmax(result["stability"].mean(axis=1)))]


def test_to_uns_carries_only_what_survives_h5ad(tmp_path):
    result = sweep_auto_k(make_blobs(), [1, 2, 3, 4, 5], max_runs=2, seed=0)
    adata = AnnData(np.zeros((result["stability"].shape[0], 1), dtype=np.float32))
    adata.uns["autok"] = to_uns(result)

    path = tmp_path / "autok.h5ad"
    adata.write_h5ad(path)
    reloaded = read_h5ad(path).uns["autok"]

    assert set(reloaded) == set(result) - {"labels"}, "`labels` is K-keyed, so uns cannot hold it"
    assert_frame_equal(reloaded["table"], result["table"])
    np.testing.assert_allclose(reloaded["stability"], result["stability"])
    assert reloaded["best_k"] == result["best_k"]
    assert reloaded["n_runs"] == result["n_runs"]
    assert reloaded["converged"] == result["converged"]


# the public entry point


def test_cluster_auto_k_on_adata():
    adata = AnnData(make_blobs())
    assert cluster_auto_k(adata, (2, 4), max_runs=3, seed=0) is None

    diagnostics = adata.uns["cluster_auto_k"]
    assert list(diagnostics["table"].index) == [1, 2, 3, 4, 5]
    assert diagnostics["best_k"] in scored(diagnostics)
    assert list(adata.obs.columns) == ["cluster_auto_k"]
    assert adata.obs["cluster_auto_k"].dtype == "category"


def test_cluster_auto_k_copy():
    adata = AnnData(make_blobs())
    out = cluster_auto_k(adata, (2, 4), max_runs=3, seed=0, copy=True)

    assert list(out.obs.columns) == ["cluster_auto_k"]
    assert "cluster_auto_k" in out.uns
    assert adata.obs.columns.empty
    assert not adata.uns


def test_cluster_auto_k_uses_the_requested_representation():
    adata = AnnData(np.zeros((90, 2)))
    adata.obsm["X_embedding"] = make_blobs()
    cluster_auto_k(adata, (2, 4), use_rep="X_embedding", max_runs=2, seed=0)
    assert adata.uns["cluster_auto_k"]["best_k"] in scored(adata.uns["cluster_auto_k"])


def test_cluster_auto_k_rejects_a_missing_representation():
    adata = AnnData(make_blobs())
    with pytest.raises(KeyError, match=r"not_there"):
        cluster_auto_k(adata, (2, 4), use_rep="not_there", max_runs=2, seed=0)


def test_cluster_auto_k_rejects_sparse_input():
    adata = AnnData(csr_matrix(make_blobs()))
    with pytest.raises(TypeError, match=r"does not support sparse input"):
        cluster_auto_k(adata, (2, 4), max_runs=2, seed=0)


def test_cluster_auto_k_keep_all_labels_adds_every_fitted_k():
    adata = AnnData(make_blobs())
    cluster_auto_k(adata, n_clusters=(2, 3), max_runs=2, seed=0, keep_all_labels=True)

    fitted = list(adata.uns["cluster_auto_k"]["table"].index)
    assert list(adata.obs.columns) == ["cluster_auto_k", *(f"cluster_auto_k_k{k}" for k in fitted)]

    best_k = adata.uns["cluster_auto_k"]["best_k"]
    pd.testing.assert_series_equal(
        adata.obs["cluster_auto_k"], adata.obs[f"cluster_auto_k_k{best_k}"], check_names=False
    )


def test_cluster_auto_k_honours_key_added():
    adata = AnnData(make_blobs())
    cluster_auto_k(adata, n_clusters=(2, 3), max_runs=2, seed=0, key_added="sweep")
    assert list(adata.obs.columns) == ["sweep"]
    assert "sweep" in adata.uns


def test_cluster_auto_k_is_reproducible_from_the_seed():
    adata = AnnData(make_blobs())
    first = cluster_auto_k(adata, (2, 4), max_runs=3, seed=0, copy=True)
    second = cluster_auto_k(adata, (2, 4), max_runs=3, seed=0, copy=True)
    pd.testing.assert_series_equal(first.obs["cluster_auto_k"], second.obs["cluster_auto_k"])


# scoring labelings that were produced elsewhere


def _runs(ks: list[int], n_runs: int, seed: int = 0) -> list[dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [{k: rng.integers(0, k, 40) for k in ks} for _ in range(n_runs)]


def test_cluster_stability_pairs_runs_like_the_sweep():
    ks = [2, 3, 4, 5]
    runs = _runs(ks, n_runs=4)
    pairs = list(zip(ks[:-1], ks[1:], strict=True))

    expected: list[list[float]] = []
    for i, new in enumerate(runs):  # the incremental pairing inside `sweep_auto_k`
        expected.extend(_score_block(new, stored, pairs, fowlkes_mallows_score) for stored in runs[:i])

    _, stability = _cluster_stability({k: [run[k] for run in runs] for k in ks})
    assert stability.shape == (len(ks) - 2, 2 * len(expected))
    # column order is arbitrary
    np.testing.assert_allclose(np.sort(mirror_stability(expected), axis=1), np.sort(stability, axis=1))


def test_cluster_stability_returns_only_interior_k():
    ks = [2, 3, 4, 5, 6]
    interior, stability = _cluster_stability({k: [run[k] for run in _runs(ks, n_runs=2)] for k in ks})
    assert interior == [3, 4, 5]
    assert stability.shape[0] == len(interior)


def test_cluster_stability_honours_score_fn():
    ks = [2, 3, 4]
    labels = {k: [run[k] for run in _runs(ks, n_runs=3)] for k in ks}
    _, stability = _cluster_stability(labels, score_fn=lambda a, b: 1.0)
    np.testing.assert_allclose(stability, 1.0)


@pytest.mark.parametrize(
    ("n_runs_per_k", "match"),
    [
        ({2: 2, 3: 2}, r"at least 3 K values"),
        ({2: 2, 3: 3, 4: 2}, r"same number of runs"),
        ({2: 1, 3: 1, 4: 1}, r"at least 2 runs"),
    ],
)
def test_cluster_stability_rejects_unusable_input(n_runs_per_k: dict[int, int], match: str):
    labels = {k: [np.zeros(4, dtype=int)] * n for k, n in n_runs_per_k.items()}
    with pytest.raises(ValueError, match=match):
        _cluster_stability(labels)


def _adata_with_runs(ks: list[int], n_runs: int) -> tuple[AnnData, dict[int, list[str]]]:
    runs = _runs(ks, n_runs=n_runs)
    adata = AnnData(np.zeros((40, 2), dtype=np.float32))
    keys = {}
    for k in ks:
        keys[k] = [f"clust_k{k}_run{r}" for r in range(n_runs)]
        for r, run in enumerate(runs):
            adata.obs[keys[k][r]] = pd.Categorical(run[k])
    return adata, keys


def test_cluster_stability_scores_obs_columns():
    adata, keys = _adata_with_runs([2, 3, 4], n_runs=3)

    df = cluster_stability(adata, keys, copy=True).uns["cluster_stability"]
    assert list(df.index) == [2, 3, 4]
    # only the interior K is scored, the bounds are a halo that is compared against but never selectable
    assert df["stability_mean"].isna().tolist() == [True, False, True]
    # the most stable K is the argmax of that column, not a separate flag
    assert df["stability_mean"].idxmax() == 3


def test_cluster_stability_writes_to_uns():
    adata, keys = _adata_with_runs([2, 3, 4], n_runs=2)
    expected = cluster_stability(adata, keys, copy=True).uns["cluster_stability"]
    assert "cluster_stability" not in adata.uns

    assert cluster_stability(adata, keys, key_added="my_stability") is None
    pd.testing.assert_frame_equal(adata.uns["my_stability"], expected)


def test_cluster_stability_rejects_a_missing_column():
    adata, keys = _adata_with_runs([2, 3, 4], n_runs=2)
    keys[3] = ["not_a_column", *keys[3][1:]]
    with pytest.raises(KeyError, match=r"not_a_column"):
        cluster_stability(adata, keys)


def test_cluster_stability_rejects_empty_cluster_keys():
    adata = AnnData(np.zeros((10, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"at least 3 K values"):
        cluster_stability(adata, {})


# parity with the reference implementation


def _cc_expand(n_clusters: tuple[int, int]) -> list[int]:
    # cellcharter 0.3.7, ClusterAutoK.__init__
    return list(range(*(max(1, n_clusters[0] - 1), n_clusters[1] + 2)))


def _cc_mirror(stability: list[float], n_clusters: list[int]) -> np.ndarray:
    # cellcharter 0.3.7, ClusterAutoK._mirror_stability
    chunks = [stability[i : i + len(n_clusters) - 1] for i in range(0, len(stability), len(n_clusters) - 1)]
    transposed = list(map(list, zip(*chunks, strict=True)))
    return np.array([transposed[i] + transposed[i - 1] for i in range(1, len(transposed))])


def _cc_best_k(stability: np.ndarray, n_clusters: list[int]) -> int:
    # cellcharter 0.3.7, ClusterAutoK.best_k
    means = np.array([np.mean(stability[k]) for k in range(len(n_clusters[1:-1]))])
    return n_clusters[int(np.argmax(means)) + 1]


@pytest.mark.parametrize("n_clusters", [(2, 10), (2, 4), (3, 7), (1, 5)])
def test_expand_n_clusters_matches_cellcharter(n_clusters):
    assert expand_n_clusters(n_clusters) == _cc_expand(n_clusters)


@pytest.mark.parametrize(("n_ks", "n_blocks"), [(5, 3), (6, 1), (4, 6), (11, 10)])
def test_mirror_stability_matches_cellcharter(n_ks, n_blocks):
    blocks = np.random.default_rng(0).random((n_blocks, n_ks - 1)).tolist()
    flat = [v for block in blocks for v in block]
    np.testing.assert_allclose(mirror_stability(blocks), _cc_mirror(flat, list(range(n_ks))))


def test_best_k_matches_cellcharter():
    ks = list(range(1, 12))
    blocks = np.random.default_rng(1).random((10, len(ks) - 1)).tolist()
    ours = pd.Series(mirror_stability(blocks).mean(axis=1), index=ks[1:-1]).idxmax()
    flat = [v for block in blocks for v in block]
    assert int(ours) == _cc_best_k(_cc_mirror(flat, ks), ks)


def test_sweep_pins_its_stability_curve():
    # guards mirror_stability/_score_block/the seeding against silent drift. these are the
    # values the criterion produces, not the ground truth: three well-separated blobs score
    # K=2 above K=3, which is a property of comparing K against K+1, not of this port.
    result = sweep_auto_k(make_blobs(), list(range(1, 7)), max_runs=5, seed=0)
    assert result["best_k"] == 2
    np.testing.assert_allclose(
        result["table"]["stability_mean"].to_numpy(),
        [np.nan, 0.720930, 0.715650, 0.669296, 0.661156, np.nan],
        rtol=1e-5,
    )


def test_seed_is_keyed_by_run_and_k_not_by_position():
    X = make_blobs()
    short = sweep_auto_k(X, [1, 2, 3, 4, 5], max_runs=3, seed=42)
    long = sweep_auto_k(X, [1, 2, 3, 4, 5, 6], max_runs=3, seed=42)
    for k in short["labels"]:
        np.testing.assert_array_equal(short["labels"][k], long["labels"][k])
