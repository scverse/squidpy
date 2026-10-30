from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from squidpy.gr import cluster_auto_k
from squidpy.gr._autok import expand_n_clusters, mirror_stability, sweep_auto_k


def scored(table: pd.DataFrame) -> list[int]:
    return table.index[table["stability_mean"].notna()].tolist()


def make_blobs(n_per_blob: int = 30, n_features: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(centre, 1.0, (n_per_blob, n_features)) for centre in (0.0, 8.0, -8.0)])


# resolving the requested K values


def test_expand_n_clusters_adds_halo_to_a_tuple():
    assert expand_n_clusters((2, 10)) == (list(range(1, 12)), list(range(2, 11)))


def test_expand_n_clusters_drops_k_below_two():
    with pytest.warns(UserWarning, match=r"K < 2"):
        assert expand_n_clusters((1, 3)) == ([1, 2, 3, 4], [2, 3])


def test_expand_n_clusters_halos_every_listed_k():
    assert expand_n_clusters([2, 5, 9]) == ([1, 2, 3, 4, 5, 6, 8, 9, 10], [2, 5, 9])
    assert expand_n_clusters([5, 2, 5, 3]) == ([1, 2, 3, 4, 5, 6], [2, 3, 5]), "sorted and deduplicated"


def test_spectral_clustering_is_swept_on_n_clusters():
    # it also takes `n_components` (the embedding width), which must not be mistaken for K
    from sklearn.cluster import SpectralClustering

    from squidpy.gr._autok import check_k_clusterer

    assert check_k_clusterer(SpectralClustering()) == "n_clusters"


@pytest.mark.parametrize(
    ("n_clusters", "match"),
    [
        ((2, 10, 3), r"must be \(min, max\)"),
        ((10, 2), r"min <= max"),
        ([0, 1], r"at least one value >= 2"),
    ],
)
@pytest.mark.filterwarnings("ignore:stability cannot be computed")
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


def test_sweep_is_reproducible_from_the_rng():
    X, ks = make_blobs(), [1, 2, 3, 4]
    first = sweep_auto_k(X, ks, max_runs=3, rng=0)
    second = sweep_auto_k(X, ks, max_runs=3, rng=0)
    assert first.best_k == second.best_k
    assert np.array_equal(first.stability, second.stability)
    assert_frame_equal(first.table, second.table)


def test_sweep_handles_non_contiguous_k_values():
    # regression: adjacent entries, not k+1
    result = sweep_auto_k(make_blobs(), [2, 5, 9], max_runs=2, rng=0)
    assert list(result.table.index) == [2, 5, 9]
    assert scored(result.table) == [5]
    assert result.best_k == 5


def test_sweep_scores_only_interior_but_fits_the_halo():
    result = sweep_auto_k(make_blobs(), [1, 2, 3, 4, 5], max_runs=2, rng=0)
    assert scored(result.table) == [2, 3, 4]
    assert result.stability.shape[0] == 3
    assert result.best_k in scored(result.table)
    # fit_loss is defined for every fitted K, including the halo
    assert result.table["fit_loss"].notna().all()
    assert sorted(result.labels) == [1, 2, 3, 4, 5]


def test_sweep_convergence():
    X, ks = make_blobs(), [1, 2, 3, 4]
    converged = sweep_auto_k(X, ks, max_runs=10, convergence_tol=np.inf, rng=0)
    assert converged.converged
    assert converged.n_runs < 10

    exhausted = sweep_auto_k(X, ks, max_runs=3, convergence_tol=0.0, rng=0)
    assert not exhausted.converged
    assert exhausted.n_runs == 3


def test_sweep_fits_the_given_mixture(monkeypatch):
    from squidpy.gr import _autok

    seen: list[str] = []
    original = _autok.clone

    def spy(est):
        seen.append(est.get_params()["init_params"])
        return original(est)

    monkeypatch.setattr(_autok, "clone", spy)

    sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=2, clusterer=GaussianMixture(init_params="random_from_data"), rng=0)
    assert set(seen) == {"random_from_data"}


def test_sweep_reg_covar_hint():
    # only two distinct points, so with the regularisation switched off every component
    # lands on a singular covariance
    X = np.repeat(np.array([[0.0, 0.0], [1.0, 1.0]]), 15, axis=0)
    with pytest.raises(ValueError, match=r"clusterer=GaussianMixture\(reg_covar"):
        sweep_auto_k(X, [2, 3, 4], max_runs=2, clusterer=GaussianMixture(reg_covar=0.0), rng=0)


def test_sweep_reg_covar_hint_not_for_other_errors():
    X = np.zeros((2, 2))  # fewer samples than components
    with pytest.raises(ValueError, match=r"the fit at K=\d+ failed") as excinfo:
        sweep_auto_k(X, [2, 3, 4], max_runs=2, rng=0)
    assert "reg_covar" not in str(excinfo.value)


def test_sweep_auto_k_keeps_labels_narrow():
    result = sweep_auto_k(make_blobs(), [2, 3, 4], max_runs=2, rng=0)
    assert {labels.dtype for labels in result.labels.values()} == {np.dtype(np.uint32)}


# the per-K table


def test_table_halo_unscored():
    result = sweep_auto_k(make_blobs(), [1, 2, 3, 4, 5], max_runs=2, rng=0)
    frame = result.table

    assert list(frame.index) == [1, 2, 3, 4, 5]
    assert frame.index.name == "k"
    assert list(frame.columns) == ["stability_mean", "stability_std", "fit_loss"]

    halo = frame.loc[[1, 5]]
    assert halo["stability_mean"].isna().all()
    assert halo["stability_std"].isna().all()
    assert not halo["fit_loss"].isna().any(), "the halo is fitted, so it has a fit_loss"

    assert scored(result.table) == [2, 3, 4], "the halo is never scored, so it can never be best"


def test_best_k_is_the_most_stable_scored_k():
    result = sweep_auto_k(make_blobs(), [1, 2, 3, 4, 5], max_runs=3, rng=0)
    assert result.best_k == scored(result.table)[int(np.argmax(result.stability.mean(axis=1)))]


# the public entry point


def test_cluster_auto_k_on_adata():
    adata = AnnData(make_blobs())
    assert cluster_auto_k(adata, use_rep="X", n_clusters=(2, 4), max_runs=3, rng=0) is None

    diagnostics = adata.uns["cluster_auto_k"]
    assert list(diagnostics["table"].index) == [1, 2, 3, 4, 5]
    assert diagnostics["best_k"] in scored(diagnostics["table"])
    assert list(adata.obs.columns) == ["cluster_auto_k"]
    assert adata.obs["cluster_auto_k"].dtype == "category"
    categories = list(adata.obs["cluster_auto_k"].cat.categories)
    assert categories == [str(i) for i in sorted(map(int, categories))], "string labels in numeric order, like leiden"


def test_cluster_auto_k_keeps_empty_components_as_categories(monkeypatch):
    from squidpy.gr import _cluster_auto_k

    real = _cluster_auto_k.sweep_auto_k

    def sweep_leaving_k3_empty(*args, **kwargs):
        result = real(*args, **kwargs)
        result.labels[3][:] = np.minimum(result.labels[3], 1)  # component 2 never used
        return result._replace(best_k=3)

    monkeypatch.setattr(_cluster_auto_k, "sweep_auto_k", sweep_leaving_k3_empty)
    adata = AnnData(make_blobs())
    cluster_auto_k(adata, use_rep="X", n_clusters=(2, 4), max_runs=2, rng=0)
    assert list(adata.obs["cluster_auto_k"].cat.categories) == ["0", "1", "2"], "as CellCharter, every K is a category"


def test_cluster_auto_k_uns_roundtrips_h5ad(tmp_path):
    adata = AnnData(make_blobs())
    cluster_auto_k(adata, use_rep="X", n_clusters=(2, 4), max_runs=2, rng=0)
    adata.write_h5ad(tmp_path / "autok.h5ad")
    original, reloaded = adata.uns["cluster_auto_k"], read_h5ad(tmp_path / "autok.h5ad").uns["cluster_auto_k"]

    assert set(reloaded) == {"table", "stability", "best_k", "n_runs", "converged"}
    assert_frame_equal(reloaded["table"], original["table"])
    assert_frame_equal(reloaded["stability"], original["stability"])
    assert list(original["stability"].index) == [2, 3, 4], "rows are the scored K"
    assert (reloaded["best_k"], reloaded["n_runs"], reloaded["converged"]) == (
        original["best_k"],
        original["n_runs"],
        original["converged"],
    )


def test_cluster_auto_k_copy():
    adata = AnnData(make_blobs())
    out = cluster_auto_k(adata, use_rep="X", n_clusters=(2, 4), max_runs=3, rng=0, copy=True)

    assert list(out.obs.columns) == ["cluster_auto_k"]
    assert "cluster_auto_k" in out.uns
    assert adata.obs.columns.empty
    assert not adata.uns


def test_cluster_auto_k_uses_the_requested_representation():
    adata = AnnData(np.zeros((90, 2)))
    adata.obsm["X_embedding"] = make_blobs()
    cluster_auto_k(adata, n_clusters=(2, 4), use_rep="X_embedding", max_runs=2, rng=0)
    assert adata.uns["cluster_auto_k"]["best_k"] in scored(adata.uns["cluster_auto_k"]["table"])


def test_cluster_auto_k_rejects_a_missing_representation():
    adata = AnnData(make_blobs())
    with pytest.raises(KeyError, match=r"not_there"):
        cluster_auto_k(adata, n_clusters=(2, 4), use_rep="not_there", max_runs=2, rng=0)


def test_cluster_auto_k_rejects_sparse_input():
    adata = AnnData(csr_matrix(make_blobs()))
    with pytest.raises(TypeError, match=r"does not support sparse input"):
        cluster_auto_k(adata, use_rep="X", n_clusters=(2, 4), max_runs=2, rng=0)


def test_cluster_auto_k_store_labels_adds_every_fitted_k():
    adata = AnnData(make_blobs())
    cluster_auto_k(adata, use_rep="X", n_clusters=(2, 3), max_runs=2, rng=0, store_labels=True)

    fitted = list(adata.uns["cluster_auto_k"]["table"].index)
    assert list(adata.obs.columns) == ["cluster_auto_k", *(f"cluster_auto_k_k{k}" for k in fitted)]

    best_k = adata.uns["cluster_auto_k"]["best_k"]
    pd.testing.assert_series_equal(
        adata.obs["cluster_auto_k"], adata.obs[f"cluster_auto_k_k{best_k}"], check_names=False
    )


def test_cluster_auto_k_honours_key_added():
    adata = AnnData(make_blobs())
    cluster_auto_k(adata, use_rep="X", n_clusters=(2, 3), max_runs=2, rng=0, key_added="sweep")
    assert list(adata.obs.columns) == ["sweep"]
    assert "sweep" in adata.uns


def test_cluster_auto_k_takes_spatialdata():
    from spatialdata import SpatialData
    from spatialdata.models import TableModel

    sdata = SpatialData(tables={"table": TableModel.parse(AnnData(make_blobs()))})
    cluster_auto_k(sdata, use_rep="X", n_clusters=(2, 4), max_runs=2, rng=0, table_key="table")
    expected = cluster_auto_k(AnnData(make_blobs()), use_rep="X", n_clusters=(2, 4), max_runs=2, rng=0, copy=True)
    np.testing.assert_array_equal(sdata.tables["table"].obs["cluster_auto_k"], expected.obs["cluster_auto_k"])


class _NoiseLabeler(KMeans):
    def predict(self, X, sample_weight=None):
        return np.full(len(X), -1)


class _OneBasedLabeler(KMeans):
    def predict(self, X, sample_weight=None):
        return super().predict(X) + 1


def test_sweep_rejects_labels_outside_range():
    with pytest.raises(ValueError, match=r"returned labels outside 0\.\.\d+"):
        sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=2, clusterer=_NoiseLabeler(n_init=1), rng=0)
    with pytest.raises(ValueError, match=r"returned labels outside 0\.\.\d+"):
        sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=2, clusterer=_OneBasedLabeler(n_init=1), rng=0)


# parity with the reference implementation


def _cc_expand(n_clusters: tuple[int, int] | list[int]) -> list[int]:
    # cellcharter main, ClusterAutoK.__init__ and _fit_n_clusters
    ks = range(n_clusters[0], n_clusters[1] + 1) if isinstance(n_clusters, tuple) else n_clusters
    ks = sorted({int(k) for k in ks if k >= 2})
    return sorted({k + d for k in ks for d in (-1, 0, 1)})


def _cc_mirror(stability: list[float], n_clusters: list[int]) -> np.ndarray:
    # cellcharter 0.3.7, ClusterAutoK._mirror_stability
    chunks = [stability[i : i + len(n_clusters) - 1] for i in range(0, len(stability), len(n_clusters) - 1)]
    transposed = list(map(list, zip(*chunks, strict=True)))
    return np.array([transposed[i] + transposed[i - 1] for i in range(1, len(transposed))])


def _cc_best_k(stability: np.ndarray, n_clusters: list[int]) -> int:
    # cellcharter 0.3.7, ClusterAutoK.best_k
    means = np.array([np.mean(stability[k]) for k in range(len(n_clusters[1:-1]))])
    return n_clusters[int(np.argmax(means)) + 1]


@pytest.mark.filterwarnings("ignore:stability cannot be computed")
@pytest.mark.parametrize("n_clusters", [(2, 10), (2, 4), (3, 7), (1, 5), [5, 10, 15, 20], [2, 3, 7]])
def test_expand_n_clusters_matches_cellcharter(n_clusters):
    assert expand_n_clusters(n_clusters)[0] == _cc_expand(n_clusters)


def test_cluster_auto_k_scores_every_listed_k():
    """Non-consecutive K values are each scored against K-1 and K+1, as in CellCharter."""
    adata = AnnData(make_blobs())
    cluster_auto_k(adata, use_rep="X", n_clusters=[2, 5], max_runs=2, rng=0)
    table = adata.uns["cluster_auto_k"]["table"]
    assert list(table.index) == [1, 2, 3, 4, 5, 6]
    assert table["stability_mean"].notna().tolist() == [False, True, False, False, True, False]
    assert adata.uns["cluster_auto_k"]["stability"].shape[0] == 2


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
    # guards mirror_stability/_score_block/the seeding/the init against silent drift. these are
    # the values this rng produces, not the ground truth: with the kmeans init, these three blobs
    # select K=4 for 17 of rngs 0-19 at max_runs=5.
    result = sweep_auto_k(make_blobs(), list(range(1, 7)), max_runs=5, rng=0)
    assert result.best_k == 4
    np.testing.assert_allclose(
        result.table["stability_mean"].to_numpy(),
        [np.nan, 0.742245, 0.831742, 0.89621, 0.807942, np.nan],
        rtol=1e-5,
    )


def test_seed_is_keyed_by_run_and_k_not_by_position():
    X = make_blobs()
    short = sweep_auto_k(X, [1, 2, 3, 4, 5], max_runs=3, rng=42)
    long = sweep_auto_k(X, [1, 2, 3, 4, 5, 6], max_runs=3, rng=42)
    for k in short.labels:
        np.testing.assert_array_equal(short.labels[k], long.labels[k])


def test_cluster_auto_k_takes_a_clusterer():
    """Any KClusterer, not only the default mixture."""
    adata = AnnData(make_blobs())
    cluster_auto_k(
        adata, use_rep="X", n_clusters=(2, 4), clusterer=KMeans(n_init=1), max_runs=3, rng=0, key_added="kmeans_k"
    )

    assert adata.obs["kmeans_k"].notna().all()
    assert adata.uns["kmeans_k"]["best_k"] in scored(adata.uns["kmeans_k"]["table"])


def test_cluster_auto_k_rejects_a_clusterer_that_cannot_be_swept():
    """A clusterer without a seed gives one labeling per K however many runs, so it is refused."""
    with pytest.raises(ValueError, match=r"AgglomerativeClustering takes no 'random_state'"):
        cluster_auto_k(
            AnnData(make_blobs()), use_rep="X", n_clusters=(2, 4), clusterer=AgglomerativeClustering(), rng=0
        )


def test_cluster_auto_k_takes_a_sparse_representation_with_a_clusterer():
    """The dense-only guard is about the default mixture, not about every clusterer."""
    adata = AnnData(csr_matrix(make_blobs()))
    cluster_auto_k(
        adata, use_rep="X", n_clusters=(2, 4), clusterer=KMeans(n_init=1), max_runs=3, rng=0, key_added="sparse_k"
    )
    assert adata.obs["sparse_k"].notna().all()
