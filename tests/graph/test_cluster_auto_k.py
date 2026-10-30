from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from squidpy.gr import cluster_auto_k
from squidpy.gr._autok import check_k_clusterer, expand_n_clusters, mirror_stability, sweep_auto_k


def make_blobs(n_per_blob: int = 30, n_features: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(centre, 1.0, (n_per_blob, n_features)) for centre in (0.0, 8.0, -8.0)])


# resolving the requested K values


@pytest.mark.parametrize(
    ("n_clusters", "fit", "scored"),
    [
        ((2, 10), list(range(1, 12)), list(range(2, 11))),
        ([2, 5, 9], [1, 2, 3, 4, 5, 6, 8, 9, 10], [2, 5, 9]),
        ([5, 2, 5, 3], [1, 2, 3, 4, 5, 6], [2, 3, 5]),  # sorted and deduplicated
        ((1, 3), [1, 2, 3, 4], [2, 3]),  # K < 2 cannot be scored
    ],
)
@pytest.mark.filterwarnings("ignore:stability cannot be computed")
def test_expand_n_clusters(n_clusters, fit, scored):
    assert expand_n_clusters(n_clusters) == (fit, scored)


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


def test_expand_n_clusters_warns_on_k_below_two():
    with pytest.warns(UserWarning, match=r"K < 2"), pytest.raises(ValueError):
        expand_n_clusters([0, 1])


# which clusterers can be swept


@pytest.mark.parametrize(
    ("clusterer", "expected"),
    [
        (GaussianMixture(), "n_components"),
        (KMeans(), "n_clusters"),
        # also takes `n_components` (its embedding width), which must not be mistaken for K
        (SpectralClustering(), "n_clusters"),
    ],
)
def test_check_k_clusterer_finds_the_k_parameter(clusterer, expected):
    assert check_k_clusterer(clusterer) == expected


@pytest.mark.parametrize(
    ("clusterer", "match"),
    [
        (object(), r"is not a clusterer squidpy can re-fit"),
        (AffinityPropagation(), r"takes neither 'n_components' nor 'n_clusters'"),
        (AgglomerativeClustering(), r"takes no 'random_state'"),
    ],
)
def test_check_k_clusterer_rejects_what_cannot_be_swept(clusterer, match):
    with pytest.raises(TypeError, match=match):
        check_k_clusterer(clusterer)


# the sweep


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_runs": 1}, r"at least 2 runs"),
        ({"scored": [1]}, r"needs a fitted neighbour on both sides"),
    ],
)
def test_sweep_rejects_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sweep_auto_k(make_blobs(), [1, 2, 3], **kwargs)


def test_sweep_convergence():
    X, ks = make_blobs(), [1, 2, 3, 4]
    converged = sweep_auto_k(X, ks, max_runs=10, convergence_tol=np.inf, rng=0)
    assert converged.converged
    assert converged.n_runs == 3, "the first curve to compare against exists after run 2"

    exhausted = sweep_auto_k(X, ks, max_runs=3, convergence_tol=0.0, rng=0)
    assert not exhausted.converged
    assert exhausted.n_runs == 3


def test_sweep_notes_the_k_of_a_failed_fit():
    with pytest.raises(ValueError) as excinfo:
        sweep_auto_k(np.zeros((2, 2)), [2, 3, 4], max_runs=2, rng=0)  # fewer samples than components
    assert any(note.startswith("while fitting K=") for note in excinfo.value.__notes__)


def _relabeled(relabel):
    """A KMeans whose labels go through *relabel*; a local class still clones, by type."""

    class _Relabeled(KMeans):
        def fit_predict(self, X, y=None, sample_weight=None):
            return relabel(super().fit_predict(X))

    return _Relabeled(n_init=1)


@pytest.mark.parametrize(
    "relabel",
    [lambda labels: np.full_like(labels, -1), lambda labels: labels + 1, lambda labels: labels.astype(str)],
    ids=["noise", "one-based", "strings"],
)
def test_sweep_rejects_labels_outside_range(relabel):
    with pytest.raises(ValueError, match=r"returned labels that are not integers in 0\.\.\d+"):
        sweep_auto_k(make_blobs(), [1, 2, 3], max_runs=2, clusterer=_relabeled(relabel), rng=0)


def test_sweep_pins_its_stability_curve():
    # guards mirror_stability/_score_block/the seeding/the init against silent drift, and with
    # it reproducibility from the rng. these are the values this rng produces, not the ground
    # truth: with the kmeans init, these three blobs select K=4 for 17 of rngs 0-19 at max_runs=5.
    result = sweep_auto_k(make_blobs(), list(range(1, 7)), max_runs=5, rng=0)
    assert result.best_k == 4
    np.testing.assert_allclose(
        result.table["stability_mean"].to_numpy(),
        [np.nan, 0.742245, 0.831742, 0.89621, 0.807942, np.nan],
        rtol=1e-5,
    )


def test_seed_is_keyed_by_run_and_k_not_by_position():
    X = make_blobs()
    short = sweep_auto_k(X, [1, 2, 3, 4, 5], max_runs=2, rng=42)
    long = sweep_auto_k(X, [1, 2, 3, 4, 5, 6], max_runs=2, rng=42)
    for k in short.labels:
        np.testing.assert_array_equal(short.labels[k], long.labels[k])


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


# the public entry point


def _auto_k(data, **kwargs):
    """`cluster_auto_k` on dense blobs with a small, fixed sweep; *kwargs* override."""
    return cluster_auto_k(data, **{"use_rep": "X", "n_clusters": (2, 4), "max_runs": 2, "rng": 0} | kwargs)


def test_cluster_auto_k_stores_labels_and_diagnostics(tmp_path):
    adata = AnnData(make_blobs())
    assert _auto_k(adata, key_added="sweep") is None

    diagnostics = adata.uns["sweep"]
    table = diagnostics["table"]
    assert set(diagnostics) == {"table", "stability", "best_k", "n_runs", "converged"}
    assert list(table.columns) == ["stability_mean", "stability_std", "fit_loss"]
    assert table.index.name == "k"
    # the +-1 halo is fitted, so it has a fit_loss, but is never scored
    assert table["stability_mean"].notna().tolist() == [False, True, True, True, False]
    assert table["fit_loss"].notna().all()
    assert list(diagnostics["stability"].index) == [2, 3, 4], "rows are the scored K"
    assert list(adata.obs.columns) == ["sweep"]
    categories = list(adata.obs["sweep"].cat.categories)
    assert categories == [str(i) for i in range(diagnostics["best_k"])], "string labels in numeric order, like leiden"

    adata.write_h5ad(tmp_path / "autok.h5ad")
    reloaded = read_h5ad(tmp_path / "autok.h5ad").uns["sweep"]
    assert_frame_equal(reloaded["table"], table)
    assert_frame_equal(reloaded["stability"], diagnostics["stability"])
    for key in ("best_k", "n_runs", "converged"):
        assert reloaded[key] == diagnostics[key]


def test_cluster_auto_k_scores_every_listed_k():
    """Non-consecutive K values are each scored against K-1 and K+1, as in CellCharter."""
    adata = AnnData(make_blobs())
    _auto_k(adata, n_clusters=[2, 5, 9])
    table = adata.uns["cluster_auto_k"]["table"]
    assert list(table.index) == [1, 2, 3, 4, 5, 6, 8, 9, 10]
    assert table.index[table["stability_mean"].notna()].tolist() == [2, 5, 9]
    assert list(adata.uns["cluster_auto_k"]["stability"].index) == [2, 5, 9]


def test_cluster_auto_k_copy_with_every_fitted_k():
    adata = AnnData(make_blobs())
    out = _auto_k(adata, n_clusters=(2, 3), store_labels=True, copy=True)
    assert adata.obs.columns.empty and not adata.uns, "copy=True leaves adata alone"

    fitted = list(out.uns["cluster_auto_k"]["table"].index)
    assert list(out.obs.columns) == ["cluster_auto_k", *(f"cluster_auto_k_k{k}" for k in fitted)]
    best_k = out.uns["cluster_auto_k"]["best_k"]
    pd.testing.assert_series_equal(out.obs["cluster_auto_k"], out.obs[f"cluster_auto_k_k{best_k}"], check_names=False)


@pytest.mark.parametrize("source", ["obsm", "spatialdata"])
def test_cluster_auto_k_reads_the_requested_data(source):
    from spatialdata import SpatialData
    from spatialdata.models import TableModel

    if source == "obsm":
        data = adata = AnnData(np.zeros((90, 2)), obsm={"X_embedding": make_blobs()})
        _auto_k(data, use_rep="X_embedding")
    else:
        data = SpatialData(tables={"table": TableModel.parse(AnnData(make_blobs()))})
        _auto_k(data, table_key="table")
        adata = data.tables["table"]
    expected = _auto_k(AnnData(make_blobs()), copy=True)
    np.testing.assert_array_equal(adata.obs["cluster_auto_k"], expected.obs["cluster_auto_k"])


@pytest.mark.parametrize(
    ("adata", "use_rep", "error", "match"),
    [
        (AnnData(make_blobs()), "not_there", KeyError, r"not_there"),
        (AnnData(csr_matrix(make_blobs())), "X", TypeError, r"dense data is required"),
        (AnnData(obs=pd.DataFrame(index=[str(i) for i in range(90)])), "X", ValueError, r"'adata.X' is None"),
    ],
)
def test_cluster_auto_k_rejects_bad_input(adata, use_rep, error, match):
    with pytest.raises(error, match=match):
        _auto_k(adata, use_rep=use_rep)


def test_cluster_auto_k_keeps_empty_components_as_categories():
    adata = AnnData(make_blobs())
    _auto_k(adata, n_clusters=[3], clusterer=_relabeled(lambda labels: np.minimum(labels, 1)))
    assert adata.uns["cluster_auto_k"]["best_k"] == 3
    assert list(adata.obs["cluster_auto_k"].cat.categories) == ["0", "1", "2"], "as CellCharter, every K is a category"
    assert set(adata.obs["cluster_auto_k"]) == {"0", "1"}


def test_cluster_auto_k_takes_a_clusterer_without_mutating_it():
    """KMeans also takes a sparse representation, which the default mixture does not."""
    clusterer = KMeans(n_init=1)
    before = clusterer.get_params()
    adata = AnnData(csr_matrix(make_blobs()))
    _auto_k(adata, clusterer=clusterer)

    assert len(adata.obs["cluster_auto_k"].cat.categories) == adata.uns["cluster_auto_k"]["best_k"]
    assert clusterer.get_params() == before, "every fit is on a clone"
