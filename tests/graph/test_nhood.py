from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from squidpy._constants._pkg_constants import Key
from squidpy.gr import (
    centrality_scores,
    interaction_matrix,
    nhood_enrichment,
    spatial_neighbors_grid,
)

_CK = "leiden"


class TestNhoodEnrichment:
    def _assert_common(self, adata: AnnData):
        key = Key.uns.nhood_enrichment(_CK)
        assert adata.uns[key]["zscore"].dtype == np.dtype("float64")
        assert adata.uns[key]["count"].dtype == np.dtype("uint32")
        assert adata.uns[key]["zscore"].shape[0] == adata.obs.leiden.cat.categories.shape[0]
        assert adata.uns[key]["count"].shape[0] == adata.obs.leiden.cat.categories.shape[0]

    def test_nhood_enrichment(self, adata: AnnData):
        spatial_neighbors_grid(adata)
        nhood_enrichment(adata, cluster_key=_CK)

        self._assert_common(adata)

    @pytest.mark.parametrize("param", ["numba_parallel", "backend"])
    def test_deprecated_params_are_ignored(self, adata: AnnData, param: str):
        """A deprecated argument is stripped before the call, so it cannot change the result."""
        spatial_neighbors_grid(adata)

        kw = {"cluster_key": _CK, "rng": 42, "n_perms": 20, "copy": True}
        expected = nhood_enrichment(adata, **kw)
        with pytest.warns(FutureWarning, match=rf"`{param}`.*is deprecated"):
            got = nhood_enrichment(adata, **kw, **{param: "loky" if param == "backend" else True})

        np.testing.assert_array_equal(got.zscore, expected.zscore)
        np.testing.assert_array_equal(got.counts, expected.counts)

    def test_no_deprecation_warning_by_default(self, adata: AnnData):
        spatial_neighbors_grid(adata)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            nhood_enrichment(adata, cluster_key=_CK, n_perms=20)

        self._assert_common(adata)

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_reproducibility(self, adata: AnnData, n_jobs: int):
        spatial_neighbors_grid(adata)

        res1 = nhood_enrichment(
            adata, cluster_key=_CK, rng=np.random.default_rng(42), n_jobs=n_jobs, n_perms=20, copy=True
        )
        res2 = nhood_enrichment(
            adata, cluster_key=_CK, rng=np.random.default_rng(42), n_jobs=n_jobs, n_perms=20, copy=True
        )
        res3 = nhood_enrichment(
            adata, cluster_key=_CK, rng=np.random.default_rng(43), n_jobs=n_jobs, n_perms=20, copy=True
        )

        assert len(res1) == len(res2)
        assert len(res2) == len(res3)

        # Test that the same seed produces the same results
        np.testing.assert_array_equal(res2.zscore, res1.zscore)
        np.testing.assert_array_equal(res2.counts, res1.counts)

        # Test that different seeds produce different z-scores but same counts
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(res3.zscore, res2.zscore)
        np.testing.assert_array_equal(res3.counts, res2.counts)

    def test_n_jobs_invariance(self, adata: AnnData):
        """The number of workers must not change the result (one seed is spawned per permutation).

        The default ``normalization='none'`` accumulates its moments in int64, where addition is
        associative, so the reduction is bit-identical regardless of thread count. See
        ``test_zscore_independent_of_n_jobs`` for the normalized modes, which are float64.
        """
        spatial_neighbors_grid(adata)

        kw = {"cluster_key": _CK, "rng": 42, "n_perms": 20, "copy": True}
        res_serial = nhood_enrichment(adata, n_jobs=1, **kw)
        res_parallel = nhood_enrichment(adata, n_jobs=2, **kw)

        np.testing.assert_array_equal(res_serial.zscore, res_parallel.zscore)
        np.testing.assert_array_equal(res_serial.counts, res_parallel.counts)


def test_centrality_scores(nhood_data: AnnData):
    adata = nhood_data
    centrality_scores(
        adata=adata,
        cluster_key=_CK,
        connectivity_key="spatial",
    )

    key = Key.uns.centrality_scores(_CK)

    assert key in adata.uns
    assert isinstance(adata.uns[key], pd.DataFrame)
    assert len(adata.obs[_CK].unique()) == adata.uns[key].shape[0]
    assert adata.uns[key]["degree_centrality"].dtype == np.dtype("float64")
    assert adata.uns[key]["average_clustering"].dtype == np.dtype("float64")
    assert adata.uns[key]["closeness_centrality"].dtype == np.dtype("float64")


def test_centrality_scores_networkx_parity(nhood_data: AnnData):
    # centrality_scores swapped networkx for rustworkx (+ a numba clustering kernel); pin the
    # numeric parity of all three group measures against networkx (still a dependency).
    import networkx as nx

    adata = nhood_data
    df = centrality_scores(adata, cluster_key=_CK, connectivity_key="spatial", copy=True)

    graph = nx.Graph(adata.obsp["spatial_connectivities"])
    clusters = adata.obs[_CK].values
    for cat in df.index:
        idx = list(np.where(clusters == cat)[0])
        np.testing.assert_allclose(df.loc[cat, "closeness_centrality"], nx.group_closeness_centrality(graph, idx))
        np.testing.assert_allclose(df.loc[cat, "degree_centrality"], nx.group_degree_centrality(graph, idx))
        np.testing.assert_allclose(df.loc[cat, "average_clustering"], nx.average_clustering(graph, idx))


@pytest.mark.parametrize("copy", [True, False])
def test_interaction_matrix_copy(nhood_data: AnnData, copy: bool):
    adata = nhood_data
    res = interaction_matrix(
        adata=adata,
        cluster_key=_CK,
        connectivity_key="spatial",
        copy=copy,
    )

    key = Key.uns.interaction_matrix(_CK)
    n_cls = adata.obs[_CK].nunique()

    if not copy:
        assert res is None
        assert key in adata.uns
        res = adata.uns[key]
    else:
        assert key not in adata.uns

    assert isinstance(res, np.ndarray)
    assert res.shape == (n_cls, n_cls)


@pytest.mark.parametrize("normalized", [True, False])
def test_interaction_matrix_normalize(nhood_data: AnnData, normalized: bool):
    adata = nhood_data
    res = interaction_matrix(
        adata=adata,
        cluster_key=_CK,
        connectivity_key="spatial",
        copy=True,
        normalized=normalized,
    )
    n_cls = adata.obs["leiden"].nunique()

    assert isinstance(res, np.ndarray)
    assert res.shape == (n_cls, n_cls)

    if normalized:
        np.testing.assert_allclose(res.sum(1), 1.0), res.sum(1)
    else:
        assert len(adata.obsp["spatial_connectivities"].data) == res.sum()


def test_interaction_matrix_values(adata_intmat: AnnData):
    result_weighted = interaction_matrix(adata_intmat, "cat", weights=True, copy=True)
    result_unweighted = interaction_matrix(adata_intmat, "cat", weights=False, copy=True)

    expected_weighted = np.array([[5, 1], [2, 3]])
    expected_unweighted = np.array([[4, 1], [2, 2]])

    np.testing.assert_array_equal(expected_weighted, result_weighted)
    np.testing.assert_array_equal(expected_unweighted, result_unweighted)


def test_interaction_matrix_nan_values(adata_intmat: AnnData):
    adata_intmat.obs.loc["0", "cat"] = np.nan
    result_weighted = interaction_matrix(adata_intmat, "cat", weights=True, copy=True)
    result_unweighted = interaction_matrix(adata_intmat, "cat", weights=False, copy=True)

    expected_weighted = np.array([[2, 1], [2, 3]])
    expected_unweighted = np.array([[1, 1], [2, 2]])

    np.testing.assert_array_equal(expected_weighted, result_weighted)
    np.testing.assert_array_equal(expected_unweighted, result_unweighted)


@pytest.mark.parametrize("normalization", ["none", "total", "conditional"])
def test_nhood_enrichment_normalization_modes(adata: AnnData, normalization: str):
    spatial_neighbors_grid(adata)
    result = nhood_enrichment(adata, cluster_key=_CK, normalization=normalization, n_jobs=1, n_perms=20, copy=True)

    z, count, ccr = result

    assert isinstance(z, np.ndarray)
    assert isinstance(count, np.ndarray)
    if normalization == "conditional":
        assert isinstance(ccr, np.ndarray)
        assert z.shape == ccr.shape
        assert count.shape == ccr.shape
    assert z.shape == count.shape
    assert z.shape[0] == adata.obs[_CK].cat.categories.shape[0]


def test_conditional_normalization_zero_division(adata: AnnData):
    adata = adata.copy()
    min_cells = 10
    adata.obs[_CK] = adata.obs[_CK].cat.add_categories("isolated")
    adata.obs.loc[adata.obs.index[0], _CK] = "isolated"
    spatial_neighbors_grid(adata)
    valid_clusters = [c for c, count in adata.obs[_CK].value_counts().items() if count >= min_cells]
    valid_idx = [i for i, cat in enumerate(adata.obs[_CK].cat.categories) if cat in valid_clusters]

    result = nhood_enrichment(adata, cluster_key=_CK, normalization="conditional", copy=True)
    assert result is not None
    zscore, count_normalized, conditional_ratio = result
    assert not np.any(np.isinf(zscore))
    assert not np.any(np.isinf(count_normalized))
    assert not np.any(np.isinf(conditional_ratio))
    assert not np.isnan(zscore[np.ix_(valid_idx, valid_idx)]).any()
    assert not np.isnan(count_normalized[np.ix_(valid_idx, valid_idx)]).any()
    assert not np.isnan(conditional_ratio[np.ix_(valid_idx, valid_idx)]).any()


def _asymmetric_adata(n: int = 200, n_cls: int = 4) -> AnnData:
    """An adata whose connectivity is deliberately asymmetric, so a transpose is visible."""
    rng = np.random.default_rng(0)
    adj = sp.random(n, n, density=0.03, format="csr", random_state=0)
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()
    assert (adj != adj.T).nnz > 0, "graph must be asymmetric for this test to mean anything"

    adata = AnnData(
        np.zeros((n, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {_CK: pd.Categorical(rng.integers(0, n_cls, n).astype(str))}, index=[f"c{i}" for i in range(n)]
        ),
        obsp={Key.obsp.spatial_conn(): adj},
    )
    return adata


@pytest.mark.parametrize(
    "convert",
    [sp.csr_matrix, sp.csr_array, sp.csc_matrix, sp.csc_array],
    ids=["csr_matrix", "csr_array", "csc_matrix", "csc_array"],
)
def test_sparse_formats_agree(convert):
    """CSC exposes ``indices``/``indptr`` too, but column-wise -- it must not yield the transpose."""
    adata = _asymmetric_adata()
    kw = {"cluster_key": _CK, "n_perms": 5, "rng": 0, "n_jobs": 1, "copy": True}
    reference = nhood_enrichment(adata, **kw)
    assert not np.array_equal(reference.counts, reference.counts.T), "asymmetry must reach the counts"

    adata.obsp[Key.obsp.spatial_conn()] = convert(adata.obsp[Key.obsp.spatial_conn()])
    np.testing.assert_array_equal(nhood_enrichment(adata, **kw).counts, reference.counts)


@pytest.mark.parametrize("densify", [np.asarray, np.asmatrix], ids=["ndarray", "matrix"])
def test_dense_connectivity_raises(densify):
    """A dense ``obsp`` has no ``indices``/``indptr``; refuse it instead of densifying or crashing."""
    adata = _asymmetric_adata()
    adata.obsp[Key.obsp.spatial_conn()] = densify(adata.obsp[Key.obsp.spatial_conn()].toarray())
    with pytest.raises(TypeError, match=r"to be a sparse matrix, found"):
        nhood_enrichment(adata, cluster_key=_CK, n_perms=5, rng=0, copy=True)


def _nan_cluster(adata: AnnData) -> None:
    adata.obs[_CK] = adata.obs[_CK].cat.add_categories("tmp")
    adata.obs.loc[adata.obs.index[0], _CK] = "tmp"
    adata.obs[_CK] = adata.obs[_CK].replace("tmp", np.nan).astype("category")


def _one_cluster(adata: AnnData) -> None:
    adata.obs[_CK] = pd.Categorical(["only"] * adata.n_obs)


@pytest.mark.parametrize(
    ("mutate", "kwargs", "match"),
    [
        (None, {"handle_nan": "nonsense"}, "Invalid `handle_nan` mode"),
        (None, {"normalization": "invalid_mode"}, "Invalid normalization mode"),
        (_nan_cluster, {}, "Found `NaN` values"),
        (_one_cluster, {}, "Expected at least `2` clusters"),
    ],
    ids=["handle_nan", "normalization", "nan_cluster", "one_cluster"],
)
def test_nhood_enrichment_rejects(adata: AnnData, mutate, kwargs, match: str):
    adata = adata.copy()
    spatial_neighbors_grid(adata)
    if mutate is not None:
        mutate(adata)

    with pytest.raises(ValueError, match=match):
        nhood_enrichment(adata, cluster_key=_CK, n_perms=5, copy=True, **kwargs)


def test_handle_nan_zero_replaces_undefined_zscores(adata: AnnData):
    """``'keep'`` leaves undefined enrichments as NaN; ``'zero'`` replaces them and nothing else."""
    spatial_neighbors_grid(adata)
    kw = {"cluster_key": _CK, "n_perms": 20, "rng": 0, "copy": True}

    kept = nhood_enrichment(adata, handle_nan="keep", **kw)
    assert np.isnan(kept.zscore).any(), "fixture should produce at least one undefined z-score"

    zeroed = nhood_enrichment(adata, handle_nan="zero", **kw)
    assert not np.isnan(zeroed.zscore).any()
    defined = ~np.isnan(kept.zscore)
    np.testing.assert_array_equal(zeroed.zscore[defined], kept.zscore[defined])


def test_interaction_matrix_all_nan_raises(adata: AnnData):
    adata = adata.copy()
    spatial_neighbors_grid(adata)
    adata.obs[_CK] = pd.Categorical([np.nan] * adata.n_obs, categories=["a", "b"])

    with pytest.raises(RuntimeError, match="none remain"):
        interaction_matrix(adata, cluster_key=_CK, copy=True)


@pytest.mark.parametrize("show_progress_bar", [False, True])
def test_centrality_scores_single_score(nhood_data: AnnData, show_progress_bar: bool):
    """A bare string selects one measure; ``show_progress_bar`` is what builds parallelize's queue."""
    df = centrality_scores(
        nhood_data, cluster_key=_CK, score="degree_centrality", show_progress_bar=show_progress_bar, copy=True
    )
    assert list(df.columns) == ["degree_centrality"]


def test_duplicate_entries_count_once():
    """scipy defines a repeated ``(i, j)`` as one edge whose value is the sum, not two edges."""
    adj = sp.csr_matrix((np.array([1.0, 1.0, 1.0, 1.0]), np.array([1, 1, 2, 0]), np.array([0, 3, 4, 4])), shape=(3, 3))
    before = (adj.nnz, adj.indices.copy(), adj.data.copy())
    adata = AnnData(
        np.zeros((3, 1), dtype=np.float32),
        obs=pd.DataFrame({_CK: pd.Categorical(["a", "b", "b"], categories=["a", "b"])}, index=list("xyz")),
        obsp={Key.obsp.spatial_conn(): adj},
    )

    result = nhood_enrichment(adata, cluster_key=_CK, n_perms=5, rng=0, copy=True)
    np.testing.assert_array_equal(result.counts, [[0, 2], [1, 0]])

    # canonicalizing must happen on a copy: `count_nonzero()` would have summed these in place
    stored = adata.obsp[Key.obsp.spatial_conn()]
    assert (stored.nnz, stored.indices.tolist(), stored.data.tolist()) == (
        before[0],
        before[1].tolist(),
        before[2].tolist(),
    )


def test_stored_zeros_are_not_edges():
    """Pruning a graph in place leaves explicit zeros behind; they must not count as neighbors."""
    adata = _asymmetric_adata(n=120, n_cls=3)
    adj = adata.obsp[Key.obsp.spatial_conn()].copy()

    rng = np.random.default_rng(0)
    adj.data[rng.random(adj.nnz) < 0.4] = 0.0  # prune, sparse-safe: structure and nnz unchanged
    adata.obsp[Key.obsp.spatial_conn()] = adj
    assert adj.nnz != adj.count_nonzero(), "the fixture must actually contain stored zeros"

    pruned = adj.copy()
    pruned.eliminate_zeros()
    expected = adata.copy()
    expected.obsp[Key.obsp.spatial_conn()] = pruned

    kw = {"cluster_key": _CK, "n_perms": 20, "rng": 0, "n_jobs": 1, "copy": True}
    np.testing.assert_array_equal(nhood_enrichment(adata, **kw).counts, nhood_enrichment(expected, **kw).counts)
    # and the caller's matrix is left as they gave it
    assert adata.obsp[Key.obsp.spatial_conn()].nnz == adj.nnz


def test_nan_library_key_raises():
    adata = _asymmetric_adata(n=60, n_cls=3)
    adata.obs["lib"] = pd.Categorical([np.nan] + ["s1"] * (adata.n_obs - 1), categories=["s1"])

    with pytest.raises(ValueError, match="Found `NaN` values"):
        nhood_enrichment(adata, cluster_key=_CK, library_key="lib", n_perms=5, copy=True)


@pytest.mark.parametrize("handle_nan", ["keep", "zero"])
def test_min_cell_count_excludes_clusters(adata: AnnData, handle_nan: str):
    """Dropped clusters are excluded, not measured: NaN whatever ``handle_nan`` says, and the
    clusters that survive must see exactly the graph they would if the dropped cells never existed.
    """
    adata = adata.copy()
    spatial_neighbors_grid(adata)
    cats = list(adata.obs[_CK].cat.categories)
    sizes = adata.obs[_CK].value_counts()
    threshold = 10

    dropped = [i for i, c in enumerate(cats) if sizes[c] < threshold]
    kept = [i for i, c in enumerate(cats) if sizes[c] >= threshold]
    assert dropped and len(kept) >= 2, f"fixture needs both sides of the threshold: {sizes.to_dict()}"

    with pytest.warns(UserWarning, match="were excluded"):
        got = nhood_enrichment(
            adata, cluster_key=_CK, min_cell_count=threshold, handle_nan=handle_nan, n_perms=20, rng=0, copy=True
        )

    for i in dropped:
        assert np.isnan(got.zscore[i, :]).all() and np.isnan(got.zscore[:, i]).all()
        assert (got.counts[i, :] == 0).all() and (got.counts[:, i] == 0).all()

    subset = adata[adata.obs[_CK].isin([cats[i] for i in kept])].copy()
    subset.obs[_CK] = subset.obs[_CK].cat.remove_unused_categories()
    expected = nhood_enrichment(subset, cluster_key=_CK, n_perms=20, rng=0, copy=True)
    np.testing.assert_array_equal(got.counts[np.ix_(kept, kept)], expected.counts)
