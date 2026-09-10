from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from fast_array_utils.conv import to_dense
from scanpy import settings
from sklearn.preprocessing import normalize

from squidpy._constants._pkg_constants import Key
from squidpy.gr import (
    _niche,
    centrality_scores,
    interaction_matrix,
    nhood_enrichment,
    nhood_entropy,
    spatial_neighbors_grid,
    spatial_neighbors_knn,
    spatial_neighbors_radius,
)
from squidpy.gr._nhood import (
    _hop_adjacencies,
    _nhood_aggregate,
    _nhood_blocks,
    _nhood_profile,
    nhood_aggregate,
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

    @pytest.mark.parametrize("backend", ["threading", "multiprocessing", "loky"])
    def test_parallel_works(self, adata: AnnData, backend: str):
        spatial_neighbors_grid(adata)

        nhood_enrichment(adata, cluster_key=_CK, n_jobs=2, n_perms=20, backend=backend)

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

    assert key in adata.uns_keys()
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
        assert key in adata.uns_keys()
        res = adata.uns[key]
    else:
        assert key not in adata.uns_keys()

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


class TestNhoodEntropy:
    @staticmethod
    def _grid(labels: list[str]) -> AnnData:
        side = int(round(len(labels) ** 0.5))
        assert side * side == len(labels)
        coords = np.array([(x, y) for y in range(side) for x in range(side)], dtype=float)
        adata = AnnData(np.zeros((len(labels), 2), dtype=np.float32), obsm={"spatial": coords})
        adata.obs["ct"] = pd.Categorical(labels)
        spatial_neighbors_grid(adata, n_neighs=8)
        return adata

    def test_homogeneous_neighborhood_scores_zero(self):
        adata = self._grid(["a"] * 36)
        np.testing.assert_allclose(nhood_entropy(adata, "ct", copy=True), 0.0)
        assert "ct_nhood_entropy" not in adata.obs

    def test_segregated_scores_below_scattered(self):
        labels = ["a"] * 50 + ["b"] * 50
        segregated = nhood_entropy(self._grid(labels), "ct", copy=True)
        scattered = nhood_entropy(self._grid(list(np.random.default_rng(0).permutation(labels))), "ct", copy=True)
        assert segregated.mean() < scattered.mean()

        # vertical stripes: an interior cell sees 2 of its own type and 6 of the other
        stripes = nhood_entropy(self._grid(["a", "b"] * 18), "ct", copy=True).to_numpy().reshape(6, 6)
        h = -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
        np.testing.assert_allclose(stripes[1:-1, 1:-1], h)

    def test_isolated_observation_is_zero_not_nan(self):
        adata = self._grid(["a", "b"] * 18)
        conn = adata.obsp["spatial_connectivities"].tolil()
        conn[0, :] = 0
        adata.obsp["spatial_connectivities"] = conn.tocsr()

        ent = nhood_entropy(adata, "ct", copy=True)
        assert not ent.isna().any()
        assert ent.iloc[0] == 0.0

    def test_writes_to_obs(self):
        adata = self._grid(["a", "b"] * 18)
        assert nhood_entropy(adata, "ct") is None
        np.testing.assert_allclose(adata.obs["ct_nhood_entropy"], nhood_entropy(adata, "ct", copy=True))


# `nhood_aggregate` is the one primitive the three niche embedders are built on; these pin
# each derivation to the embedder it replaced, since the flavors' output depends on it.


@pytest.fixture
def aggregate_adata() -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.random((80, 7)))
    adata.obsm["spatial"] = rng.random((80, 2)) * 10
    adata.obs["celltype"] = pd.Categorical(rng.choice(list("abcd"), 80))
    spatial_neighbors_knn(adata, n_neighs=6)
    return adata


@pytest.mark.parametrize(
    ("distance", "hop_weights", "abs_nhood"),
    [(1, None, False), (3, None, False), (3, [1.0, 0.5, 0.25], False), (2, None, True)],
)
def test_nhood_aggregate_derives_the_neighborhood_profile(
    aggregate_adata: AnnData, distance: int, hop_weights: list[float] | None, abs_nhood: bool
):
    """Categories summed over the matrix powers of the graph."""
    expected = _niche._nhood_profile_embedding(
        aggregate_adata,
        groups="celltype",
        spatial_connectivities_key="spatial_connectivities",
        scale=False,
        distance=distance,
        abs_nhood=abs_nhood,
        n_hop_weights=hop_weights,
    )

    got = _nhood_aggregate(
        aggregate_adata,
        groups="celltype",
        hops=range(1, distance + 1),
        hop_weights=hop_weights,
        aggregation="sum" if abs_nhood else "mean",
    )
    np.testing.assert_allclose(to_dense(got), expected)


def test_nhood_aggregate_derives_utag(aggregate_adata: AnnData):
    """One hop, mean-aggregated: a row-normalized graph times the features."""
    expected = normalize(aggregate_adata.obsp["spatial_connectivities"], norm="l1", axis=1) @ aggregate_adata.X
    np.testing.assert_allclose(to_dense(_nhood_aggregate(aggregate_adata, hops=(1,))), expected)


@pytest.mark.parametrize(("distance", "aggregation"), [(1, "mean"), (3, "mean"), (2, "variance")])
def test_nhood_aggregate_derives_cellcharter(aggregate_adata: AnnData, distance: int, aggregation: str):
    """Disjoint hop rings, concatenated, with the observation's own features as hop 0."""
    blocks = _nhood_blocks(aggregate_adata, hops=range(distance + 1), hop_mode="shell", aggregation=aggregation)
    got = np.hstack([to_dense(block) for block in blocks])
    assert got.shape == (aggregate_adata.n_obs, aggregate_adata.n_vars * (distance + 1))
    # hop 0 is the features themselves, not an aggregate of them
    np.testing.assert_allclose(got[:, : aggregate_adata.n_vars], aggregate_adata.X)


def test_nhood_aggregate_writes_obsm(aggregate_adata: AnnData):
    assert nhood_aggregate(aggregate_adata, groups="celltype", key_added="X_profile") is None
    assert aggregate_adata.obsm["X_profile"].shape == (aggregate_adata.n_obs, 4)


def test_nhood_aggregate_rejects_conflicting_features(aggregate_adata: AnnData):
    with pytest.raises(ValueError, match=r"at most one of 'groups', 'use_rep' and 'layer'"):
        nhood_aggregate(aggregate_adata, groups="celltype", layer="counts")


def test_nhood_aggregate_warns_on_short_hop_weights(aggregate_adata: AnnData, capsys):
    """A short list is more likely a mistake than an intention; see scverse/squidpy#1277."""
    # scanpy's logger needs pointing at the captured stream, as in `test_ligrec.py`
    settings.logfile = sys.stderr
    _nhood_aggregate(aggregate_adata, groups="celltype", hops=(1, 2, 3), hop_weights=[1.0])
    assert "padding with 1.0" in capsys.readouterr().err


def test_nhood_aggregate_rejects_too_many_hop_weights(aggregate_adata: AnnData):
    with pytest.raises(ValueError, match=r"'hop_weights' has 4 values but there are 2 hops"):
        _nhood_aggregate(aggregate_adata, groups="celltype", hops=(1, 2), hop_weights=[1.0, 1.0, 1.0, 1.0])


def test_nhood_aggregate_excludes_unassigned_neighbours(aggregate_adata: AnnData):
    """An observation with no category is missing data, not a neighbor of no type.

    It leaves each neighborhood's denominator, so the shares still sum to 1. Every
    observation being labeled makes the two denominators equal, so this needs unassigned
    ones to test anything at all.
    """
    labels = aggregate_adata.obs["celltype"].copy()
    labels.iloc[[3, 17, 42]] = np.nan
    aggregate_adata.obs["celltype"] = labels

    got = _nhood_aggregate(aggregate_adata, groups="celltype", aggregation="mean")
    expected = _nhood_profile(labels, aggregate_adata.obsp["spatial_connectivities"], normalize=True)
    np.testing.assert_allclose(to_dense(got), expected.to_numpy())
    np.testing.assert_allclose(np.asarray(to_dense(got)).sum(axis=1), 1.0)


def test_nhood_aggregate_masks_after_expanding_the_hops(aggregate_adata: AnnData):
    """An unassigned observation still relays paths; it only stops being counted.

    Masking the graph up front instead would drop it as a stepping stone too, which shows
    up from two hops out.
    """
    labels = aggregate_adata.obs["celltype"].copy()
    labels.iloc[[3, 17, 42]] = np.nan
    aggregate_adata.obs["celltype"] = labels
    adj = aggregate_adata.obsp["spatial_connectivities"]

    hops = (1, 2, 3)
    got = _nhood_aggregate(aggregate_adata, groups="celltype", hops=hops)
    by_hop = _hop_adjacencies(adj, hops, "power")
    expected = sum(_nhood_profile(labels, by_hop[hop], normalize=True).to_numpy() for hop in hops) / len(hops)
    np.testing.assert_allclose(to_dense(got), expected)


@pytest.mark.parametrize("max_hop", [1, 2, 3, 4])
@pytest.mark.parametrize("weight", [1.0, 0.5])
def test_bfs_shells_match_the_matmul_definition(max_hop: int, weight: float):
    rng = np.random.default_rng(0)
    points = np.vstack([rng.random((120, 2)) * 10, rng.random((120, 2)) * 10 + [60, 0], rng.random((4, 2)) + [30, 30]])
    adata = AnnData(X=np.zeros((len(points), 1), dtype=np.float32))
    adata.obsm["spatial"] = points
    spatial_neighbors_radius(adata, radius=1.6)
    adj = adata.obsp["spatial_connectivities"].astype(float) * weight

    # the definition the search replaced: boolean matmul minus everything already reached
    boolean = adj.astype(bool)
    hop, visited = boolean.copy(), boolean.copy()
    hop.setdiag(0)
    hop.eliminate_zeros()
    visited.setdiag(1)
    expected = {1: hop}
    for h in range(2, max_hop + 1):
        hop = (hop @ boolean) > visited
        visited = visited + hop
        expected[h] = hop

    got = _hop_adjacencies(adj, range(1, max_hop + 1), "shell")
    assert got[0] is None, "hop 0 is the observation itself, not a neighborhood"
    for h in range(1, max_hop + 1):
        assert (expected[h] != got[h]).nnz == 0, f"hop {h} differs"
