from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from squidpy._constants._pkg_constants import Key
from squidpy.gr import (
    centrality_scores,
    interaction_matrix,
    nhood_enrichment,
    nhood_entropy,
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

    @pytest.mark.parametrize("backend", ["threading", "multiprocessing", "loky"])
    def test_parallel_works(self, adata: AnnData, backend: str):
        spatial_neighbors_grid(adata)

        nhood_enrichment(adata, cluster_key=_CK, n_jobs=2, n_perms=20, backend=backend)

        self._assert_common(adata)

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_reproducibility(self, adata: AnnData, n_jobs: int):
        spatial_neighbors_grid(adata)

        res1 = nhood_enrichment(adata, cluster_key=_CK, seed=42, n_jobs=n_jobs, n_perms=20, copy=True)
        res2 = nhood_enrichment(adata, cluster_key=_CK, seed=42, n_jobs=n_jobs, n_perms=20, copy=True)
        res3 = nhood_enrichment(adata, cluster_key=_CK, seed=43, n_jobs=n_jobs, n_perms=20, copy=True)

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

        kw = {"cluster_key": _CK, "seed": 42, "n_perms": 20, "copy": True}
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
