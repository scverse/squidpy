from typing import Tuple

import anndata as ad
import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import isspmatrix_csr
from squidpy._constants._pkg_constants import Key
from squidpy.gr import spatial_neighbors
from squidpy.gr._build import _build_connectivity


class TestSpatialNeighbors:
    # ground-truth Delaunay distances
    _gt_ddist = np.array(
        [
            [0.0, 2.0, 0.0, 4.12310563],
            [2.0, 0.0, 6.32455532, 5.0],
            [0.0, 6.32455532, 0.0, 5.38516481],
            [4.12310563, 5.0, 5.38516481, 0.0],
        ]
    )
    # ground-truth Delaunay graph
    _gt_dgraph = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])

    @staticmethod
    def _adata_concat(adata1, adata2):
        adata2.uns["spatial"] = {"library2": None}  # needed to trigger grid building
        batch1, batch2 = "batch1", "batch2"
        adata_concat = ad.concat(
            {batch1: adata1, batch2: adata2},
            label="library_id",
            uns_merge="unique",
        )
        return adata_concat, batch1, batch2

    # TODO: add edge cases
    # TODO(giovp): test with reshuffling
    @pytest.mark.parametrize(("n_rings", "n_neigh", "sum_dist"), [(1, 6, 0), (2, 18, 30), (3, 36, 84)])
    def test_spatial_neighbors_visium(self, visium_adata: AnnData, n_rings: int, n_neigh: int, sum_dist: int):
        """
        check correctness of neighborhoods for visium coordinates
        """
        spatial_neighbors(visium_adata, n_rings=n_rings)
        assert visium_adata.obsp[Key.obsp.spatial_conn()][0].sum() == n_neigh
        assert visium_adata.uns[Key.uns.spatial_neighs()]["distances_key"] == Key.obsp.spatial_dist()
        if n_rings > 1:
            assert visium_adata.obsp[Key.obsp.spatial_dist()][0].sum() == sum_dist

        # test for library_key
        visium_adata2 = visium_adata.copy()
        adata_concat, batch1, batch2 = TestSpatialNeighbors._adata_concat(visium_adata, visium_adata2)
        spatial_neighbors(visium_adata2, n_rings=n_rings)
        spatial_neighbors(adata_concat, library_key="library_id", n_rings=n_rings)
        assert adata_concat.obsp[Key.obsp.spatial_conn()][0].sum() == n_neigh
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch1].obsp[Key.obsp.spatial_conn()].A,
            visium_adata.obsp[Key.obsp.spatial_conn()].A,
        )
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch2].obsp[Key.obsp.spatial_conn()].A,
            visium_adata2.obsp[Key.obsp.spatial_conn()].A,
        )

    @pytest.mark.parametrize(("n_rings", "n_neigh", "sum_neigh"), [(1, 4, 4), (2, 4, 12), (3, 4, 24)])
    def test_spatial_neighbors_squaregrid(self, adata_squaregrid: AnnData, n_rings: int, n_neigh: int, sum_neigh: int):
        """
        check correctness of neighborhoods for visium coordinates
        """
        adata = adata_squaregrid
        spatial_neighbors(adata, n_neighs=n_neigh, n_rings=n_rings, coord_type="grid")
        assert np.diff(adata.obsp[Key.obsp.spatial_conn()].indptr).max() == sum_neigh
        assert adata.uns[Key.uns.spatial_neighs()]["distances_key"] == Key.obsp.spatial_dist()

        # test for library_key
        adata2 = adata.copy()
        adata_concat, batch1, batch2 = TestSpatialNeighbors._adata_concat(adata, adata2)
        spatial_neighbors(adata2, n_neighs=n_neigh, n_rings=n_rings, coord_type="grid")
        spatial_neighbors(adata_concat, library_key="library_id", n_neighs=n_neigh, n_rings=n_rings, coord_type="grid")
        assert np.diff(adata_concat.obsp[Key.obsp.spatial_conn()].indptr).max() == sum_neigh
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch1].obsp[Key.obsp.spatial_conn()].A,
            adata.obsp[Key.obsp.spatial_conn()].A,
        )
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch2].obsp[Key.obsp.spatial_conn()].A,
            adata2.obsp[Key.obsp.spatial_conn()].A,
        )

    @pytest.mark.parametrize("type_rings", [("grid", 1), ("grid", 6), ("generic", 1)])
    @pytest.mark.parametrize("set_diag", [False, True])
    def test_set_diag(self, adata_squaregrid: AnnData, set_diag: bool, type_rings: tuple[str, int]):
        typ, n_rings = type_rings
        spatial_neighbors(adata_squaregrid, coord_type=typ, set_diag=set_diag, n_rings=n_rings)
        G = adata_squaregrid.obsp[Key.obsp.spatial_conn()]
        D = adata_squaregrid.obsp[Key.obsp.spatial_dist()]

        np.testing.assert_array_equal(G.diagonal(), float(set_diag))
        np.testing.assert_array_equal(D.diagonal(), 0.0)

    def test_spatial_neighbors_non_visium(self, non_visium_adata: AnnData):
        """
        check correctness of neighborhoods for non-visium coordinates
        """
        correct_knn_graph = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )

        correct_radius_graph = np.array(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )

        spatial_neighbors(non_visium_adata, n_neighs=3, coord_type=None)
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].A

        np.testing.assert_array_equal(spatial_graph, correct_knn_graph)

        spatial_neighbors(non_visium_adata, radius=5.0, coord_type=None)
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].A

        np.testing.assert_array_equal(spatial_graph, correct_radius_graph)

        spatial_neighbors(non_visium_adata, delaunay=True, coord_type=None)
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].A
        spatial_dist = non_visium_adata.obsp[Key.obsp.spatial_dist()].A

        np.testing.assert_array_equal(spatial_graph, self._gt_dgraph)
        np.testing.assert_allclose(spatial_dist, self._gt_ddist)

        # test for library_key
        non_visium_adata2 = non_visium_adata.copy()
        adata_concat, batch1, batch2 = TestSpatialNeighbors._adata_concat(non_visium_adata, non_visium_adata2)
        spatial_neighbors(adata_concat, library_key="library_id", delaunay=True, coord_type=None)
        spatial_neighbors(non_visium_adata2, delaunay=True, coord_type=None)

        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch1].obsp[Key.obsp.spatial_conn()].A,
            non_visium_adata.obsp[Key.obsp.spatial_conn()].A,
        )
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch2].obsp[Key.obsp.spatial_conn()].A,
            non_visium_adata2.obsp[Key.obsp.spatial_conn()].A,
        )

    @pytest.mark.parametrize("set_diag", [False, True])
    @pytest.mark.parametrize("radius", [(0, np.inf), (2.0, 4.0), (-42, -420), (100, 200)])
    def test_radius_min_max(self, non_visium_adata: AnnData, radius: tuple[float, float], set_diag: bool):
        gt_ddist = self._gt_ddist.copy()
        gt_dgraph = self._gt_dgraph.copy()

        minn, maxx = sorted(radius)
        mask = (gt_ddist < minn) | (gt_ddist > maxx)
        gt_ddist[mask] = 0.0
        gt_dgraph[mask] = 0.0
        if set_diag:
            ixs = np.arange(len(gt_dgraph))
            gt_dgraph[ixs, ixs] = 1.0

        spatial_neighbors(non_visium_adata, delaunay=True, coord_type=None, radius=radius, set_diag=set_diag)
        spatial_dist = non_visium_adata.obsp[Key.obsp.spatial_dist()].A
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].A

        np.testing.assert_allclose(spatial_graph, gt_dgraph)
        np.testing.assert_allclose(spatial_dist, gt_ddist)

    def test_copy(self, non_visium_adata: AnnData):
        conn, dist = spatial_neighbors(non_visium_adata, delaunay=True, coord_type=None, copy=True)

        assert isspmatrix_csr(conn)
        assert isspmatrix_csr(dist)
        assert Key.obsp.spatial_conn() not in non_visium_adata.obsp
        assert Key.obsp.spatial_dist() not in non_visium_adata.obsp
        np.testing.assert_allclose(dist.A, self._gt_ddist)
        np.testing.assert_allclose(conn.A, self._gt_dgraph)

    @pytest.mark.parametrize("percentile", [99.0, 95.0])
    def test_percentile_filtering(self, adata_hne: AnnData, percentile: float, coord_type="generic"):
        conn, dist = spatial_neighbors(adata_hne, coord_type=coord_type, copy=True)
        conn_filtered, dist_filtered = spatial_neighbors(
            adata_hne, coord_type=coord_type, percentile=percentile, copy=True
        )

        # check whether there are less connectivities in the filtered graph and whether the max distance is smaller
        assert not ((conn != conn_filtered).nnz == 0)
        assert dist.max() > dist_filtered.max()

        Adj, Dst = _build_connectivity(adata_hne.obsm["spatial"], n_neighs=6, return_distance=True, set_diag=False)
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0
        Adj.eliminate_zeros()
        Dst.eliminate_zeros()

        assert dist_filtered.max() == Dst.max()

    @pytest.mark.parametrize("n_neighs", [5, 10, 20])
    def test_spatial_neighbors_generic(self, n_neighs: int):
        rng = np.random.default_rng(42)
        adata = ad.AnnData(shape=(512, 1))
        adata.obsm["spatial"] = rng.random(size=(512, 2))

        spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic", radius=None)
        graph = adata.obsp[Key.obsp.spatial_conn()]
        actual = np.array(graph.sum(axis=1)).flatten()

        np.testing.assert_array_equal(actual, n_neighs)
