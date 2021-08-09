from typing import Tuple
import pytest

from anndata import AnnData

from scipy.sparse import isspmatrix_csr
import numpy as np

from squidpy.gr import spatial_neighbors
from squidpy._constants._pkg_constants import Key


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

    # TODO: add edge cases
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

    @pytest.mark.parametrize(("n_rings", "n_neigh", "sum_neigh"), [(1, 4, 4), (2, 4, 12), (3, 4, 24)])
    def test_spatial_neighbors_squaregrid(self, adata_squaregrid: AnnData, n_rings: int, n_neigh: int, sum_neigh: int):
        """
        check correctness of neighborhoods for visium coordinates
        """
        adata = adata_squaregrid
        spatial_neighbors(adata, n_neighs=n_neigh, n_rings=n_rings, coord_type="grid")
        assert np.diff(adata.obsp[Key.obsp.spatial_conn()].indptr).max() == sum_neigh
        assert adata.uns[Key.uns.spatial_neighs()]["distances_key"] == Key.obsp.spatial_dist()

    @pytest.mark.parametrize("type_rings", [("grid", 1), ("grid", 6), ("generic", 1)])
    @pytest.mark.parametrize("set_diag", [False, True])
    def test_set_diag(self, adata_squaregrid: AnnData, set_diag: bool, type_rings: Tuple[str, int]):
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

    @pytest.mark.parametrize("set_diag", [False, True])
    @pytest.mark.parametrize("radius", [(0, np.inf), (2.0, 4.0), (-42, -420), (100, 200)])
    def test_radius_min_max(self, non_visium_adata: AnnData, radius: Tuple[float, float], set_diag: bool):
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
