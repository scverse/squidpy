import pytest

from anndata import AnnData

import numpy as np

from squidpy.gr import spatial_neighbors
from squidpy._constants._pkg_constants import Key


# todo add edge cases
@pytest.mark.parametrize(("n_rings", "n_neigh", "sum_dist"), [(1, 6, 0), (2, 18, 30), (3, 36, 84)])
def test_spatial_neighbors_visium(visium_adata: AnnData, n_rings: int, n_neigh: int, sum_dist: int):
    """
    check correctness of neighborhoods for visium coordinates
    """
    spatial_neighbors(visium_adata, n_rings=n_rings)
    assert visium_adata.obsp[Key.obsp.spatial_conn()][0].sum() == n_neigh
    assert visium_adata.uns[Key.uns.spatial_neighs()]["distances_key"] == Key.obsp.spatial_dist()
    if n_rings > 1:
        assert visium_adata.obsp[Key.obsp.spatial_dist()][0].sum() == sum_dist


def test_spatial_neighbors_non_visium(non_visium_adata: AnnData):
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

    spatial_neighbors(non_visium_adata, n_neigh=3, coord_type=None)
    spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].toarray()
    assert np.array_equal(spatial_graph, correct_knn_graph)

    spatial_neighbors(non_visium_adata, radius=5.0, coord_type=None)
    spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].toarray()
    assert np.array_equal(spatial_graph, correct_radius_graph)
