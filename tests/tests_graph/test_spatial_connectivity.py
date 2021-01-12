from anndata import AnnData

import numpy as np

from squidpy.gr import spatial_neighbors


# todo add edge cases
def test_spatial_neighbors_visium(visium_adata: AnnData):
    """
    check correctness of neighborhoods for visium coordinates
    """
    for i, n_neigh in enumerate((6, 18, 36)):
        spatial_neighbors(visium_adata, n_rings=i + 1)
        assert visium_adata.obsp["spatial_neighbors_connectivities"][0].sum() == n_neigh


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
    spatial_graph = non_visium_adata.obsp["spatial_neighbors_connectivities"].toarray()
    assert np.array_equal(spatial_graph, correct_knn_graph)

    spatial_neighbors(non_visium_adata, radius=5.0, coord_type=None)
    spatial_graph = non_visium_adata.obsp["spatial_neighbors_connectivities"].toarray()
    assert np.array_equal(spatial_graph, correct_radius_graph)
