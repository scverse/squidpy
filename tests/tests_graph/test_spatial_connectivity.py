from anndata import AnnData

import numpy as np

from squidpy.graph import spatial_connectivity


# todo add edge cases
def test_spatial_connectivity_visium(visium_adata: AnnData):
    """
    check correctness of neighborhoods for visium coordinates
    """
    for i, n_neigh in enumerate((6, 18, 36)):
        spatial_connectivity(visium_adata, n_rings=i + 1)
        assert visium_adata.obsp["spatial_connectivities"][0].sum() == n_neigh


def test_spatial_connectivity_non_visium(non_visium_adata: AnnData):
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

    spatial_connectivity(non_visium_adata, n_neigh=3, coord_type="non-visium")
    spatial_graph = non_visium_adata.obsp["spatial_connectivities"].toarray()
    assert np.array_equal(spatial_graph, correct_knn_graph)

    spatial_connectivity(non_visium_adata, radius=5.0, coord_type="non-visium")
    spatial_graph = non_visium_adata.obsp["spatial_connectivities"].toarray()
    assert np.array_equal(spatial_graph, correct_radius_graph)
