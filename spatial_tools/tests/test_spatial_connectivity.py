import pytest
import numpy as np
from anndata import AnnData
from spatial_tools.graph import spatial_connectivity


@pytest.fixture
def visium_adata():
    visium_coords = np.array([
        [4193, 7848],
        [4469, 7848], [4400, 7968], [4262, 7729], [3849, 7968],
        [4124, 7729], [4469, 7609], [3987, 8208], [4331, 8088],
        [4262, 7968], [4124, 7968], [4124, 7489], [4537, 7968],
        [4469, 8088], [4331, 7848], [4056, 7848], [3849, 7729],
        [4262, 7489], [4400, 8208], [4056, 7609], [3987, 7489],
        [4262, 8208], [4400, 7489], [4537, 7729], [4606, 7848],
        [3987, 7968], [3918, 8088], [3918, 7848], [4193, 8088],
        [4056, 8088], [4193, 7609], [3987, 7729], [4331, 7609],
        [4124, 8208], [3780, 7848], [3918, 7609], [4400, 7729]
    ])
    adata = AnnData(X=np.ones((visium_coords.shape[0], 3)))
    adata.obsm['spatial'] = visium_coords
    return adata


@pytest.fixture
def non_visium_adata():
    non_visium_coords = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])
    adata = AnnData(X=non_visium_coords)
    adata.obsm['spatial'] = non_visium_coords
    return adata


# todo add edge cases
def test_spatial_connectivity_visium(visium_adata):
    """
    check correctness of neighborhoods for visium coordinates
    """
    for i, n_neigh in enumerate((6, 18, 36)):
        spatial_connectivity(visium_adata, n_rings=i+1)
        assert visium_adata.obsp['spatial_connectivity'][0].sum() == n_neigh


def test_spatial_connectivity_non_visium(non_visium_adata):
    """
    check correctness of neighborhoods for non-visium coordinates
    """
    correct_knn_graph = np.array([
        [0., 1., 1., 1.],
        [1., 0., 1., 1.],
        [1., 1., 0., 1.],
        [1., 1., 1., 0.]
    ])

    correct_radius_graph = np.array([
        [0., 1., 0., 1.],
        [1., 0., 0., 1.],
        [0., 0., 0., 0.],
        [1., 1., 0., 0.]
    ])

    spatial_connectivity(non_visium_adata, n_neigh=3, coord_type='non-visium')
    spatial_graph = non_visium_adata.obsp['spatial_connectivity'].toarray()
    assert np.array_equal(spatial_graph, correct_knn_graph)

    spatial_connectivity(non_visium_adata, radius=5.0, coord_type='non-visium')
    spatial_graph = non_visium_adata.obsp['spatial_connectivity'].toarray()
    assert np.array_equal(spatial_graph, correct_radius_graph)
