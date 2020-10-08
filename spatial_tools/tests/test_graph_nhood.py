import pytest
from spatial_tools.graph.build import spatial_connectivity
from spatial_tools.graph.nhood import _count_observations_by_pairs
from spatial_tools.graph.nhood import *


def test_nhood_permtest_toydata():
    """
    i) Verify that the permutation works in a simple connectivity graph
    ii) expected values n_nodes = 5 and n_edges = 6
    """
    import numpy as np
    positions = np.arange(5)
    leiden = np.array([1, 1, 2, 2, 2])
    conn = np.array([[0, 0, 1, 1, 1],
                     [0, 0, 1, 1, 1],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0]])
    edges = _count_observations_by_pairs(conn, leiden, positions, count_option='edges')
    nodes = _count_observations_by_pairs(conn, leiden, positions, count_option='nodes')
    n_edges = list(edges['n.obs'])[0] 
    n_nodes = list(nodes['n.obs'])[0]
    assert n_nodes == 5 and n_edges == 6

    
def test_nhood_permtest_realdata():
    """
    Try to run a more complex test and report the respective statistics as a table.
    """
    # path to "raw" dataset folder
    BASE_PATH = "/storage/groups/ml01/datasets/raw/20200909_PublicVisium_giovanni.palla"
    dataset_name = "V1_Adult_Mouse_Brain"
    
    import scanpy as sc
    import os
    dataset_folder = os.path.join(
        BASE_PATH, "20191205_10XVisium_MouseBrainCoronal_giovanni.palla"
    )
    adata = sc.read_visium(
        dataset_folder, count_file=f"{dataset_name}_filtered_feature_bc_matrix.h5"
    )
    adata.var_names_make_unique()
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    
    n_permutations = 25
    spatial_connectivity(adata, n_degree=3)
    try:
        permtest_leiden_pairs(adata, n_permutations=n_permutations, print_log_each=25, count_option='nodes')
        result = adata.uns['nhood_permtest'].copy()
        print(result.head())
        print(result.shape)
    except Exception:
        assert False
    assert True


