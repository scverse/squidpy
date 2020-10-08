import pytest
import scanpy
import numpy as np
from spatial_tools.graph import spatial_connectivity, cluster_centrality_scores, cluster_interactions
  
from spatial_tools.graph.nhood import permtest_leiden_pairs, _count_observations_by_pairs

def get_sample_data():
    adata = scanpy.datasets.visium_sge()
    scanpy.tl.louvain(adata, adjacency=adata.obsp['spatial_connectivity'])
    spatial_connectivity(adata)
    return adata


def test_cluster_centrality_scores():
    """
    check that scores fit the expected shape + content
    """
    adata = get_sample_data()
    cluster_centrality_scores(
        adata=adata,
        clusters_key='louvain',
        connectivity_key='spatial_connectivity',
        key_added='cluster_centrality_scores'
    )
    # assert saving in .uns
    assert 'cluster_centrality_scores' in adata.uns_keys()
    # assert centrality scores are computed for each cluster
    assert len(adata.obs['louvain'].unique()) == adata.uns['cluster_centrality_scores'].shape[0]
    assert adata.uns['cluster_centrality_scores']['cluster'].dtype == np.dtype('O')
    assert adata.uns['cluster_centrality_scores']['degree centrality'].dtype == np.dtype('float64')
    assert adata.uns['cluster_centrality_scores']['clustering coefficient'].dtype == np.dtype('float64')
    assert adata.uns['cluster_centrality_scores']['closeness centrality'].dtype == np.dtype('float64')
    assert adata.uns['cluster_centrality_scores']['betweenness centrality'].dtype == np.dtype('float64')


def test_cluster_interactions():
    """
    check that interaction matrix fits the expected shape
    """
    adata = get_sample_data()
    cluster_interactions(
        adata=adata,
        clusters_key='louvain',
        connectivity_key='spatial_connectivity',
        normalized=True,
        key_added='cluster_interactions'
    )
    # assert saving in .uns
    assert 'cluster_interactions' in adata.uns_keys()
    assert type(adata.uns['cluster_interactions']) == tuple
    assert type(adata.uns['cluster_interactions'][0]) == np.matrix
    assert len(adata.obs['louvain'].unique()) == adata.uns['cluster_interactions'][0].shape[0]
    assert len(adata.obs['louvain'].unique()) == adata.uns['cluster_interactions'][0].shape[1]
    assert type(adata.uns['cluster_interactions'][1]) == list
    assert len(adata.obs['louvain'].unique()) == len(adata.uns['cluster_interactions'][1])

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
    adata = scanpy.datasets.visium_sge()
    adata.var_names_make_unique()
    scanpy.pp.neighbors(adata)
    scanpy.tl.leiden(adata)
    
    n_permutations = 25
    spatial_connectivity(adata, n_rings=3)
    try:
        permtest_leiden_pairs(adata, n_permutations=n_permutations, print_log_each=25, count_option='nodes')
        result = adata.uns['nhood_permtest'].copy()
        print(result.head())
        print(result.shape)
    except Exception:
        assert False
    assert True