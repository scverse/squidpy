import pytest
import scanpy
import numpy as np
from spatial_tools.graph import spatial_connectivity, cluster_centrality_scores, cluster_interactions


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

