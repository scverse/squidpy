from anndata import AnnData

import numpy as np

from spatial_tools.graph import (
    nhood_enrichment,
    interaction_matrix,
    spatial_connectivity,
    centrality_scores,
)


def test_nhood_enrichment(adata: AnnData):

    ckey = "leiden"
    spatial_connectivity(adata)
    nhood_enrichment(adata, cluster_key=ckey)

    assert adata.uns[f"{ckey}_nhood_enrichment"]["zscore"].dtype == np.dtype("float64")
    assert adata.uns[f"{ckey}_nhood_enrichment"]["count"].dtype == np.dtype("uint32")
    assert adata.uns[f"{ckey}_nhood_enrichment"]["zscore"].shape[0] == adata.obs.leiden.cat.categories.shape[0]
    assert adata.uns[f"{ckey}_nhood_enrichment"]["count"].shape[0] == adata.obs.leiden.cat.categories.shape[0]


# nhood_data is now in conftest.py
def test_centrality_scores(nhood_data: AnnData):
    """
    check that scores fit the expected shape + content
    """
    adata = nhood_data
    cluster_key = "leiden"
    centrality_scores(
        adata=adata,
        cluster_key=cluster_key,
    )
    # assert saving in .uns
    assert f"{cluster_key}_centrality_scores" in adata.uns_keys()
    # assert centrality scores are computed for each cluster
    assert len(adata.obs[cluster_key].unique()) == adata.uns[f"{cluster_key}_centrality_scores"].shape[0]
    assert adata.uns[f"{cluster_key}_centrality_scores"]["cluster_key"].dtype == np.dtype("O")
    assert adata.uns[f"{cluster_key}_centrality_scores"]["degree_centrality"].dtype == np.dtype("float64")
    assert adata.uns[f"{cluster_key}_centrality_scores"]["clustering_coefficient"].dtype == np.dtype("float64")
    assert adata.uns[f"{cluster_key}_centrality_scores"]["closeness_centrality"].dtype == np.dtype("float64")
    assert adata.uns[f"{cluster_key}_centrality_scores"]["betweenness_centrality"].dtype == np.dtype("float64")


def test_interaction_matrix(nhood_data: AnnData):
    """
    check that interaction matrix fits the expected shape
    """
    adata = nhood_data
    cluster_key = "leiden"
    interaction_matrix(
        adata=adata,
        cluster_key=cluster_key
    )
    # assert saving in .uns
    assert f"{cluster_key}_interactions" in adata.uns_keys()
    assert type(adata.uns[f"{cluster_key}_interactions"]) == np.matrix
    assert len(adata.obs[cluster_key].unique()) == adata.uns[f"{cluster_key}_interactions"].shape[0]
    assert len(adata.obs[cluster_key].unique()) == adata.uns[f"{cluster_key}_interactions"].shape[1]
