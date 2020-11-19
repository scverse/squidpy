from anndata import AnnData

import numpy as np

from spatial_tools.graph import (
    nhood_enrichment,
    cluster_interactions,
    spatial_connectivity,
    cluster_centrality_scores,
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
def test_cluster_centrality_scores(nhood_data: AnnData):
    """
    check that scores fit the expected shape + content
    """
    adata = nhood_data
    cluster_centrality_scores(
        adata=adata,
        clusters_key="leiden",
        connectivity_key="spatial_connectivities",
        key_added="cluster_centrality_scores",
    )
    # assert saving in .uns
    assert "cluster_centrality_scores" in adata.uns_keys()
    # assert centrality scores are computed for each cluster
    assert len(adata.obs["leiden"].unique()) == adata.uns["cluster_centrality_scores"].shape[0]
    assert adata.uns["cluster_centrality_scores"]["cluster"].dtype == np.dtype("O")
    assert adata.uns["cluster_centrality_scores"]["degree centrality"].dtype == np.dtype("float64")
    assert adata.uns["cluster_centrality_scores"]["clustering coefficient"].dtype == np.dtype("float64")
    assert adata.uns["cluster_centrality_scores"]["closeness centrality"].dtype == np.dtype("float64")
    assert adata.uns["cluster_centrality_scores"]["betweenness centrality"].dtype == np.dtype("float64")


def test_cluster_interactions(nhood_data: AnnData):
    """
    check that interaction matrix fits the expected shape
    """
    adata = nhood_data
    cluster_interactions(
        adata=adata,
        clusters_key="leiden",
        connectivity_key="spatial_connectivities",
        normalized=True,
        key_added="cluster_interactions",
    )
    # assert saving in .uns
    assert "cluster_interactions" in adata.uns_keys()
    assert type(adata.uns["cluster_interactions"]) == tuple
    assert type(adata.uns["cluster_interactions"][0]) == np.matrix
    assert len(adata.obs["leiden"].unique()) == adata.uns["cluster_interactions"][0].shape[0]
    assert len(adata.obs["leiden"].unique()) == adata.uns["cluster_interactions"][0].shape[1]
    assert type(adata.uns["cluster_interactions"][1]) == list
    assert len(adata.obs["leiden"].unique()) == len(adata.uns["cluster_interactions"][1])
