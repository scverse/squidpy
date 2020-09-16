"""Functions for node clustering incorporating spatial information
"""

import scipy
import scanpy as sc

def prepare_data_for_clustering(adata):
    """
    Does all data transformations that are independent of the ratio of the spatial and feature graphs and the degree
    of node feature smoothing.

    Parameters
    ----------
    adata
        The AnnData object.
    """
    # add self-connections to adjacency matrix and normalize
    adj = adata.obsp["spatial_connectivity"]
    diag = scipy.sparse.diags([1] * adj.shape[0])
    adj += diag
    adj /= adj.sum(axis=1)
    adata.obsp["self_connected_adj"] = adj

def compute_louvain_on_joined_connectivities(adata, alpha, nr_steps):
    """
    Computes louvain clusters based on a weighted sum of a spatial connectivity graph and a connectivity graph
    based on node feature similarities.

    Parameters
    ----------
    adata
        The AnnData object.
    alpha
        Ratio of spatial graph and feature graph information. Feature information are weighted by alpha,
        spatial connectivity matrix is multiplied by (1 - alpha).
    nr_steps
        Number of times the spatial adjacency matrix is multiplied to the node features to smoothen them locally.
        nr_steps = 0 corresponds to working on the original node features.
    """
    # compute smoothened node features
    adata.obsm["smooth_features"] = adata.X
    for i in range(nr_steps):
        adata.obsm["smooth_features"] = adata.obsp["self_connected_adj"] @ adata.obsm["smooth_features"]
    adata.obsm["smooth_features"] = sc.pp.pca(adata.obsm["smooth_features"])

    # compute knn-graph on smoothened node features
    # TODO give more options for this graph construction? Use spatial_connectivity?
    sc.pp.neighbors(
        adata,
        use_rep="smooth_features",
        key_added="feature",
    )

    # take weighted mean of spatial and feature-based adjacency matrices
    joined_graph = (
        alpha * adata.obsp["feature_connectivities"]
        + (1 - alpha) * adata.obsp["spatial_connectivity"]
    )
    adata.obsp["joined_connectivities"] = joined_graph
    sc.tl.louvain(
        adata,
        obsp="joined_connectivities",
        key_added="joined_louvain",
        use_weights=True,
    )

def plot_louvain_on_graph(adata):
    """
    Plots the computed louvain cluster assignments on top of the spatial graph.

    Parameters
    ----------
    adata
        The AnnData object.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.cm as cmx
    import matplotlib.colors as colors

    g = nx.from_numpy_matrix(adata.obsp["spatial_connectivity"].todense())
    dict_nodes = {i: p for i, p in enumerate(adata.obsm["spatial"])}
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(g, pos=dict_nodes, width=0.1)
    cluster_name = 'joined_louvain'
    vmax = len(np.unique(adata.obs[cluster_name]))
    cNorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("hsv"))
    for cluster_id in np.unique(adata.obs[cluster_name]):
        color = [scalarMap.to_rgba(int(cluster_id))]
        idx_c = list(np.where(adata.obs[cluster_name] == cluster_id)[0])
        nx.draw_networkx_nodes(
            g,
            with_labels=False,
            node_size=10,
            nodelist=idx_c,
            node_color=color,
            pos=dict_nodes,
            label=cluster_id,
        )
    plt.gca().invert_yaxis()

def plot_louvain_on_umap(adata):
    """
    Plots the computed louvain cluster assignments on top of the UMAP of the, potentially processed, node features.

    Parameters
    ----------
    adata
        The AnnData object.
    """
    sc.pp.neighbors(adata=adata, use_rep='smooth_features', key_added='umap')
    sc.tl.umap(adata=adata, neighbors_key='umap')
    sc.pl.umap(adata, color="joined_louvain")


