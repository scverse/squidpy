"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)
"""

import pandas as pd
import networkx as nx
from typing import List


def get_centrality_scores(
        adata: "AnnData",
        connectivity_key: str,
        clusters_key: List[str]
):
    """
    Computes centrality scores per cluster. If no list of of clusters is provided centrality scores will
    be evaluated per node. Results are stored in a pandas DataFrame.

    Params
    ------
    adata
        The AnnData object.
    connectivity_key
        Key to connectivity_matrix in obsp.
    """
    graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])
    clusters = adata.obs[clusters_key].unique().tolist()

    degree_centrality = []
    betweenness_centrality = []
    clustering_coefficient = []

    for c in clusters:
        cluster_node_idx = adata[adata.obs[clusters_key] == c].obs.index.tolist()
        # ensuring that cluster_node_idx are List[int]
        cluster_node_idx = [int(x) for x in cluster_node_idx]

        centrality = nx.algorithms.centrality.group_degree_centrality(graph, cluster_node_idx)
        degree_centrality.append(centrality)

        clustering = nx.algorithms.cluster.clustering(graph, cluster_node_idx)
        clustering_coefficient.append(sum(clustering.values()) / len(clustering.values()))

        subgraph = graph.subgraph(cluster_node_idx)
        betweenness = nx.algorithms.centrality.betweenness_centrality(subgraph)
        betweenness_centrality.append(sum(betweenness.values()) / len(betweenness.values()))

    df = pd.DataFrame(list(zip(clusters, degree_centrality, betweenness_centrality, clustering_coefficient)),
                      columns=['cluster', 'degree\ncentrality', 'betweenness\ncentrality',
                               'clustering\ncoefficient']
                      )
    return df
