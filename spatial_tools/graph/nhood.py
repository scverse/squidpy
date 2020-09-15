"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)
"""

import pandas as pd
import networkx as nx
import scanpy as sc


def get_cluster_scores_from_adata(
        adata,
        clusters: list[str]
):
    """
    Computes degree centrality per cluster of AnnData object.
    :param::
    :return:
    """
    sc.pp.neighbors(adata=adata, n_neighbors=15)
    sc.tl.umap(adata=adata)

    graph = nx.from_scipy_sparse_matrix(adata.obsp['connectivities'])

    degree_centrality = []
    betweenness_centrality = []
    clustering_coefficient = []

    for c in clusters:
        class_node_indices = adata[adata.obs[clusters] == c].obs.index.tolist()
        class_node_indices = [int(x) for x in class_node_indices]

        centrality = nx.algorithms.centrality.group_degree_centrality(graph, class_node_indices)
        degree_centrality.append(centrality)

        clustering = nx.algorithms.cluster.clustering(graph, class_node_indices)
        clustering_coefficient.append(sum(clustering.values()) / len(clustering.values()))

        subgraph = graph.subgraph(class_node_indices)
        betweenness = nx.algorithms.centrality.betweenness_centrality(subgraph)
        betweenness_centrality.append(sum(betweenness.values()) / len(betweenness.values()))

    df = pd.DataFrame(list(zip(clusters, degree_centrality, betweenness_centrality, clustering_coefficient)),
                      columns=['cluster', 'degree\ncentrality', 'betweenness\ncentrality',
                               'clustering\ncoefficient']
                      )
    return df
