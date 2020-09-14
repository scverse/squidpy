import pandas as pd
import anndata as ad
import os
import networkx as nx


def read_seqfish(base_path: str, dataset: str):

    if dataset == "ob":
        counts_path = os.path.join(base_path, "sourcedata", "ob_counts.csv")
        clusters_path = os.path.join(base_path, "OB_cell_type_annotations.csv")
        centroids_path = os.path.join(base_path, "sourcedata", "ob_cellcentroids.csv")
    elif dataset == "svz":
        counts_path = os.path.join(base_path, "sourcedata", "cortex_svz_counts.csv")
        clusters_path = os.path.join(base_path, "cortex_svz_cell_type_annotations.csv")
        centroids_path = os.path.join(
            base_path, "sourcedata", "cortex_svz_cellcentroids.csv"
        )
    else:
        print("Dataset not available")

    counts = pd.read_csv(counts_path)
    clusters = pd.read_csv(clusters_path)
    centroids = pd.read_csv(centroids_path)

    adata = ad.AnnData(counts, obs=pd.concat([clusters, centroids], axis=1))
    adata.obsm["spatial"] = adata.obs[["X", "Y"]].to_numpy()

    adata.obs["louvain"] = pd.Categorical(adata.obs["louvain"])

    return adata


def get_dict_node_attributes(adata, node_attributes):
    """
    Utils function that stores an observation of an AnnData object in dict to be used in networkx graph as node
    attribute.
    :param adata: AnnData object with has a node attributes object stored in obs.
    :param node_attributes: Name of the node attributes object stored in obs.
    :return: dict = {node_index : value of node_attribute}
    """
    attributes = {i: {'louvain': adata.obs[node_attributes].tolist()[i]} for i in range(node_attributes.n_obs)}
    return attributes


def get_networkx_graph(adjacency_matrix, node_attributes):
    """
    Function to create networkx graph from adjacency matrix a with node attributes attr.
    Networkx graph is needed to calculate assortativity coefficients with networkx built-in functions.
    :param adjacency_matrix: sparse adjacency matrix
    :param node_attributes: dict of nodes
    :return: networkx graph G with node_attributes
    """
    graph = nx.from_scipy_sparse_matrix(adjacency_matrix)
    if node_attributes is not None:
        nx.set_node_attributes(graph, node_attributes)
    self.graph = graph
    return self.graph