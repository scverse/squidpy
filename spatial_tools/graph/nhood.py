"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)
"""

import numpy as np
import networkx as nx


class AssortativityMeasures:
    """
    Class for assortivity measures.
    """
    def get_networkx_graph(
            self,
            a,
            node_attributes
    ):
        """
        Function to create networkx graph from adjacency matrix a with node attributes attr.
        Networkx graph is needed to calculate assortativity coefficients with networkx built-in functions.
        :param a: adjacency matrix
        :param node_attributes: dict of nodes
        :return: networkx graph G
        """

        self.G = nx.from_scipy_sparse_matrix(a)
        if attr is not None:
            nx.set_node_attributes(G, node_attributes)
        return self.G
