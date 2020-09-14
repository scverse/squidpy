"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)
"""
import networkx as nx
import warnings


class AssortativityMeasures:
    """
    Class for assortivity measures.
    """
    def calculate_degree_assortativity_coefficient(
            self,
            graph
    ):
        """
        Calculates degree assortativity coefficient for a graph G
        :param graph: networkx graph
        :return: degree assortativity coefficient r (int)
        """
        r = nx.degree_assortativity_coefficient(graph)
        return r

    def calculate_assortativity_coefficient(
            self,
            graph,
            node_attributes: str
    ):
        """
        Calculates degree assortativity coefficient for a graph G with node attributes node_attributes
        :param graph: networkx graph
        :param node_attributes: node attributes stored in networkx graph
        :return: assortativity coefficient r (int)
        """
        attr = nx.get_node_attributes(graph, node_attributes)
        if not attr:  # checking if graph has node_attributes
            warnings.warn("The given node attribute is not given in the graph, choose different node attributes or "
                          "connect the given attrbute to the graph.")

        # ToDo: difference in calculation of numeric assortativity coefficient and categorical coefficient - check why
        #  or implement version which uses the respective function
        r = nx.attribute_assortativity_coefficient(graph, node_attributes)
        return r

    def _add_r_to_anndata_object(self):
        """
        Function that adds assortativity coefficient as unstructured annotation to AnnData object
        :return:
        """
        #  ToDo: add assortativity coefficient as unstructzred annotation to the AnnData object
        raise NotImplementedError
