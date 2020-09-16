"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)
"""

import numpy as np
from itertools import product, combinations
import pandas as pd
import networkx as nx


def _count_observations_by_pairs(conn, leiden, positions):
    obs = []
    leiden_labels_unique = set(leiden)
    for i, j in combinations(leiden_labels_unique, r=2):
        x = positions[leiden == i]
        y = positions[leiden == j]
        x, y = np.array(list(product(x, y)))[:, 0].flatten(), np.array(list(product(x, y)))[:, 1].flatten()
        # print(conn[x, y])
        obs.append([i, j, int(sum(conn[x, y]).sum())])

    obs = pd.DataFrame(obs, columns=['leiden.i', 'leiden.j', 'n.obs'])
    obs['k'] = obs['leiden.i'].astype(str) + ":" + obs['leiden.j'].astype(str) 
    obs = obs.sort_values('n.obs', ascending=False)
    return obs


def permutation_test_leiden_pairs(adata: "AnnData",
                            n_permutations: int = 10,
                            key_added: str ='nhood_permutation_test'
                           ):
    leiden = adata.obs['leiden']
    conn = adata.obsp['spatial_connectivity']
    
    """
    Calculate enrichment/depletion of observed leiden pairs in the spatial connectivity graph, versus permutations as background.
    Params
    ------
    adata
        The AnnData object.
    n_permutations
        Number of shuffling and recalculations to be done.
    key_added
        Key added to output dataframe in adata.uns.
    """
    
    conn = adata.obsp['spatial_connectivity']
    N = adata.shape[0]
    positions = np.arange(N) # .reshape(w, h)
    X = np.array(leiden).astype(int) # np.random.randint(1, 10, size=(w, h))

    # real observations
    print('calculating pairwise enrichment/depletion on real data...')
    df = _count_observations_by_pairs(conn, leiden, positions)
    
    # permutations
    leiden_rand = leiden.copy()
    perm = []
    print('calculating pairwise enrichment/depletion permutations...')
    for pi in range(n_permutations):
        if (pi + 1) % 2 == 0:
            print('%i out of %i permutations' % (pi, n_permutations))
        np.random.shuffle(leiden_rand)
        obs_perm = _count_observations_by_pairs(conn, leiden_rand, positions)
        obs_perm['permutation.i'] = True
        perm.append(obs_perm)
    perm = pd.concat(perm)
    
    # statistics
    n_by_leiden = leiden.to_dict()
    mean_by_k = perm.groupby('k').mean()['n.obs'].to_dict()
    std_by_k = perm.groupby('k').std()['n.obs'].to_dict()
    
    
    mu = df['k'].map(mean_by_k)
    sigma = df['k'].map(std_by_k)
    df['n.i'] = df['leiden.i'].map(n_by_leiden)
    df['n.j'] = df['leiden.j'].map(n_by_leiden)
    df['z.score'] = (df['n.obs'] - mu) / sigma
    df['n.exp'] = mu
    df['sigma'] = sigma
    df.sort_values('z.score', ascending=False)
    
    adata.uns[key_added] = df


def cluster_centrality_scores(
        adata: "AnnData",
        connectivity_key: str,
        clusters_key: str,
        key_added: str = 'cluster_centrality_scores',
        save_networkx_graph: bool=True
):
    """
    Computes centrality scores per cluster. Results are stored in .uns in the AnnData object.
    Params
    ------
    adata
        The AnnData object.
    connectivity_key
        Key to connectivity_matrix in obsp.
    clusters_key
        Key to clusters in obs.
    key_added
        Key added to output dataframe in adata.uns.
    save_networkx_graph
        Whether to add networkx to adata.uns under key 'networkx_graph'.
    """
    graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])
    if save_networkx_graph:
        adata.uns['networkx_graph'] = graph

    clusters = adata.obs[clusters_key].unique().tolist()

    degree_centrality = []
    clustering_coefficient = []
    closeness_centrality = []

    for c in clusters:
        cluster_node_idx = adata[adata.obs[clusters_key] == c].obs.index.tolist()
        # ensuring that cluster_node_idx are List[int]
        cluster_node_idx = [i for i, x in enumerate(cluster_node_idx)]

        centrality = nx.algorithms.centrality.group_degree_centrality(graph, cluster_node_idx)
        degree_centrality.append(centrality)

        clustering = nx.algorithms.cluster.average_clustering(graph, cluster_node_idx)
        clustering_coefficient.append(clustering)

        closeness = nx.algorithms.centrality.group_closeness_centrality(graph, cluster_node_idx)
        closeness_centrality.append(closeness)

    df = pd.DataFrame(list(zip(clusters, degree_centrality, clustering_coefficient, closeness_centrality)),
                      columns=['cluster', 'degree centrality', 'clustering coefficient', 'closeness centrality']
                      )
    adata.uns[key_added] = df


def plot_cluster_centrality_scores(
        adata: "AnnData",
        centrality_scores_key: str
):
    """
    Plots centrality scores per cluster.
    Params
    ------
    adata
        The AnnData object.
    connectivity_key
        Key to centrality_scores in uns.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = adata.uns[centrality_scores_key]
    df = df.rename(columns={"degree centrality": "degree\ncentrality",
                            "clustering coefficient": "clustering\ncoefficient",
                            'closeness centrality': 'closeness\ncentrality'}
                   )
    values = ["degree\ncentrality", "clustering\ncoefficient", 'closeness\ncentrality']
    for i, value in zip([1, 2, 3], values):
        plt.subplot(1, 3, i)
        ax = sns.stripplot(data=df, y="cluster", x=value, size=10, orient="h", linewidth=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, left=False, right=False, top=False)
        if i > 1:
            plt.ylabel(None)
            ax.tick_params(labelleft=False)

