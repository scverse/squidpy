"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)
"""

import numpy as np
from itertools import product, combinations
import pandas as pd

def _count_observations_by_pairs(conn, leiden, positions, count_option='edges'):
    obs = []
    leiden_labels_unique = set(leiden)
    positions_by_leiden = {li: positions[leiden == li] for li in leiden_labels_unique}
    
    if count_option == 'edges':        
        masks = []
        for cat in pd.Series(leiden).cat.categories:
            masks.append((leiden == cat).tolist())
        # _, masks = sc._utils.select_groups(adata, list(adata.obs['leiden'].cat.categories), 'leiden')
        
        N = len(pd.Series(leiden).cat.categories)
        cluster_counts = np.zeros((N, N), dtype=int)
        for i, mask in enumerate(masks):
            cluster_counts[i] = [np.ravel(conn[mask].sum(0))[j_mask].sum() for j_mask in masks]
        
        for i, j in combinations(range(cluster_counts.shape[0]), r=2):
            n_edges = cluster_counts[i][j]
            obs.append([i, j, n_edges, 'edges'])
    elif count_option == 'nodes':
        conn_array = conn.toarray()
        for i, j in combinations(leiden_labels_unique, r=2):
            x = positions[leiden == i]
            y = positions[leiden == j]
            x, y = np.array(list(product(x, y)))[:, 0].flatten(), np.array(list(product(x, y)))[:, 1].flatten()

            edges = conn_array[x, y]
            x_nodes = x[edges == 1]
            y_nodes = y[edges == 1]
            n_nodes_x, n_nodes_y = x_nodes.shape[0], y_nodes.shape[0]
            nx_uniq, ny_uniq = np.unique(x_nodes).shape[0], np.unique(y_nodes).shape[0]
            obs.append([int(i), int(j), nx_uniq + ny_uniq, 'nodes'])
            
            
    

    obs = pd.DataFrame(obs, columns=['leiden.i', 'leiden.j', 'n.obs', 'mode'])
    obs['k'] = obs['leiden.i'].astype(str) + ":" + obs['leiden.j'].astype(str) 
    obs = obs.sort_values('n.obs', ascending=False)
    return obs

def permutation_test_leiden_pairs(adata: "AnnData",
                            n_permutations: int = 10,
                            key_added: str ='nhood_permutation_test',
                            print_log_each=25,
                            count_option: str = 'edges',
                           ):
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
    count_option
        counting option (edges = count edges, nodes = count nodes)
    """
    
    leiden = adata.obs['leiden']
    conn = adata.obsp['spatial_connectivity']
    N = adata.shape[0]
    positions = np.arange(N) # .reshape(w, h)
    X = np.array(leiden).astype(int) # np.random.randint(1, 10, size=(w, h))

    # real observations
    print('calculating pairwise enrichment/depletion on real data...')
    df = _count_observations_by_pairs(conn, leiden, positions,
                                      count_option=count_option)
    
    # permutations
    leiden_rand = leiden.copy()
    perm = []
    print('calculating pairwise enrichment/depletion permutations...')
    for pi in range(n_permutations):
        if (pi + 1) % print_log_each == 0:
            print('%i out of %i permutations' % (pi + 1, n_permutations))
        leiden_rand = leiden_rand[np.random.permutation(leiden_rand.shape[0])]
        obs_perm = _count_observations_by_pairs(conn, leiden_rand, positions,
                                               count_option=count_option)
        obs_perm['permutation.i'] = True
        perm.append(obs_perm)
    perm = pd.concat(perm)
    
    # statistics
    n_by_leiden = adata.obs['leiden'].astype(int).value_counts().to_dict()
    mean_by_k = perm.groupby('k').mean()['n.obs'].to_dict()
    std_by_k = perm.groupby('k').std()['n.obs'].to_dict()
    
    
    mu = df['k'].map(mean_by_k)
    sigma = df['k'].map(std_by_k)
    
    df['n.i'] = df['leiden.i'].map(n_by_leiden)
    df['n.j'] = df['leiden.j'].map(n_by_leiden)

    # print((df['n.obs']))
    # print(mu)
    # print(sigma)
    # print((df['n.obs'] - mu) / sigma)
    df['z.score'] = (df['n.obs'] - mu) / sigma
    df['n.exp'] = mu
    df['sigma'] = sigma
    df.sort_values('z.score', ascending=False)
    
    adata.uns[key_added] = df