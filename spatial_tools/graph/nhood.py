"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)
"""

import numpy as np
from itertools import product, combinations
import pandas as pd
from spatial_tools.graph.build import spatial_connectivity


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    
    Based on discussion from here
    https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def _count_observations_by_pairs(conn, leiden, positions, count_option='nodes'):
    obs = []
    masks = []
    leiden_labels_unique = set(map(int, set(leiden)))
    # assert min(leiden_labels_unique) == 1
    
    positions_by_leiden = {li: positions[leiden == li] for li in leiden_labels_unique}
    
    if count_option == 'edges':        
        # for cat in pd.Series(leiden).cat.categories:
        for cat in set(leiden):
            masks.append((leiden == cat).tolist())
        # _, masks = sc._utils.select_groups(adata, list(adata.obs['leiden'].cat.categories), 'leiden')
        
        # N = len(pd.Series(leiden).cat.categories)
        N = len(set(leiden))
        cluster_counts = np.zeros((N, N), dtype=int)
        for i, mask in enumerate(masks):
            cluster_counts[i] = [np.ravel(conn[mask].sum(0))[j_mask].sum() for j_mask in masks]
        
        for i, j in combinations(leiden_labels_unique, r=2):
            n_edges = cluster_counts[i - 1][j - 1]
            obs.append([i, j, n_edges, 'edges'])
    elif count_option == 'nodes':
        conn_array = conn.toarray() if (type(conn) != np.ndarray) else conn
        for i, j in combinations(set(leiden), r=2):
            x = positions[leiden == i]
            y = positions[leiden == j]
            
            xy = cartesian([x, y])
            x, y = xy[:, 0].flatten(), xy[:, 1].flatten()

            edges = conn_array[x, y]
            x_nodes = x[edges == 1]
            y_nodes = y[edges == 1]
            n_nodes_x, n_nodes_y = x_nodes.shape[0], y_nodes.shape[0]
            nx_uniq, ny_uniq = np.unique(x_nodes).shape[0], np.unique(y_nodes).shape[0]
            obs.append([int(i), int(j), nx_uniq + ny_uniq, count_option])
    elif count_option == 'nodes-dev':
        for cat in np.unique(leiden):
            masks.append((leiden == cat).tolist())
        # _, masks = sc._utils.select_groups(adata, list(adata.obs['leiden'].cat.categories), 'leiden')

        N = len(np.unique(leiden))
        cluster_counts = np.zeros((N, N), dtype=int)
        for i, mask in enumerate(masks):
            cluster_counts[i] = [(np.ravel(conn[mask].sum(0))>0).astype(int)[j_mask].sum() for j_mask in masks]

        for i, j in combinations(leiden_labels_unique, r=2):
            n_nodes = cluster_counts[i - 1][j - 1] + cluster_counts[j - 1][i - 1]
            # print([i + 1, j + 1, n_edges])
            obs.append([i, j, n_nodes, count_option])

    obs = pd.DataFrame(obs, columns=['leiden.i', 'leiden.j', 'n.obs', 'mode'])
    obs['k'] = obs['leiden.i'].astype(str) + ":" + obs['leiden.j'].astype(str)
    obs = obs.sort_values('n.obs', ascending=False)
    
    return obs

def _get_output_symmetrical(df):
    """
    It assures the output is symmetrical given the permutation paired-events that are calculated
    """
    res = df
    res2 = res.copy()
    li = res2['leiden.i'].astype(str)
    res2['leiden.i'] = res2['leiden.j']
    res2['leiden.j'] = li
    res = pd.concat([res, res2])

    res['k.sorted'] = np.where(res['leiden.i'].astype(int) < res['leiden.j'].astype(int),
                    res['leiden.i'].astype(str) + ":" + res['leiden.j'].astype(str),
                    res['leiden.j'].astype(str) + ":" + res['leiden.i'].astype(str))
    
    res['leiden.i'] = res['leiden.i'].astype(int)
    res['leiden.j'] = res['leiden.j'].astype(int)
    
    res['k'] = res['leiden.i'].astype(str) + ":" + res['leiden.j'].astype(str)
    res = res.drop_duplicates('k')
    return res

def permutation_test_leiden_pairs_complex(adata, degree_start=1, degree_end=6, n_perm=100):
    res = []
    for n_degree in range(degree_start, degree_end):
        print('# degree', n_degree)
        print('calculating connectivity graph with degree %i...' % n_degree)
        spatial_connectivity(adata, n_degree=n_degree)
        print('permutations...')
        for count_option in ['edges', 'nodes', 'nodes-dev']:
            permutation_test_leiden_pairs(adata, n_permutations=n_perm,
                                          print_log_each=25, log=False,
                                          count_option=count_option)
            df = adata.uns['nhood_permutation_test'].copy()
            df['n.degree'] = n_degree
            df['n.perm'] = n_perm
            res.append(df)
    res = pd.concat(res)        
    return(res)

def permutation_test_leiden_pairs(adata: "AnnData",
                            n_permutations: int = 10,
                            key_added: str ='nhood_permutation_test',
                            print_log_each: int = 25,
                            log: bool = True,
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
    if log:
        print('calculating pairwise enrichment/depletion on real data...')
    df = _count_observations_by_pairs(conn, leiden, positions,
                                      count_option=count_option)
    
    # permutations
    leiden_rand = leiden.copy()
    perm = []
    
    if log:
        print('calculating pairwise enrichment/depletion permutations...')
    for pi in range(n_permutations):
        if (pi + 1) % print_log_each == 0 and log:
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
    
    print(df.shape)
    df = _get_output_symmetrical(df)
    print(df.shape)
    
    adata.uns[key_added] = df