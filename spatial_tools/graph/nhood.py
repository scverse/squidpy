"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)."""

import random
from typing import Union
from itertools import combinations

from anndata import AnnData

import numpy as np
import pandas as pd

import networkx as nx

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
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def _count_observations_by_pairs(conn, leiden, positions, count_option="nodes"):
    obs = []
    masks = []

    leiden = leiden.astype(int)

    leiden_labels_unique = set(map(int, set(leiden)))
    # assert min(leiden_labels_unique) == 1

    # positions_by_leiden = {li: positions[leiden == li] for li in leiden_labels_unique}

    if count_option == "edges":
        # for cat in pd.Series(leiden).cat.categories:
        for cat in set(leiden):
            masks.append((leiden == cat).tolist())
        # _, masks = sc._utils.select_groups(adata, list(adata.obs['leiden'].cat.categories), 'leiden')

        # N = len(pd.Series(leiden).cat.categories)
        N = len(set(leiden))
        cluster_counts = np.zeros((N, N), dtype=int)
        for i, mask in enumerate(masks):
            # initialize
            cluster_counts[i] = [0 for j_mask in masks]
            for j, j_mask in enumerate(masks):
                # subset for i and j to avoid duplicate edges counts
                m = conn[mask, :]
                m = m[:, j_mask]
                cluster_counts[i][j] = m.sum()

        for i, j in combinations(leiden_labels_unique, r=2):
            n_edges = cluster_counts[i][j]
            obs.append([i, j, n_edges, "edges"])

    elif count_option == "nodes":
        conn_array = conn.toarray() if (type(conn) != np.ndarray) else conn
        for i, j in combinations(set(leiden), r=2):
            x = positions[leiden == i]
            y = positions[leiden == j]

            xy = cartesian([x, y])
            x, y = xy[:, 0].flatten(), xy[:, 1].flatten()

            edges = conn_array[x, y]
            x_nodes = x[edges == 1]
            y_nodes = y[edges == 1]
            nx_uniq, ny_uniq = np.unique(x_nodes).shape[0], np.unique(y_nodes).shape[0]
            obs.append([int(i), int(j), nx_uniq + ny_uniq, count_option])

    obs = pd.DataFrame(obs, columns=["leiden.i", "leiden.j", "n.obs", "mode"])
    obs["k"] = obs["leiden.i"].astype(str) + ":" + obs["leiden.j"].astype(str)
    obs = obs.sort_values("n.obs", ascending=False)

    return obs


def _get_output_symmetrical(df):
    """Assure that the output is symmetrical given the permutation paired-events that are calculated."""
    res = df
    res2 = res.copy()
    li = res2["leiden.i"].astype(str)
    res2["leiden.i"] = res2["leiden.j"]
    res2["leiden.j"] = li
    res = pd.concat([res, res2])

    res["k.sorted"] = np.where(
        res["leiden.i"].astype(int) < res["leiden.j"].astype(int),
        res["leiden.i"].astype(str) + ":" + res["leiden.j"].astype(str),
        res["leiden.j"].astype(str) + ":" + res["leiden.i"].astype(str),
    )

    res["leiden.i"] = res["leiden.i"].astype(int)
    res["leiden.j"] = res["leiden.j"].astype(int)

    res["k"] = res["leiden.i"].astype(str) + ":" + res["leiden.j"].astype(str)
    res = res.drop_duplicates("k")
    return res


def permtest_leiden_pairs_complex(adata, rings_start=1, rings_end=6, n_perm=100, random_seed=500):
    """Todo."""
    res = []
    random.seed(random_seed)
    for n_rings in range(rings_start, rings_end):
        print("# degree", n_rings)
        print("calculating connectivity graph with degree %i..." % n_rings)
        spatial_connectivity(adata, n_rings=n_rings)
        for count_option in ["edges", "nodes"]:
            print("permutations with mode %s..." % count_option)
            permtest_leiden_pairs(adata, n_permutations=n_perm, print_log_each=25, log=False, count_option=count_option)
            df = adata.uns["nhood_permtest"].copy()
            df["n.rings"] = n_rings
            df["n.perm"] = n_perm
            res.append(df)
    res = pd.concat(res)
    return res


def permtest_leiden_pairs(
    adata: AnnData,
    n_permutations: int = 100,
    key_added: str = "nhood_permtest",
    print_log_each: int = 25,
    log: bool = True,
    count_option: str = "edges",
    random_seed: int = 500,
) -> None:
    """
    Calculate enrichment/depletion of observed leiden pairs in the spatial connectivity graph, \
    versus permutations as background.

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
    random_seed
        The number to initialize a pseudorandom number generator.
    """
    leiden = adata.obs["leiden"]
    conn = adata.obsp["spatial_connectivities"]
    N = adata.shape[0]
    positions = np.arange(N)  # .reshape(w, h)

    # real observations
    if log:
        print("calculating pairwise enrichment/depletion on real data...")
    df = _count_observations_by_pairs(conn, leiden, positions, count_option=count_option)

    # permutations
    leiden_rand = leiden.copy()
    perm = []

    random.seed(random_seed)
    if log:
        print("calculating pairwise enrichment/depletion permutations (n=%i)..." % n_permutations)
        if n_permutations <= 100:
            print("Please consider a high permutation value for reliable Z-score estimates (>500)...")

    for pi in range(n_permutations):
        if (pi + 1) % print_log_each == 0 and log:
            print("%i permutations (out of %i)..." % (pi + 1, n_permutations))
        leiden_rand = leiden_rand[np.random.permutation(leiden_rand.shape[0])]
        obs_perm = _count_observations_by_pairs(conn, leiden_rand, positions, count_option=count_option)
        obs_perm["permutation.i"] = True
        perm.append(obs_perm)
    perm = pd.concat(perm)

    # statistics
    n_by_leiden = adata.obs["leiden"].astype(int).value_counts().to_dict()

    mean_by_k = perm.groupby("k").mean()["n.obs"].to_dict()
    std_by_k = perm.groupby("k").std()["n.obs"].to_dict()

    mu = df["k"].map(mean_by_k)
    sigma = df["k"].map(std_by_k)

    df["n.i"] = df["leiden.i"].map(n_by_leiden)
    df["n.j"] = df["leiden.j"].map(n_by_leiden)

    # print((df['n.obs']))
    # print(mu)
    # print(sigma)
    # print((df['n.obs'] - mu) / sigma)
    df["z.score"] = (df["n.obs"] - mu) / sigma
    df["n.exp"] = mu
    df["sigma"] = sigma
    df.sort_values("z.score", ascending=False)

    # this ensures output is symmetrical
    df = _get_output_symmetrical(df)

    adata.uns[key_added] = df


def centrality_scores(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Union[str, None] = "spatial_connectivities",
    key_added: str = "centrality_scores",
):
    """
    Compute centrality scores per cluster or cell type in AnnData object.

    Results are stored in .uns in the AnnData object under the key specified in key_added.

    Based among others on methods used for Gene Regulatory Networks (GRNs) in:
    'CellOracle: Dissecting cell identity via network inference and in silico gene perturbation'
    Kenji Kamimoto, Christy M. Hoffmann, Samantha A. Morris
    bioRxiv 2020.02.17.947416; doi: https://doi.org/10.1101/2020.02.17.947416

    Parameters
    ----------
    adata
        The AnnData object.
    cluster_key
        Key to clusters in obs.
    connectivity_key
        (Optional) Key to connectivity_matrix in obsp.
    key_added
        (Optional) Key added to output dataframe in adata.uns.

    Returns
    -------
    None
    """
    if cluster_key not in adata.obs_keys():
        raise ValueError(
            "cluster_key %s not recognized. Choose a different key refering to a cluster in .obs." % cluster_key
        )

    if connectivity_key in adata.obsp:
        graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])
        print("Saving networkx graph based on %s in adata.obsp" % connectivity_key)
        adata.uns["networkx_graph"] = graph
    else:
        raise ValueError(
            "Choose a different connectivity_key or run first "
            "build.spatial_connectivity(adata) on the AnnData object." % connectivity_key
        )

    clusters = adata.obs[cluster_key].unique().tolist()

    degree_centrality = []
    clustering_coefficient = []
    betweenness_centrality = []
    closeness_centrality = []

    for c in clusters:
        cluster_node_idx = adata[adata.obs[cluster_key] == c].obs.index.tolist()
        # ensuring that cluster_node_idx are List[int]
        cluster_node_idx = [i for i, x in enumerate(cluster_node_idx)]
        subgraph = graph.subgraph(cluster_node_idx)

        centrality = nx.algorithms.centrality.group_degree_centrality(graph, cluster_node_idx)
        degree_centrality.append(centrality)

        clustering = nx.algorithms.cluster.average_clustering(graph, cluster_node_idx)
        clustering_coefficient.append(clustering)

        closeness = nx.algorithms.centrality.group_closeness_centrality(graph, cluster_node_idx)
        closeness_centrality.append(closeness)

        betweenness = nx.betweenness_centrality(subgraph)
        betweenness_centrality.append(sum(betweenness.values()))

    df = pd.DataFrame(
        list(zip(clusters, degree_centrality, clustering_coefficient, closeness_centrality, betweenness_centrality)),
        columns=[
            "cluster",
            "degree centrality",
            "clustering coefficient",
            "closeness centrality",
            "betweenness centrality",
        ],
    )
    adata.uns[key_added] = df


def interaction_matrix(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Union[str, None] = None,
    normalized: bool = False,
    key_added: str = "interaction_matrix",
):
    """
    Compute interaction matrix for clusters. Results are stored in .uns in the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object.
    cluster_key
        Key to clusters in obs.
    connectivity_key
        (Optional) Key to connectivity_matrix in obsp.
    normalized
        (Optional) If True, then each row is normalized by the summation of its values.
    key_added
        (Optional) Key added to output in adata.uns.

    Returns
    -------
    None
    """
    if cluster_key not in adata.obs_keys():
        raise ValueError(
            f"cluster_key {cluster_key} not recognized. Choose a different key refering to a cluster in .obs."
        )

    if "networkx_graph" in adata.uns_keys():
        print("Using saved networkx graph stored under .uns in AnnData object.")
        graph = adata.uns["networkx_graph"]
    elif connectivity_key in adata.obsp:
        graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])
        print(
            f"Saving networkx graph build on adjacency matrix of connectivity_key in .uns under key: "
            f"networkx_graph{connectivity_key}"
        )
        adata.uns["networkx_graph"] = graph
    else:
        raise ValueError(
            f"Networkx graph not found in .uns and connectivity_key {connectivity_key} not recognized. "
            "Choose a different connectivity_key or run first "
            "build.spatial_connectivity(adata) on the AnnData object."
        )

    cluster = {i: {cluster_key: str(x)} for i, x in enumerate(adata.obs[cluster_key].tolist())}
    nx.set_node_attributes(graph, cluster)
    adata.uns[key_added] = nx.attr_matrix(graph, node_attr=cluster_key, normalized=normalized)
