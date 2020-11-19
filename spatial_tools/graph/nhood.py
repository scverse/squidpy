"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)."""

from typing import Union, Callable, Optional

import numba.types as nt
from numba import njit, prange  # noqa: F401

from anndata import AnnData

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_categorical_dtype

import networkx as nx

dt = nt.uint32  # data type aliases (both for numpy and numba should match)
ndt = np.uint32
_template = """
@njit(dt[:, :](dt[:], dt[:], dt[:]), parallel={parallel}, fastmath=True)
def _nenrich_{n_cls}_{parallel}(indices: np.ndarray, indptr: np.ndarray, clustering: np.ndarray) -> np.ndarray:
    '''
    Count how many times clusters :math:`i` and :math:`j` are connected.

    Parameters
    ----------
    indices
        :attr:`scipy.sparse.csr_matrix.indices`.
    indptr
        :attr:`scipy.sparse.csr_matrix.indptr`.
    clustering
        Array of shape ``(n_cells,)`` containig cluster labels ranging from `0` to `n_clusters - 1` inclusive.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of shape ``(n_clusters, n_clusters)`` containing the pairwise counts.
    '''
    res = np.zeros((indptr.shape[0] - 1, {n_cls}), dtype=ndt)

    for i in prange(res.shape[0]):
        xs, xe = indptr[i], indptr[i + 1]
        cols = indices[xs:xe]
        for c in cols:
            res[i, clustering[c]] += 1
    {init}
    {loop}
    {finalize}
"""


def _create_function(n_cls: int, parallel: bool = False) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Create a :mod:`numba` function which counts the number of connections between clusters.

    Parameters
    ----------
    n_cls
        Number of clusters. We're assuming that cluster labels are `0`, `1`, ..., `n_cls - 1`.
    parallel
        Whether to enable :mod:`numba` parallelization.

    Returns
    -------
    callable
        The aforementioned function.
    """
    if n_cls <= 1:
        raise ValueError(f"Expected at least `2` clusters, found `{n_cls}`.")

    rng = range(n_cls)
    init = "".join(
        f"""
    g{i} = np.zeros(({n_cls},), dtype=ndt)"""
        for i in rng
    )

    loop_body = """
        if cl == 0:
            g0 += res[row]"""
    loop_body = loop_body + "".join(
        f"""
        elif cl == {i}:
            g{i} += res[row]"""
        for i in range(1, n_cls)
    )
    loop = f"""
    for row in prange(res.shape[0]):
        cl = clustering[row]
        {loop_body}
        else:
            assert False, "Unhandled case."
    """
    finalize = ", ".join(f"g{i}" for i in rng)
    finalize = f"return np.stack(({finalize}))"  # must really be a tuple

    fn_key = f"_nenrich_{n_cls}_{parallel}"
    if fn_key not in globals():
        template = _template.format(init=init, loop=loop, finalize=finalize, n_cls=n_cls, parallel=parallel)
        exec(compile(template, "", "exec"), globals())

    return globals()[fn_key]


def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Union[str, None] = "spatial_connectivities",
    n_perms: int = 1000,
    numba_parallel: Optional[bool] = False,
    seed: Optional[int] = None,
    copy: bool = False,
) -> None:
    """
    Compute neighborhood enrichment by permutation test. Results are stored in .uns in the AnnData.

    Parameters
    ----------
    adata
        The AnnData object.
    cluster_key
        Key to clusters in obs.
    connectivity_key
        (Optional) Key to connectivity_matrix in obsp.
    n_perms
        number of permutations (deafult 1000).
    numba_parallel
        whether to pass parallel=True in numba code
    seed
        Random seed.
    copy
        If `True`, return the result, otherwise save it to the ``adata`` object.

    Returns
    -------
    zscore, nenrich_count
    """
    if cluster_key not in adata.obs.keys():
        raise KeyError(f"Cluster key `{cluster_key}` not found in `adata.obs`.")
    if not is_categorical_dtype(adata.obs[cluster_key]):
        raise TypeError(
            f"Expected `adata.obs[{cluster_key}]` to be `categorical`, "
            f"found `{infer_dtype(adata.obs[cluster_key])}`."
        )

    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"{connectivity_key} not present in `adata.obs`"
            "Choose a different connectivity_key or run first "
            "build.spatial_connectivity(adata) on the AnnData object."
        )
    adj = adata.obsp[connectivity_key]

    original_clust = adata.obs[cluster_key]
    # map categories
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    int_clust = np.array([clust_map[c] for c in original_clust], dtype=ndt)

    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    n_cls = len(clust_map)

    _test = _create_function(n_cls, parallel=numba_parallel)

    perms = np.zeros((n_cls, n_cls, n_perms), dtype=ndt)
    count = _test(indices, indptr, int_clust)

    np.random.seed(seed)  # better way is to use random state (however, it can't be used in the numba function)
    for perm in range(n_perms):
        np.random.shuffle(int_clust)
        perms[:, :, perm] = _test(indices, indptr, int_clust)

    zscore = count - perms.mean(axis=-1) / perms.std(axis=-1)

    adata.uns[f"{cluster_key}_nhood_enrichment"] = {"zscore": zscore, "count": count}

    return (zscore, count) if copy else None


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
    connectivity_key: Union[str, None] = "spatial_connectivities",
    normalized: bool = True,
    copy: bool = False,
) -> None:
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
    copy
        If `True`, return the result, otherwise save it to the ``adata`` object.

    Returns
    -------
    None
    """
    if cluster_key not in adata.obs_keys():
        raise ValueError(
            f"cluster_key {cluster_key} not recognized. Choose a different key refering to a cluster in .obs."
        )

    if connectivity_key in adata.obsp:
        graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])
    else:
        raise ValueError(
            "Choose a different connectivity_key or run first "
            "build.spatial_connectivity(adata) on the AnnData object."
        )

    cluster = {i: {cluster_key: str(x)} for i, x in enumerate(adata.obs[cluster_key].tolist())}
    nx.set_node_attributes(graph, cluster)
    int_mat = nx.attr_matrix(
        graph, node_attr=cluster_key, normalized=normalized, rc_order=adata.obs[cluster_key].cat.categories
    )
    adata.uns[f"{cluster_key}_interactions"] = int_mat

    return None if copy is False else int_mat
