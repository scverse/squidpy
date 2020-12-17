"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)."""

from typing import Tuple, Callable, Optional

from anndata import AnnData

import numpy as np
import pandas as pd
import numba.types as nt
from numba import njit, prange  # noqa: F401
from pandas.api.types import infer_dtype, is_categorical_dtype

import networkx as nx

from squidpy._docs import d
from squidpy.constants._pkg_constants import Key

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


@d.get_sections(base="nhood_ench", sections=["Parameters"])
@d.dedent
def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Optional[str] = Key.obsp.spatial_conn(),
    n_perms: int = 1000,
    numba_parallel: bool = False,
    seed: Optional[int] = None,
    copy: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute neighborhood enrichment by permutation test.

    Parameters
    ----------
    cluster_key
        Cluster key in :attr:`anndata.AnnData.obs`.
    connectivity_key
        Connectivity matrix key in :attr:`anndata.AnnData.obsp`.
    n_perms
        Number of permutations.
    %(numba_parallel)s
    seed
        Random seed.
    %(copy)s

    Returns
    -------
    :class:`tuple`
        zscore and nenrich_count.  TODO: be more verbose
    None
        If ``copy = False``.  TODO: explain where the result is saved.
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

    zscore = (count - perms.mean(axis=-1)) / perms.std(axis=-1)

    if copy:
        return zscore, count

    adata.uns[f"{cluster_key}_nhood_enrichment"] = {"zscore": zscore, "count": count}


# TODO:
# d.keep_params("nhood_ench.parameters", "cluster_key", "connectivity_key")
# https://github.com/Chilipp/docrep/issues/21


@d.dedent
def centrality_scores(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Optional[str] = Key.obsp.spatial_conn(),
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Compute centrality scores per cluster or cell type.

    Based among others on methods used for Gene Regulatory Networks (GRNs) in [CellOracle20]_.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Cluster key in :attr:`anndata.AnnData.obs`.
    connectivity_key
        Connectivity matrix key in :attr:`anndata.AnnData.obsp`.
    %(copy)s

    Returns
    -------
    :class:`pandas.DataFrame`
        The result.
    None
        If ``copy = False``.  TODO: rephrase (e.g. what columns to expect).
        Results are stored in :attr:`anndata.AnnData.uns` [``{cluster_key}_centrality_scores``].
    """
    if cluster_key not in adata.obs_keys():
        raise ValueError(
            f"`cluster_key` {cluster_key!r} not recognized. Choose a different key referring to a cluster in .obs."
        )
    # TODO: check for categorical dtype

    # TODO: unify error messages
    if connectivity_key not in adata.obsp:
        raise KeyError("Choose a different `connectivity_key` or run first `squidpy.gr.spatial_connectivity()`.")
    graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])

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
            "cluster_key",
            "degree_centrality",
            "clustering_coefficient",
            "closeness_centrality",
            "betweenness_centrality",
        ],
    )

    if copy:
        return df

    adata.uns[f"{cluster_key}_centrality_scores"] = df


@d.dedent
def interaction_matrix(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Optional[str] = Key.obsp.spatial_conn(),
    normalized: bool = True,
    copy: bool = False,
) -> Optional[np.matrix]:
    """
    Compute interaction matrix for clusters.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Cluster key in :attr:`anndata.AnnData.obs`.
    connectivity_key
        Connectivity matrix key in :attr:`anndata.AnnData.obsp`.
    normalized
        If `True`, then each row is normalized by the sum of its values.
    %(copy)s

    Returns
    -------
    :class:`np.matrix`
        The interaction matrix.
    None
        If ``copy = False``. Results are in :attr:`anndata.AnnData.uns`. TODO: rephrase
    """
    if cluster_key not in adata.obs_keys():
        raise ValueError(
            f"`cluster_key` {cluster_key!r} not recognized. Choose a different key referring to a cluster in .obs."
        )

    if connectivity_key not in adata.obsp:
        raise KeyError("Choose a different `connectivity_key` or run first `squidpy.build.spatial_connectivity()`.")
    graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])

    cluster = {i: {cluster_key: x} for i, x in enumerate(adata.obs[cluster_key].tolist())}
    # TODO: convert to np.ndarray?
    nx.set_node_attributes(graph, cluster)
    int_mat = nx.attr_matrix(
        graph, node_attr=cluster_key, normalized=normalized, rc_order=adata.obs[cluster_key].cat.categories
    )

    if copy:
        return int_mat

    adata.uns[f"{cluster_key}_interactions"] = int_mat
