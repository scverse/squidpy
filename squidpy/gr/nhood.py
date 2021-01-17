"""Functions for neighborhood enrichment analysis (permutation test, assortativity measures etc.)."""

from typing import Tuple, Callable, Optional

from anndata import AnnData

from numba import njit, prange  # noqa: F401
import numpy as np
import pandas as pd
import numba.types as nt

import networkx as nx

from squidpy._docs import d
from squidpy.gr._utils import (
    _save_data,
    _assert_positive,
    _assert_categorical_obs,
    _assert_connectivity_key,
)
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

    return globals()[fn_key]  # type: ignore[no-any-return]


@d.get_sections(base="nhood_ench", sections=["Parameters"])
@d.dedent
def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Optional[str] = None,
    n_perms: int = 1000,
    numba_parallel: bool = False,
    seed: Optional[int] = None,
    copy: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute neighborhood enrichment by permutation test.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(conn_key)s
    %(n_perms)s
    %(numba_parallel)s
    %(seed)s
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the z-score and the enrichment count.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['zscore']`` - the enrichment z-score.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['count']`` - the enrichment count.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_positive(n_perms, name="n_perms")

    adj = adata.obsp[connectivity_key]
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}  # map categories
    int_clust = np.array([clust_map[c] for c in original_clust], dtype=ndt)

    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    n_cls = len(clust_map)

    _test = _create_function(n_cls, parallel=numba_parallel)

    perms = np.zeros((n_cls, n_cls, n_perms), dtype=ndt)
    count = _test(indices, indptr, int_clust)

    rs = np.random.RandomState(seed=seed)
    # TODO: parallelize? alternatively incremental mean/std so that we don't need to save all the perms?
    for perm in range(n_perms):
        rs.shuffle(int_clust)
        perms[:, :, perm] = _test(indices, indptr, int_clust)

    zscore = (count - perms.mean(axis=-1)) / perms.std(axis=-1)

    if copy:
        return zscore, count
    _save_data(adata, attr="uns", key=Key.uns.nhood_enrichment(cluster_key), data={"zscore": zscore, "count": count})


@d.dedent
def centrality_scores(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Optional[str] = None,
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Compute centrality scores per cluster or cell type.

    Based among others on methods used for Gene Regulatory Networks (GRNs) in [CellOracle20]_.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(conn_key)s
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the following keys:

        - `'degree_centrality'` - TODO.
        - `'clustering_coefficient'` - TODO.
        - `'closeness_centrality'` - TODO.
        - `'betweenness_centrality'` - TODO.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_centrality_scores']`` - the centrality scores,
          as mentioned above.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])
    clusters = adata.obs[cluster_key].unique()

    degree_centrality = []
    clustering_coefficient = []
    betweenness_centrality = []
    closeness_centrality = []

    for c in clusters:
        cluster_node_idx = np.where(adata.obs[cluster_key] == c)[0]
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
        {
            cluster_key: clusters,
            "degree_centrality": degree_centrality,
            "clustering_coefficient": clustering_coefficient,
            "closeness_centrality": closeness_centrality,
            "betweenness_centrality": betweenness_centrality,
        }
    )

    if copy:
        return df
    _save_data(adata, attr="uns", key=Key.uns.centrality_scores(cluster_key), data=df)


@d.dedent
def interaction_matrix(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Optional[str] = None,
    normalized: bool = True,
    copy: bool = False,
) -> Optional[np.ndarray]:
    """
    Compute interaction matrix for clusters.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(conn_key)s
    normalized
        If `True`, each row is normalized to sum to 1.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns the interaction matrix.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_interactions']`` - the interaction matrix.
    """
    # TODO: improve the return docstring
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])
    cluster = {i: {cluster_key: x} for i, x in enumerate(adata.obs[cluster_key])}

    nx.set_node_attributes(graph, cluster)
    int_mat = np.asarray(
        nx.attr_matrix(
            graph, node_attr=cluster_key, normalized=normalized, rc_order=adata.obs[cluster_key].cat.categories
        )
    )

    if copy:
        return int_mat
    _save_data(adata, attr="uns", key=Key.uns.interaction_matrix(cluster_key), data=int_mat)
