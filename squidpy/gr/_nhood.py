"""Functions for neighborhood enrichment analysis (permutation test, centralities measures etc.)."""

from typing import Any, Tuple, Union, Callable, Iterable, Optional, Sequence
from functools import partial

from scanpy import logging as logg
from anndata import AnnData

from numba import njit, prange  # noqa: F401
import numpy as np
import pandas as pd
import numba.types as nt

import networkx as nx

from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, parallelize, _get_n_cores
from squidpy.gr._utils import (
    _save_data,
    _assert_positive,
    _assert_categorical_obs,
    _assert_connectivity_key,
)
from squidpy._constants._constants import Centrality
from squidpy._constants._pkg_constants import Key

__all__ = ["nhood_enrichment", "centrality_scores", "interaction_matrix"]

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
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
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
    %(parallelize)s

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
    count = _test(indices, indptr, int_clust)

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating neighborhood enrichment using `{n_jobs}` core(s)")

    perms = parallelize(
        _nhood_enrichment_helper,
        collection=np.arange(n_perms),
        extractor=np.vstack,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(callback=_test, indices=indices, indptr=indptr, int_clust=int_clust, n_cls=n_cls, seed=seed)
    zscore = (count - perms.mean(axis=0)) / perms.std(axis=0)

    if copy:
        return zscore, count

    _save_data(
        adata,
        attr="uns",
        key=Key.uns.nhood_enrichment(cluster_key),
        data={"zscore": zscore, "count": count},
        time=start,
    )


@d.dedent
@inject_docs(c=Centrality)
def centrality_scores(
    adata: AnnData,
    cluster_key: str,
    score: Optional[Union[str, Iterable[str]]] = None,
    connectivity_key: Optional[str] = None,
    copy: bool = False,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Compute centrality scores per cluster or cell type.

    Inspired by usage in Gene Regulatory Networks (GRNs) in :cite:`celloracle`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    score
        Centrality measures as described in :class:`networkx.algorithms.centrality` :cite:`networkx`.
        If `None`, use all the options below. Valid options are:

            - `{c.CLOSENESS.s!r}` - measure of how close the group is to other nodes.
            - `{c.CLUSTERING.s!r}` - measure of the degree to which nodes cluster together.
            - `{c.DEGREE.s!r}` - fraction of non-group members connected to group members.

    %(conn_key)s
    %(copy)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame`. Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{{cluster_key}}_centrality_scores']`` - the centrality scores,
          as mentioned above.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    if isinstance(score, (str, Centrality)):
        centrality = [score]
    elif score is None:
        centrality = [c.s for c in Centrality]

    centralities = [Centrality(c) for c in centrality]

    graph = nx.from_scipy_sparse_matrix(adata.obsp[connectivity_key])

    cat = adata.obs[cluster_key].cat.categories.values
    clusters = adata.obs[cluster_key].values

    fun_dict = {}
    for c in centralities:
        if c == Centrality.CLOSENESS:
            fun_dict[c.s] = partial(nx.algorithms.centrality.group_closeness_centrality, graph)
        elif c == Centrality.DEGREE:
            fun_dict[c.s] = partial(nx.algorithms.centrality.group_degree_centrality, graph)
        elif c == Centrality.CLUSTERING:
            fun_dict[c.s] = partial(nx.algorithms.cluster.average_clustering, graph)
        else:
            raise NotImplementedError(f"Centrality `{c}` is not yet implemented.")

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating centralities `{centralities}` using `{n_jobs}` core(s)")

    res_list = []
    for k, v in fun_dict.items():
        df = parallelize(
            _centrality_scores_helper,
            collection=cat,
            extractor=pd.concat,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )(clusters=clusters, fun=v, method=k)
        res_list.append(df)

    df = pd.concat(res_list, axis=1)

    if copy:
        return df
    _save_data(adata, attr="uns", key=Key.uns.centrality_scores(cluster_key), data=df, time=start)


@d.dedent
def interaction_matrix(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: Optional[str] = None,
    normalized: bool = False,
    copy: bool = False,
    weights: bool = False,
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
    weights
        Whether to use edge weights or binarize.

    Returns
    -------
    If ``copy = True``, returns the interaction matrix.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_interactions']`` - the interaction matrix.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    cats = adata.obs[cluster_key]
    mask = ~pd.isnull(cats).values
    cats = cats.loc[mask]
    if not len(cats):
        raise RuntimeError(f"After removing NaNs in `adata.obs[{cluster_key!r}]`, none remain.")

    g = adata.obsp[connectivity_key]
    g = g[mask, :][:, mask]
    n_cats = len(cats.cat.categories)

    if weights:
        g_data = g.data
    else:
        g_data = np.broadcast_to(1, shape=len(g.data))
    if pd.api.types.is_bool_dtype(g.dtype) or pd.api.types.is_integer_dtype(g.dtype):
        dtype = np.intp
    else:
        dtype = np.float_
    output = np.zeros((n_cats, n_cats), dtype=dtype)

    _interaction_matrix(g_data, g.indices, g.indptr, cats.cat.codes.to_numpy(), output)

    if normalized:
        output = output / output.sum(axis=1).reshape((-1, 1))

    if copy:
        return output
    _save_data(adata, attr="uns", key=Key.uns.interaction_matrix(cluster_key), data=output)


@njit
def _interaction_matrix(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, cats: np.ndarray, output: np.ndarray
) -> np.ndarray:
    indices_list = np.split(indices, indptr[1:-1])
    data_list = np.split(data, indptr[1:-1])
    for i in range(len(data_list)):
        cur_row = cats[i]
        cur_indices = indices_list[i]
        cur_data = data_list[i]
        for j, val in zip(cur_indices, cur_data):
            cur_col = cats[j]
            output[cur_row, cur_col] += val
    return output


def _centrality_scores_helper(
    cat: Iterable[Any],
    clusters: Sequence[str],
    fun: Callable[..., float],
    method: str,
    queue: Optional[SigQueue] = None,
) -> pd.DataFrame:
    res_list = []
    for c in cat:
        idx = np.where(clusters == c)[0]
        res = fun(idx)
        res_list.append(res)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame(res_list, columns=[method], index=cat)


def _nhood_enrichment_helper(
    ixs: np.ndarray,
    callback: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    indices: np.ndarray,
    indptr: np.ndarray,
    int_clust: np.ndarray,
    n_cls: int,
    seed: Optional[int] = None,
    queue: Optional[SigQueue] = None,
) -> np.ndarray:
    perms = np.empty((len(ixs), n_cls, n_cls), dtype=np.float64)
    int_clust = int_clust.copy()  # threading
    rs = np.random.RandomState(seed=None if seed is None else seed + ixs[0])

    for i in range(len(ixs)):
        rs.shuffle(int_clust)
        perms[i, ...] = callback(indices, indptr, int_clust)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return perms
