"""Functions for neighborhood enrichment analysis (permutation test, centralities measures etc.)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, Literal, NamedTuple

import numba.types as nt
import numpy as np
import pandas as pd
import rustworkx as rx
from anndata import AnnData
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase
from fast_array_utils.types import HasArrayNamespace as Array
from numba import get_num_threads, njit, prange
from numpy.typing import NDArray
from pandas import CategoricalDtype
from scanpy import logging as logg
from scipy.sparse import csr_array, csr_matrix, diags, issparse, spmatrix
from sklearn.preprocessing import normalize
from spatialdata import SpatialData

from squidpy._constants._constants import Centrality
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import (
    NDArrayA,
    RNGLike,
    SeedLike,
    Signal,
    SigQueue,
    deprecated_randomness_param,
    get_n_processes,
    parallelize,
)
from squidpy._validators import assert_key_in_adata, assert_positive
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_connectivity_key,
    _save_data,
    _shuffle_group,
    extract_adata_if_sdata,
)

__all__ = ["nhood_enrichment", "centrality_scores", "interaction_matrix"]


class NhoodEnrichmentResult(NamedTuple):
    """Result of nhood_enrichment function."""

    zscore: NDArray[np.number]
    counts: NDArray[np.number]  # NamedTuple inherits from tuple so cannot use 'count' as attribute name


# data type aliases (both for numpy and numba should match)
dt = nt.uint32
ndt = np.uint32
_template = """
from __future__ import annotations

from numba import njit, prange
import numpy as np

@njit(dt[:, :](dt[:], dt[:], dt[:]), parallel={parallel}, fastmath=True)
def _nenrich_{n_cls}_{parallel}(indices: NDArrayA, indptr: NDArrayA, clustering: NDArrayA) -> np.ndarray:
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


def _create_function(n_cls: int, parallel: bool = False) -> Callable[[NDArrayA, NDArrayA, NDArrayA], NDArrayA]:
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
@deprecated_randomness_param
def nhood_enrichment(
    adata: AnnData | SpatialData,
    cluster_key: str,
    library_key: str | None = None,
    connectivity_key: str | None = None,
    n_perms: int = 1000,
    numba_parallel: bool = False,
    rng: SeedLike | RNGLike | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    *,
    table_key: str | None = None,
) -> NhoodEnrichmentResult | None:
    """
    Compute neighborhood enrichment by permutation test.

    %(seed_versionchanged)s

    %(rng_versionchanged)s

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    %(cluster_key)s
    %(library_key)s
    %(conn_key)s
    %(n_perms)s
    %(numba_parallel)s
    %(rng)s
    %(copy)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns a :class:`~squidpy.gr.NhoodEnrichmentResult` with the z-score and the enrichment count.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['zscore']`` - the enrichment z-score.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['count']`` - the enrichment count.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)
    assert_positive(n_perms, name="n_perms")

    adj = adata.obsp[connectivity_key]
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}  # map categories
    int_clust = np.array([clust_map[c] for c in original_clust], dtype=ndt)

    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        libraries: pd.Series | None = adata.obs[library_key]
    else:
        libraries = None

    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    n_cls = len(clust_map)

    _test = _create_function(n_cls, parallel=numba_parallel)
    count = _test(indices, indptr, int_clust)

    n_jobs = get_n_processes(n_jobs)
    start = logg.info(f"Calculating neighborhood enrichment using `{n_jobs}` core(s)")
    generators = np.random.default_rng(rng).spawn(n_perms)

    perms = parallelize(
        _nhood_enrichment_helper,
        collection=np.arange(n_perms).tolist(),
        extractor=np.vstack,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(
        callback=_test,
        indices=indices,
        indptr=indptr,
        int_clust=int_clust,
        libraries=libraries,
        n_cls=n_cls,
        generators=generators,
    )
    zscore = (count - perms.mean(axis=0)) / perms.std(axis=0)

    if copy:
        return NhoodEnrichmentResult(zscore=zscore, counts=count)

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
    adata: AnnData | SpatialData,
    cluster_key: str,
    score: str | Iterable[str] | None = None,
    connectivity_key: str | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = False,
    *,
    table_key: str | None = None,
) -> pd.DataFrame | None:
    """
    Compute centrality scores per cluster or cell type.

    Inspired by usage in Gene Regulatory Networks (GRNs) in :cite:`celloracle`.

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    %(cluster_key)s
    score
        Group centrality measures as implemented in ``rustworkx`` :cite:`rustworkx`.
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
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    if isinstance(score, str | Centrality):
        centrality = [score]
    elif score is None:
        centrality = [c.s for c in Centrality]

    centralities = [Centrality(c) for c in centrality]

    # a rustworkx graph mirrors the undirected connectivity graph for the group closeness/degree
    # measures; a symmetric, self-loop-free CSR feeds the clustering-coefficient kernel.
    graph, adj = _build_graph(adata.obsp[connectivity_key])

    cat = adata.obs[cluster_key].cat.categories.values
    clusters = adata.obs[cluster_key].values

    fun_dict = {}
    for c in centralities:
        if c == Centrality.CLOSENESS:
            fun_dict[c.s] = partial(rx.group_closeness_centrality, graph)
        elif c == Centrality.DEGREE:
            fun_dict[c.s] = partial(rx.group_degree_centrality, graph)
        elif c == Centrality.CLUSTERING:
            # average the per-node clustering coefficients over the group (0 if the group is empty).
            node_clustering = _local_clustering(adj.indptr, adj.indices, adj.shape[0])
            fun_dict[c.s] = lambda idx, cc=node_clustering: float(cc[idx].mean()) if len(idx) else 0.0
        else:
            raise NotImplementedError(f"Centrality `{c}` is not yet implemented.")

    n_jobs = get_n_processes(n_jobs)
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
    _save_data(
        adata,
        attr="uns",
        key=Key.uns.centrality_scores(cluster_key),
        data=df,
        time=start,
    )


@d.dedent
def interaction_matrix(
    adata: AnnData | SpatialData,
    cluster_key: str,
    connectivity_key: str | None = None,
    normalized: bool = False,
    copy: bool = False,
    weights: bool = False,
    *,
    table_key: str | None = None,
) -> NDArrayA | None:
    """
    Compute interaction matrix for clusters.

    Parameters
    ----------
    %(adata)s
    %(table_key)s
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
    adata = extract_adata_if_sdata(adata, table_key=table_key)
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

    g_data = g.data if weights else np.broadcast_to(1, shape=len(g.data))
    dtype = int if pd.api.types.is_bool_dtype(g.dtype) or pd.api.types.is_integer_dtype(g.dtype) else float
    output: NDArrayA = np.zeros((n_cats, n_cats), dtype=dtype)

    _interaction_matrix(g_data, g.indices, g.indptr, cats.cat.codes.to_numpy(), output)

    if normalized:
        output = output / output.sum(axis=1).reshape((-1, 1))

    if copy:
        return output

    _save_data(adata, attr="uns", key=Key.uns.interaction_matrix(cluster_key), data=output)


@njit
def _interaction_matrix(
    data: NDArrayA,
    indices: NDArrayA,
    indptr: NDArrayA,
    cats: NDArrayA,
    output: NDArrayA,
) -> NDArrayA:
    indices_list = np.split(indices, indptr[1:-1])
    data_list = np.split(data, indptr[1:-1])
    for i in range(len(data_list)):
        cur_row = cats[i]
        cur_indices = indices_list[i]
        cur_data = data_list[i]
        for j, val in zip(cur_indices, cur_data):  # noqa: B905
            cur_col = cats[j]
            output[cur_row, cur_col] += val
    return output


def _build_graph(conn: Any) -> tuple[rx.PyGraph, csr_matrix]:
    """Build the graph representations used by :func:`centrality_scores`.

    Returns a :class:`rustworkx.PyGraph` mirroring the undirected connectivity graph
    (used by the group closeness/degree measures) and a symmetric, self-loop-free,
    index-sorted CSR matrix feeding the clustering-coefficient kernel.
    """
    from scipy.sparse import triu

    adj = csr_matrix(conn)
    # undirected, unweighted, no self-loops: matches ``networkx.Graph(conn)`` topology.
    adj = (adj + adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.sort_indices()  # the clustering kernel merges neighbor lists, which must be sorted.

    n = adj.shape[0]
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(n))
    # the strict upper triangle lists each undirected edge exactly once.
    rows, cols = triu(adj, k=1).nonzero()
    graph.add_edges_from_no_data([(int(i), int(j)) for i, j in zip(rows, cols, strict=True)])
    return graph, adj


@njit(parallel=True, cache=True)
def _local_clustering(indptr: NDArrayA, indices: NDArrayA, n: int) -> NDArrayA:
    """Local clustering coefficient per node over a symmetric, index-sorted CSR graph.

    ``c_v = 2 * T(v) / (k_v * (k_v - 1))`` where ``T(v)`` is the number of edges among the
    neighbors of ``v`` and ``k_v`` its degree; ``c_v = 0`` when ``k_v < 2``. Triangles are
    counted by intersecting sorted neighbor lists, so only existing edges are ever visited
    (no length-2-path matrix is materialized). Matches :func:`networkx.clustering`.
    """
    out = np.zeros(n, dtype=np.float64)
    for v in prange(n):
        start = indptr[v]
        end = indptr[v + 1]
        k = end - start
        if k < 2:
            continue
        # summing |N(u) ∩ N(v)| over u in N(v) counts each neighbor-neighbor edge twice,
        # so it already equals 2 * T(v) -> c_v = that sum / (k * (k - 1)).
        two_triangles = 0
        for a in range(start, end):
            u = indices[a]
            i = start
            j = indptr[u]
            u_end = indptr[u + 1]
            while i < end and j < u_end:
                if indices[i] == indices[j]:
                    two_triangles += 1
                    i += 1
                    j += 1
                elif indices[i] < indices[j]:
                    i += 1
                else:
                    j += 1
        out[v] = two_triangles / (k * (k - 1))
    return out


def _centrality_scores_helper(
    cat: Iterable[Any],
    clusters: Sequence[str],
    fun: Callable[..., float],
    method: str,
    queue: SigQueue | None = None,
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
    ixs: NDArrayA,
    callback: Callable[[NDArrayA, NDArrayA, NDArrayA], NDArrayA],
    indices: NDArrayA,
    indptr: NDArrayA,
    int_clust: NDArrayA,
    libraries: pd.Series[CategoricalDtype] | None,
    n_cls: int,
    generators: Sequence[np.random.Generator],
    queue: SigQueue | None = None,
) -> NDArrayA:
    perms = np.empty((len(ixs), n_cls, n_cls), dtype=np.float64)
    int_clust = int_clust.copy()  # threading; used as a read-only base for each permutation

    for i, ix in enumerate(ixs):
        # shuffle from the same base with a per-permutation generator, so each permutation is
        # independent of the others and of how the permutations are split across jobs
        rng = generators[ix]
        if libraries is not None:
            shuffled = _shuffle_group(int_clust, libraries, rng)
        else:
            shuffled = int_clust.copy()
            rng.shuffle(shuffled)
        perms[i, ...] = callback(indices, indptr, shuffled)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return perms


@njit(parallel=True, cache=True)
def _bfs_shells(
    indptr: NDArrayA,
    indices: NDArrayA,
    max_hop: int,
    n_threads: int,
    counts: NDArrayA,
    base: NDArrayA,
    rowptr: NDArrayA,
    out: NDArrayA,
    fill: bool,
) -> None:
    """Breadth-first search from every observation, recording the hop each is first reached at.

    Run twice: once with ``fill=False`` to size the output, once with ``fill=True`` to write
    it. Sharing one traversal between the two passes is why the counting and filling logic
    cannot drift apart.
    """
    n = indptr.shape[0] - 1
    stamp = np.full((n_threads, n), -1, dtype=np.int64)
    frontier = np.empty((n_threads, n), dtype=np.int64)
    nxt = np.empty((n_threads, n), dtype=np.int64)
    written = np.zeros((n_threads, max_hop + 1), dtype=np.int64)

    for thread in prange(n_threads):
        for src in range(thread, n, n_threads):
            for hop in range(max_hop + 1):
                written[thread, hop] = 0
            # the source counts as already reached, which is what keeps an out-and-back
            # walk from putting an observation in its own neighborhood
            stamp[thread, src] = src
            frontier[thread, 0] = src
            n_frontier = 1

            for hop in range(1, max_hop + 1):
                n_next = 0
                for f in range(n_frontier):
                    node = frontier[thread, f]
                    for p in range(indptr[node], indptr[node + 1]):
                        neighbor = indices[p]
                        if stamp[thread, neighbor] == src:
                            continue  # a nearer hop already reached it
                        stamp[thread, neighbor] = src
                        nxt[thread, n_next] = neighbor
                        n_next += 1
                        if fill:
                            at = base[hop - 1] + rowptr[hop - 1, src] + written[thread, hop]
                            out[at] = neighbor
                            written[thread, hop] += 1
                        else:
                            counts[hop - 1, src] += 1
                for g in range(n_next):
                    frontier[thread, g] = nxt[thread, g]
                n_frontier = n_next
                if n_frontier == 0:
                    break


def _shell_adjacencies(adj: spmatrix, max_hop: int) -> list[spmatrix]:
    adj = adj.tocsr()
    n = adj.shape[0]
    indptr, indices = adj.indptr.astype(np.int64), adj.indices.astype(np.int64)

    counts = np.zeros((max_hop, n), dtype=np.int64)
    empty = np.zeros(1, dtype=np.int64)
    n_threads = get_num_threads()
    _bfs_shells(indptr, indices, max_hop, n_threads, counts, empty, counts, empty, False)

    rowptr = np.zeros((max_hop, n + 1), dtype=np.int64)
    np.cumsum(counts, axis=1, out=rowptr[:, 1:])
    per_hop = rowptr[:, -1]
    base = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(per_hop)))

    out = np.empty(int(base[-1]), dtype=np.int64)
    _bfs_shells(indptr, indices, max_hop, n_threads, counts, base, rowptr, out, True)

    shells: list[spmatrix] = []
    for hop in range(max_hop):
        lo, hi = int(base[hop]), int(base[hop + 1])
        shell = csr_matrix(
            (np.ones(hi - lo, dtype=bool), out[lo:hi], rowptr[hop]),
            shape=(n, n),
        )
        shell.sort_indices()  # breadth-first order is not sorted order
        shells.append(shell)

    # The search marks the source visited before it starts, which is what stops an
    # out-and-back walk from putting an observation in its own later hops. That also drops
    # a self-loop the caller gave us, so hop 1 is taken from the graph itself: it is the
    # direct neighborhood by definition, self-loops included.
    shells[0] = adj.astype(bool)
    return shells


def _compute_hop_adjacency_matrices(
    adjacency_matrix_orig: spmatrix | NDArrayA,
    max_hop: int,
) -> list[spmatrix]:
    """Compute a sequence of 'new-connections-only' adjacency matrices for increasing hop distances.

    Parameters
    ----------
    adjacency_matrix
        The 1-hop (direct neighbor) adjacency matrix. Used as-is: if it has an
        explicit self-loop (diagonal == 1), that is respected and preserved in
        the output.
    max_hop
        Number of hop levels to compute (>= 1).

    Returns
    -------
    A list ``adj_mat_list`` of length ``max_hop`` where:

    - ``adj_mat_list[0]`` is ``adjacency_matrix``, as booleans.
    - ``adj_mat_list[k]`` (k >= 1) has a 1 at ``(i, j)`` iff cell ``i`` and ``j``
      are reachable in exactly ``k + 1`` hops *and* were not already connected
      in any of ``adj_mat_list[0], ..., adj_mat_list[k-1]``.

    Notes
    -----
    A breadth-first search reaches each cell once, at its shortest
    distance, so the hops are disjoint by construction -- there is no "visited" matrix to
    subtract, and a cell cannot reach itself via an out-and-back path.
    """
    if max_hop < 1:
        raise ValueError(f"max_hop must be >= 1, got {max_hop}.")

    adjacency_matrix = adjacency_matrix_orig if issparse(adjacency_matrix_orig) else csr_array(adjacency_matrix_orig)
    return _shell_adjacencies(adjacency_matrix, max_hop)


def _onehot(labels: pd.Series) -> csr_matrix:
    """Indicator matrix of ``labels``, one column per category."""
    codes = labels.astype("category").cat.codes.to_numpy()
    keep = codes >= 0
    return csr_matrix(
        (np.ones(keep.sum()), (np.flatnonzero(keep), codes[keep])),
        shape=(len(codes), len(labels.astype("category").cat.categories)),
    )


def _aggregate_over(
    adj: CSBase, features: Array | CSBase, aggregation: Literal["mean", "sum", "variance"]
) -> Array | CSBase:
    """Aggregate *features* over the neighborhood each row of *adj* defines."""
    if aggregation == "sum":
        return adj @ features
    normalized = normalize(adj, norm="l1", axis=1)
    if aggregation == "mean":
        return normalized @ features
    if aggregation == "variance":
        mean = to_dense(normalized @ features)
        dense = to_dense(features)
        return to_dense(normalized @ (dense * dense)) - mean * mean
    raise ValueError(f"'aggregation' must be 'mean', 'sum' or 'variance', got {aggregation!r}")


def _nhood_blocks(
    adata: AnnData,
    *,
    groups: str | None = None,
    use_rep: str | None = None,
    layer: str | None = None,
    connectivity_key: str = Key.obsp.spatial_conn(),
    hops: Sequence[int] = (1,),
    hop_mode: Literal["shell", "power"] = "shell",
    aggregation: Literal["mean", "sum", "variance"] = "mean",
) -> list[Array | CSBase]:
    """One aggregated block per requested hop, in the order given."""
    _assert_connectivity_key(adata, connectivity_key)
    if not len(hops):
        raise ValueError("'hops' must name at least one hop")
    if any(hop < 0 for hop in hops):
        raise ValueError(f"'hops' must be non-negative, got {list(hops)!r}")

    given = [name for name, value in (("groups", groups), ("use_rep", use_rep), ("layer", layer)) if value is not None]
    if len(given) > 1:
        raise ValueError(f"pass at most one of 'groups', 'use_rep' and 'layer', got {given}")
    # `has_value` says which observations have something to contribute; only a category can
    # be unassigned, so a feature matrix leaves every row valid
    has_value = None
    if groups is not None:
        assert_key_in_adata(adata, groups, attr="obs")
        features = _onehot(adata.obs[groups])
        has_value = np.asarray(features.sum(axis=1)).ravel() != 0
    elif use_rep is not None:
        assert_key_in_adata(adata, use_rep, attr="obsm")
        features = adata.obsm[use_rep]
    elif layer is not None:
        assert_key_in_adata(adata, layer, attr="layers")
        features = adata.layers[layer]
    else:
        features = adata.X

    by_hop: dict[int, CSBase | None] = {0: None}
    if max(hops) >= 1:
        adj = adata.obsp[connectivity_key]
        if hop_mode == "shell":
            adjacencies = _compute_hop_adjacency_matrices(adj, max(hops))
        elif hop_mode == "power":
            # numeric on purpose: hop k counts every walk of length k, and that
            # multiplicity is what weighs a near neighbor more in a summed profile
            adjacencies, power = [adj], adj
            for _ in range(1, max(hops)):
                power = power @ adj
                adjacencies.append(power)
        else:
            raise ValueError(f"'hop_mode' must be 'shell' or 'power', got {hop_mode!r}")
        by_hop |= dict(enumerate(adjacencies, start=1))
    if has_value is not None:
        keep = diags(has_value.astype(float))
        by_hop = {hop: adj if adj is None else (adj @ keep).tocsr() for hop, adj in by_hop.items()}

    # hop 0 is the observation itself, so it contributes its features unaggregated
    return [features if hop == 0 else _aggregate_over(by_hop[hop], features, aggregation) for hop in hops]


def nhood_aggregate(
    adata: AnnData,
    *,
    groups: str | None = None,
    use_rep: str | None = None,
    layer: str | None = None,
    connectivity_key: str = Key.obsp.spatial_conn(),
    hops: Sequence[int] = (1,),
    aggregation: Literal["mean", "sum", "variance"] = "mean",
    hop_weights: Sequence[float] | None = None,
) -> Array | CSBase:
    """Summarise each neighborhood into one block of ``n_features`` columns.

    The hops are summed, weighted by *hop_weights*. Disjoint rings are the other way to
    read a neighborhood and belong with the stacking path, so this one always takes
    matrix powers; see ``_nhood_blocks``.
    """
    blocks = _nhood_blocks(
        adata,
        groups=groups,
        use_rep=use_rep,
        layer=layer,
        connectivity_key=connectivity_key,
        hops=hops,
        # matrix powers, not disjoint rings: the hops are summed here, so a cell
        # reachable by several short paths is meant to weigh more
        hop_mode="power",
        aggregation=aggregation,
    )
    weights = [1.0] * len(blocks) if hop_weights is None else list(hop_weights)
    if len(weights) < len(blocks):
        raise ValueError(
            f"Number of weights provided is less than hops requested. n_hop_weights = {weights} "
            f"is less than the {len(blocks)} hops"
        )
    if len(weights) > len(blocks):
        raise ValueError(f"'hop_weights' has {len(weights)} values but there are {len(blocks)} hops")
    # keep the container the features came in; `variance` has already densified, so a
    # mixed set of blocks has to be densified whole
    if not all(issparse(block) for block in blocks):
        blocks = [to_dense(block) for block in blocks]
    total = sum(weight * block for weight, block in zip(weights, blocks, strict=True))
    # a weighted mean over the hops; counts are meant to stay counts
    return total if aggregation == "sum" else total / sum(weights)
