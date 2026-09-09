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
from numba import njit, prange
from numpy.typing import NDArray
from pandas import CategoricalDtype
from scanpy import logging as logg
from scipy.sparse import csr_matrix, issparse, spmatrix
from scipy.stats import entropy
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

__all__ = ["nhood_enrichment", "centrality_scores", "interaction_matrix", "nhood_entropy", "nhood_aggregate"]


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


def _onehot(labels: pd.Series) -> csr_matrix:
    """Indicator matrix of ``labels``, one column per category.

    Observations whose label is unassigned (``NaN``) get an all-zero row, so they are
    counted in nobody's neighborhood.
    """
    codes = labels.astype("category").cat.codes.to_numpy()
    keep = codes >= 0
    return csr_matrix(
        (np.ones(keep.sum()), (np.flatnonzero(keep), codes[keep])),
        shape=(len(codes), len(labels.astype("category").cat.categories)),
    )


def _nhood_profile(labels: pd.Series, adj: csr_matrix, *, normalize: bool = True) -> pd.DataFrame:
    """Frequency of every ``labels`` category in each observation's neighborhood.

    This is ``adj @ onehot(labels)``. Observations whose label is unassigned (``NaN``) are
    counted in nobody's neighborhood, and with ``normalize`` an observation without neighbors
    gets an all-zero row rather than ``NaN``.
    """
    labels = labels.astype("category")
    profile = pd.DataFrame((adj @ _onehot(labels)).toarray(), index=labels.index, columns=labels.cat.categories)
    if not normalize:
        return profile
    return profile.div(profile.sum(axis=1), axis=0).fillna(0.0)


@d.dedent
def nhood_entropy(
    adata: AnnData | SpatialData,
    cluster_key: str,
    connectivity_key: str | None = None,
    copy: bool = False,
    *,
    table_key: str | None = None,
) -> pd.Series | None:
    """
    Compute the Shannon entropy of each observation's neighborhood composition.

    High entropy marks a mixed neighborhood, low entropy a homogeneous domain; the mean over
    all observations summarises how spatially coherent a clustering is.

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    %(cluster_key)s
    %(conn_key)s
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.Series`. Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.obs` ``['{cluster_key}_nhood_entropy']`` - the per-observation entropy, in nats.

    Notes
    -----
    The neighborhood is whatever ``connectivity_key`` holds; keep it fixed when sweeping a
    clustering parameter.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    start = logg.info(f"Calculating neighborhood entropy of `{cluster_key}`")
    profile = _nhood_profile(adata.obs[cluster_key], adata.obsp[connectivity_key])
    # observations without neighbors give 0/0 in `entropy`
    ent = pd.Series(np.nan_to_num(entropy(profile.to_numpy(), axis=1)), index=adata.obs_names)

    if copy:
        return ent
    _save_data(adata, attr="obs", key=f"{cluster_key}_nhood_entropy", data=ent, time=start)
    return None


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


def _shell_hop(adj_hop: spmatrix, adj: spmatrix, adj_visited: spmatrix) -> tuple[spmatrix, spmatrix]:
    """One step out, dropping what earlier hops already reached."""
    adj_hop = adj_hop @ adj
    adj_hop = adj_hop > adj_visited
    return adj_hop, adj_visited + adj_hop


def _densify(matrix: Any) -> NDArrayA:
    """Dense view of *matrix*, which may already be dense."""
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def _hop_adjacencies(
    adj: spmatrix, hops: Sequence[int], hop_mode: Literal["power", "shell"]
) -> dict[int, spmatrix | None]:
    """The adjacency of each requested hop; ``None`` for hop 0, which is no neighborhood.

    ``power`` takes matrix powers of *adj*, so hop *k* counts every walk of length *k* and
    a near neighbor keeps contributing to the far hops. ``shell`` subtracts what earlier
    hops already reached, as CellCharter does, making the hops disjoint rings.
    """
    if any(hop < 0 for hop in hops):
        raise ValueError(f"'hops' must be non-negative, got {list(hops)!r}")

    by_hop: dict[int, spmatrix | None] = {0: None}
    if max(hops, default=0) < 1:
        return by_hop

    if hop_mode == "power":
        current = adj
        by_hop[1] = current
        for hop in range(2, max(hops) + 1):
            current = current @ adj
            by_hop[hop] = current
    elif hop_mode == "shell":
        # CellCharter starts from the graph without self-loops, and counts every
        # observation as already having visited itself. `setdiag` on the CSR directly:
        # the diagonal of a kNN graph is empty, and filling it neither warns nor differs
        # from the `tolil()` roundtrip it used to take.
        adj_hop, adj_visited = adj.copy(), adj.copy()
        adj_hop.setdiag(0)
        adj_hop.eliminate_zeros()
        adj_visited.setdiag(1)
        by_hop[1] = adj_hop
        for hop in range(2, max(hops) + 1):
            adj_hop, adj_visited = _shell_hop(adj_hop, adj, adj_visited)
            by_hop[hop] = adj_hop
    else:
        raise ValueError(f"'hop_mode' must be 'power' or 'shell', got {hop_mode!r}")
    return by_hop


def _aggregate_over(adj: spmatrix, features: Any, aggregation: Literal["mean", "sum", "variance"]) -> Any:
    """Aggregate *features* over the neighborhood each row of *adj* defines."""
    if aggregation == "sum":
        return adj @ features
    # rows sum to 1, so a high degree does not dominate the aggregate
    normalized = normalize(adj, norm="l1", axis=1)
    if aggregation == "mean":
        return normalized @ features
    if aggregation == "variance":
        mean = _densify(normalized @ features)
        dense = _densify(features)
        return _densify(normalized @ (dense * dense)) - mean * mean
    raise ValueError(f"'aggregation' must be 'mean', 'sum' or 'variance', got {aggregation!r}")


def _resolve_hop_weights(hop_weights: Sequence[float] | None, n_hops: int) -> list[float]:
    """One weight per hop, padding a short list with its last value."""
    if hop_weights is None:
        return [1.0] * n_hops
    weights = list(hop_weights)
    if len(weights) > n_hops:
        raise ValueError(f"'hop_weights' has {len(weights)} values but there are {n_hops} hops")
    if len(weights) < n_hops:
        # a short list is more likely a mistake than an intention, so say so out loud
        logg.warning(f"'hop_weights' has {len(weights)} values for {n_hops} hops; padding with {weights[-1]}")
        weights += [weights[-1]] * (n_hops - len(weights))
    return weights


def _nhood_features(adata: AnnData, groups: str | None, use_rep: str | None, layer: str | None) -> Any:
    """The matrix the neighborhoods are aggregated over."""
    given = [name for name, value in (("groups", groups), ("use_rep", use_rep), ("layer", layer)) if value is not None]
    if len(given) > 1:
        raise ValueError(f"pass at most one of 'groups', 'use_rep' and 'layer', got {given}")
    if groups is not None:
        # any dtype: `_onehot` coerces, as the neighborhood profile always has
        assert_key_in_adata(adata, groups, attr="obs")
        return _onehot(adata.obs[groups])
    if use_rep is not None:
        assert_key_in_adata(adata, use_rep, attr="obsm")
        return adata.obsm[use_rep]
    if layer is not None:
        assert_key_in_adata(adata, layer, attr="layers")
        return adata.layers[layer]
    return adata.X


def _nhood_aggregate(
    adata: AnnData,
    *,
    groups: str | None = None,
    use_rep: str | None = None,
    layer: str | None = None,
    connectivity_key: str = "spatial_connectivities",
    hops: Sequence[int] = (1,),
    hop_mode: Literal["power", "shell"] = "power",
    combine: Literal["concat", "sum"] = "concat",
    hop_weights: Sequence[float] | None = None,
    aggregation: Literal["mean", "sum", "variance"] = "mean",
) -> NDArrayA:
    """The aggregated matrix, without touching *adata*. See :func:`nhood_aggregate`."""
    _assert_connectivity_key(adata, connectivity_key)
    if not len(hops):
        raise ValueError("'hops' must name at least one hop")

    features = _nhood_features(adata, groups, use_rep, layer)
    by_hop = _hop_adjacencies(adata.obsp[connectivity_key], hops, hop_mode)
    # hop 0 is the observation itself, so it contributes the features unaggregated -- which
    # is what makes `variance` over it 0 rather than meaningful
    blocks = [features if hop == 0 else _aggregate_over(by_hop[hop], features, aggregation) for hop in hops]

    if combine == "concat":
        return np.hstack([_densify(block) for block in blocks])
    if combine != "sum":
        raise ValueError(f"'combine' must be 'concat' or 'sum', got {combine!r}")

    weights = _resolve_hop_weights(hop_weights, len(blocks))
    total = sum(weight * _densify(block) for weight, block in zip(weights, blocks, strict=True))
    # a weighted mean over the hops, so the scale does not depend on how many there are.
    # `sum` is counts, which are meant to stay counts.
    return total if aggregation == "sum" else total / sum(weights)


@d.dedent
def nhood_aggregate(
    data: AnnData | SpatialData,
    *,
    groups: str | None = None,
    use_rep: str | None = None,
    layer: str | None = None,
    connectivity_key: str = "spatial_connectivities",
    hops: Sequence[int] = (1,),
    hop_mode: Literal["power", "shell"] = "power",
    combine: Literal["concat", "sum"] = "concat",
    hop_weights: Sequence[float] | None = None,
    aggregation: Literal["mean", "sum", "variance"] = "mean",
    key_added: str = "X_nhood",
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Summarise each observation's spatial neighborhood into a feature matrix.

    The step every niche-calling method starts from: what is around an observation,
    expressed as numbers it can be clustered on. The flavors of
    :func:`~squidpy.gr.calculate_niche` differ in how they answer that, and each is a
    choice of the arguments below.

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    groups
        Column in :attr:`~anndata.AnnData.obs` whose categories are counted in each
        neighborhood -- cell types, typically. Mutually exclusive with *use_rep* and
        *layer*; the features default to :attr:`~anndata.AnnData.X`.
    use_rep
        Key in :attr:`~anndata.AnnData.obsm` holding the features to aggregate.
    layer
        Key in :attr:`~anndata.AnnData.layers` holding the features to aggregate.
    connectivity_key
        Key in :attr:`~anndata.AnnData.obsp` holding the spatial graph.
    hops
        Which neighborhood hops to aggregate. ``0`` is the observation itself, and
        contributes its own features unaggregated.
    hop_mode
        ``'power'`` takes matrix powers of the graph, so hop *k* counts every walk of
        length *k*. ``'shell'`` subtracts what nearer hops already reached, so the hops are
        disjoint rings.
    combine
        ``'concat'`` puts the hops side by side, giving one block of columns each;
        ``'sum'`` adds them into one block, weighted by *hop_weights*.
    hop_weights
        One weight per hop for ``combine='sum'``. A short list is padded with its last
        value, and defaults to equal weights.
    aggregation
        How the neighbors' features are combined: ``'mean'``, ``'sum'`` (counts), or the
        ``'variance'`` over the neighborhood.
    key_added
        Key in :attr:`~anndata.AnnData.obsm` to write the matrix to.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata``. Otherwise, modifies the ``adata``
    with the following key:

        - :attr:`anndata.AnnData.obsm` ``['{key_added}']`` - the aggregated matrix, one
          row per observation.

    Notes
    -----
    The three built-in niche flavors are each one call of this function followed by a
    scaling step:

    - ``neighborhood``: ``groups=...``, ``hops=range(1, k + 1)``, ``combine='sum'``,
      then :func:`~scanpy.pp.scale`.
    - ``utag``: the defaults, then :func:`~scanpy.tl.pca`.
    - ``cellcharter``: ``hops=range(0, k + 1)``, ``hop_mode='shell'``, then
      :func:`~scanpy.tl.pca`.

    See Also
    --------
    calculate_niche : Niche calling, which starts from this.
    nhood_entropy : How mixed each neighborhood is, over the same graph.
    """
    adata = extract_adata_if_sdata(data, table_key=table_key)
    adata = adata.copy() if copy else adata

    start = logg.info(f"Aggregating neighborhoods over hops `{list(hops)}`")
    aggregated = _nhood_aggregate(
        adata,
        groups=groups,
        use_rep=use_rep,
        layer=layer,
        connectivity_key=connectivity_key,
        hops=hops,
        hop_mode=hop_mode,
        combine=combine,
        hop_weights=hop_weights,
        aggregation=aggregation,
    )
    _save_data(adata, attr="obsm", key=key_added, data=aggregated, time=start)
    return adata if copy else None
