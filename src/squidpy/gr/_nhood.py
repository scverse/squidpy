"""Functions for neighborhood enrichment analysis (permutation test, centralities measures etc.)."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, NamedTuple

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit, prange, set_num_threads
from numba_progress import ProgressBar
from numpy.typing import NDArray
from pandas import CategoricalDtype
from scanpy import logging as logg
from spatialdata import SpatialData

from squidpy._constants._constants import Centrality
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, Signal, SigQueue, _get_n_cores, parallelize
from squidpy._validators import assert_positive
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_connectivity_key,
    _save_data,
    extract_adata_if_sdata,
)

__all__ = ["nhood_enrichment", "centrality_scores", "interaction_matrix"]


class NhoodEnrichmentResult(NamedTuple):
    """Result of nhood_enrichment function."""

    zscore: NDArray[np.number]
    """Z-score values of enrichment statistic."""
    counts: NDArray[np.number]  # NamedTuple inherits from tuple so cannot use 'count' as attribute name
    """Enrichment count."""
    conditional_ratio: NDArray[np.number] | None = None
    """Conditional ratio. Only present if ``normalization='conditional'``."""


# integer dtype used for cluster labels and CSR index arrays (numpy/numba must match)
ndt = np.uint32


@njit(nogil=True, cache=True)
def _nenrich(indices: NDArrayA, indptr: NDArrayA, clustering: NDArrayA, n_cls: int) -> NDArrayA:
    """Count how many times clusters are connected.

    Parameters
    ----------
    indices
        :attr:`scipy.sparse.csr_matrix.indices`.
    indptr
        :attr:`scipy.sparse.csr_matrix.indptr`.
    clustering
        Array of shape ``(n_cells,)`` containing cluster labels ranging from ``0`` to ``n_cls - 1`` inclusive.
    n_cls
        Number of clusters.

    Returns
    -------
    Array of shape ``(n_cls, n_cls)`` where entry ``(a, b)`` is the number of directed edges
    from a cluster-``a`` cell to a cluster-``b`` neighbor.
    """
    out = np.zeros((n_cls, n_cls), dtype=np.uint32)
    for i in range(indptr.shape[0] - 1):
        a = clustering[i]
        for c in indices[indptr[i] : indptr[i + 1]]:
            out[a, clustering[c]] += 1
    return out


@njit(nogil=True, cache=True)
def _conditional_counts(indices: NDArrayA, indptr: NDArrayA, clustering: NDArrayA, n_cls: int) -> NDArrayA:
    """Count, per cluster pair ``(a, b)``, how many cluster-``a`` cells have at least one cluster-``b`` neighbor.

    This is the COZI conditional denominator. Returns an ``(n_cls, n_cls)`` float array.
    """
    cond = np.zeros((n_cls, n_cls), dtype=np.float64)
    seen = np.zeros(n_cls, dtype=np.bool_)
    for i in range(indptr.shape[0] - 1):
        seen[:] = False
        for c in indices[indptr[i] : indptr[i + 1]]:
            seen[clustering[c]] = True
        a = clustering[i]
        for b in range(n_cls):
            if seen[b]:
                cond[a, b] += 1.0
    return cond


@njit(parallel=True, nogil=True)
def _permutation_counts(
    indices: NDArrayA,
    indptr: NDArrayA,
    int_clust: NDArrayA,
    group_offsets: NDArrayA,
    group_indices: NDArrayA,
    n_cls: int,
    n_perms: int,
    seed: int,
    norm_code: int,
    progress: ProgressBar,
) -> NDArrayA:
    perms = np.empty((n_perms, n_cls, n_cls), dtype=np.float64)
    for p in prange(n_perms):
        np.random.seed(seed + p)
        shuffled = int_clust.copy()
        for g in range(group_offsets.shape[0] - 1):
            s, e = group_offsets[g], group_offsets[g + 1]
            sub = np.empty(e - s, dtype=int_clust.dtype)
            for t in range(e - s):
                sub[t] = int_clust[group_indices[s + t]]
            np.random.shuffle(sub)
            for t in range(e - s):
                shuffled[group_indices[s + t]] = sub[t]

        out = np.zeros((n_cls, n_cls), dtype=np.float64)
        for i in range(indptr.shape[0] - 1):
            a = shuffled[i]
            for c in indices[indptr[i] : indptr[i + 1]]:
                out[a, shuffled[c]] += 1.0

        if norm_code == 1:  # total
            for a in range(n_cls):
                s = 0.0
                for b in range(n_cls):
                    s += out[a, b]
                if s == 0.0:
                    s = 1.0
                for b in range(n_cls):
                    out[a, b] /= s
        elif norm_code == 2:  # conditional
            cond = _conditional_counts(indices, indptr, shuffled, n_cls)
            for a in range(n_cls):
                for b in range(n_cls):
                    d = cond[a, b] if cond[a, b] != 0.0 else 1.0
                    out[a, b] /= d

        perms[p] = out
        progress.update(1)
    return perms


_NORM_CODES = {"none": 0, "total": 1, "conditional": 2}


def _filter_clusters_by_min_cell_count(
    adata: AnnData,
    int_clust: NDArrayA,
    connectivity_key: str,
    min_cell_count: int,
) -> tuple[NDArrayA, NDArrayA, NDArrayA]:
    clust_sizes = pd.Series(int_clust).value_counts()
    valid_clusters = clust_sizes[clust_sizes >= min_cell_count].index.to_numpy()

    valid_mask = np.isin(int_clust, valid_clusters)
    valid_cells_idx = np.where(valid_mask)[0]
    int_clust = int_clust[valid_mask]

    adj = adata.obsp[connectivity_key][np.ix_(valid_cells_idx, valid_cells_idx)]
    return int_clust, adj, valid_mask


@d.get_sections(base="nhood_ench", sections=["Parameters"])
@d.dedent
def nhood_enrichment(
    adata: AnnData | SpatialData,
    cluster_key: str,
    library_key: str | None = None,
    connectivity_key: str | None = None,
    n_perms: int = 1000,
    numba_parallel: bool = False,
    seed: int | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str | None = None,
    normalization: str = "none",
    min_cell_count: int = 0,
    handle_nan: str = "keep",
    show_progress_bar: bool = True,
    *,
    table_key: str | None = None,
) -> NhoodEnrichmentResult | None:
    """
    Compute neighborhood enrichment by permutation test.

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    %(cluster_key)s
    %(library_key)s
    %(conn_key)s
    %(n_perms)s
    %(numba_parallel)s
    %(seed)s
    %(copy)s
    %(parallelize)s
    normalization
        Normalization mode to use:
        - ``'none'``: No normalization of neighbor counts
        - ``'total'``: Normalize neighbor counts by total number of cells per cluster (SEA)
        - ``'conditional'``: Normalize neighbor counts by number of cells with at least one neighbor of given type (COZI)
    min_cell_count
        Minimum number of cells a cluster must contain to be included. Clusters with fewer cells are
        dropped before counting (default ``0`` keeps all clusters).
    handle_nan
        How to handle NaN values in z-scores:
        - ``'zero'``: Replace NaN values with 0
        - ``'keep'``: Keep NaN values (undefined enrichment)

    Returns
    -------
    If ``copy = True``, returns a :class:`~squidpy.gr.NhoodEnrichmentResult` with the z-score and the enrichment count.
    If normalization = "conditional", also contains the conditional ratio, otherwise it is None.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['zscore']`` - the enrichment z-score.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['count']`` - the enrichment count.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['conditional_ratio']`` - the ratio of cells of type A that neighbor type B.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)
    assert_positive(n_perms, name="n_perms")

    if numba_parallel:
        warnings.warn(
            "`numba_parallel` is deprecated and no longer has any effect; permutations are now "
            "parallelized across threads. It will be removed in a future version.",
            FutureWarning,
            stacklevel=2,
        )
    if backend is not None:
        warnings.warn(
            "`backend` is deprecated and no longer has any effect; permutations now run on a "
            "thread pool. It will be removed in a future version.",
            FutureWarning,
            stacklevel=2,
        )

    adj = adata.obsp[connectivity_key]
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    int_clust = np.array([clust_map[c] for c in original_clust], dtype=ndt)
    n_total_cells = len(int_clust)

    int_clust, adj, valid_mask = _filter_clusters_by_min_cell_count(
        adata=adata,
        int_clust=int_clust,
        connectivity_key=connectivity_key,
        min_cell_count=min_cell_count,
    )
    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        # subset to the kept cells so the per-cell series stays aligned with the filtered
        libraries: pd.Series | None = adata.obs[library_key].iloc[valid_mask].cat.remove_unused_categories()
    else:
        libraries = None

    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    n_cls = len(clust_map)
    if n_cls <= 1:
        raise ValueError(f"Expected at least `2` clusters, found `{n_cls}`.")

    count = _nenrich(indices, indptr, int_clust, n_cls)
    conditional_ratio = np.full((n_cls, n_cls), np.nan, dtype=np.float64)

    if normalization == "total":
        row_sums = count.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        count_normalized = count / row_sums
    elif normalization == "conditional":
        cond_counts = _conditional_counts(indices, indptr, int_clust, n_cls)

        cluster_sizes = np.bincount(int_clust, minlength=n_cls).astype(np.float64)
        nonempty = cluster_sizes > 0
        conditional_ratio[nonempty] = cond_counts[nonempty] / cluster_sizes[nonempty, None]

        safe_cond_counts = cond_counts.copy()
        safe_cond_counts[safe_cond_counts == 0] = 1.0

        count_normalized = count / safe_cond_counts

        n_retained_cells = len(int_clust)
        n_filtered = n_total_cells - n_retained_cells
        frac_filtered = n_filtered / n_total_cells * 100

        if n_filtered > 0:
            warnings.warn(
                f"{frac_filtered:.3f}% of cells were excluded because their clusters had fewer than {min_cell_count} cells.",
                UserWarning,
                stacklevel=2,
            )

    elif normalization == "none":
        count_normalized = count.copy()
    else:
        raise ValueError(f"Invalid normalization mode `{normalization}`. Choose from 'none', 'total', 'conditional'.")

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating neighborhood enrichment using `{n_jobs}` core(s)")

    # Every permutation is seeded by its global index (``seed + p``), so results are reproducible
    # and independent of the thread count. A user seed is used directly; otherwise a random base
    # seed is drawn once so a single call is internally consistent.
    norm_code = _NORM_CODES[normalization]
    base_seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0]) % 1_000_000_000

    # Group structure for within-group shuffling, as a CSR-like (offsets, indices) pair in category
    # order with ascending indices per group (matching `_shuffle_group`). Without a `library_key`,
    # a single group spanning all cells reproduces a plain global shuffle.
    group_offsets, group_indices = _build_shuffle_groups(libraries, len(int_clust))

    # A single numba ``prange`` kernel shuffles + counts + normalizes per thread with the GIL
    # released, and ticks the progress bar from inside the loop; numba owns the parallelism.
    set_num_threads(n_jobs)
    with ProgressBar(total=n_perms, unit="perm", desc="nhood_enrichment", disable=not show_progress_bar) as progress:
        perms = _permutation_counts(
            indices, indptr, int_clust, group_offsets, group_indices, n_cls, n_perms, base_seed, norm_code, progress
        )

    std = perms.std(axis=0)
    std[std == 0] = np.nan
    zscore = (count_normalized - perms.mean(axis=0)) / std

    if handle_nan == "zero":
        zscore = np.nan_to_num(zscore, nan=0.0)
    elif handle_nan == "keep":
        pass
    else:
        raise ValueError("handle_nan must be 'keep' or 'zero'")

    result_kwargs = {"zscore": zscore, "count": count}
    if normalization == "conditional":
        result_kwargs["conditional_ratio"] = conditional_ratio

    if copy:
        return NhoodEnrichmentResult(
            zscore=result_kwargs["zscore"],
            counts=result_kwargs["count"],
            conditional_ratio=result_kwargs.get("conditional_ratio"),
        )

    _save_data(
        adata,
        attr="uns",
        key=Key.uns.nhood_enrichment(cluster_key),
        data=result_kwargs,
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
        Centrality measures as described in :mod:`networkx.algorithms.centrality` :cite:`networkx`.
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

    graph = nx.Graph(adata.obsp[connectivity_key])

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


def _build_shuffle_groups(
    libraries: pd.Series[CategoricalDtype] | None,
    n_cells: int,
) -> tuple[NDArrayA, NDArrayA]:
    """Build a CSR-like ``(offsets, indices)`` description of the within-group shuffling.

    ``indices[offsets[g]:offsets[g + 1]]`` are the cell indices of group ``g`` in ascending order,
    with groups in category order — matching :func:`squidpy.gr._utils._shuffle_group`. Without a
    ``library_key`` there is a single group spanning all cells, which reproduces a global shuffle.
    """
    if libraries is None:
        return np.array([0, n_cells], dtype=np.int64), np.arange(n_cells, dtype=np.int64)

    codes = libraries.cat.codes.to_numpy()
    n_groups = len(libraries.cat.categories)
    group_indices = np.argsort(codes, kind="stable").astype(np.int64)
    group_offsets = np.concatenate(([0], np.cumsum(np.bincount(codes, minlength=n_groups)))).astype(np.int64)
    return group_offsets, group_indices
