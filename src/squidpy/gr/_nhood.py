"""Functions for neighborhood enrichment analysis (permutation test, centralities measures etc.)."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import rustworkx as rx
from anndata import AnnData
from fast_array_utils import stats as fau_stats
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase
from fast_array_utils.types import HasArrayNamespace as Array
from numba import get_num_threads, njit, prange
from numba.typed import List
from numba_progress import ProgressBar
from numpy.typing import NDArray
from pandas import CategoricalDtype
from scanpy import logging as logg
from scipy.sparse import csr_array, csr_matrix, issparse
from scipy.stats import entropy
from spatialdata import SpatialData

from squidpy._compat import old_positionals
from squidpy._constants._constants import Centrality
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import (
    NDArrayA,
    RNGLike,
    SeedLike,
    Signal,
    SigQueue,
    deprecated_params,
    deprecated_randomness_param,
    get_n_numba_threads,
    get_n_processes,
    numba_threads,
    parallelize,
)
from squidpy._validators import assert_key_in_adata, assert_positive
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_connectivity_key,
    _save_data,
    extract_adata_if_sdata,
)

__all__ = ["nhood_enrichment", "NhoodEnrichmentResult", "centrality_scores", "interaction_matrix", "nhood_entropy"]


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
def _counts_and_conditional(
    indices: NDArrayA, indptr: NDArrayA, clustering: NDArrayA, n_cls: int
) -> tuple[NDArrayA, NDArrayA]:
    """One traversal yielding both the edge counts and the COZI conditional denominator.

    ``normalization='conditional'`` needs :func:`_nenrich`'s directed-edge counts *and*, per cluster
    pair ``(a, b)``, how many cluster-``a`` cells have at least one cluster-``b`` neighbor. Both are
    row-local, so a single pass produces them. ``stamp[b]`` holds the row that last touched cluster
    ``b``, which makes "already counted for this cell" a comparison rather than a per-row array to
    wipe and sweep — so the cost stays O(nnz) instead of growing with ``n_cls``.

    Returns ``(counts, cond)`` as ``(n_cls, n_cls)`` ``uint32`` / ``float64`` arrays. The counts are
    bit-identical to :func:`_nenrich`: the same additions happen in the same per-row order.
    """
    out = np.zeros((n_cls, n_cls), dtype=np.uint32)
    cond = np.zeros((n_cls, n_cls), dtype=np.float64)
    stamp = np.full(n_cls, -1, dtype=np.int64)  # -1 sits below every row index, so row 0 is correct
    for i in range(indptr.shape[0] - 1):
        a = clustering[i]
        for c in indices[indptr[i] : indptr[i + 1]]:
            b = clustering[c]
            out[a, b] += 1
            if stamp[b] != i:
                stamp[b] = i
                cond[a, b] += 1.0
    return out, cond


@njit(nogil=True, cache=True)
def _normalize_total(cnt: NDArrayA, sizes: NDArrayA) -> NDArrayA:
    """Divide edge counts by the number of index cells (SEA), i.e. neighbors per cell of type ``a``.

    ``sizes[a]`` is the number of cluster-``a`` cells, which label permutation leaves unchanged --
    so unlike the conditional denominator this one is a constant of the null. An empty cluster
    divides by ``1`` and stays zero.
    """
    out = np.zeros(cnt.shape, dtype=np.float64)
    for a in range(cnt.shape[0]):
        s = sizes[a] if sizes[a] != 0.0 else 1.0
        for b in range(cnt.shape[1]):
            out[a, b] = cnt[a, b] / s
    return out


@njit(nogil=True, cache=True)
def _normalize_conditional(cnt: NDArrayA, cond: NDArrayA) -> NDArrayA:
    """Divide edge counts by the COZI denominator (COZI). A zero denominator divides by ``1``."""
    out = np.zeros(cnt.shape, dtype=np.float64)
    for a in range(cnt.shape[0]):
        for b in range(cnt.shape[1]):
            d = cond[a, b] if cond[a, b] != 0.0 else 1.0
            out[a, b] = cnt[a, b] / d
    return out


@njit(nogil=True, cache=True)
def _shuffled_labels(
    int_clust: NDArrayA,
    group_offsets: NDArrayA,
    group_indices: NDArrayA,
    rng: Any,
) -> NDArrayA:
    """Shuffle cluster labels within each group, drawing once per group from ``rng``.

    Groups are visited in category order with ascending indices, so for a given generator state
    the draw sequence, and hence the result, is fully determined.
    """
    shuffled = int_clust.copy()
    # one group covering every cell has ascending indices `0..n-1`, so the gather/scatter below is
    # the identity; shuffling in place draws from ``rng`` identically and skips two passes
    if group_offsets.shape[0] == 2 and group_offsets[1] == int_clust.shape[0]:
        rng.shuffle(shuffled)
        return shuffled
    for g in range(group_offsets.shape[0] - 1):
        s, e = group_offsets[g], group_offsets[g + 1]
        sub = np.empty(e - s, dtype=int_clust.dtype)
        for t in range(e - s):
            sub[t] = int_clust[group_indices[s + t]]
        rng.shuffle(sub)
        for t in range(e - s):
            shuffled[group_indices[s + t]] = sub[t]
    return shuffled


@njit(parallel=True, nogil=True, cache=True)
def _permutation_moments_counts(  # noqa: PLR0917, numba requires positional arguments
    indices: NDArrayA,
    indptr: NDArrayA,
    int_clust: NDArrayA,
    group_offsets: NDArrayA,
    group_indices: NDArrayA,
    n_cls: int,
    observed: NDArrayA,
    generators: Any,
    progress: Any,
) -> tuple[NDArrayA, NDArrayA]:
    """Exact integer moments of the permutation distribution for ``normalization='none'``.

    The unnormalized statistic is a directed-edge count, so every ``d = permuted - observed`` is a
    whole number. Accumulating in :obj:`numpy.int64` makes the ``prange`` reduction exactly
    order-independent — the result is bit-identical for any thread count by construction rather
    than by luck — and keeps ``sum(d * d)`` exact up to ``2**63`` instead of float64's ``2**53``,
    which a large graph can genuinely exceed.

    Returns ``(sum_d, sum_d2)``; the caller turns these into the mean, std and z-score.
    """
    n_perms = len(generators)
    sum_d = np.zeros((n_cls, n_cls), dtype=np.int64)
    sum_d2 = np.zeros((n_cls, n_cls), dtype=np.int64)
    for p in prange(n_perms):
        # explicit int64 index: under prange the loop var is uint64 and indexing the typed list
        # would otherwise trigger a (harmless) uint64->int64 NumbaTypeSafetyWarning
        rng = generators[np.int64(p)]
        shuffled = _shuffled_labels(int_clust, group_offsets, group_indices, rng)
        out = _nenrich(indices, indptr, shuffled, n_cls)

        # the temporaries are needed because numba only recognizes whole-array in-place updates as
        # a reduction -- `sum_d[a, b] += dev` here would race instead
        local_d = np.zeros((n_cls, n_cls), dtype=np.int64)
        local_d2 = np.zeros((n_cls, n_cls), dtype=np.int64)
        for a in range(n_cls):
            for b in range(n_cls):
                dev = np.int64(out[a, b]) - observed[a, b]
                local_d[a, b] = dev
                local_d2[a, b] = dev * dev
        sum_d += local_d
        sum_d2 += local_d2
        progress.update(1)
    return sum_d, sum_d2


@njit(parallel=True, nogil=True, cache=True)
def _permutation_moments_normalized(  # noqa: PLR0917, numba requires positional arguments
    indices: NDArrayA,
    indptr: NDArrayA,
    int_clust: NDArrayA,
    group_offsets: NDArrayA,
    group_indices: NDArrayA,
    n_cls: int,
    norm_code: int,
    sizes: NDArrayA,
    observed: NDArrayA,
    generators: Any,
    progress: Any,
) -> tuple[NDArrayA, NDArrayA]:
    """Moments of the permutation distribution for the ``'total'`` / ``'conditional'`` modes.

    Normalizing divides by a row sum or a conditional denominator, so the statistic is fractional
    and has to be accumulated in float64. Deviations are still taken against ``observed``, which
    sits on the same scale as the null distribution: a raw sum-of-squares would cancel badly.
    The summation order depends on the thread count, so the result matches to rounding rather than
    bit-for-bit (measured at <= 1e-14 relative).

    Returns ``(sum_d, sum_d2)``; the caller turns these into the mean, std and z-score.
    """
    n_perms = len(generators)
    sum_d = np.zeros((n_cls, n_cls), dtype=np.float64)
    sum_d2 = np.zeros((n_cls, n_cls), dtype=np.float64)
    for p in prange(n_perms):
        rng = generators[np.int64(p)]
        shuffled = _shuffled_labels(int_clust, group_offsets, group_indices, rng)

        if norm_code == 1:  # total
            out = _normalize_total(_nenrich(indices, indptr, shuffled, n_cls), sizes)
        else:  # conditional: one fused walk yields both the numerator and its denominator
            cnt, cond = _counts_and_conditional(indices, indptr, shuffled, n_cls)
            out = _normalize_conditional(cnt, cond)

        local_d = np.zeros((n_cls, n_cls), dtype=np.float64)
        local_d2 = np.zeros((n_cls, n_cls), dtype=np.float64)
        for a in range(n_cls):
            for b in range(n_cls):
                dev = out[a, b] - observed[a, b]
                local_d[a, b] = dev
                local_d2[a, b] = dev * dev
        sum_d += local_d
        sum_d2 += local_d2
        progress.update(1)
    return sum_d, sum_d2


_NORM_CODES = {"none": 0, "total": 1, "conditional": 2}


def _filter_clusters_by_min_cell_count(
    adj: csr_matrix,
    int_clust: NDArrayA,
    min_cell_count: int,
) -> tuple[NDArrayA, NDArrayA, NDArrayA]:
    clust_sizes = pd.Series(int_clust).value_counts()
    valid_clusters = clust_sizes[clust_sizes >= min_cell_count].index.to_numpy()

    valid_mask = np.isin(int_clust, valid_clusters)
    valid_cells_idx = np.where(valid_mask)[0]
    int_clust = int_clust[valid_mask]

    return int_clust, adj[np.ix_(valid_cells_idx, valid_cells_idx)], valid_mask


@d.get_sections(base="nhood_ench", sections=["Parameters"])
@d.dedent
@deprecated_randomness_param
@deprecated_params({"numba_parallel": "1.10.0", "backend": "1.10.0"})
@old_positionals(
    "cluster_key",
    "library_key",
    "connectivity_key",
    "n_perms",
    "rng",
    "copy",
    "n_jobs",
    "show_progress_bar",
    "normalization",
    "min_cell_count",
    "handle_nan",
)
def nhood_enrichment(
    adata: AnnData | SpatialData,
    *,
    cluster_key: str,
    library_key: str | None = None,
    connectivity_key: str | None = None,
    n_perms: int = 1000,
    rng: SeedLike | RNGLike | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
    show_progress_bar: bool = True,
    normalization: str = "none",
    min_cell_count: int = 0,
    handle_nan: Literal["keep", "zero"] = "keep",
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
    %(rng)s
    %(copy)s
    %(n_jobs_threads)s
    %(show_progress_bar)s
    normalization
        Normalization mode to use, as compared in :cite:`schiller2025`:

        - ``'none'``: No normalization of neighbor counts.
        - ``'total'``: Divide by the number of cells of the index cluster (SEA). Cluster sizes are
          unchanged by the permutation, so this rescales the statistic without changing the z-score.
        - ``'conditional'``: Divide by the number of index-cluster cells having at least one
          neighbor of the given type (COZI).
    min_cell_count
        Minimum number of cells a cluster must contain to be included. Clusters with fewer cells are
        dropped before counting (default ``0`` keeps all clusters) and their z-scores are `NaN`
        whatever ``handle_nan`` says. Worth raising: a cluster too small to give the permutation
        null a spread of values yields an unreliable z-score, and in ``'conditional'`` mode a rare
        cluster pair can leave the denominator at zero, which is reported as ``0``.
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

    if normalization not in _NORM_CODES:
        raise ValueError(f"Invalid normalization mode `{normalization}`. Choose from {sorted(_NORM_CODES)}.")
    if handle_nan not in ("keep", "zero"):
        raise ValueError(f"Invalid `handle_nan` mode `{handle_nan}`. Choose from 'keep', 'zero'.")

    adj = adata.obsp[connectivity_key]
    if not issparse(adj):
        raise TypeError(
            f"Expected `adata.obsp[{connectivity_key!r}]` to be a sparse matrix, found `{type(adj).__name__}`."
        )
    # CSC has `indices`/`indptr` too, but column-wise: without this the counts come out transposed
    adj = adj.tocsr()
    # The kernels read `indices`/`indptr` only, so anything the CSR stores is counted as an edge:
    # a stored zero (what pruning in place, `adj.data[mask] = 0`, leaves behind) and each half of a
    # duplicated `(i, j)` entry, which scipy defines as one edge whose value is the sum. Copy before
    # canonicalizing -- `tocsr()` hands back the caller's own matrix when it is already CSR, and
    # `count_nonzero()` cannot be used to test for this because it sums duplicates in place.
    if not adj.has_canonical_format or (adj.data == 0).any():
        adj = adj.copy()
        adj.sum_duplicates()
        adj.eliminate_zeros()
    original_clust = adata.obs[cluster_key]
    # `.cat.codes` already holds each cell's index into `cat.categories`. NaN shows up as `-1`,
    # which `ndt` would wrap into a huge cluster id, so reject it rather than let it index out of
    # range.
    codes = original_clust.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError(f"Found `NaN` values in `adata.obs[{cluster_key!r}]`; every cell needs a cluster.")
    int_clust = codes.astype(ndt)
    n_total_cells = len(int_clust)

    # `min_cell_count=0` keeps every cluster, so the filter would rebuild `adj` into an identical
    # copy -- an `nnz`-sized allocation on the default path.
    if min_cell_count > 0:
        int_clust, adj, valid_mask = _filter_clusters_by_min_cell_count(adj, int_clust, min_cell_count)
    else:
        valid_mask = np.ones(n_total_cells, dtype=bool)
    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        if (adata.obs[library_key].cat.codes.to_numpy() < 0).any():
            raise ValueError(f"Found `NaN` values in `adata.obs[{library_key!r}]`; every cell needs a library.")
        # subset to the kept cells so the per-cell series stays aligned with the filtered
        libraries: pd.Series | None = adata.obs[library_key].iloc[valid_mask].cat.remove_unused_categories()
    else:
        libraries = None

    n_filtered = n_total_cells - len(int_clust)
    if n_filtered > 0:
        warnings.warn(
            f"{n_filtered / n_total_cells * 100:.3f}% of cells were excluded because their clusters "
            f"had fewer than {min_cell_count} cells.",
            UserWarning,
            # +2 for the `deprecated_randomness_param` and `deprecated_params` wrappers
            stacklevel=4,
        )

    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    n_cls = len(original_clust.cat.categories)
    if n_cls <= 1:
        raise ValueError(f"Expected at least `2` clusters, found `{n_cls}`.")

    conditional_ratio = np.full((n_cls, n_cls), np.nan, dtype=np.float64)

    # label permutation preserves cluster sizes, so this is a constant of the null
    cluster_sizes = np.bincount(int_clust, minlength=n_cls).astype(np.float64)

    if normalization == "conditional":
        # one fused walk: this mode is the only one that needs the conditional denominator too
        count, cond_counts = _counts_and_conditional(indices, indptr, int_clust, n_cls)

        nonempty = cluster_sizes > 0
        conditional_ratio[nonempty] = cond_counts[nonempty] / cluster_sizes[nonempty, None]

        count_normalized = _normalize_conditional(count, cond_counts)
    else:
        count = _nenrich(indices, indptr, int_clust, n_cls)
        if normalization == "total":
            count_normalized = _normalize_total(count, cluster_sizes)
        else:  # "none"
            count_normalized = count.copy()

    n_jobs = get_n_numba_threads(n_jobs)
    start = logg.info(f"Calculating neighborhood enrichment using `{n_jobs}` thread(s)")
    norm_code = _NORM_CODES[normalization]

    generators = List(np.random.default_rng(rng).spawn(n_perms))

    # Group structure for within-group shuffling, as a CSR-like (offsets, indices) pair in category
    # order with ascending indices per group. Without a `library_key` there is a single group
    # spanning all cells, which reproduces a plain global shuffle.
    group_offsets, group_indices = _build_shuffle_groups(libraries, len(int_clust))

    # A single numba ``prange`` kernel shuffles + counts + normalizes per thread with the GIL
    # released, and ticks the progress bar from inside the loop; numba owns the parallelism.
    # Unnormalized counts go through the integer kernel, which is exactly order-independent.
    with (
        numba_threads(n_jobs),
        ProgressBar(total=n_perms, unit="perm", desc="nhood_enrichment", disable=not show_progress_bar) as progress,
    ):
        if norm_code == 0:
            sum_d, sum_d2 = _permutation_moments_counts(
                indices,
                indptr,
                int_clust,
                group_offsets,
                group_indices,
                n_cls,
                np.ascontiguousarray(count_normalized, dtype=np.int64),
                generators,
                progress,
            )
        else:
            sum_d, sum_d2 = _permutation_moments_normalized(
                indices,
                indptr,
                int_clust,
                group_offsets,
                group_indices,
                n_cls,
                norm_code,
                cluster_sizes,
                np.ascontiguousarray(count_normalized, dtype=np.float64),
                generators,
                progress,
            )

    # ``sum_d``/``sum_d2`` are moments of ``permuted - observed``, so the mean deviation *is* the
    # (negated) numerator of the z-score and no permutation ever has to be kept around. The int64
    # sums are exact, so converting here is a single deterministic rounding, not an accumulated one.
    n = float(n_perms)
    mean_d = sum_d / n
    var = (sum_d2 - sum_d * mean_d) / n  # population variance, i.e. ddof=0
    std = np.sqrt(np.maximum(var, 0.0))  # clamp: rounding can push an all-equal column just below 0
    std[std == 0] = np.nan
    zscore = -mean_d / std

    if handle_nan == "zero":
        zscore = np.nan_to_num(zscore, nan=0.0)

    # `handle_nan` governs enrichments the permutation test leaves undefined. A cluster dropped by
    # `min_cell_count` was never measured at all, so it stays NaN either way -- otherwise `'zero'`
    # would render "excluded" and "no enrichment" as the same number.
    dropped = cluster_sizes == 0
    if dropped.any():
        zscore[dropped, :] = np.nan
        zscore[:, dropped] = np.nan

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
@old_positionals("cluster_key", "score", "connectivity_key", "copy", "n_jobs", "backend", "show_progress_bar")
def centrality_scores(
    adata: AnnData | SpatialData,
    *,
    cluster_key: str,
    score: str | Iterable[str] | None = None,
    connectivity_key: str | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = False,
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
@old_positionals("cluster_key", "connectivity_key", "normalized", "copy", "weights")
def interaction_matrix(
    adata: AnnData | SpatialData,
    *,
    cluster_key: str,
    connectivity_key: str | None = None,
    normalized: bool = False,
    copy: bool = False,
    weights: bool = False,
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
    %(cluster_key)s
    %(conn_key)s
    %(copy)s
    %(table_key)s

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
    profile = to_dense(
        nhood_aggregate(adata, groups=cluster_key, connectivity_key=connectivity_key, aggregation="mean")
    )
    # observations without neighbors give 0/0 in `entropy`
    ent = pd.Series(np.nan_to_num(entropy(np.asarray(profile), axis=1)), index=adata.obs_names)

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


def _build_shuffle_groups(
    libraries: pd.Series[CategoricalDtype] | None,
    n_cells: int,
) -> tuple[NDArrayA, NDArrayA]:
    """Build a CSR-like ``(offsets, indices)`` description of the within-group shuffling.

    ``indices[offsets[g]:offsets[g + 1]]`` are the cell indices of group ``g`` in ascending order,
    with groups in category order. Without a ``library_key`` there is a single group spanning all
    cells, which reproduces a global shuffle.
    """
    if libraries is None:
        return np.array([0, n_cells], dtype=np.int64), np.arange(n_cells, dtype=np.int64)

    codes = libraries.cat.codes.to_numpy()
    n_groups = len(libraries.cat.categories)
    group_indices = np.argsort(codes, kind="stable").astype(np.int64)
    group_offsets = np.concatenate(([0], np.cumsum(np.bincount(codes, minlength=n_groups)))).astype(np.int64)
    return group_offsets, group_indices


@njit(inline="always", cache=True)
def _expand(  # noqa: PLR0917, numba requires positional arguments
    indptr: NDArrayA,
    indices: NDArrayA,
    stamp: NDArrayA,
    tag: int,
    queue: NDArrayA,
    head: int,
    tail: int,
) -> int:
    """Advance one breadth-first level, returning the new tail.

    Discoveries land contiguously at ``queue[tail:new_tail]``. ``stamp`` holds ``tag`` instead of
    a boolean so one buffer serves many searches without being cleared.
    """
    new_tail = tail
    for i in range(head, tail):
        node = queue[i]
        for p in range(indptr[node], indptr[node + 1]):
            neighbor = indices[p]
            if stamp[neighbor] == tag:
                continue  # a nearer level already reached it
            stamp[neighbor] = tag
            queue[new_tail] = neighbor
            new_tail += 1
    return new_tail


@njit(parallel=True, cache=True)
def _bfs_shells(  # noqa: PLR0917, numba requires positional arguments
    indptr: NDArrayA,
    indices: NDArrayA,
    max_hop: int,
    counts: NDArrayA,
    base: NDArrayA,
    rowptr: NDArrayA,
    out: NDArrayA,
    fill: bool,
) -> None:
    n = indptr.shape[0] - 1
    n_threads = get_num_threads()
    stamp = np.full((n_threads, n), -1, dtype=indices.dtype)
    queue = np.empty((n_threads, n), dtype=indices.dtype)

    for thread in prange(n_threads):
        for src in range(thread, n, n_threads):
            # source pre-marked, so an out-and-back walk cannot reach it
            stamp[thread, src] = src
            queue[thread, 0] = src
            head, tail = 0, 1

            for hop in range(1, max_hop + 1):
                new_tail = _expand(indptr, indices, stamp[thread], src, queue[thread], head, tail)
                found = new_tail - tail
                # hop 1 is the input graph, which the caller keeps as ring 0, so it is expanded
                # only as the frontier for hop 2 and nothing is written for it
                ring = hop - 2
                if ring >= 0:
                    if fill:
                        # contiguous, so the row is a straight copy with no write cursor
                        at = base[ring] + rowptr[ring, src]
                        for g in range(found):
                            out[at + g] = queue[thread, tail + g]
                    else:
                        counts[ring, src] += found
                head, tail = tail, new_tail
                if found == 0:
                    break


def compute_hop_adjacency_matrices(
    adjacency_matrix_orig: CSBase | Array,
    max_hop: int,
    n_jobs: int | None = None,
) -> list[CSBase]:
    """Disjoint adjacency rings, one per hop up to *max_hop*.

    Ring ``k`` holds the pairs first reached at hop ``k + 1``, so the rings never restate
    each other. Ring 0 is the input as booleans, self-loops included.
    """
    # CellCharter builds these as iterated products masked by a visited set (`adj_hop @ adj`,
    # then `adj_hop > adj_visited`) in scipy; a numba BFS gives the same disjoint rings without
    # that comparison, which is degree-dependent on a weighted graph. See
    # https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/gr/_aggr.py
    if max_hop < 1:
        raise ValueError(f"max_hop must be >= 1, got {max_hop}.")

    shape = np.shape(adjacency_matrix_orig)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"'adjacency_matrix' must be square, got {shape}")

    adj = (adjacency_matrix_orig if issparse(adjacency_matrix_orig) else csr_array(adjacency_matrix_orig)).tocsr()
    adj = adj.astype(bool)
    adj.eliminate_zeros()
    if max_hop == 1:
        return [csr_matrix(adj)]
    n = adj.shape[0]
    indptr, indices = adj.indptr, adj.indices

    # one row per ring past the first; ring 0 is the input itself
    counts = np.zeros((max_hop - 1, n), dtype=np.int64)
    no_base = np.zeros(1, dtype=np.int64)
    no_out = np.zeros(1, dtype=indices.dtype)
    n_jobs = get_n_numba_threads(n_jobs)
    with numba_threads(n_jobs):
        _bfs_shells(indptr, indices, max_hop, counts, no_base, counts, no_out, False)

        rowptr = np.zeros((max_hop - 1, n + 1), dtype=np.int64)
        np.cumsum(counts, axis=1, out=rowptr[:, 1:])
        base = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(rowptr[:, -1])))

        out = np.empty(int(base[-1]), dtype=indices.dtype)  # shell column indices, same dtype as the input's
        _bfs_shells(indptr, indices, max_hop, counts, base, rowptr, out, True)

    shells: list[CSBase] = [csr_matrix(adj)]
    for ring in range(max_hop - 1):
        lo, hi = int(base[ring]), int(base[ring + 1])
        shell = csr_matrix((np.ones(hi - lo, dtype=bool), out[lo:hi], rowptr[ring]), shape=(n, n))
        shell.sort_indices()  # breadth-first order is not sorted order
        shells.append(shell)
    return shells


def _power_adjacencies(adj: CSBase, max_hop: int) -> list[CSBase]:
    if max_hop < 1:
        raise ValueError(f"max_hop must be >= 1, got {max_hop}.")

    # Edge weights apply to edges, so hop 1 enters as the caller gave it. Past it there are
    # none to apply: scipy's product counts walks, and a cell reached by two paths is still
    # one cell, so the reach is a set. `bool @ bool` saturates to True, which is exactly that.
    reach = adj.astype(bool)
    reach.eliminate_zeros()
    adjacencies, power = [adj], reach
    for _ in range(1, max_hop):
        power = power @ reach
        adjacencies.append(power)
    return adjacencies


def _onehot(labels: pd.Series) -> csr_matrix:
    """Indicator matrix of ``labels``, one column per category."""
    # TODO: move to fast-array-utils; scanpy keeps its own private copy as `get._aggregated.sparse_indicator`
    cat = labels.astype("category")
    codes = cat.cat.codes.to_numpy()
    keep = codes >= 0
    return csr_matrix(
        (np.ones(keep.sum(), dtype=np.float64), (np.flatnonzero(keep), codes[keep])),
        shape=(len(codes), len(cat.cat.categories)),
    )


def _aggregate_over(
    adj: CSBase,
    features: Array | CSBase,
    aggregation: Literal["mean", "sum", "variance"],
    *,
    counted: NDArray[np.bool_] | None = None,
) -> Array | CSBase:
    """Aggregate *features* over the neighborhood each row of *adj* defines.

    *counted* marks the observations that count as neighbors. The others must have all-zero
    features, so they already add nothing to a sum and only the neighbor count leaves them out.
    """
    if aggregation == "sum":
        return adj @ features

    # scale after the sum, not before: normalizing `adj` rounds 1/k to `adj.dtype`, which
    # `spatial_neighbors` leaves float32, so a mean that is exactly representable at that width
    # still does not come back exact. Dividing by the signed sum rather than the L1 norm is also
    # the weighted mean a negative edge weight asks for.
    if counted is None:
        total = np.asarray(fau_stats.sum(adj, axis=1, dtype=np.float64)).reshape(-1, 1)
    else:
        total = np.asarray(adj @ counted.astype(np.float64)).reshape(-1, 1)
    inv = np.reciprocal(total, where=total != 0, out=np.zeros_like(total))  # an isolated row stays 0

    def mean_over(x: Array | CSBase) -> Array | CSBase:
        """Row-mean of ``adj @ x``, scaled in place -- the product is ours."""
        # widen before the product, not after: `bool @ bool` saturates to True instead of
        # summing, so by the time an integer product exists the counts are already gone
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float64)
        product = adj @ x
        if issparse(product):
            product = product.tocsr()
            scale = np.repeat(inv.ravel(), np.diff(product.indptr))  # each entry against its own row
            product.data = np.multiply(product.data, scale, dtype=np.float64).astype(product.dtype, copy=False)
            return product
        return np.multiply(product, inv, out=product, casting="same_kind")

    if aggregation == "mean":
        return mean_over(features)
    if aggregation == "variance":
        mean = to_dense(mean_over(features))
        dense = to_dense(features)
        return to_dense(mean_over(dense * dense)) - mean * mean
    raise ValueError(f"'aggregation' must be 'mean', 'sum' or 'variance', got {aggregation!r}")


def _assert_hop_request(adata: AnnData, connectivity_key: str, hops: Sequence[int]) -> None:
    """Verify a hop request against the graph it is about to run on."""
    _assert_connectivity_key(adata, connectivity_key)
    if len(hops) == 0:
        raise ValueError("'hops' must name at least one hop")
    if any(hop < 0 for hop in hops):
        raise ValueError(f"'hops' must be non-negative, got {list(hops)!r}")


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

    Matrix powers, not disjoint rings, so a hop restates the ones below it. Each cell in
    reach counts once: a cell two paths away is still one cell.
    """
    _assert_hop_request(adata, connectivity_key, hops)
    if aggregation not in ("mean", "sum", "variance"):
        raise ValueError(f"'aggregation' must be 'mean', 'sum' or 'variance', got {aggregation!r}")
    weights = [1.0] * len(hops) if hop_weights is None else list(hop_weights)
    if len(weights) != len(hops):
        raise ValueError(f"'hop_weights' has {len(weights)} value(s) but there are {len(hops)} hop(s)")
    if aggregation != "sum" and sum(weights) == 0:
        raise ValueError("'hop_weights' must not sum to zero, since the hops are averaged over it")

    given = [name for name, value in (("groups", groups), ("use_rep", use_rep), ("layer", layer)) if value is not None]
    if len(given) > 1:
        raise ValueError(f"pass at most one of 'groups', 'use_rep' and 'layer', got {given}")

    # only a category can be missing, so a feature matrix leaves every observation a neighbor
    has_value = None
    if groups is not None:
        assert_key_in_adata(adata, groups, attr="obs")
        features = _onehot(adata.obs[groups])
        # an observation with no category is not a neighbor either, so it leaves every denominator
        has_value = adata.obs[groups].notna().to_numpy()
    elif use_rep is not None:
        if use_rep == "X":  # the spelling `scanpy.pp.neighbors` takes
            features = adata.X
        else:
            assert_key_in_adata(adata, use_rep, attr="obsm")
            features = adata.obsm[use_rep]
    elif layer is not None:
        assert_key_in_adata(adata, layer, attr="layers")
        features = adata.layers[layer]
    else:
        features = adata.X

    by_hop: dict[int, CSBase | None] = {0: None}
    if max(hops) >= 1:
        by_hop |= dict(enumerate(_power_adjacencies(adata.obsp[connectivity_key], max(hops)), start=1))
    # an unlabelled observation's one-hot row is zero, so it only has to leave the neighbor count;
    # the hop matrices are not copied to drop it, and paths through it still reach past it
    counted = None if has_value is None or has_value.all() else has_value
    # hop 0 is the observation itself, so it contributes its features unaggregated
    blocks = [
        features if hop == 0 else _aggregate_over(by_hop[hop], features, aggregation, counted=counted) for hop in hops
    ]

    # keep the container the features came in; `variance` has already densified, so a
    # mixed set of blocks has to be densified whole
    if not all(issparse(block) for block in blocks):
        blocks = [to_dense(block) for block in blocks]
    total = sum(weight * block for weight, block in zip(weights, blocks, strict=True))
    if aggregation == "sum":  # counts are meant to stay counts
        return total
    # in place, since `total / scalar` promotes a sparse matrix to float64
    total /= sum(weights)
    return total
