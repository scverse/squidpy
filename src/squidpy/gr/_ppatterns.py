"""Functions for point patterns spatial statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numba.types as nt
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit, prange
from scanpy import logging as logg
from scanpy.metrics import gearys_c, morans_i
from scipy import stats
from scipy.sparse import issparse, spmatrix
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from spatialdata import SpatialData
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from squidpy._compat import old_positionals
from squidpy._constants._constants import SpatialAutocorr
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import (
    NDArrayA,
    RNGLike,
    SeedLike,
    deprecated_params,
    deprecated_randomness_param,
    get_n_numba_threads,
    numba_threads,
    thread_map,
)
from squidpy._validators import assert_key_in_adata, assert_positive
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_connectivity_key,
    _assert_spatial_basis,
    _save_data,
    extract_adata_if_sdata,
)

__all__ = ["spatial_autocorr", "co_occurrence"]


it = nt.int32
ft = nt.float32
tt = nt.UniTuple
ip = np.int32
fp = np.float32
bl = nt.boolean

# Permutation entries held at once. Permutations are reused across features, so they are
# materialized rather than redrawn per feature; the cap keeps that buffer off `n_perms`, which
# users raise for FDR resolution. Splitting costs ~3% at 1M cells and ~10% at 10M (four blocks
# either way), and only becomes expensive past ~8 blocks. It changes no result, see
# `test_spatial_autocorr_perm_blocks`.
_PERM_BLOCK_SIZE = 2**28


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn(), sp=SpatialAutocorr)
@old_positionals(
    "connectivity_key",
    "genes",
    "mode",
    "transformation",
    "n_perms",
    "two_tailed",
    "corr_method",
    "attr",
    "layer",
    "seed",
    "use_raw",
    "copy",
    "n_jobs",
    "backend",
    "show_progress_bar",
)
@deprecated_randomness_param
@deprecated_params({"backend": "1.10.0"})
def spatial_autocorr(
    adata: AnnData | SpatialData,
    *,
    connectivity_key: str = Key.obsp.spatial_conn(),
    genes: str | int | Sequence[str] | Sequence[int] | None = None,
    mode: SpatialAutocorr | Literal["moran", "geary"] = "moran",
    transformation: bool = True,
    n_perms: int | None = None,
    two_tailed: bool = False,
    corr_method: str | None = "fdr_bh",
    attr: Literal["obs", "X", "obsm"] = "X",
    layer: str | None = None,
    rng: SeedLike | RNGLike | None = None,
    use_raw: bool = False,
    copy: bool = False,
    n_jobs: int | None = None,
    show_progress_bar: bool = True,
    table_key: str | None = None,
) -> pd.DataFrame | None:
    """
    Calculate Global Autocorrelation Statistic (Moran’s I  or Geary's C).

    See :cite:`pysal` for reference.

    .. versionchanged:: 1.8.2
        The analytic (normality-assumption) variance for Geary's C was corrected; previously the
        Moran's I variance was reused for ``mode = 'geary'``. As a result, ``'var_norm'`` and
        ``'pval_norm'`` for Geary's C differ from earlier versions. Permutation-based p-values
        (``'pval_sim'``, ``'pval_z_sim'``) are unaffected.
        See `#1183 <https://github.com/scverse/squidpy/issues/1183>`_.

    %(seed_versionchanged)s

    %(rng_versionchanged)s

    .. versionchanged:: 1.8.4
        Permutations run on a thread pool, and ``n_jobs = None`` now uses all ``NUMBA_NUM_THREADS``
        threads instead of one process. Pass ``n_jobs = 1`` for the old serial default.

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    %(conn_key)s
    genes
        Depending on the ``attr``:

            - if ``attr = 'X'``, it corresponds to genes stored in :attr:`anndata.AnnData.var_names`.
              If `None`, it's computed :attr:`anndata.AnnData.var` ``['highly_variable']``,
              if present. Otherwise, it's computed for all genes.
            - if ``attr = 'obs'``, it corresponds to a list of columns in :attr:`anndata.AnnData.obs`.
              If `None`, use all numerical columns.
            - if ``attr = 'obsm'``, it corresponds to indices in :attr:`anndata.AnnData.obsm` ``['{{layer}}']``.
              If `None`, all indices are used.

    mode
        Mode of score calculation:

            - `{sp.MORAN.s!r}` - `Moran's I autocorrelation <https://en.wikipedia.org/wiki/Moran%27s_I>`_.
            - `{sp.GEARY.s!r}` - `Geary's C autocorrelation <https://en.wikipedia.org/wiki/Geary%27s_C>`_.

    transformation
        If `True`, weights in :attr:`anndata.AnnData.obsp` ``['{key}']`` are row-normalized,
        advised for analytic p-value calculation.
    %(n_perms)s
        If `None`, only p-values under normality assumption are computed.
    two_tailed
        If `True`, p-values are two-tailed, otherwise they are one-tailed.
    %(corr_method)s
    use_raw
        Whether to access :attr:`anndata.AnnData.raw`. Only used when ``attr = 'X'``.
    layer
        Depending on ``attr``:
        Layer in :attr:`anndata.AnnData.layers` to use. If `None`, use :attr:`anndata.AnnData.X`.
    attr
        Which attribute of :class:`~anndata.AnnData` to access. See ``genes`` parameter for more information.
    %(rng)s
    %(copy)s
    %(n_jobs_threads)s
    %(show_progress_bar)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the following keys:

        - `'I' or 'C'` - Moran's I or Geary's C statistic.
        - `'pval_norm'` - p-value under normality assumption.
        - `'var_norm'` - variance of `'score'` under normality assumption.
        - `'{{p_val}}_{{corr_method}}'` - the corrected p-values if ``corr_method != None`` .

    If ``n_perms != None``, additionally returns the following columns:

        - `'pval_z_sim'` - p-value based on standard normal approximation from permutations.
        - `'pval_sim'` - p-value based on permutations.
        - `'var_sim'` - variance of `'score'` from permutations.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['moranI']`` - the above mentioned dataframe, if ``mode = {sp.MORAN.s!r}``.
        - :attr:`anndata.AnnData.uns` ``['gearyC']`` - the above mentioned dataframe, if ``mode = {sp.GEARY.s!r}``.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    _assert_connectivity_key(adata, key=connectivity_key)

    def extract_X(adata: AnnData, genes: str | Sequence[str] | None) -> tuple[NDArrayA | spmatrix, Sequence[Any]]:
        if genes is None:
            if "highly_variable" in adata.var:
                genes = adata[:, adata.var["highly_variable"]].var_names.values
            else:
                genes = adata.var_names.values
        elif isinstance(genes, str):
            genes = [genes]

        if not use_raw:
            # full var_names in order: `adata[:, genes]` would copy X to hand back the same
            # columns it was given.
            if len(genes) == adata.n_vars and np.array_equal(np.asarray(genes), adata.var_names.values):
                return (adata.X if layer is None else adata.layers[layer]).T, genes
            subset = adata[:, genes]
            return (subset.X if layer is None else subset.layers[layer]).T, genes
        if adata.raw is None:
            raise AttributeError("No `.raw` attribute found. Try specifying `use_raw=False`.")
        genes = list(set(genes) & set(adata.raw.var_names))
        return adata.raw[:, genes].X.T, genes

    def extract_obs(adata: AnnData, cols: str | Sequence[str] | None) -> tuple[NDArrayA | spmatrix, Sequence[Any]]:
        if cols is None:
            df = adata.obs.select_dtypes(include=np.number)
            return df.T.to_numpy(), df.columns
        if isinstance(cols, str):
            cols = [cols]
        return adata.obs[cols].T.to_numpy(), cols

    def extract_obsm(adata: AnnData, ixs: int | Sequence[int] | None) -> tuple[NDArrayA | spmatrix, Sequence[Any]]:
        assert_key_in_adata(adata, layer, attr="obsm")
        if ixs is None:
            ixs = list(np.arange(adata.obsm[layer].shape[1]))
        ixs = list(np.ravel([ixs]))

        return adata.obsm[layer][:, ixs].T, ixs

    if attr == "X":
        vals, index = extract_X(adata, genes)  # type: ignore
    elif attr == "obs":
        vals, index = extract_obs(adata, genes)  # type: ignore
    elif attr == "obsm":
        vals, index = extract_obsm(adata, genes)  # type: ignore
    else:
        raise NotImplementedError(f"Extracting from `adata.{attr}` is not yet implemented.")

    mode = SpatialAutocorr(mode)
    params = {"mode": mode.s, "transformation": transformation, "two_tailed": two_tailed}

    if mode == SpatialAutocorr.MORAN:
        params["func"] = morans_i
        params["stat"] = "I"
        params["expected"] = -1.0 / (adata.shape[0] - 1)  # expected score
        params["ascending"] = False
    elif mode == SpatialAutocorr.GEARY:
        params["func"] = gearys_c
        params["stat"] = "C"
        params["expected"] = 1.0
        params["ascending"] = True
    else:
        raise NotImplementedError(f"Mode `{mode}` is not yet implemented.")

    g = adata.obsp[connectivity_key].tocsr(copy=True)
    if transformation:  # row-normalize
        normalize(g, norm="l1", axis=1, copy=False)

    n_jobs = get_n_numba_threads(n_jobs)
    start = logg.info(f"Calculating {mode}'s statistic for `{n_perms}` permutations using `{n_jobs}` thread(s)")
    if n_perms is not None:
        assert_positive(n_perms, name="n_perms")
        # the observed score comes from the same kernel as the permuted ones, so exact ties
        # in the tally below stay exact
        score, score_perms = _score_perms(
            g, vals, mode=mode, n_perms=n_perms, rng=rng, n_jobs=n_jobs, show_progress_bar=show_progress_bar
        )
    else:
        score, score_perms = params["func"](g, vals), None  # type: ignore

    with np.errstate(divide="ignore"):
        pval_results = _p_value_calc(score, score_perms, g, params)

    data_dict: dict[str, Any] = {str(params["stat"]): score, **pval_results}
    df = pd.DataFrame(data_dict, index=index)

    if corr_method is not None:
        for pv in filter(lambda x: "pval" in x, df.columns):
            pvals = df[pv].to_numpy(dtype=np.float64)
            # a degenerate feature has no p-value to correct, and `multipletests` would spread its
            # NaN over every other gene, so correct the defined ones among themselves
            defined = np.isfinite(pvals)
            adj = np.full(pvals.shape, np.nan)
            if defined.any():
                _, adj[defined], _, _ = multipletests(pvals[defined], alpha=0.05, method=corr_method)
            df[f"{pv}_{corr_method}"] = adj

    df.sort_values(by=params["stat"], ascending=params["ascending"], inplace=True)

    if copy:
        logg.info("Finish", time=start)
        return df

    mode_str = str(params["mode"])
    stat_str = str(params["stat"])
    _save_data(adata, attr="uns", key=mode_str + stat_str, data=df, time=start)


@njit(parallel=False, nogil=True, cache=True)
def _autocorr_perms(  # noqa: PLR0917, numba requires positional arguments
    tptr: NDArrayA,
    tind: NDArrayA,
    tdat: NDArrayA,
    w_sum: NDArrayA,
    col_sum: NDArrayA,
    nz: NDArrayA,
    xv: NDArrayA,
    w: float,
    perms: NDArrayA,
    moran: bool,
) -> NDArrayA:
    """Permutation scores of one feature over the row-permuted graph, one per row of ``perms``.

    Both statistics sum, over the rows of ``g[perm, :]``, a per-row quantity that depends on the
    permutation only through *which* row is visited, so those quantities are accumulated once per
    row of the unpermuted graph rather than re-derived for every permutation.

    ``nz`` lists the cells where this feature is non-zero. Both statistics split into a term that
    only touches those cells and a term that is permutation-invariant (``sum(perm)`` over a bijection
    is ``sum``), so the per-permutation loop costs ``O(len(nz))`` rather than ``O(n_cells)``. Pass
    ``arange(n)`` for a dense feature.

    ``xv`` must already be :class:`numpy.float64`, and ``tdat`` the float64 graph weights,
    matching the casts :mod:`scanpy` applies before its own kernels.

    Compiled serial on purpose: `_score_perms` parallelizes across features instead, which measured
    faster than an in-kernel ``prange`` even with fewer features than workers. ``nogil`` is what
    lets that pool enter it concurrently, which a ``parallel=True`` kernel cannot survive under
    numba's default threading layer.
    """
    n_perms, n = perms.shape
    out = np.empty(n_perms, dtype=np.float64)

    # Independence from ``n_jobs`` comes from `_score_perms`: permutation ``p`` is always drawn
    # from ``rngs[p]``, and each feature is owned by one worker.
    n_nz = nz.shape[0]
    x_bar = 0.0
    for t in range(n_nz):
        x_bar += xv[t]
    x_bar /= n

    # sum_i (x_i - x_bar)^2: Moran's denominator, and Geary's.
    css = 0.0
    for t in range(n_nz):  # looped rather than ((xv - x_bar) ** 2).sum() to skip the temporary
        d = xv[t] - x_bar
        css += d * d
    css += (n - n_nz) * x_bar * x_bar  # every zero entry contributes x_bar^2

    if moran:
        # I = n / W * sum_ij w_ij z_i z_j / sum_i z_i^2. The inner row sum is the spatial lag,
        # which the permutation only reindexes.
        lag = np.zeros(n, dtype=np.float64)
        # lag = g @ z = (g @ x) - x_bar * rowsum(g). Only columns where x != 0 contribute, so
        # this scatters over the feature's non-zeros rather than sweeping the whole graph.
        for t in range(n_nz):
            j = nz[t]
            xj = xv[t]
            for e in range(tptr[j], tptr[j + 1]):
                lag[tind[e]] += tdat[e] * xj
        for k in range(n):
            lag[k] -= x_bar * w_sum[k]
        # sum_i lag[perm[i]] * z[i] = sum_{x[i]!=0} lag[perm[i]] * x[i] - x_bar * sum_k lag[k].
        # The second term is permutation-invariant, so it is hoisted out of the loop.
        # sum_k lag[k] = sum_j x[j] * colsum(g)[j] - x_bar * W: O(nnz), not O(n_cells).
        lag_tot = 0.0
        for t in range(n_nz):
            lag_tot += xv[t] * col_sum[nz[t]]
        lag_tot -= x_bar * w
        for p in range(n_perms):
            inum = 0.0
            for t in range(n_nz):
                inum += lag[perms[p, nz[t]]] * xv[t]
            out[p] = n / w * (inum - x_bar * lag_tot) / css
    else:
        # C = (n - 1) * sum_ij w_ij (x_i - x_j)^2 / (2 W sum_i (x_i - x_bar)^2). Expanding the
        # square splits each row into (sum w, sum w x_j, sum w x_j^2), all permutation-independent.
        wx = np.zeros(n, dtype=np.float64)
        wx2 = np.zeros(n, dtype=np.float64)
        for t in range(n_nz):
            j = nz[t]
            xj = xv[t]
            xj2 = xj * xj
            for e in range(tptr[j], tptr[j + 1]):
                k = tind[e]
                weight = tdat[e]
                wx[k] += weight * xj
                wx2[k] += weight * xj2
        denom = 2.0 * w * css
        # sum_i wx2[perm[i]] is permutation-invariant; the other two terms vanish where x[i] == 0.
        wx2_tot = 0.0
        for t in range(n_nz):
            wx2_tot += xv[t] * xv[t] * col_sum[nz[t]]
        for p in range(n_perms):
            total = 0.0
            for t in range(n_nz):
                k = perms[p, nz[t]]
                xi = xv[t]
                total += xi * xi * w_sum[k] - 2.0 * xi * wx[k]
            out[p] = (n - 1) * (total + wx2_tot) / denom
    return out


def _chunks(n: int, workers: int) -> list[range]:
    """Contiguous index runs over ``n``, ~4 per worker so the pool can balance uneven items."""
    step = max(1, -(-n // (workers * 4)))
    return [range(s, min(s + step, n)) for s in range(0, n, step)]


def _score_perms(
    g: spmatrix,
    vals: NDArrayA | spmatrix,
    *,
    mode: SpatialAutocorr,
    n_perms: int,
    rng: SeedLike | RNGLike | None,
    n_jobs: int,
    show_progress_bar: bool,
) -> tuple[NDArrayA, dict[str, Any]]:
    """Observed scores ``(n_features,)`` and the permutation accumulators `_p_value_calc` needs.

    The permutation scores are only ever reduced along the permutation axis, so the
    ``(n_perms, n_features)`` matrix is never materialized: a tally, plus sums of the scores
    shifted by the observed one, carry the same information in ``O(n_features)``.
    """
    n_cells = g.shape[0]
    # Match the casts scanpy applies to its own inputs, so the kernel sees the same numbers.
    g = g.astype(np.float64, copy=False)
    w = g.data.sum()
    rngs = np.random.default_rng(rng).spawn(n_perms)
    # Blocks of at most `_PERM_BLOCK_SIZE` entries; every feature is revisited once per block, so
    # the per-block scatter is re-paid. More than one block only once n_perms * n_cells exceeds it.
    block = int(np.clip(_PERM_BLOCK_SIZE // max(n_cells, 1), 1, n_perms))
    buffer = np.empty((block, n_cells), dtype=np.int32)
    identity = np.arange(n_cells, dtype=np.int32)[None]
    all_cells = np.arange(n_cells, dtype=np.int32)  # `nz` for a dense feature
    w_sum = np.asarray(g.sum(axis=1)).ravel()  # feature-independent: compute once, not per call
    col_sum = np.asarray(g.sum(axis=0)).ravel()  # ditto; turns the O(n) totals into O(nnz)

    moran = mode == SpatialAutocorr.MORAN
    sparse_vals = issparse(vals)
    if sparse_vals:
        # ``vals`` arrives as ``X.T``, i.e. CSC, whose row slicing is O(nnz) rather than O(nnz_row);
        # one conversion here makes the per-feature extraction below ~180x cheaper.
        vals = vals.tocsr()
    n_features = vals.shape[0]
    # The kernel walks g by column, so it needs the transpose. One O(nnz) pass, and it removes
    # the per-feature dense vector the row-wise form needed.
    gt = g.T.tocsr()
    tptr, tind, tdat = gt.indptr, gt.indices, gt.data
    # Constant features have a zero denominator; scanpy drops them and reports `nan`, so seed with it.
    score = np.full(n_features, np.nan, dtype=np.float64)
    count_ge = np.zeros(n_features, dtype=np.int64)  # integer tally of `sims >= score`, so exact
    # Mean and variance accumulate as sums of `sims - score`. Both are plain sums, so blocks merge
    # with `+=` and no Welford state is carried. `s1/n` is the observed score's distance from the
    # permutation mean, so `s2/n - (s1/n)**2` loses about eps*z**2: ~1e-13 at z=27, against ~1e-16
    # for a two-pass variance. Worth it for the simpler merge, but it is a real difference, and it
    # makes the variance exactly 0 when every permutation scores alike (see `_p_value_calc`).
    s1 = np.zeros(n_features, dtype=np.float64)
    s2 = np.zeros(n_features, dtype=np.float64)
    n_blocks = -(-n_perms // block)
    # Features are independent and the kernel is nogil, so the parallelism is here rather than
    # inside the kernel. That held even with fewer features than workers, where an in-kernel
    # `prange` over permutations still measured slower.
    pool_workers = max(1, min(n_jobs, n_features))

    def run_feature(m: int, perms: NDArrayA, first: bool) -> None:
        if sparse_vals:
            # `vals` is CSR, so this feature's non-zero cells and values are already contiguous.
            # The kernel reads them directly and never densifies the feature.
            lo_m, hi_m = vals.indptr[m], vals.indptr[m + 1]
            nz = vals.indices[lo_m:hi_m].astype(np.int32, copy=False)
            xv = np.ascontiguousarray(vals.data[lo_m:hi_m], dtype=np.float64)
        else:
            nz = all_cells
            xv = np.ascontiguousarray(vals[m], dtype=np.float64)
        if len(nz) == 0 or (xv.min() == xv.max() and (len(nz) == n_cells or xv[0] == 0.0)):
            return  # constant feature: scanpy drops it and reports `nan`
        args = (tptr, tind, tdat, w_sum, col_sum, nz, xv, w)
        if first:
            score[m] = _autocorr_perms(*args, identity, moran)[0]
        sims = _autocorr_perms(*args, perms, moran)
        count_ge[m] += int((sims >= score[m]).sum())
        shifted = sims - score[m]
        s1[m] += shifted.sum()
        s2[m] += (shifted * shifted).sum()

    with (
        numba_threads(1),  # the kernel is serial; the pool below owns the parallelism
        tqdm(total=n_blocks * n_features, unit="feature", disable=not show_progress_bar) as pbar,
    ):
        for lo in range(0, n_perms, block):
            perms = buffer[: min(block, n_perms - lo)]
            first = lo == 0

            def fill(chunk: range, perms: NDArrayA = perms, lo: int = lo) -> None:
                # numpy releases the GIL inside `permutation`, so these draws spread over the
                # pool rather than serializing on it.
                for i in chunk:
                    perms[i] = rngs[lo + i].permutation(n_cells)

            # Drained before the feature map below starts, so no worker reads `perms` while
            # another is still filling it.
            thread_map(fill, _chunks(len(perms), pool_workers), n_jobs=pool_workers)

            # Each worker takes a contiguous run of features rather than one at a time; a single
            # feature is short enough that per-item dispatch cost about as much as the work. Each
            # `m` belongs to one worker, so the writes in `run_feature` never collide. The defaults
            # bind this block's values rather than the loop's last ones.
            def run_chunk(chunk: range, perms: NDArrayA = perms, first: bool = first) -> int:
                for m in chunk:
                    run_feature(m, perms, first)
                return len(chunk)

            for done in thread_map(run_chunk, _chunks(n_features, pool_workers), n_jobs=pool_workers):
                pbar.update(done)
    # Constant features never ran, so they keep scanpy's `nan` rather than a zero mean/variance.
    ran = ~np.isnan(score)
    mean_shift = np.where(ran, s1 / n_perms, np.nan)
    return score, {
        "n_perms": n_perms,
        "count_ge": count_ge,
        "mean": score + mean_shift,
        "var": np.where(ran, s2 / n_perms - mean_shift**2, np.nan),
    }


@njit(parallel=True, fastmath=True, cache=True)
def _occur_count(  # noqa: PLR0917, numba requires positional arguments
    spatial_x: NDArrayA, spatial_y: NDArrayA, thresholds: NDArrayA, label_idx: NDArrayA, n: int, k: int, l_val: int
) -> NDArrayA:
    # Allocate a 2D array to store a flat local result per point.
    k2 = k * k
    local_results = np.zeros((n, l_val * k2), dtype=np.int32)

    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            dx = spatial_x[i] - spatial_x[j]
            dy = spatial_y[i] - spatial_y[j]
            d2 = dx * dx + dy * dy

            pair = label_idx[i] * k + label_idx[j]  # fixed in r-loop
            base = pair * l_val  # first cell for that pair

            for r in range(l_val):
                if d2 <= thresholds[r]:
                    local_results[i][base + r] += 1

    # reduction and reshape stay the same
    result_flat = local_results.sum(axis=0)
    result: NDArrayA = result_flat.reshape(k, k, l_val)

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _co_occurrence_helper(v_x: NDArrayA, v_y: NDArrayA, v_radium: NDArrayA, labs: NDArrayA) -> NDArrayA:
    """
    Fast co-occurrence probability computation using the new numba-accelerated counting.

    Parameters
    ----------
    v_x : np.ndarray, float64
         x-coordinates.
    v_y : np.ndarray, float64
         y-coordinates.
    v_radium : np.ndarray, float64
         Distance thresholds (in ascending order).
    labs : np.ndarray
         Cluster labels (as integers).

    Returns
    -------
    occ_prob : np.ndarray
         A 3D array of shape (k, k, len(v_radium)-1) containing the co-occurrence probabilities.
    labs_unique : np.ndarray
         Array of unique labels.
    """
    n = len(v_x)
    labs_unique = np.unique(labs)
    k = len(labs_unique)
    # l_val is the number of bins; here we assume the thresholds come from v_radium[1:].
    l_val = len(v_radium) - 1
    # Compute squared thresholds from the interval (skip the first value)
    thresholds = (v_radium[1:]) ** 2

    # Compute co-occurence counts.
    counts = _occur_count(v_x, v_y, thresholds, labs, n, k, l_val)

    occ_prob = np.zeros((k, k, l_val), dtype=np.float64)
    row_sums = counts.sum(axis=0)
    totals = row_sums.sum(axis=0)

    for r in prange(l_val):
        probs = row_sums[:, r] / totals[r]
        for c in range(k):
            for i in range(k):
                if probs[i] != 0.0 and row_sums[c, r] != 0.0:
                    occ_prob[i, c, r] = (counts[c, i, r] / row_sums[c, r]) / probs[i]

    return occ_prob


@d.dedent
@old_positionals("cluster_key", "spatial_key", "interval", "copy")
@deprecated_params({"n_splits": "1.10.0", "n_jobs": "1.10.0", "backend": "1.10.0", "show_progress_bar": "1.10.0"})
def co_occurrence(
    adata: AnnData | SpatialData,
    *,
    cluster_key: str,
    spatial_key: str = Key.obsm.spatial,
    interval: int | NDArrayA = 50,
    copy: bool = False,
    table_key: str | None = None,
) -> tuple[NDArrayA, NDArrayA] | None:
    """
    Compute co-occurrence probability of clusters.

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    %(cluster_key)s
    %(spatial_key)s
    interval
        Distances interval at which co-occurrence is computed. If :class:`int`, uniformly spaced interval
        of the given size will be used.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns the co-occurrence probability and the distance thresholds intervals.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['occ']`` - the co-occurrence probabilities
          across interval thresholds.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['interval']`` - the distance thresholds
          computed at ``interval``.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)

    spatial = adata.obsm[spatial_key].astype(fp)
    original_clust = adata.obs[cluster_key]
    labs = original_clust.cat.codes.to_numpy().astype(ip)  # same mapping, without a per-cell loop
    if (labs < 0).any():
        # code -1 would index out of bounds in the kernel (the old per-cell dict raised KeyError)
        raise ValueError(f"`adata.obs[{cluster_key!r}]` contains missing values.")

    # create intervals thresholds
    if isinstance(interval, int):
        thresh_min, thresh_max = _find_min_max(spatial)
        interval = np.linspace(thresh_min, thresh_max, num=interval, dtype=fp)
    else:
        interval = np.array(sorted(interval), dtype=fp, copy=True)
    if len(interval) <= 1:
        raise ValueError(f"Expected interval to be of length `>= 2`, found `{len(interval)}`.")

    spatial_x = spatial[:, 0]
    spatial_y = spatial[:, 1]

    # Compute co-occurrence probabilities using the fast numba routine.
    out = _co_occurrence_helper(spatial_x, spatial_y, interval, labs)
    start = logg.info(f"Calculating co-occurrence probabilities for `{len(interval)}` intervals")

    if copy:
        logg.info("Finish", time=start)
        return out, interval

    _save_data(
        adata, attr="uns", key=Key.uns.co_occurrence(cluster_key), data={"occ": out, "interval": interval}, time=start
    )


def _find_min_max(spatial: NDArrayA) -> tuple[float, float]:
    coord_sum = np.sum(spatial, axis=1)
    min_idx, min_idx2 = np.argpartition(coord_sum, 2)[:2]
    max_idx = np.argmax(coord_sum)
    # fmt: off
    thres_max = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1))[0, 0] / 2.0
    thres_min = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[min_idx2, :].reshape(1, -1))[0, 0]
    # fmt: on

    return thres_min.astype(fp), thres_max.astype(fp)


def _p_value_calc(
    score: NDArrayA,
    sims: dict[str, Any] | None,
    weights: spmatrix | NDArrayA,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Handle p-value calculation for spatial autocorrelation function.

    Parameters
    ----------
    score
        (n_features,).
    sims
        Permutation accumulators from `_score_perms`: ``n_perms`` and, per feature, the
        ``count_ge`` tally plus the ``mean``/``var`` of the permuted scores.
    weights
        The spatial connectivity graph, used for the analytic (normality) p-value.
    params
        Object to store relevant function parameters.

    Returns
    -------
    pval_norm
        p-value under normality assumption
    pval_sim
        p-values based on permutations
    pval_z_sim
        p-values based on standard normal approximation from permutations
    """
    p_norm, var_norm = _analytic_pval(score, weights, params)
    results = {"pval_norm": p_norm, "var_norm": var_norm}

    if sims is None:
        return results

    n_perms = sims["n_perms"]
    large_perm = sims["count_ge"].copy()  # copy: the fold below writes in place
    # subtract total perm for negative values
    large_perm[(n_perms - large_perm) < large_perm] = n_perms - large_perm[(n_perms - large_perm) < large_perm]
    # get p-value based on permutation
    p_sim: NDArrayA = (large_perm + 1) / (n_perms + 1)

    # get p-value based on standard normal approximation from permutations
    e_score_sim = sims["mean"]
    var_sim = sims["var"]
    se_score_sim = np.sqrt(var_sim)
    z_sim = (score - e_score_sim) / se_score_sim
    # NaN where a feature's permutations are all identical, so `var_sim` is exactly 0; the masks
    # below skip those and `np.empty` would leave them uninitialized.
    p_z_sim = np.full(z_sim.shape, np.nan)

    p_z_sim[z_sim > 0] = 1 - stats.norm.cdf(z_sim[z_sim > 0])
    p_z_sim[z_sim <= 0] = stats.norm.cdf(z_sim[z_sim <= 0])

    results["pval_z_sim"] = p_z_sim
    results["pval_sim"] = p_sim
    results["var_sim"] = var_sim

    return results


def _analytic_pval(score: NDArrayA, g: spmatrix | NDArrayA, params: dict[str, Any]) -> tuple[NDArrayA, float]:
    """
    Analytic p-value computation.

    See `Moran's I <https://pysal.org/esda/_modules/esda/moran.html#Moran>`_ and
    `Geary's C <https://pysal.org/esda/_modules/esda/geary.html#Geary>`_ implementation.
    """
    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0

    match params["mode"]:
        case SpatialAutocorr.GEARY.s:
            # Geary's C and Moran's I have different sampling variances under the
            # normality assumption (Cliff & Ord 1981). Use the Geary's C variance
            # (matching pysal/esda ``Geary``); reusing Moran's variance here gives a
            # miscalibrated analytic p-value (see #1183).
            Vscore_norm = ((2 * s1 + s2) * (n - 1) - 4 * s02) / (2 * (n + 1) * s02)
        case SpatialAutocorr.MORAN.s:
            # Moran's I normality variance (Cliff & Ord 1981; pysal/esda ``Moran``).
            n2 = n * n
            v_num = n2 * s1 - n * s2 + 3 * s02
            v_den = (n - 1) * (n + 1) * s02
            Vscore_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
        case mode:
            raise AssertionError(f"Unexpected mode `{mode}`.")

    seScore_norm = Vscore_norm ** (1 / 2.0)

    z_norm = (score - params["expected"]) / seScore_norm
    p_norm = np.full(score.shape, np.nan)  # constant features have a NaN score and match neither mask
    p_norm[z_norm > 0] = 1 - stats.norm.cdf(z_norm[z_norm > 0])
    p_norm[z_norm <= 0] = stats.norm.cdf(z_norm[z_norm <= 0])

    if params["two_tailed"]:
        p_norm *= 2.0

    return p_norm, Vscore_norm


def _g_moments(w: spmatrix | NDArrayA) -> tuple[float, float, float]:
    """
    Compute moments of adjacency matrix for analytic p-value calculation.

    See `pysal <https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W>`_ implementation.
    """
    # s0
    s0 = w.sum()

    # s1
    t = w.transpose() + w
    t2 = t.multiply(t) if isinstance(t, spmatrix) else t * t
    s1 = t2.sum() / 2.0

    # s2
    s2array: NDArrayA = np.array(w.sum(1) + w.sum(0).transpose()) ** 2
    s2 = s2array.sum()

    return s0, s1, s2
