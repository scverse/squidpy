"""Functions for point patterns spatial statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numba.types as nt
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit, prange
from numpy.random import default_rng
from scanpy import logging as logg
from scanpy.metrics import gearys_c, morans_i
from scipy import stats
from scipy.sparse import spmatrix
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from spatialdata import SpatialData
from statsmodels.stats.multitest import multipletests

from squidpy._constants._constants import SpatialAutocorr
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, Signal, SigQueue, _get_n_cores, parallelize
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_connectivity_key,
    _assert_positive,
    _assert_spatial_basis,
    _save_data,
)

__all__ = ["spatial_autocorr", "co_occurrence"]


it = nt.int32
ft = nt.float32
tt = nt.UniTuple
ip = np.int32
fp = np.float32
bl = nt.boolean


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn(), sp=SpatialAutocorr)
def spatial_autocorr(
    adata: AnnData | SpatialData,
    connectivity_key: str = Key.obsp.spatial_conn(),
    genes: str | int | Sequence[str] | Sequence[int] | None = None,
    mode: SpatialAutocorr | Literal["moran", "geary"] = "moran",
    transformation: bool = True,
    n_perms: int | None = None,
    two_tailed: bool = False,
    corr_method: str | None = "fdr_bh",
    attr: Literal["obs", "X", "obsm"] = "X",
    layer: str | None = None,
    seed: int | None = None,
    use_raw: bool = False,
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> pd.DataFrame | None:
    """
    Calculate Global Autocorrelation Statistic (Moran’s I  or Geary's C).

    See :cite:`pysal` for reference.

    Parameters
    ----------
    %(adata)s
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
    %(seed)s
    %(copy)s
    %(parallelize)s

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
    if isinstance(adata, SpatialData):
        adata = adata.table
    _assert_connectivity_key(adata, connectivity_key)

    def extract_X(adata: AnnData, genes: str | Sequence[str] | None) -> tuple[NDArrayA | spmatrix, Sequence[Any]]:
        if genes is None:
            if "highly_variable" in adata.var:
                genes = adata[:, adata.var["highly_variable"]].var_names.values
            else:
                genes = adata.var_names.values
        elif isinstance(genes, str):
            genes = [genes]

        if not use_raw:
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
        if layer not in adata.obsm:
            raise KeyError(f"Key `{layer!r}` not found in `adata.obsm`.")
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

    g = adata.obsp[connectivity_key].copy()
    if transformation:  # row-normalize
        normalize(g, norm="l1", axis=1, copy=False)

    score = params["func"](g, vals)  # type: ignore

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating {mode}'s statistic for `{n_perms}` permutations using `{n_jobs}` core(s)")
    if n_perms is not None:
        _assert_positive(n_perms, name="n_perms")
        perms = list(np.arange(n_perms))

        score_perms = parallelize(
            _score_helper,
            collection=perms,
            extractor=np.concatenate,
            use_ixs=True,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )(mode=mode, g=g, vals=vals, seed=seed)
    else:
        score_perms = None

    with np.errstate(divide="ignore"):
        pval_results = _p_value_calc(score, score_perms, g, params)

    data_dict: dict[str, Any] = {str(params["stat"]): score, **pval_results}
    df = pd.DataFrame(data_dict, index=index)

    if corr_method is not None:
        for pv in filter(lambda x: "pval" in x, df.columns):
            _, pvals_adj, _, _ = multipletests(df[pv].values, alpha=0.05, method=corr_method)
            df[f"{pv}_{corr_method}"] = pvals_adj

    df.sort_values(by=params["stat"], ascending=params["ascending"], inplace=True)

    if copy:
        logg.info("Finish", time=start)
        return df

    mode_str = str(params["mode"])
    stat_str = str(params["stat"])
    _save_data(adata, attr="uns", key=mode_str + stat_str, data=df, time=start)


def _score_helper(
    ix: int,
    perms: Sequence[int],
    mode: SpatialAutocorr,
    g: spmatrix,
    vals: NDArrayA,
    seed: int | None = None,
    queue: SigQueue | None = None,
) -> pd.DataFrame:
    score_perms = np.empty((len(perms), vals.shape[0]))
    rng = default_rng(None if seed is None else ix + seed)
    func = morans_i if mode == SpatialAutocorr.MORAN else gearys_c

    for i in range(len(perms)):
        idx_shuffle = rng.permutation(g.shape[0])
        score_perms[i, :] = func(g[idx_shuffle, :], vals)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return score_perms


@njit(parallel=True, fastmath=True, cache=True)
def _occur_count(
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

            pair = label_idx[i] * k + label_idx[j]  # fixed in r–loop
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
         x–coordinates.
    v_y : np.ndarray, float64
         y–coordinates.
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
def co_occurrence(
    adata: AnnData | SpatialData,
    cluster_key: str,
    spatial_key: str = Key.obsm.spatial,
    interval: int | NDArrayA = 50,
    copy: bool = False,
    n_splits: int | None = None,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> tuple[NDArrayA, NDArrayA] | None:
    """
    Compute co-occurrence probability of clusters.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(spatial_key)s
    interval
        Distances interval at which co-occurrence is computed. If :class:`int`, uniformly spaced interval
        of the given size will be used.
    %(copy)s
    n_splits
        Number of splits in which to divide the spatial coordinates in
        :attr:`anndata.AnnData.obsm` ``['{spatial_key}']``.
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns the co-occurrence probability and the distance thresholds intervals.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['occ']`` - the co-occurrence probabilities
          across interval thresholds.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['interval']`` - the distance thresholds
          computed at ``interval``.
    """

    if isinstance(adata, SpatialData):
        adata = adata.table
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)

    spatial = adata.obsm[spatial_key].astype(fp)
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    labs = np.array([clust_map[c] for c in original_clust], dtype=ip)

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
    start = logg.info(
        f"Calculating co-occurrence probabilities for `{len(interval)}` intervals using `{n_jobs}` core(s) and `{n_splits}` splits"
    )

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
    sims: NDArrayA | None,
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
        (n_simulations, n_features).
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

    n_perms = sims.shape[0]
    large_perm = (sims >= score).sum(axis=0)
    # subtract total perm for negative values
    large_perm[(n_perms - large_perm) < large_perm] = n_perms - large_perm[(n_perms - large_perm) < large_perm]
    # get p-value based on permutation
    p_sim: NDArrayA = (large_perm + 1) / (n_perms + 1)

    # get p-value based on standard normal approximation from permutations
    e_score_sim = sims.sum(axis=0) / n_perms
    se_score_sim = sims.std(axis=0)
    z_sim = (score - e_score_sim) / se_score_sim
    p_z_sim = np.empty(z_sim.shape)

    p_z_sim[z_sim > 0] = 1 - stats.norm.cdf(z_sim[z_sim > 0])
    p_z_sim[z_sim <= 0] = stats.norm.cdf(z_sim[z_sim <= 0])

    var_sim = np.var(sims, axis=0)

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
    n2 = n * n
    v_num = n2 * s1 - n * s2 + 3 * s02
    v_den = (n - 1) * (n + 1) * s02

    Vscore_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
    seScore_norm = Vscore_norm ** (1 / 2.0)

    z_norm = (score - params["expected"]) / seScore_norm
    p_norm = np.empty(score.shape)
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
