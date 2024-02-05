"""Functions for point patterns spatial statistics."""

from __future__ import annotations

from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Literal,  # < 3.8
    Sequence,
    Union,  # noqa: F401
)

import numba.types as nt
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit
from numpy.random import default_rng
from scanpy import logging as logg
from scanpy.get import _get_obs_rep
from scanpy.metrics._gearys_c import _gearys_c
from scanpy.metrics._morans_i import _morans_i
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
    _assert_non_empty_sequence,
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


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn(), sp=SpatialAutocorr)
def spatial_autocorr(
    adata: AnnData | SpatialData,
    connectivity_key: str = Key.obsp.spatial_conn(),
    genes: str | int | Sequence[str] | Sequence[int] | None = None,
    mode: Literal["moran", "geary"] = SpatialAutocorr.MORAN.s,  # type: ignore[assignment]
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
            return _get_obs_rep(adata[:, genes], use_raw=False, layer=layer).T, genes
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
            ixs = np.arange(adata.obsm[layer].shape[1])  # type: ignore[assignment]
        ixs = list(np.ravel([ixs]))
        return adata.obsm[layer][:, ixs].T, ixs

    if attr == "X":
        vals, index = extract_X(adata, genes)  # type: ignore[arg-type]
    elif attr == "obs":
        vals, index = extract_obs(adata, genes)  # type: ignore[arg-type]
    elif attr == "obsm":
        vals, index = extract_obsm(adata, genes)  # type: ignore[arg-type]
    else:
        raise NotImplementedError(f"Extracting from `adata.{attr}` is not yet implemented.")

    mode = SpatialAutocorr(mode)  # type: ignore[assignment]
    if TYPE_CHECKING:
        assert isinstance(mode, SpatialAutocorr)
    params = {"mode": mode.s, "transformation": transformation, "two_tailed": two_tailed}

    if mode == SpatialAutocorr.MORAN:
        params["func"] = _morans_i
        params["stat"] = "I"
        params["expected"] = -1.0 / (adata.shape[0] - 1)  # expected score
        params["ascending"] = False
    elif mode == SpatialAutocorr.GEARY:
        params["func"] = _gearys_c
        params["stat"] = "C"
        params["expected"] = 1.0
        params["ascending"] = True
    else:
        raise NotImplementedError(f"Mode `{mode}` is not yet implemented.")

    g = adata.obsp[connectivity_key].copy()
    if transformation:  # row-normalize
        normalize(g, norm="l1", axis=1, copy=False)

    score = params["func"](g, vals)

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating {mode}'s statistic for `{n_perms}` permutations using `{n_jobs}` core(s)")
    if n_perms is not None:
        _assert_positive(n_perms, name="n_perms")
        perms = np.arange(n_perms)

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

    df = pd.DataFrame({params["stat"]: score, **pval_results}, index=index)

    if corr_method is not None:
        for pv in filter(lambda x: "pval" in x, df.columns):
            _, pvals_adj, _, _ = multipletests(df[pv].values, alpha=0.05, method=corr_method)
            df[f"{pv}_{corr_method}"] = pvals_adj

    df.sort_values(by=params["stat"], ascending=params["ascending"], inplace=True)

    if copy:
        logg.info("Finish", time=start)
        return df

    _save_data(adata, attr="uns", key=params["mode"] + params["stat"], data=df, time=start)


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
    func = _morans_i if mode == SpatialAutocorr.MORAN else _gearys_c

    for i in range(len(perms)):
        idx_shuffle = rng.permutation(g.shape[0])
        score_perms[i, :] = func(g[idx_shuffle, :], vals)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return score_perms


@njit(
    ft[:, :, :](tt(it[:], 2), ft[:, :], it[:], ft[:]),
    parallel=False,
    fastmath=True,
)
def _occur_count(
    clust: tuple[NDArrayA, NDArrayA],
    pw_dist: NDArrayA,
    labs_unique: NDArrayA,
    interval: NDArrayA,
) -> NDArrayA:
    num = labs_unique.shape[0]
    out = np.zeros((num, num, interval.shape[0] - 1), dtype=ft)

    for idx in range(interval.shape[0] - 1):
        co_occur = np.zeros((num, num), dtype=ft)
        probs_con = np.zeros((num, num), dtype=ft)

        thres_min = interval[idx]
        thres_max = interval[idx + 1]
        clust_x, clust_y = clust

        idx_x, idx_y = np.nonzero((pw_dist <= thres_max) & (pw_dist > thres_min))
        x = clust_x[idx_x]
        y = clust_y[idx_y]
        for i, j in zip(x, y):
            co_occur[i, j] += 1

        probs_matrix = co_occur / np.sum(co_occur)
        probs = np.sum(probs_matrix, axis=1)

        for c in labs_unique:
            probs_conditional = co_occur[c] / np.sum(co_occur[c])
            probs_con[c, :] = probs_conditional / probs

        out[:, :, idx] = probs_con

    return out


def _co_occurrence_helper(
    idx_splits: Iterable[tuple[int, int]],
    spatial_splits: Sequence[NDArrayA],
    labs_splits: Sequence[NDArrayA],
    labs_unique: NDArrayA,
    interval: NDArrayA,
    queue: SigQueue | None = None,
) -> pd.DataFrame:
    out_lst = []
    for t in idx_splits:
        idx_x, idx_y = t
        labs_x = labs_splits[idx_x]
        labs_y = labs_splits[idx_y]
        dist = pairwise_distances(spatial_splits[idx_x], spatial_splits[idx_y])

        out = _occur_count((labs_x, labs_y), dist, labs_unique, interval)
        out_lst.append(out)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return out_lst


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

    # annotate cluster idx
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    labs = np.array([clust_map[c] for c in original_clust], dtype=ip)
    labs_unique = np.array(list(clust_map.values()), dtype=ip)

    # create intervals thresholds
    if isinstance(interval, int):
        thresh_min, thresh_max = _find_min_max(spatial)
        interval = np.linspace(thresh_min, thresh_max, num=interval, dtype=fp)
    else:
        interval = np.array(sorted(interval), dtype=fp, copy=True)
    if len(interval) <= 1:
        raise ValueError(f"Expected interval to be of length `>= 2`, found `{len(interval)}`.")

    n_obs = spatial.shape[0]
    if n_splits is None:
        size_arr = (n_obs**2 * spatial.itemsize) / 1024 / 1024  # calc expected mem usage
        if size_arr > 2000:
            n_splits = 1
            while 2048 < (n_obs / n_splits):
                n_splits += 1
            logg.warning(
                f"`n_splits` was automatically set to `{n_splits}` to "
                f"prevent `{n_obs}x{n_obs}` distance matrix from being created"
            )
        else:
            n_splits = 1
    n_splits = max(min(n_splits, n_obs), 1)

    # split array and labels
    spatial_splits = tuple(s for s in np.array_split(spatial, n_splits, axis=0) if len(s))  # type: ignore[arg-type]
    labs_splits = tuple(s for s in np.array_split(labs, n_splits, axis=0) if len(s))  # type: ignore[arg-type]
    # create idx array including unique combinations and self-comparison
    x, y = np.triu_indices_from(np.empty((n_splits, n_splits)))  # type: ignore[arg-type]
    idx_splits = list(zip(x, y))

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(
        f"Calculating co-occurrence probabilities for `{len(interval)}` intervals "
        f"`{len(idx_splits)}` split combinations using `{n_jobs}` core(s)"
    )

    out_lst = parallelize(
        _co_occurrence_helper,
        collection=idx_splits,
        extractor=chain.from_iterable,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(
        spatial_splits=spatial_splits,
        labs_splits=labs_splits,
        labs_unique=labs_unique,
        interval=interval,
    )
    out = list(out_lst)[0] if len(idx_splits) == 1 else sum(list(out_lst)) / len(idx_splits)

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
    t2 = t.multiply(t)  # type: ignore[union-attr]
    s1 = t2.sum() / 2.0

    # s2
    s2array: NDArrayA = np.array(w.sum(1) + w.sum(0).transpose()) ** 2
    s2 = s2array.sum()

    return s0, s1, s2
