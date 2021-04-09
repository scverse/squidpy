"""Functions for point patterns spatial statistics."""
from typing import Any, Dict, Tuple, Union, Callable, Iterable, Optional, Sequence
from itertools import chain
from typing_extensions import Literal  # < 3.8

from scanpy import logging as logg
from anndata import AnnData
from scanpy.get import _get_obs_rep
from scanpy.metrics._gearys_c import _gearys_c
from scanpy.metrics._morans_i import _morans_i

from numba import njit
from scipy import stats
from numpy.random import default_rng
from scipy.sparse import spmatrix
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import numba.types as nt

from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, parallelize, _get_n_cores
from squidpy.gr._utils import (
    _save_data,
    _assert_positive,
    _assert_spatial_basis,
    _assert_categorical_obs,
    _assert_connectivity_key,
    _assert_non_empty_sequence,
)
from squidpy._constants._pkg_constants import Key

__all__ = ["ripley_k", "spatial_autocorr", "co_occurrence"]


it = nt.int32
ft = nt.float32
tt = nt.UniTuple
ip = np.int32
fp = np.float32


@d.dedent
@inject_docs(key=Key.obsm.spatial)
def ripley_k(
    adata: AnnData,
    cluster_key: str,
    spatial_key: str = Key.obsm.spatial,
    mode: str = "ripley",
    support: int = 100,
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    r"""
    Calculate `Ripley's K <https://en.wikipedia.org/wiki/Spatial_descriptive_statistics#Ripley's_K_and_L_functions>`_
    statistics for each cluster in the tissue coordinates.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(spatial_key)s
    mode
        Keyword which indicates the method for edge effects correction.
        See :class:`astropy.stats.RipleysKEstimator` for valid options.
    support
        Number of points where Ripley's K is evaluated between a fixed radii with :math:`min=0`,
        :math:`max=\sqrt{{area \over 2}}`.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the following keys:

        - `'ripley_k'` - the Ripley's K statistic.
        - `'distance'` - set of distances where the estimator was evaluated.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{{cluster_key}}_ripley_k']`` - the above mentioned dataframe.
    """  # noqa: D205, D400
    try:
        # from pointpats import ripley, hull
        from astropy.stats import RipleysKEstimator
    except ImportError:
        raise ImportError("Please install `astropy` as `pip install astropy`.") from None

    _assert_spatial_basis(adata, key=spatial_key)
    coord = adata.obsm[spatial_key]

    # set coordinates
    y_min = int(coord[:, 1].min())
    y_max = int(coord[:, 1].max())
    x_min = int(coord[:, 0].min())
    x_max = int(coord[:, 0].max())
    area = int((x_max - x_min) * (y_max - y_min))
    r = np.linspace(0, (area / 2) ** 0.5, support)

    # set estimator
    Kest = RipleysKEstimator(area=area, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
    df_lst = []

    # TODO: how long does this take (i.e. does it make sense to measure the elapse time?)
    logg.info("Calculating Ripley's K")
    for c in adata.obs[cluster_key].unique():
        idx = adata.obs[cluster_key].values == c
        coord_sub = coord[idx, :]
        est = Kest(data=coord_sub, radii=r, mode=mode)
        df_est = pd.DataFrame(np.stack([est, r], axis=1))
        df_est.columns = ["ripley_k", "distance"]
        df_est[cluster_key] = c
        df_lst.append(df_est)

    df = pd.concat(df_lst, axis=0)
    # filter by min max dist
    minmax_dist = df.groupby(cluster_key)["ripley_k"].max().min()
    df = df[df.ripley_k < minmax_dist].copy()

    if copy:
        return df

    adata.uns[f"ripley_k_{cluster_key}"] = df
    _save_data(adata, attr="uns", key=Key.uns.ripley_k(cluster_key), data=df)


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn())
def spatial_autocorr(
    adata: AnnData,
    connectivity_key: str = Key.obsp.spatial_conn(),
    genes: Optional[Union[str, Sequence[str]]] = None,
    mode: Literal["moran", "geary"] = "moran",
    transformation: bool = True,
    n_perms: Optional[int] = None,
    two_tailed: bool = False,
    corr_method: Optional[str] = "fdr_bh",
    layer: Optional[str] = None,
    seed: Optional[int] = None,
    use_raw: bool = False,
    copy: bool = False,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Calculate Global Autocorrelation Statistic (Moran’s I  or Geary's C).

    See  :cite:`pysal` for reference.

    Parameters
    ----------
    %(adata)s
    %(conn_key)s
    genes
        List of gene names, as stored in :attr:`anndata.AnnData.var_names`, used to compute global
        spatial autocorrelation statistic.

        If `None`, it's computed :attr:`anndata.AnnData.var` ``['highly_variable']``, if present. Otherwise,
        it's computed for all genes.
    mode
        Mode of score calculation:
        - `'moran'` - `Moran's I autocorrelation <https://en.wikipedia.org/wiki/Moran%27s_I>`_ .
        - `'geary'` - `Geary's C autocorrelation <https://en.wikipedia.org/wiki/Geary%27s_C>`_ .
    transformation
        If `True`, weights in :attr:`anndata.AnnData.obsp` ``['{connectivity_key}']`` are row-normalized,
        advised for analytic p-value calculation.
    %(n_perms)s
         If `None`, only p-values under normality assumption are computed.
    two_tailed
        If true, pval_norm is two-tailed, otherwise it is one-tailed.
    %(corr_method)s
    layer
        Layer in :attr:`anndata.AnnData.layers` to use. If `None`, use :attr:`anndata.AnnData.X`.
    %(seed)s
    %(copy)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the following keys:

        - `'I' or 'C'` - Moran's I or Geary's C statistic.
        - `'pval_norm'` - p-value under normality assumption.
        - `'var_norm'` - variance of `'score'` under normality assumption.
        - `'pval_z_sim'` - p-value based on standard normal approximation from permutations.
        - `'pval_sim'` - p-value based on permutations.
        - `'var_sim'` - variance of `'score'` from permutations.
        - `'{{p_val}}_{{corr_method}}'` - the corrected p-values if ``corr_method != None`` .

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['score_statistic']`` - the above mentioned dataframe.
    """
    _assert_connectivity_key(adata, connectivity_key)

    if genes is None:
        if "highly_variable" in adata.var.columns:
            genes = adata[:, adata.var.highly_variable.values].var_names.values
        else:
            genes = adata.var_names.values
    genes = _assert_non_empty_sequence(genes, name="genes")

    params: Dict[str, Any] = {"mode": mode, "transformation": transformation, "two_tailed": two_tailed}

    if params["mode"] == "moran":
        params["func"] = _morans_i
        params["stat"] = "I"
        params["expected"] = -1.0 / (adata.shape[0] - 1)  # expected score
    elif params["mode"] == "geary":
        params["func"] = _gearys_c
        params["stat"] = "C"
        params["expected"] = 1.0
    else:
        raise NotImplementedError(f"mode: `{mode}` is not available.")

    n_jobs = _get_n_cores(n_jobs)

    vals = _get_obs_rep(adata[:, genes], use_raw=use_raw, layer=layer).T
    g = adata.obsp[connectivity_key].copy()
    # row-normalize
    if transformation:
        normalize(g, norm="l1", axis=1, copy=False)

    res = np.empty((vals.shape[0], 6), dtype=np.float32) * np.nan

    score = params["func"](
        g,
        vals,
    )

    start = logg.info(f"Calculating for `{n_perms}` permutations using `{n_jobs}` core(s)")
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
        )(func=params["func"], g=g, vals=vals, seed=seed)
    else:
        score_perms = None

    p_norm, var_norm, p_sim, p_z_sim, var_sim = _p_value_calc(score, score_perms, g, params)

    res[:, 1] = p_norm
    res[:, 2] = var_norm
    res[:, 3] = p_z_sim
    res[:, 4] = p_sim
    res[:, 5] = var_sim

    res[:, 0] = score
    df = pd.DataFrame(
        res, columns=[params["stat"], "pval_norm", "var_norm", "pval_z_sim", "pval_sim", "var_sim"], index=genes
    )  # fix names

    if corr_method is not None:
        for pv in filter(lambda x: "pval" in x, df.columns):
            _, pvals_adj, _, _ = multipletests(df[pv].values, alpha=0.05, method=corr_method)
            df[f"{pv}_{corr_method}"] = pvals_adj

    if params["mode"] == "moran":
        df.sort_values(by=params["stat"], ascending=False, inplace=True)
    elif params["mode"] == "geary":
        df.sort_values(by=params["stat"], ascending=True, inplace=True)

    if copy:
        logg.info("Finish", time=start)
        return df

    _save_data(adata, attr="uns", key=params["mode"] + params["stat"], data=df, time=start)


def _score_helper(
    ix: int,
    perms: Sequence[int],
    func: Callable[..., np.ndarray],
    g: spmatrix,
    vals: np.ndarray,
    seed: Optional[int] = None,
    queue: Optional[SigQueue] = None,
) -> pd.DataFrame:

    score_perms = np.empty((len(perms), vals.shape[0]))
    rng = default_rng(None if seed is None else ix + seed)
    for i in range(len(perms)):
        idx_shuffle = rng.permutation(g.shape[0])
        score_perms[i, :] = func(
            g[idx_shuffle, :],
            vals,
        )

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
    clust: Tuple[np.ndarray, np.ndarray],
    pw_dist: np.ndarray,
    labs_unique: np.ndarray,
    interval: np.ndarray,
) -> np.ndarray:
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
    idx_splits: Iterable[Tuple[int, int]],
    spatial_splits: Sequence[np.ndarray],
    labs_splits: Sequence[np.ndarray],
    labs_unique: np.ndarray,
    interval: np.ndarray,
    queue: Optional[SigQueue] = None,
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
    adata: AnnData,
    cluster_key: str,
    spatial_key: str = Key.obsm.spatial,
    n_steps: int = 50,
    copy: bool = False,
    n_splits: Optional[int] = None,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute co-occurrence probability of clusters.

    The co-occurrence is computed across ``n_steps`` distance thresholds in spatial dimensions.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(spatial_key)s
    n_steps
        Number of distance thresholds at which co-occurrence is computed.

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
          computed at ``n_steps``.
    """
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)

    spatial = adata.obsm[spatial_key].astype(fp)
    original_clust = adata.obs[cluster_key]

    # find minimum, second minimum and maximum for thresholding
    thres_min, thres_max = _find_min_max(spatial)

    # annotate cluster idx
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    labs = np.array([clust_map[c] for c in original_clust], dtype=ip)

    labs_unique = np.array(list(clust_map.values()), dtype=ip)

    # create intervals thresholds
    interval = np.linspace(thres_min, thres_max, num=n_steps, dtype=fp)

    n_obs = spatial.shape[0]
    if n_splits is None:
        size_arr = (n_obs ** 2 * 4) / 1024 / 1024  # calc expected mem usage
        if size_arr > 2_000:
            s = 1
            while 2_048 < (n_obs / s):
                s += 1
            n_splits = s
            logg.warning(
                f"`n_splits` was automatically set to: {n_splits}\n"
                f"preventing a NxN with N={n_obs} distance matrix to be created"
            )
        else:
            n_splits = 1

    n_splits = max(min(n_splits, n_obs), 1)

    # split array and labels
    spatial_splits = tuple(s for s in np.array_split(spatial, n_splits, axis=0) if len(s))
    labs_splits = tuple(s for s in np.array_split(labs, n_splits, axis=0) if len(s))
    # create idx array including unique combinations and self-comparison
    x, y = np.triu_indices_from(np.empty((n_splits, n_splits)))
    idx_splits = [(i, j) for i, j in zip(x, y)]

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(
        f"Calculating co-occurrence probabilities for\
            `{len(interval)}` intervals\
            `{len(idx_splits)}` split combinations\
            using `{n_jobs}` core(s)"
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

    if len(idx_splits) == 1:
        out = list(out_lst)[0]
    else:
        out = sum(list(out_lst)) / len(idx_splits)

    if copy:
        logg.info("Finish", time=start)
        return out, interval

    _save_data(
        adata, attr="uns", key=Key.uns.co_occurrence(cluster_key), data={"occ": out, "interval": interval}, time=start
    )


def _find_min_max(spatial: np.ndarray) -> Tuple[float, float]:

    coord_sum = np.sum(spatial, axis=1)
    min_idx, min_idx2 = np.argpartition(coord_sum, 2)[0:2]
    max_idx = np.argmax(coord_sum)
    thres_max = (
        pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1),)[
            0
        ][0]
        / 2.0
    ).astype(fp)
    thres_min = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[min_idx2, :].reshape(1, -1),)[0][
        0
    ].astype(fp)

    return thres_min, thres_max


def _p_value_calc(
    score: np.ndarray,
    sims: Union[np.ndarray, None],
    weights: Union[spmatrix, np.ndarray],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, Union[Any, np.ndarray]]:
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
    if sims is not None:
        n_perms = sims.shape[0]
        large_perm = (sims >= score).sum(axis=0)
        # subtract total perm for negative values
        large_perm[(n_perms - large_perm) < large_perm] = n_perms - large_perm[(n_perms - large_perm) < large_perm]
        # get p-value based on permutation
        p_sim: np.ndarray = (large_perm + 1) / (n_perms + 1)

        # get p-value based on standard normal approximation from permutations
        e_score_sim = sims.sum(axis=0) / n_perms
        se_score_sim = sims.std(axis=0)
        z_sim = (score - e_score_sim) / se_score_sim
        p_z_sim = np.empty(z_sim.shape)

        p_z_sim[z_sim > 0] = 1 - stats.norm.cdf(z_sim[z_sim > 0])
        p_z_sim[z_sim <= 0] = stats.norm.cdf(z_sim[z_sim <= 0])

        var_sim = np.var(sims, axis=0)
    else:
        p_sim = p_z_sim = var_sim = np.empty(score.shape[0]) * np.nan

    p_norm, var_norm = _analytic_pval(score, weights, params)

    return p_norm, var_norm, p_sim, p_z_sim, var_sim


def _analytic_pval(
    score: np.ndarray, g: Union[spmatrix, np.ndarray], params: Dict[str, Any]
) -> Tuple[np.ndarray, float]:
    """
    Analytic pvalue computation.

    See `Moran's I <https://pysal.org/esda/_modules/esda/moran.html#Moran>`_
    and `Geary's C <https://pysal.org/esda/_modules/esda/geary.html#Geary>`_
    implementation.
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


def _g_moments(w: Union[spmatrix, np.ndarray]) -> Tuple[np.float_, np.float_, np.float_]:
    """
    Compute moments of adjacency matrix for analytic p-value calculation.

    see `Pysal <https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W>`_ implementation.
    """
    # s0
    s0 = w.sum()

    # s1
    t = w.transpose() + w
    t2 = t.multiply(t)
    s1 = t2.sum() / 2.0

    # s2
    s2array = np.array(w.sum(1) + w.sum(0).transpose()) ** 2
    s2 = s2array.sum()

    return s0, s1, s2
