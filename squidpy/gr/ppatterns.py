"""Functions for point patterns spatial statistics."""
from __future__ import annotations

from squidpy.gr._utils import (
    _save_data,
    _assert_positive,
    _assert_spatial_basis,
    _assert_categorical_obs,
    _assert_connectivity_key,
    _assert_non_empty_sequence,
)

try:
    # [ py<3.8 ]
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Literal

from typing import Tuple, Union, Iterable, Optional, Sequence
from itertools import combinations
import warnings

from scanpy import logging as logg
from anndata import AnnData

from numba import njit
from scipy.sparse import isspmatrix_lil
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import numba.types as nt

from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, parallelize, _get_n_cores
from squidpy._constants._pkg_constants import Key

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from libpysal.weights import W
        import esda
        import libpysal
except ImportError:
    esda = None
    libpysal = None
    W = None


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
    Calculate Ripley's K statistics for each cluster in the tissue coordinates.

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
    """
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
def moran(
    adata: AnnData,
    connectivity_key: str = Key.obsp.spatial_conn(),
    genes: Optional[Union[str, Sequence[str]]] = None,
    transformation: Literal["r", "B", "D", "U", "V"] = "r",  # type: ignore[name-defined]
    n_perms: int = 1000,
    corr_method: Optional[str] = "fdr_bh",
    layer: Optional[str] = None,
    seed: Optional[int] = None,
    copy: bool = False,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Calculate Moran’s I Global Autocorrelation Statistic.

    Parameters
    ----------
    %(adata)s
    %(conn_key)s
    genes
        List of gene names, as stored in :attr:`anndata.AnnData.var_names`, used to compute Moran's I statistics
        [Moran50]_.

        If `None`, it's computed for `'highly_variable'` in :attr:`anndata.AnnData.var`, if present.
        Otherwise, it's computed for all genes.
    transformation
        Transformation to be used, as reported in :class:`esda.Moran`. Default is `"r"`, row-standardized.
    %(n_perms)s
    %(corr_method)s
    layer
        Layer in :attr:`anndata.AnnData.layers` to use. If `None`, use :attr:`anndata.AnnData.X`.
    %(seed)s
    %(copy)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the following keys:

        - `'I'` - Moran's I statistic.
        - `'pval_sim'` - p-value based on permutations.
        - `'VI_sim'` - variance of `'I'` from permutations.
        - `'pval_sim_{{corr_method}}'` - the corrected p-values if ``corr_method != None`` .

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['moranI']`` - the above mentioned dataframe.
    """
    if esda is None or libpysal is None:
        raise ImportError("Please install `esda` and `libpysal` as `pip install esda libpysal`.")

    _assert_positive(n_perms, name="n_perms")
    _assert_connectivity_key(adata, connectivity_key)

    if genes is None:
        if "highly_variable" in adata.var.columns:
            genes = adata[:, adata.var.highly_variable.values].var_names.values
        else:
            genes = adata.var_names.values
    genes = _assert_non_empty_sequence(genes)  # type: ignore[assignment]

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating for `{len(genes)}` genes using `{n_jobs}` core(s)")

    w = _set_weight_class(adata, key=connectivity_key)  # init weights
    df = parallelize(
        _moran_helper,
        collection=genes,
        extractor=pd.concat,
        use_ixs=True,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(adata=adata, weights=w, transformation=transformation, permutations=n_perms, layer=layer, seed=seed)

    if corr_method is not None:
        _, pvals_adj, _, _ = multipletests(df["pval_sim"].values, alpha=0.05, method=corr_method)
        df[f"pval_sim_{corr_method}"] = pvals_adj

    df.sort_values(by="I", ascending=False, inplace=True)

    if copy:
        logg.info("Finish", time=start)
        return df

    _save_data(adata, attr="uns", key="moranI", data=df, time=start)


def _moran_helper(
    ix: int,
    gen: Iterable[str],
    adata: AnnData,
    weights: W,
    transformation: Literal["r", "B", "D", "U", "V"] = "B",  # type: ignore[name-defined]
    permutations: int = 1000,
    layer: Optional[str] = None,
    seed: Optional[int] = None,
    queue: Optional[SigQueue] = None,
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed + ix)

    moran_list = []
    for g in gen:
        mi = _compute_moran(adata.obs_vector(g, layer=layer), weights, transformation, permutations)
        moran_list.append(mi)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame(moran_list, columns=["I", "pval_sim", "VI_sim"], index=gen)


def _compute_moran(y: np.ndarray, w: W, transformation: str, permutations: int) -> Tuple[float, float, float]:
    mi = esda.moran.Moran(y, w, transformation=transformation, permutations=permutations)
    return mi.I, mi.p_z_sim, mi.VI_sim


def _set_weight_class(adata: AnnData, key: str) -> W:
    X = adata.obsp[key]
    if not isspmatrix_lil(X):
        X = X.tolil()

    neighbors = dict(enumerate(X.rows))
    weights = dict(enumerate(X.data))

    return libpysal.weights.W(neighbors, weights, ids=adata.obs.index.values)


@njit(
    ft[:, :](tt(it[:], 2), ft[:, :], tt(ft, 2), it[:]),
    parallel=False,
    fastmath=True,
)
def _occur_count(
    clust: Tuple[np.ndarray[np.int32], np.ndarray[np.int32]],
    pw_dist: np.ndarray[np.float32],
    thres: Tuple[np.float32, np.float32],
    labs_unique: np.ndarray[np.int32],
) -> np.ndarray[np.float32]:

    num = labs_unique.shape[0]
    co_occur = np.zeros((num, num), dtype=ft)
    probs_con = np.zeros((num, num), dtype=ft)

    thres_min, thres_max = thres
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

    return probs_con


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
    Compute co-occurrence probability of clusters across `n_steps` distance thresholds in spatial dimensions.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(spatial_key)s
    n_steps
        Number of distance thresholds at which co-occurrence is computed.

    %(copy)s

    Returns
    -------
    If ``copy = True``, returns the co-occurence probability and the distance thresholds intervals.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['occ']`` - the co-occurrence probabilities
          across interval thresholds.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['interval']`` - the distance thresholds
          computed at ``n_steps``.
    """
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)

    spatial = adata.obsm[spatial_key]
    original_clust = adata.obs[cluster_key]

    # find minimum, second minimum and maximum for thresholding
    coord_sum = np.sum(spatial, axis=1)
    min_idx = np.argmin(coord_sum)
    min2_idx = np.argmin(np.array(coord_sum)[coord_sum != np.amin(coord_sum)])
    max_idx = np.argmax(coord_sum)
    thres_max = (
        pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1),)[
            0
        ][0]
        / 2.0
    ).astype(fp)
    thres_min = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[min2_idx, :].reshape(1, -1),)[0][
        0
    ].astype(fp)

    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    labs = np.array([clust_map[c] for c in original_clust], dtype=ip)

    labs_unique = np.array(list(clust_map.values()), dtype=ip)
    n_cls = labs_unique.shape[0]

    interval = np.linspace(thres_min, thres_max, num=n_steps, dtype=fp)

    if n_splits is None:
        n_splits = 1

    # split array and labels
    spatial_splits = np.array_split(spatial, n_splits, axis=0)
    labs_split = np.array_split(labs, n_splits, axis=0)
    # create idx array including unique combinations and self-comparison
    idx_splits = list(combinations(np.arange(n_splits), 2))
    idx_splits.extend([(i, j) for i, j in zip(np.arange(n_splits), np.arange(n_splits))])

    n_jobs = _get_n_cores(n_jobs)

    start = logg.info(
        f"Calculating co-occurrence probabilities for \
            `{len(interval)}` intervals \
            `{len(idx_splits)}` split combinations \
            using `{n_jobs}` core(s)"
    )

    out_lst = []
    for t in idx_splits:
        print(t)
        idx_x, idx_y = t
        labs_x = labs_split[idx_x]
        labs_y = labs_split[idx_y]
        dist = pairwise_distances(spatial_splits[idx_x], spatial_splits[idx_y]).astype(fp)

        out = np.empty((n_cls, n_cls, interval.shape[0] - 1))
        for i in range(interval.shape[0] - 1):
            cond_prob = _occur_count((labs_x, labs_y), dist, (interval[i], interval[i + 1]), labs_unique)
            out[:, :, i] = cond_prob
        out_lst.append(out)

    out = sum(out_lst) / len(idx_splits)

    if copy:
        logg.info("Finish", time=start)
        return out, interval

    _save_data(
        adata, attr="uns", key=Key.uns.co_occurrence(cluster_key), data={"occ": out, "interval": interval}, time=start
    )


# def _co_occurrence_helper(
#     splits: Iterable,
#     gen: Iterable[str],
#     adata: AnnData,
#     interval: Iterable[float],
#     n_cls: int,
#     layer: Optional[str] = None,
#     queue: Optional[SigQueue] = None,
# ) -> pd.DataFrame:

#     for s in splits:
#         # TODO: parallelize (i.e. what's the interval length?)
#         dist = pairwise_distances(s).astype(fp)
#         out = np.empty((n_cls, n_cls, interval.shape[0] - 1))

#         for i in range(interval.shape[0] - 1):
#             cond_prob = _occur_count(labs, dist, (interval[i], interval[i + 1]), labs_unique)
#             out[:, :, i] = cond_prob

#         if queue is not None:
#             queue.put(Signal.UPDATE)

#     if queue is not None:
#         queue.put(Signal.FINISH)

#     return
