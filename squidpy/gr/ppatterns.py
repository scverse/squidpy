"""Functions for point patterns spatial statistics."""
from __future__ import annotations

try:
    # [ py<3.8 ]
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Literal
from typing import Any, Tuple, Union, Iterable, Optional, Sequence
import warnings

from scanpy import logging as logg
from anndata import AnnData

from numba import njit
from scipy.sparse import isspmatrix_lil
from sklearn.metrics import pairwise_distances
from pandas.api.types import infer_dtype, is_categorical_dtype
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import numba.types as nt

from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, parallelize, _get_n_cores
from squidpy.constants._pkg_constants import Key

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


@d.dedent
@inject_docs(key=Key.obsm.spatial)
def ripley_k(
    adata: AnnData,
    cluster_key: str,
    spatial_key: str = Key.obsm.spatial,
    mode: str = "ripley",
    support: int = 100,
    copy: bool = False,
) -> Union[AnnData, pd.DataFrame]:
    r"""
    Calculate Ripley's K statistics for each cluster in the tissue coordinates.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(spatial_key)s
    mode
        Keyword which indicates the method for edge effects correction, as reported in
        :class:`astropy.stats.RipleysKEstimator`.
    support
        Number of points where Ripley's K is evaluated between a fixed radii with :math:`min=0`,
        :math:`max=\sqrt{{area \over 2}}`.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame`. Otherwise, it modifies ``adata`` and store Ripley's K stat
    for each cluster in ``adata.uns['ripley_k_{{cluster_key}}']``.
    """
    try:
        # from pointpats import ripley, hull
        from astropy.stats import RipleysKEstimator
    except ImportError:
        raise ImportError("Please install `astropy` as `pip install astropy`.") from None

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


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn())
def moran(
    adata: AnnData,
    connectivity_key: str = Key.obsp.spatial_conn(),
    genes: Optional[Union[str, Sequence[str]]] = None,
    transformation: Literal["r", "B", "D", "U", "V"] = "B",  # type: ignore[name-defined]
    permutations: int = 1000,
    corr_method: Optional[str] = "fdr_bh",
    layer: Optional[Union[str, None]] = None,
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
        [Moran50]_. If None, it's computed for `highly_variable` in attr:`anndata.AnnData.var`,
        if present, else it is computed for all genes.
    transformation
        Transformation to be used, as reported in :class:`esda.Moran`. Default: `"B"` binary.
    permutations
        Number of permutations to be performed, as reported in :class:`esda.Moran`.
    corr_method
        Correction method for multiple testing. See :func:`statsmodels.stats.multitest.multipletests` for available
        methods.
    layer
        What layer values should be returned from. If None, X is used.
    %(copy)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame`. Otherwise, it modifies ``adata`` in place and stores
    Global Moran's I stats in :attr:`anndata.uns["moranI"]`.
    """
    if esda is None or libpysal is None:
        raise ImportError("Please install `esda` and `libpysal` as `pip install esda libpysal`.")

    if genes is None:
        if "highly_variable" in adata.var.columns:
            genes = adata[:, adata.var.highly_variable.values].var_names.values.tolist()
        else:
            genes = adata.var_names.values.tolist()
    else:
        if isinstance(genes, str):
            genes = [
                genes,
            ]
    if not isinstance(genes, Sequence):
        raise TypeError(f"Expected `genes` to be `Sequence`, found `{type(genes).__name__}`.")

    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"{connectivity_key} not present in `adata.obsp`."
            "Choose a different connectivity_key or run first "
            "gr.spatial_neighbors(adata) on the AnnData object."
        )

    w = _set_weight_class(adata, key=connectivity_key)  # init weights

    n_jobs = _get_n_cores(n_jobs)
    logg.info(f"Calculating `{len(genes)}` genes using `{n_jobs}` core(s)")

    df = parallelize(
        _moran_helper,
        collection=genes,
        extractor=pd.concat,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(adata=adata, weights=w, transformation=transformation, permutations=permutations, layer=layer)

    if corr_method is not None:
        _, pvals_adj, _, _ = multipletests(df["pval_sim"].values, alpha=0.05, method=corr_method)
        df[f"pval_sim_{corr_method}"] = pvals_adj

    # df.index = genes
    df.sort_values(by="I", ascending=False, inplace=True)

    if copy:
        return df

    adata.uns["moranI"] = df


def _moran_helper(
    gen: Iterable[Any],
    adata: AnnData,
    weights: W,
    transformation: Literal["r", "B", "D", "U", "V"] = "B",  # type: ignore[name-defined]
    permutations: int = 1000,
    layer: Optional[Union[str, None]] = None,
    queue: Optional[SigQueue] = None,
) -> pd.DataFrame:

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


it = nt.int32
ft = nt.float32
tt = nt.UniTuple


@njit(
    ft[:, :](it[:], ft[:, :], tt(ft, 2), it[:]),
    parallel=False,
    fastmath=True,
)
def _occur_count(
    clust: np.ndarray[np.int32],
    pw_dist: np.ndarray[np.float32],
    thres: Tuple[np.float32, np.float32],
    labs_unique: np.ndarray[np.int32],
) -> np.ndarray[np.float32]:

    num = labs_unique.shape[0]
    co_occur = np.zeros((num, num), dtype=ft)
    probs_con = np.zeros((num, num), dtype=ft)

    thres_min, thres_max = thres

    idx_x, idx_y = np.nonzero((pw_dist <= thres_max) & (pw_dist > thres_min))
    x = clust[idx_x]
    y = clust[idx_y]
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
    spatial_key: Optional[str] = Key.obsm.spatial,
    steps: int = 50,
    copy: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute co-occurrence probability of clusters across spatial dimensions.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(spatial_key)s
    steps
        Number of step to compute radius for co-occurrence.
    %(copy)s

    Returns
    -------
    If ``copy = True`` returns two :class:`numpy.array`. Otherwise, it modifies the ``adata`` object with the
    following keys:

        - :attr:`anndata.AnnData.uns` ``[{cluster_key}_co_occurrence]`` - the centrality scores.
    """
    ip = np.int32
    fp = np.float32
    if cluster_key not in adata.obs.keys():
        raise KeyError(f"Cluster key `{cluster_key}` not found in `adata.obs`.")
    if not is_categorical_dtype(adata.obs[cluster_key]):
        raise TypeError(
            f"Expected `adata.obs[{cluster_key}]` to be `categorical`, "
            f"found `{infer_dtype(adata.obs[cluster_key])}`."
        )
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial key `{spatial_key}` not found in `adata.obsm`.")

    spatial = adata.obsm[spatial_key]
    original_clust = adata.obs[cluster_key]
    dist = pairwise_distances(spatial).astype(fp)

    thres_max = dist.max() / 2.0
    thres_min = np.amin(np.array(dist)[dist != np.amin(dist)]).astype(fp)
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}

    labs = np.array([clust_map[c] for c in original_clust], dtype=ip)

    labs_unique = np.array(list(clust_map.values()), dtype=ip)
    n_cls = labs_unique.shape[0]

    interval = np.linspace(thres_min, thres_max, num=steps, dtype=fp)

    out = np.empty((n_cls, n_cls, interval.shape[0] - 1))
    for i in range(interval.shape[0] - 1):
        cond_prob = _occur_count(labs, dist, (interval[i], interval[i + 1]), labs_unique)
        out[:, :, i] = cond_prob

    if copy:
        return out, interval

    adata.uns[f"{cluster_key}_co_occurrence"] = {"occ": out, "interval": interval}
