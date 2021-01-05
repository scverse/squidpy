"""Functions for point patterns spatial statistics."""
from typing import Tuple, Union, Iterable, Optional
import warnings

from anndata import AnnData

from numba import njit
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances
from pandas.api.types import infer_dtype, is_categorical_dtype
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import numba.types as nt

from squidpy._docs import d, inject_docs
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
    mode: str = "ripley",
    support: int = 100,
    copy: Optional[bool] = False,
) -> Union[AnnData, pd.DataFrame]:
    r"""
    Calculate Ripley's K statistics for each cluster in the tissue coordinates.

    Parameters
    ----------
    %(adata)s
        The function will use coordinates in ``adata.obsm[{key!r}]``.
        TODO: expose key.
    cluster_key
        Key of cluster labels saved in :attr:`anndata.AnnData.obs`.
        TODO: docrep.
    mode
        Keyword which indicates the method for edge effects correction, as reported in
        :class:`astropy.stats.RipleysKEstimator`.
    support
        Number of points where Ripley's K is evaluated between a fixed radii with :math:`min=0`,
        :math:`max=\sqrt{{area \over 2}}`.
    copy
        If an :class:`anndata.AnnData` is passed, determines whether a copy is returned.

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

    coord = adata.obsm[Key.obsm.spatial]
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
    gene_names: Optional[Iterable[str]] = None,
    transformation: str = "r",
    permutations: int = 1000,
    corr_method: Optional[str] = "fdr_bh",
    copy: Optional[bool] = False,
) -> Optional[pd.DataFrame]:
    """
    Calculate Moran’s I Global Autocorrelation Statistic.

    Parameters
    ----------
    adata
        The function will use connectivities in ``adata.obsp[{key!r}]``. TODO: expose key
    gene_names
        List of gene names, as stored in :attr:`anndata.AnnData.var_names`, used to compute Moran's I statistics
        [Moran50]_. If None, it's computed for all genes.
    transformation
        Keyword which indicates the transformation to be used, as reported in :class:`esda.Moran`.
    permutations
        Keyword which indicates the number of permutations to be performed, as reported in :class:`esda.Moran`.
    corr_method
        Correction method for multiple testing. See :func:`statsmodels.stats.multitest.multipletests` for available
        methods.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame`. Otherwise, it modifies ``adata`` in place and stores
    Global Moran's I stats in :attr:`anndata.AnnData.var`.
    """
    if esda is None or libpysal is None:
        raise ImportError("Please install `esda` and `libpysal` as `pip install esda libpysal`.")

    # TODO: use_raw?
    # init weights
    w = _set_weight_class(adata)

    lst_mi = []

    if gene_names is None:
        gene_names = adata.var_names
    if not isinstance(gene_names, Iterable):
        raise TypeError(f"Expected `gene_names` to be `Iterable`, found `{type(gene_names).__name__}`.")

    sparse = issparse(adata.X)

    for v in gene_names:
        y = adata[:, v].X.todense() if sparse else adata[:, v].X
        mi = _compute_moran(y, w, transformation, permutations)
        lst_mi.append(mi)

    df = pd.DataFrame(lst_mi, index=gene_names, columns=["I", "pval_sim", "VI_sim"])

    if corr_method is not None:
        _, pvals_adj, _, _ = multipletests(df["pval_sim"].values, alpha=0.05, method=corr_method)
        df[f"pval_sim_{corr_method}"] = pvals_adj

    if copy:
        return df

    adata.var = adata.var.join(df, how="left")


def _compute_moran(y: np.ndarray, w: W, transformation: str, permutations: int) -> Tuple[float, float, float]:
    mi = esda.moran.Moran(y, w, transformation=transformation, permutations=permutations)
    return mi.I, mi.p_z_sim, mi.VI_sim


# TODO: expose the key?
# TODO: is return type correct?
def _set_weight_class(adata: AnnData) -> W:

    try:
        a = adata.obsp[Key.obsp.spatial_conn()].tolil()
    except KeyError:
        raise KeyError(
            f"`adata.obsp[{Key.obsp.spatial_conn()!r}]` is empty, run `squidpy.graph.spatial_connectivity()` first."
        ) from None

    neighbors = dict(enumerate(a.rows))
    weights = dict(enumerate(a.data))

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
    clust: np.ndarray,
    pw_dist: np.ndarray,
    thres: Tuple[np.float32, np.float32],
    labs_unique: np.int32,
) -> np.ndarray:

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
    cluster_key
        Cluster key in :attr:`anndata.AnnData.obs`.
    spatial_key
        Spatial key in :attr:`anndata.AnnData.obsm`.
    steps
        number of step to compute radius for co-occurrence.
    %(copy)s

    Returns
    -------
    If ``copy = True`` returns two :class:`numpy.ndarray`. Otherwise, it modifies the ``adata`` object with the
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
        raise KeyError(f"{spatial_key} not present in `adata.obs`" "Choose a different spatial_key or run first ")

    spatial = adata.obsm[spatial_key]
    original_clust = adata.obs[cluster_key]
    dist = pairwise_distances(spatial).astype(fp)

    thres_max = dist.max() / 2
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
        return (out, interval)

    adata.uns[f"{cluster_key}_co_occurrence"] = {"occ": out, "interval": interval}
