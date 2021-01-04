"""Functions for point patterns spatial statistics."""
from typing import Tuple, Union, Iterable, Optional
import warnings

from anndata import AnnData

from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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


def neighborhood_plot(
        adata: AnnData,
        spatial_key: str,
        cluster_key: str,
        condition: str,
        max_distance: int,
        step: int,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[Union[str, Path]] = None,
        **kwargs,
):
    """
    Neighborhood Plot
    Parameters
    ----------
    adata: `~anndata.AnnData`
            Annotated data matrix.
    spatial_key: `str`
            String label for the spatial data
    cluster_key: `str`
            String label for the clusters
    condition: `str`
            String specifying the cluster used for conditioning
    max_distance: `int`
            Integer specifying the maximum offset distance
    step: `int`
            Integer for generating intervals sequence from 0 to max_distance
    kwargs
        Keyword arguments to :func:`seaborn.lineplot`.
    Returns
    -------
    Neighborhood plot
    """
    pairwise_dis = pairwise_distances(adata.obsm[spatial_key])

    features = np.unique(adata.obs[cluster_key])
    f_num = features.size

    le = LabelEncoder()
    le.fit(features)

    df = pd.DataFrame({'label': features, 'ratio': 0, 'distance': 0})

    intervals = list(range(0, max_distance + 1, step))

    for i in range(1, len(intervals)):
        co_occur = np.zeros((f_num, f_num))

        rows = adata.obs[cluster_key][np.where((pairwise_dis <= intervals[i]) & (pairwise_dis > intervals[i - 1]))[0]]
        cols = adata.obs[cluster_key][np.where((pairwise_dis <= intervals[i]) & (pairwise_dis > intervals[i - 1]))[1]]

        rows_idx = le.transform(rows)
        cols_idx = le.transform(cols)

        np.add.at(co_occur, [rows_idx, cols_idx], 1)
        probs_matrix = co_occur / np.sum(co_occur)
        probs = np.sum(probs_matrix, axis=1)

        idx = le.transform([condition]).item()
        probs_conditional = co_occur[idx] / np.sum(co_occur[idx])

        df = df.append(pd.DataFrame({'label': features, 'ratio': probs_conditional / probs, 'distance': intervals[i]}))

    df_wide = df.pivot("distance", "label", "ratio")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    g = sns.lineplot(data=df_wide, dashes=False)
    g.set_xticks(intervals)
    g.set_xticklabels(g.get_xticks(), size=7)
    plt.ylabel("Probability Ratio")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if save is not None:
        save_fig(fig, path=save)
