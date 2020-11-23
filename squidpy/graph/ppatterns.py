"""Functions for point patterns spatial statistics."""
from typing import Union, Optional

from anndata import AnnData

import numpy as np
import pandas as pd
from scipy.sparse import issparse

try:
    import esda
    import libpysal
except ImportError:
    esda = None
    libpysal = None


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
    adata
        :mod:`anndata` object of spatial transcriptomics data. The function will use coordinates in
        adata.obsm['X_spatial'].
    cluster_key
        Key of cluster labels saved in :attr:`anndata.AnnData.obs`.
    mode
        Keyword which indicates the method for edge effects correction, as reported in
        :class:`astropy.stats.RipleysKEstimator`.
    support
        Number of points where Ripley's K is evaluated between a fixed radii with :math:`min=0`,
        :math:`max=\sqrt{area \over 2}`.
    copy
        If an :class:`anndata.AnnData` is passed, determines whether a copy is returned.

    Returns
    -------
    :class:`anndata.AnnData`
        Modifies ``adata`` and store Ripley's K stat for each cluster in ``adata.uns['ripley_k_{cluster_key}']``.
    :class:`pandas.DataFrame`
        Return a dataframe if ``copy = True``.
    """
    try:
        # from pointpats import ripley, hull
        from astropy.stats import RipleysKEstimator
    except ImportError:
        raise ImportError("\nplease install astropy: \n\n" "\tpip install astropy\n")

    coord = adata.obsm["X_spatial"]
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
    print(df.head())
    minmax_dist = df.groupby(cluster_key)["ripley_k"].max().min()
    df = df[df.ripley_k < minmax_dist].copy()

    adata.uns[f"ripley_k_{cluster_key}"] = df

    return df if copy else adata


def moran(
    adata: AnnData,
    gene_names: Union[list, None] = None,
    transformation: Optional[str] = "r",
    permutations: Optional[int] = 1000,
    corr_method: Optional[str] = "fdr_bh",
    copy: Optional[bool] = False,
) -> Union[AnnData, pd.DataFrame]:
    """
    Calculate Moran’s I Global Autocorrelation Statistic.

    Wraps :class:`esda.Moran`.

    Parameters
    ----------
    adata
        anndata object of spatial transcriptomics data. The function will use connectivities in
        adata.obsp["spatial_connectivities"]
    gene_names
        list of gene names, as stored in adata.var_names, used to compute Moran's I statistics.
        If none, it's computed for all.
    transformation
        Keyword which indicates the transformation to be used, as reported in :class:`esda.Moran`.
    permutations
        Keyword which indicates the number of permutations to be performed, as reported in :class:`esda.Moran`.
    corr_method
        Correction method for multiple testing. Any of the methods listed in
        :func:`statsmodels.stats.multitest.multipletests`.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned. Otherwise returns dataframe.

    Returns
    -------
    :class:`anndata.AnnData`
        Modifies anndata in place and store Global Moran's I stats in `adata.var`.
        if copy = False
    :class:`pandas.DataFrame`
        Return dataframe if copy = True.
    """
    if esda is None or libpysal is None:
        raise ImportError("Please install esda and libpysal: \n\n\tpip install esda and libpysal\n")

    # init weights
    w = _set_weight_class(adata)

    lst_mi = []

    if not isinstance(gene_names, list):
        gene_names = adata.var_names

    sparse = issparse(adata.X)

    for v in gene_names:
        y = adata[:, v].X.todense() if sparse else adata[:, v].X
        mi = _compute_moran(y, w, transformation, permutations)
        lst_mi.append(mi)

    df = pd.DataFrame(lst_mi)
    df.columns = ["I", "pval_sim", "VI_sim"]
    df.index = gene_names

    if corr_method is not None:
        pvals = df.pval_sim.values
        from statsmodels.stats.multitest import multipletests

        _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method=corr_method)
        df[f"pval_sim_{corr_method}"] = pvals_adj

    if copy:
        return df

    adata.var = adata.var.join(df, how="left")
    return adata


def _compute_moran(y, w, transformation, permutations):
    mi = esda.moran.Moran(y, w, transformation=transformation, permutations=permutations)
    return (mi.I, mi.p_z_sim, mi.VI_sim)


def _set_weight_class(adata: AnnData):

    try:
        a = adata.obsp["spatial_connectivities"].tolil()
    except KeyError:
        raise KeyError(
            "\n`adata.obsp['spatial_connectivities']` is empty, run " "`squidpy.graph.spatial_connectivity()` first"
        )

    neighbors = dict(enumerate(a.rows))
    weights = dict(enumerate(a.data))

    w = libpysal.weights.W(neighbors, weights, ids=adata.obs.index.values)

    return w
