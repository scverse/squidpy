"""Functions for point patterns spatial statistics."""
# flake8: noqa

from typing import Union, Optional

from anndata import AnnData

import numpy as np
import pandas as pd


def ripley_k(
    adata: AnnData,
    cluster_key: str,
    mode: str = "ripley",
    support: int = 100,
    copy: Optional[bool] = False,
) -> Union[AnnData, pd.DataFrame]:

    """
    Calculate Ripley's K statistics for each cluster in the tissue coordinates.

    Parameters
    ----------
    adata : anndata.AnnData
        anndata object of spatial transcriptomics data. The function will use coordinates in adata.obsm["spatial]
    cluster_key : str
        Key of cluster labels saved in adata.obs.
    mode: str
        Keyword which indicates the method
        for edge effects correction, as reported in
        https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator.
    support: int
        Number of points where Ripley's K is evaluated
        between a fixed radii with min=0, max=(area/2)**0.5 .
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned. Otherwise returns dataframe.

    Returns
    -------
    adata : anndata.AnnData
        modifies anndata in place and store Ripley's K stat for each cluster in adata.uns[f"ripley_k_{cluster_key}"].
        if copy = False
    df: pandas.DataFrame
        return dataframe if copy = True
    """
    try:
        # from pointpats import ripley, hull
        from astropy.stats import RipleysKEstimator
    except ImportError:
        raise ImportError("\nplease install astropy: \n\n" "\tpip install astropy\n")

    coord = adata.obsm["spatial"]
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

    return adata if copy is False else df


def moran(
    adata: AnnData,
    gene_names: Union[list, None] = None,
    transformation: Optional[str] = "r",
    permutations: Optional[int] = 999,
    corr_method: Optional[str] = "fdr_bh",
    copy: Optional[bool] = False,
) -> Union[AnnData, list]:

    """
    Calculate Moran’s I Global Autocorrelation Statistic.
    Wraps esda.moran.Moran https://pysal.org/esda/generated/esda.Moran.html#esda.Moran

    Parameters
    ----------
    adata : anndata.AnnData
        anndata object of spatial transcriptomics data. The function will use connectivities in adata.obsp["spatial_connectivities"]
    gene_names: list
        list of gene names, as stored in adata.var_names, used to compute Moran's I statistics. If none, it's computed for all.
    transformation: str
        Keyword which indicates the transformation to be used, as reported in
        https://pysal.org/esda/generated/esda.Moran.html#esda.Moran.
    permutations: str
        Keyword which indicates the number of permutations to be performed, as reported in
        https://pysal.org/esda/generated/esda.Moran.html#esda.Moran.
    corr_method:
        Correction method for multiple testing.
        Any of the methods listed here: https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned. Otherwise returns dataframe.

    Returns
    -------
    adata : anndata.AnnData
        modifies anndata in place and store Global Moran's I stats in `adata.var`.
        if copy = False
    df: pandas.DataFrame
        return dataframe if copy = True
    """

    try:
        # from pointpats import ripley, hull
        import esda
    except ImportError:
        raise ImportError("\nplease install esda: \n\n" "\tpip install esda\n")

    # init weights
    w = _set_weight_class(adata)

    lst_mi = []

    if not isinstance(gene_names, list):
        gene_names = adata.var_names

    for v in gene_names:
        y = adata[:, v].X.todense()
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

    if copy is False:
        adata.var = adata.var.join(df, how="left")
        return adata
    else:
        df


def _compute_moran(y, w, transformation, permutations):
    mi = esda.moran.Moran(y, w, transformation=transformation, permutations=permutations)
    return (mi.I, mi.p_z_sim, mi.VI_sim)


def _set_weight_class(adata: AnnData):

    try:
        a = adata.obsp["spatial_connectivity"].tolil()
    except ValueError:
        raise VAlueError("\n`adata.obsp['spatial_connectivity']` is empty, run `spatial_connectivity` first")

    neighbors = dict(enumerate(a.rows))
    weights = dict(enumerate(a.data))

    w = libpysal.weights.W(neighbors, weights, ids=adata.obs.index.values)

    return w
