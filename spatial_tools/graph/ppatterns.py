"""Functions for point patterns spatial statistics
"""

from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import pairwise_distances


def ripleyK(
    adata: ad.AnnData, cluster_key: str, mode: str = "ripley", support: int = 100, copy: bool = False, return_info: bool = False,
) -> pd.DataFrame:

    """
    Calculate Ripley's K statistics for each cluster in the tissue coordinates .

    Parameters
    ----------
    adata : anndata.AnnData
        anndata object of spatial transcriptomics data. The function will use coordinates in adata.obsm["spatial]
    cluster_key : str
        Key of cluster labels saved in adata.obs.
    mode: str
        Keyword which indicates the method for edge effects correction, as reported in
        https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator.
    support: int
        Number of points where Ripley's K is evaluated
        between a fixed radii with min=0, max=(area/2)**0.5 .
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned. Is ignored otherwise.
    return_info
        Only relevant when not passing an :class:`~anndata.AnnData`:
        see “**Returns**”.

    Returns
    -------
    adata : anndata.AnnData
        modifies anndata in place and store Ripley's K stat for each cluster in adata.uns[f"ripley_k_{cluster_key}"].
        if copy = False and return_info = False
    df: pandas.DataFrame
        return dataframe if return_info = True
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
    r = np.linspace(0, ((area / 2)) ** 0.5, support)

    # set estimator
    Kest = RipleysKEstimator(
        area=area, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min
    )
    df_lst = []
    for c in adata.obs[cluster_key].unique():
        idx = adata.obs[cluster_key].values == c
        coord_sub = coord[idx, :]
        est = Kest(data=coord_sub, radii=r, mode=mode)
        df_est = pd.DataFrame(np.stack([est, r], axis=1))
        df_est.columns = ["ripley_k", "distance"]
        df_est["leiden"] = c
        df_lst.append(df_est)

    df = pd.concat(df_lst, axis=0)
    # filter by min max dist
    minmax_dist = df.groupby(cluster_key)["ripley_k"].max().min()
    df = df[df.ripley_k < minmax_dist].copy()

    if return_info:
        return df
    else:
        adata.uns[f"ripley_k_{cluster_key}"] = df
        return adata if copy else None


## this was implementation with pointpats

# def ripley_c(adata: ad.AnnData, dist_key: str, cluster_key: str, r_name: str, support: int):

#     """
#     Calculate Ripley values (k and l implemented) for each cluster in the tissue coordinates .
#     Params
#     ------
#     adata
#         The AnnData object.
#     dist_key
#         Distance key in adata.obsp, available choices according to sklearn.metrics.pairwise_distances.
#     cluster_key
#         Cluster key in adata.obs.
#     r_name
#         Ripley's function to be used.
#     support
#         Number of points for the distance moving threshold.
#     """

#     df_lst = []
#     # calculate convex hull and pairwise distances
#     coord = adata.obsm["spatial"]
#     h = hull(coord)
#     pws_dist = pairwise_distances(coord, metric=dist_key)

#     for c in adata.obs[cluster_key].unique():
#         idx = adata.obs[cluster_key].values==c
#         coord_sub = coord[idx,:]
#         dist_sub =  pws_dist[idx,:]

#         rip = _ripley_fun(coord_sub, dist_sub[:,idx] , name=r_name, support=support)
#         df_rip = pd.DataFrame(rip)
#         df_rip.columns = ["distance",f"ripley_{r_name}"]
#         df_rip[cluster_key]=c
#         df_lst.append(df_rip)

#     df = pd.concat(df_lst,axis=0)
#     # filter by min max dist
#     # minmax_dist = df.groupby(cluster_key)["distance"].max().min()
#     # df = df[df.distance < minmax_dist].copy()
#     return df


# def _ripley_fun(coord: np.array, dist: np.array, name: str, support: int):

#     if name == "k":
#         rip = ripley.k_function(coord, distances=dist, support=support,)
#     elif name == "l":
#         rip = ripley.l_function(coord, distances=dist, support=support,)
#     else:
#         print("Function not implemented")
#     return np.stack([*rip], axis=1)

