from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from functools import reduce
from itertools import product

from scanpy import logging as logg
from anndata import AnnData

from numpy.typing import ArrayLike
from sklearn.cluster import DBSCAN
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from squidpy._docs import d
from squidpy._utils import NDArrayA
from squidpy.gr._utils import _save_data
from squidpy._constants._pkg_constants import Key

__all__ = ["exp_dist"]


@d.dedent
def exp_dist(
    adata: AnnData,
    groups: str | List[str] | NDArrayA,
    library_key: str | None = None,
    cluster_key: str = None,
    design_matrix_key: str = "design_matrix",
    covariates: str | List[str] | None = None,
    metric: str = "euclidean",
    spatial_key: str = "spatial",
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Build a design matrix consisting of gene expression by distance to selected anchor point(s).

    Parameters
    ----------
    adata
        Annotated data matrix.
    groups
        Anchor points to calculate distances from, can be a single gene,
        a list of genes or a set of coordinates.
    cluster_key
        Annotation column in `.obs` to take anchor point(s) from.
    library_key
        specify slide location in '.obs' which contain identical anchor points.
    design_matrix
        Name of the design matrix saved to `.obsm`, defaults to "design_matrix".
    covariates
        Additional covariates from `.obs` to include in the design matrix.
    metric
        Distance metric, defaults to "euclidean".
    spatial_key:
        Source of spatial coordinates for each observation.
    copy
        Whether to modify copied input object.

    Returns
    -------
    If ``copy = True``, returns the design_matrix and the distance thresholds intervals
    Otherwise, stores design_matrix in .obsm
    """
    start = logg.info(f"Creating {design_matrix_key}")
    # list of columns which will be categorical later on
    categorical_columns = [cluster_key]
    # save initial metadata to adata.uns
    adata.uns[design_matrix_key] = _add_metadata(cluster_key, groups, metric=metric, library_key=library_key, covariates=covariates)

    if isinstance(groups, str) or isinstance(groups, np.ndarray):
        anchor: List[Any] = [groups]
    elif isinstance(groups, list):
        anchor = groups
    else:
        raise TypeError(f"Invalid type for groups: {type(groups)}.")

    # prepare batch key for iteration (Nonetype alone in product will result in neutral element)
    if library_key is None:
        batch = [None]
    else:
        batch = adata.obs[library_key].unique()
        categorical_columns.append(library_key)

    batch_design_matrices = {}
    max_distances = {}
    anchor_col_id = 2

    # iterate over slide + anchor combinations (anchor only possible as well)
    for anchor_var, batch_var in product(anchor, batch):
        # initialize dataframe and anndata depending on whether multiple slides are used or not
        if batch_var is not None:
            if anchor_var in pd.unique(adata[adata.obs[library_key] == batch_var].obs[cluster_key]):
                df = _init_design_matrix(adata, cluster_key, spatial_key, library_key=library_key, batch_var=batch_var)
                anchor_coord, batch_coord, nan_ids = _get_coordinates(
                    adata[adata.obs[library_key] == batch_var], anchor_var, cluster_key, spatial_key)
            else:
                continue

        else:
            df = _init_design_matrix(adata, cluster_key, spatial_key)
            anchor_coord, batch_coord, nan_ids = _get_coordinates(adata, anchor_var, cluster_key, spatial_key)
            anchor_col_id = 1

        tree = KDTree(anchor_coord, metric=DistanceMetric.get_metric(metric))
        mindist, _ = tree.query(batch_coord)

        if isinstance(anchor_var, np.ndarray):
            anchor_var = "custom_anchor"
            anchor = ["custom_anchor"]
        if nan_ids.size != 0:
            mindist = np.insert(mindist, nan_ids - np.arange(len(nan_ids)), np.nan)
        df.insert(loc=anchor_col_id, column=str(anchor_var), value=mindist)
        if batch_var is not None:
            df["obs"] = adata[adata.obs[library_key] == batch_var].obs_names
        else:
            df["obs"] = adata.obs_names
        batch_design_matrices[(batch_var, anchor_var)] = df
        max_distances[(batch_var, anchor_var)] = df[anchor_var].max()

    # scale euclidean distances by slide
    batch_design_matrices = _normalize_distances(batch_design_matrices, anchor=anchor, slides=batch, max_distances=max_distances)

    # combine individual data frames
    # merge if multiple anchor points are used but there is no separation by slides
    if library_key is None and len(anchor) > 1:
        df = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=[cluster_key, "obs"]),
            list(batch_design_matrices.values()),
        )
        df.set_index("obs", inplace=True)
        df.index.name = None
    # concatenate if a single anchor point is used within multiple slides
    elif library_key is not None and len(anchor) == 1:
        df = pd.concat(list(batch_design_matrices.values()))
        df = df.reindex(adata.obs_names)
        df = df.drop("obs", axis=1)
    # merge if multiple anchor points are used within multiple slides
    elif library_key is not None and len(anchor) > 1:
        df_by_anchor = []
        for a in anchor:
            df = pd.concat([value for key, value in batch_design_matrices.items() if a == key[1]])
            df = df.reindex(adata.obs_names)
            df = df.drop("obs", axis=1)
            df_by_anchor.append(df)
        df = reduce(lambda df1,df2: pd.merge(df1,df2[df2.columns.difference(df1.columns)], left_index=True, right_index=True, how='outer'), df_by_anchor)
    # if a single anchor point is used and there is no separation by slides, no combination needs to be applied
    else:
        df = batch_design_matrices[(batch_var, anchor_var)].drop("obs", axis=1)

    # add additional covariates to design matrix
    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        df[covariates] = adata.obs[covariates].copy()
    df.columns = df.columns.str.replace(' ', '_')
    if copy:
        logg.info("Finish", time=start)
        return df
    else:
        _save_data(adata, attr="obsm", key=design_matrix_key, data=df, time=start)


def _add_metadata(
    cluster_key: str,
    groups: str | List[str] | NDArrayA,
    library_key: str | None = None,
    covariates: str | List[str] | None = None,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """Add metadata to adata.uns."""
    metadata = {}
    if isinstance(groups, np.ndarray):
        metadata["anchor_scaled"] = "custom_anchor"
        metadata["anchor_raw"] = "custom_anchor_raw"
    elif isinstance(groups, list):
        for i, anchor in enumerate(groups):
            metadata["anchor_scaled_" + str(i)] = anchor
            metadata["anchor_raw_" + str(i)] = anchor + "_raw"
    else:
        metadata["anchor_scaled"] = groups
        metadata["anchor_raw"] = groups + "_raw"

    metadata["annotation"] = cluster_key

    if library_key is not None:
        metadata["library_key"] = library_key

    metadata["metric"] = metric

    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        for i, covariate in enumerate(covariates):
            metadata["covariate_" + str(i)] = covariate

    return metadata


def _init_design_matrix(
    adata: AnnData, cluster_key: str, spatial_key: str, library_key: Optional[str] = None, batch_var: Optional[str] = None
) -> pd.DataFrame:
    """Initialize design matrix."""
    if library_key is not None and batch_var is not None:
        df = adata[adata.obs[library_key] == batch_var].obs[[cluster_key]].copy()
        df[library_key] = adata[adata.obs[library_key] == batch_var].obs[library_key].copy()
    else:
        df = adata.obs[[cluster_key]].copy()

    return df


def _get_coordinates(adata: AnnData, anchor: str, annotation: str, spatial_key: str) -> Tuple[NDArrayA, NDArrayA]:
    """Get anchor point coordinates and coordinates of all observations, excluding nan values."""
    #since amount of distances have to match n_obs, the nan id's are stored an inserted after KDTree construction
    nan_ids,_ = np.split(np.argwhere(np.isnan(adata.obsm[spatial_key])),2,axis=1)

    if nan_ids.size != 0:
        nan_ids = np.unique(nan_ids)

    batch_coord = adata.obsm["spatial"][~np.isnan(adata.obsm["spatial"]).any(axis=1)]

    if isinstance(anchor, np.ndarray):
        anchor_coord = anchor[~np.isnan(anchor).any(axis=1)]
        return (anchor, batch_coord, nan_ids)

    else:
        anchor_arr = np.array(adata[adata.obs[annotation] == anchor].obsm["spatial"])
        anchor_coord = anchor_arr[~np.isnan(anchor_arr).any(axis=1)]
        return (anchor_coord, batch_coord, nan_ids)

def _normalize_distances(
    df: dict,
    anchor: List[str],
    slides: List[str],
    max_distances: dict
) -> pd.DataFrame:
    """Normalize distances to anchor."""
    if not isinstance(anchor, list):
        anchor = [anchor]

    #save raw distances, set 0 distances to NaN and smallest non-zero distance to 0 for scaling
    for key, value in df.items():
        value[f"{key[1]}_raw"] = value[key[1]]
        value[key[1]].replace(0, np.nan, inplace=True)
        value.loc[value[key[1]].idxmin(),key[1]] = 0
    #for each anchor point, get the slide with the highest maximum distance
    for a in anchor:
        pairs = list(product(slides, [a]))
        subset = dict((k, max_distances[k]) for k in pairs if k in max_distances)
        max_pair = max(subset, key=subset.get)
        scaler = MinMaxScaler()
        scaler.fit(df[max_pair][[max_pair[1]]].values)
        df[max_pair][max_pair[1]] = scaler.transform(df[max_pair][[max_pair[1]]].values)
        del subset[max_pair]  
        for key in subset.keys():
            df[key][key[1]] = scaler.transform(df[key][[key[1]]].values)
    
    return df
