from __future__ import annotations

from collections.abc import Iterator
from functools import reduce
from itertools import product
from typing import Any, Union, cast

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler

from squidpy._docs import d
from squidpy._utils import NDArrayA
from squidpy.gr._utils import _save_data

__all__ = ["var_by_distance"]


@d.dedent
def var_by_distance(
    adata: AnnData,
    groups: str | list[str] | NDArrayA,
    cluster_key: str | None = None,
    library_key: str | None = None,
    library_id: str | list[str] | None = None,
    design_matrix_key: str = "design_matrix",
    covariates: str | list[str] | None = None,
    metric: str = "euclidean",
    spatial_key: str = "spatial",
    copy: bool = False,
) -> AnnData:
    """
    Build a design matrix consisting of distance measurements to selected anchor point(s) for each observation.

    Parameters
    ----------
    %(adata)s
    groups
        Anchor point(s) to calculate distances to. Can be a single or multiple observations as long as they are annotated in an .obs column with name given by `cluster_key`.
        Alternatively, a numpy array of coordinates can be passed.
    cluster_key
        Name of annotation column in `.obs` where the observation used as anchor points are located.
    %(library_key)s
    %(library_id)s
    design_matrix_key
        Name of the design matrix saved to `.obsm`.
    covariates
        Additional covariates from `.obs` to include in the design matrix.
    metric
        Distance metric, defaults to "euclidean".
    %(spatial_key)s
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns the design_matrix with the distances to an anchor point
    Otherwise, stores design_matrix in `.obsm`.
    """
    start = logg.info(f"Creating {design_matrix_key}")

    if isinstance(groups, str):
        anchor = [groups]
    elif isinstance(groups, list):
        anchor = groups
    elif isinstance(groups, np.ndarray):
        # can't be a 2D array
        if groups.ndim != 1:
            raise ValueError(f"Expected a 1D array for 'groups', but got shape {groups.shape}.")
        anchor = cast(list[str], groups.astype(str).tolist())
    else:
        raise TypeError(f"Invalid type for groups: {type(groups)}.")

    # prepare batch key for iteration (None alone in product will result in neutral element)
    batch: list[str] | list[None]  # mypy
    if library_key is None:
        batch = [None]
    elif isinstance(library_key, str):
        if library_id is not None:
            if isinstance(library_id, str):
                library_id = [library_id]
            for x in library_id:
                if x not in adata.obs[library_key].unique():
                    raise ValueError(f"library id {x} not in {library_key}")
            batch = [slide for slide in library_id if slide in adata.obs[library_key].unique()]
        else:
            batch = adata.obs[library_key].unique().tolist()
    else:
        raise TypeError(f"Invalid type for library_key: {type(library_key)}.")

    batch_design_matrices = {}
    max_distances = {}
    anchor_col_id = 2
    # iterate over slide + anchor combinations (anchor only possible as well)
    combinations: Iterator[tuple[str, str | None]] | zip[tuple[list[int | float], str]]  # mypy
    if all(isinstance(x, list) for x in anchor) and isinstance(batch, list):
        if len(anchor) == len(batch):
            combinations = zip(anchor, batch, strict=True)
        else:
            raise ValueError(
                f"There must be one anchor point per slide. A numpy array with {len(anchor)} anchor points was provided, but {len(batch)} slides were found."
            )
    else:
        combinations = product(anchor, batch)
    for anchor_var, batch_var in combinations:
        # initialize dataframe and anndata depending on whether multiple slides are used or not
        if batch_var is not None:
            if cluster_key is not None:
                if anchor_var in pd.unique(adata[adata.obs[library_key] == batch_var].obs[cluster_key]):
                    df = _init_design_matrix(adata, cluster_key, library_key, batch_var)
                    anchor_coord, batch_coord, nan_ids = _get_coordinates(
                        adata[adata.obs[library_key] == batch_var], anchor_var, cluster_key, spatial_key
                    )
                else:
                    continue
            else:
                df = _init_design_matrix(adata, cluster_key, library_key, batch_var)
                anchor_coord, batch_coord, nan_ids = _get_coordinates(
                    adata[adata.obs[library_key] == batch_var], anchor_var, cluster_key, spatial_key
                )
                anchor_col_id = 1

        else:
            df = _init_design_matrix(adata, cluster_key, None, None)
            anchor_coord, batch_coord, nan_ids = _get_coordinates(adata, anchor_var, cluster_key, spatial_key)
            anchor_col_id = 1

        tree = KDTree(
            anchor_coord, metric=DistanceMetric.get_metric(metric)
        )  # build KDTree of anchor point coordinates
        mindist, _ = tree.query(
            batch_coord
        )  # calculate the closest distance from any observation to an observation within anchor point

        if all(isinstance(x, (int | float)) for x in anchor_var):  # adjust anchor column name if it is a numpy array
            anchor_var = "custom_anchor"
            anchor = ["custom_anchor"]
        if nan_ids.size != 0:  # in case there were nan coordinates before building the tree, add them back in
            mindist = np.insert(mindist, nan_ids - np.arange(len(nan_ids)), np.nan)
        df.insert(loc=anchor_col_id, column=str(anchor_var), value=mindist)  # add distance measurements to dataframe
        if batch_var is not None:
            df["obs"] = adata[adata.obs[library_key] == batch_var].obs_names
        else:
            df["obs"] = adata.obs_names

        # store dataframes by (slide, anchor) combination and also the corresponding maximum distance for normalization
        batch_design_matrices[(batch_var, anchor_var)] = df
        max_distances[(batch_var, anchor_var)] = df[anchor_var].max()

    # normalize euclidean distances by slide
    batch_design_matrices_ = _normalize_distances(batch_design_matrices, anchor, batch, max_distances)

    # combine individual data frames
    # merge if multiple anchor points are used but there is no separation by slides
    if library_key is None and len(anchor) > 1:
        df = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=[cluster_key, "obs"]),
            batch_design_matrices_,
        )
        df.set_index("obs", inplace=True)
        df.index.name = None
    # concatenate if a single anchor point is used within multiple slides
    elif library_key is not None and len(anchor) == 1:
        df = pd.concat(batch_design_matrices_)
        df = df.reindex(adata.obs_names)
        df = df.drop("obs", axis=1)
    # merge if multiple anchor points are used within multiple slides
    elif library_key is not None and len(anchor) > 1:
        df_by_anchor = []
        for a in anchor:
            df = pd.concat([i for i in batch_design_matrices_ if a in i.columns])
            df = df.reindex(adata.obs_names)
            df = df.drop("obs", axis=1)
            df_by_anchor.append(df)
        df = reduce(
            lambda df1, df2: pd.merge(
                df1, df2[df2.columns.difference(df1.columns)], left_index=True, right_index=True, how="outer"
            ),
            df_by_anchor,
        )
    # if a single anchor point is used and there is no separation by slides, no combination needs to be applied
    else:
        df = batch_design_matrices_[0].drop("obs", axis=1)

    # add additional covariates to design matrix
    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        df[covariates] = adata.obs[covariates].copy()
    # match index with .obs
    if isinstance(groups, list):
        df = df.reindex(adata.obs.index)
    if copy:
        logg.info("Finish", time=start)
        return df
    else:
        _save_data(adata, attr="obsm", key=design_matrix_key, data=df, time=start)


def _init_design_matrix(
    adata: AnnData,
    cluster_key: str | None,
    library_key: str | None,
    batch_var: str | None,
) -> pd.DataFrame:
    """Initialize design matrix."""
    if library_key is not None and batch_var is not None:
        if cluster_key is not None:
            df = adata[adata.obs[library_key] == batch_var].obs[[cluster_key]].copy()
            df[library_key] = adata[adata.obs[library_key] == batch_var].obs[library_key].copy()
        else:
            df = adata.obs[adata.obs[library_key] == batch_var][[library_key]].copy()
    else:
        if cluster_key is not None:
            df = adata.obs[[cluster_key]].copy()
        else:
            df = pd.DataFrame(index=adata.obs.index)
    return df


def _get_coordinates(
    adata: AnnData, anchor: str | list[int | float], annotation: str | None, spatial_key: str
) -> tuple[Any, Any, Any]:
    """Get anchor point coordinates and coordinates of all observations, excluding nan values."""
    # since amount of distances have to match n_obs, the nan ids are stored an inserted after KDTree construction
    nan_ids, _ = np.split(np.argwhere(np.isnan(adata.obsm[spatial_key])), 2, axis=1)

    if nan_ids.size != 0:
        nan_ids = np.unique(nan_ids)

    batch_coord = adata.obsm["spatial"][~np.isnan(adata.obsm["spatial"]).any(axis=1)]

    if isinstance(anchor, list):
        anchor_coord = np.array(anchor).reshape(1, -1)
    else:
        anchor_arr = np.array(adata[adata.obs[annotation] == anchor].obsm["spatial"])
        anchor_coord = anchor_arr[~np.isnan(anchor_arr).any(axis=1)]
    return anchor_coord, batch_coord, nan_ids


def _normalize_distances(
    mapping_design_matrix: dict[tuple[Any | None, Any], pd.DataFrame],
    anchor: str | list[str],
    slides: list[str] | list[None],
    mapping_max_distances: dict[tuple[Any | None, Any], float],
) -> list[pd.DataFrame]:
    """Normalize distances to anchor."""
    if not isinstance(anchor, list):
        anchor = [anchor]

    # save raw distances, set 0 distances to NaN and smallest non-zero distance to 0 for scaling
    for (_, anchor_point), design_matrix in mapping_design_matrix.items():  # (slide, anchor_point) , design_matrix
        design_matrix[f"{anchor_point}_raw"] = design_matrix[anchor_point]
        design_matrix.replace({anchor_point: 0}, np.nan, inplace=True)
        design_matrix.loc[design_matrix[anchor_point].idxmin(), anchor_point] = 0

    # normalize each slide separately will result in all conditions having a maximum distance of 1, which makes them comparable across all distances
    for (_, anchor_point), design_matrix in mapping_design_matrix.items():
        scaler = MinMaxScaler()
        scaler.fit(design_matrix[[anchor_point]].values)
        design_matrix[anchor_point] = scaler.transform(design_matrix[[anchor_point]].values)
    return list(mapping_design_matrix.values())
