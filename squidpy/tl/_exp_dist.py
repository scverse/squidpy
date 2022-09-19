from __future__ import annotations

from anndata import AnnData
from scanpy import logging as logg
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d
from squidpy._utils import NDArrayA
from squidpy.gr._utils import _save_data

__all__ = ["exp_dist"]

from functools import reduce
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler


@d.dedent
def exp_dist(
    adata: AnnData,
    groups: str | List[str] | NDArrayA,
    cluster_key: str,
    design_matrix_key: str = "design_matrix",
    batch_key: str | None = None,
    covariates: str | List[str] | None = None,
    spatial_key: str = Key.obsm.spatial,
    metric: str = "euclidean",
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Build a design matrix consisting of gene expression by distance to selected anchor point(s).
    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    groups
        anchor points to calculate distances from, can be a single str, a list of str or a numpy array of coordinates
    cluster_key
        annotation in .obs to take anchor point from
    metric
        distance metric, defaults to "euclidean"
    design_matrix
        name of the design matrix saved to .obsm, defaults to "design_matrix"
    batch_key
        optional: specifiy batches which contain identical anchor points
    covariates
        additional covariates from .obs which can be used for modelling
    Returns
    -------
    If ``copy = True``, returns the design_matrix and the distance thresholds intervals
    Otherwise, stores design_matrix in .obsm
    """
    start = logg.info(f"Creating {design_matrix_key}")
    # list of columns which will be categorical later on
    categorical_columns = [cluster_key]

    # save initial metadata to adata.uns
    adata.uns[design_matrix_key] = _add_metadata(
        adata, cluster_key, groups, metric=metric, batch_key=batch_key, covariates=covariates
    )

    # anchor_count = 0
    # if isinstance(groups, np.ndarray):
    # anchor: List[str | None] = [None] # if None, then assumed ArrayLike
    # else:
    anchor = (
        [groups] if isinstance(groups, str) or isinstance(groups, np.ndarray) else groups
    )  # type: ignore [assignment]

    # prepare batch key for iteration (Nonetype alone in product will result in neutral element)
    # batch = [batch_key] if batch_key is None else adata.obs[batch_key].cat.categorical
    if batch_key is None:
        batch = [None]
    else:
        batch = adata.obs[batch_key].unique()
        categorical_columns.append(batch_key)

    batch_design_matrices = {}

    # iterate over slide + anchor combinations (anchor only possible as well)
    for batch_var, anchor_var in product(batch, anchor):
        # initialize dataframe and anndata depending on whether batches are used or not
        if batch_var is not None:
            df = _init_design_matrix(adata, cluster_key, spatial_key, batch_key=batch_key, batch_var=batch_var)
            anchor_coord, batch_coord = _get_coordinates(
                adata[adata.obs[batch_key] == batch_var], anchor_var, cluster_key
            )

        else:
            df = _init_design_matrix(adata, cluster_key, spatial_key)
            anchor_coord, batch_coord = _get_coordinates(adata, anchor_var, cluster_key)

        anchor_coord = _prune_anchor_tree(anchor_coord, 0.05, 4, metric)

        tree = KDTree(anchor_coord, metric=DistanceMetric.get_metric(metric))
        mindist, _ = tree.query(batch_coord)

        # test later
        # ARRAY = ARRAY[ARRAY >= some_condition]
        if isinstance(anchor_var, np.ndarray):
            anchor_var = "custom_anchor"
            anchor = ["custom_anchor"]
        df.insert(loc=1, column=str(anchor_var), value=mindist)
        if batch_var is not None:
            df["obs"] = adata[adata.obs[batch_key] == batch_var].obs_names
        else:
            df["obs"] = adata.obs_names
        batch_design_matrices[str((batch_var, anchor_var))] = df

    # merge individual data frames
    # use merge when several anchor points were used and concat when one anchor but several slides were used
    # if a single anchor point with a single batch is used take design matrix directly

    if batch_key is None and len(anchor) > 1:
        df = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=[cluster_key, "obs"]),
            list(batch_design_matrices.values()),
        )
        df.set_index("obs", inplace=True)
        df.index.name = None
    elif batch_key is not None:
        df = pd.concat(list(batch_design_matrices.values()))
        df = df.reindex(adata.obs_names)
        df = df.drop("obs", axis=1)
    else:
        df = batch_design_matrices[str((batch_var, anchor_var))].drop("obs", axis=1)

    # normalize euclidean distances column(s)
    df = _normalize_distances(df, anchor)

    # add additional covariates to design matrix
    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        df[covariates] = adata.obs[covariates].copy()

    # organize data frames after merging, depending if batches were used or not
    # adata.obsm[design_matrix_key + "_raw_dist"] = adata.obsm[design_matrix_key + "_raw_dist"][[value for key, value in adata.uns[design_matrix_key].items() if key not in ["metric"]]]
    # df = df[[value for key, value in adata.uns[design_matrix_key].items() if key not in ["metric"]]]

    # make sure that columns without numerical values are of type categorical
    # for cat_name in categorical_columns:
    #    if isinstance(cat_name, CategoricalDtype):
    #        continue
    #    else:
    #        df[cat_name] = pd.Categorical(df[cat_name])

    if copy:
        return df
    else:
        # adapted from https://github.com/scverse/squidpy/blob/2cf664ffd9a1654b6d921307a76f5732305a371c/squidpy/gr/_ppatterns.py#L398-L404
        return _save_data(adata, attr="obsm", key=design_matrix_key, data=df, time=start)


def _add_metadata(
    adata: AnnData,
    cluster_key: str,
    groups: str | List[str] | NDArrayA,
    batch_key: str | None = None,
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
            metadata["anchor_scaled" + str(i)] = anchor
            metadata["anchor_raw" + str(i)] = anchor + "_raw"
    else:
        metadata["anchor_scaled"] = groups
        metadata["anchor_raw"] = groups + "_raw"

    metadata["annotation"] = cluster_key

    if batch_key is not None:
        metadata["batch_key"] = batch_key

    metadata["metric"] = metric

    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        for i, covariate in enumerate(covariates):
            metadata["covariate_" + str(i)] = covariate

    return metadata


def _init_design_matrix(
    adata: AnnData, cluster_key: str, spatial_key: str, batch_key: Optional[str] = None, batch_var: Optional[str] = None
) -> pd.DataFrame:
    """Initialize design matrix."""
    if batch_key is not None and batch_var is not None:
        df = adata[adata.obs[batch_key] == batch_var].obs[[cluster_key]].copy()
        df[batch_key] = adata[adata.obs[batch_key] == batch_var].obs[batch_key].copy()
    else:
        df = adata.obs[[cluster_key]].copy()

    return df


def _get_coordinates(adata: AnnData, anchor: str = None, annotation: str = None) -> Tuple(NDArrayA, NDArrayA):
    """Get anchor coordinates and coordinates of all observations."""
    if isinstance(anchor, np.ndarray):
        return (anchor, adata.obsm["spatial"])
    else:
        return (np.array(adata[adata.obs[annotation] == anchor].obsm["spatial"]), adata.obsm["spatial"])


def _prune_anchor_tree(anchor_coord: np.ndarray, pct_node_to_filter: float, nth_nbor: int, metric: str) -> np.ndarray:
    """
    Prune the anchor coordinate tree to keep the anchor point dense.
    ----------
    pct_node_to_filter
        Percentage of anchor point spots to be removed.
    nth_nbor
        Filter based on the distance to nth neighbor. If a single spot belonging to an anchor point
        is distant to the main region of the anchor point, then a value of 1 will filter the spot
        if there is a cluster of spots distant to the main region of the anchor point, the value needs to be greater
        than the amount of spots in the cluster, to filter the entire cluster.
    Returns
    -------
    A np.ndarray of anchor coordinates.
    """

    # query the anchor point spots against each other to get the distances between all spots within the anchor point
    anchor_tree = KDTree(anchor_coord, metric=DistanceMetric.get_metric(metric))
    anchor_dist, anchor_id = anchor_tree.query(anchor_coord, k=nth_nbor)

    # m amount of nodes to filter in the tree
    top_n_dist = round(len(anchor_dist) * pct_node_to_filter)

    # for all nodes in the tree, get the distances to the nth neighbor
    nth_nbor_dist = anchor_dist[..., (nth_nbor - 1)].ravel()

    # get the nodes which have the top m highest distances to the nth neighbor
    top_indices = np.argpartition(nth_nbor_dist, -top_n_dist)[-top_n_dist:]

    # from the tree remove the nodes with the m highest distances to the nth neighbor
    to_keep = np.ones(len(anchor_dist), dtype=bool)
    to_keep[top_indices] = False
    anchor_coord = anchor_coord[to_keep]

    return anchor_coord


def _normalize_distances(
    df: pd.DataFrame,
    anchor: list = None,
) -> pd.DataFrame:
    """Normalize distances to anchor."""
    scaler = MinMaxScaler()
    if len(anchor) > 1:
        max_dist_anchor = df[anchor].columns[np.where(df[anchor].values == np.max(df[anchor].values))[1]][0]
        scaler.fit(df[[max_dist_anchor]].values)
        df[f"{max_dist_anchor}_raw"] = df[max_dist_anchor]
        df[max_dist_anchor] = scaler.transform(df[[max_dist_anchor]].values)
        anchor.remove(max_dist_anchor)
        for a in anchor:
            df[f"{a}_raw"] = df[a]
            df[a] = scaler.transform(df[[a]].values)

    else:
        df[f"{anchor[0]}_raw"] = df[anchor]
        df[anchor] = scaler.fit_transform(df[anchor].values)
    return df
