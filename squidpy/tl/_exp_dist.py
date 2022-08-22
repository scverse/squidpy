from __future__ import annotations

from typing import Any, Dict, List, Optional
from functools import reduce
from itertools import product

from anndata import AnnData

from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from squidpy._docs import d
from squidpy._utils import NDArrayA
from squidpy._constants._pkg_constants import Key

__all__ = ["exp_dist"]


@d.dedent
def exp_dist(
    adata: AnnData,
    cluster_key: str,
    groups: str | List[str] | NDArrayA,
    design_matrix_key: str = "design_matrix",
    batch_key: str | None = None,
    covariates: str | List[str] | None = None,
    spatial_key: str = Key.obsm.spatial,
    metric: str = "euclidean",
    copy: bool = True,  # TODO(LLehner): once save_adata implemented, default to False
) -> Optional[AnnData]:
    """
    Build a design matrix consisting of gene expression by distance to selected anchor point(s).

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    groups
        TODO(LLehner)
    metric
        TODO(LLehner)
    design_matrix
        TODO(LLehner)
    batch_key
        TODO(LLehner)
    covariates
        TODO(LLehner)

    Returns
    -------
    TODO(LLehner)
    If ``copy = True``, returns the design_matrix and the distance thresholds intervals TODO(LLehner)....

    Otherwise, TODO(LLehner)
    """
    # TODO(LLehner): if a key should be a categorical series, check it with :
    # TODO(LLehner): squidpy.gr._utils import _assert_categorical_obs
    # TODO(LLehner): e.g. batch_key, cluster_key
    # TODO(LLehner):  also check if entries are valid with
    # TODO(LLehner): squidpy.gr._utils import _get_valid_values

    # TODO(LLehner): consider removing?
    # list of columns which will be categorical later on
    # categorical_columns = [cluster_key]

    # save initial metadata to adata.uns
    adata.uns[design_matrix_key] = _add_metadata(
        cluster_key,
        groups,
        batch_key=batch_key,
        covariates=covariates,
        metric=metric,
    )

    # prepare batch key for iteration (Nonetype alone in product will result in neutral element)
    batch = [batch_key] if batch_key is None else adata.obs[batch_key].cat.categorical

    if isinstance(groups, np.ndarray):
        anchor: List[str | None] = [None]  # if None, then assumed ArrayLike
    else:
        anchor = [groups] if isinstance(groups, str) else groups  # type: ignore [assignment]

    batch_design_matrices = {}

    # iterate over slide + anchor combinations (anchor only possible as well)
    for batch_var, anchor_var in product(batch, anchor):
        # initialize dataframe and anndata depending on whether batches are used or not
        df = _init_design_matrix(
            adata,
            cluster_key,
            spatial_key,
            batch_key=batch_key,
            batch_var=batch_var,
        )
        if batch_var is not None:
            batch_coord = adata[adata.obs[batch_key] == batch_var].obsm[spatial_key]
        else:
            batch_coord = adata.obsm["spatial"]

        if anchor_var is not None:
            anchor_coord = adata[adata.obs[cluster_key] == anchor_var].obsm["spatial"]
        elif isinstance(groups, np.ndarray):
            anchor_coord = groups
        else:
            raise ValueError("TODO(LLehner).")

        # TODO(LLehner): does from sklearn.metrics.DistanceMetric import get_metric works?
        # TODO(LLehner): if so, please do that instead of using class
        tree = KDTree(anchor_coord, metric=DistanceMetric.get_metric(metric))
        mindist, _ = tree.query(batch_coord)

        if anchor_var is None:
            anchor_var = "custom_anchor"
        df[anchor_var] = mindist
        if batch_var is not None:
            df.index = adata[adata.obs[batch_key] == batch_var].obs_names.copy()
        else:
            df.index = adata.obs_names.copy()

        batch_design_matrices[str((batch_var, anchor_var))] = df

    # merge individual data frames
    # use merge when several anchor points were used and concat when one anchor but several slides were used
    # if a single anchor point with a single batch is used take design matrix directly
    if batch_key is None and len(anchor) == 1:
        df = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=[cluster_key, "x", "y", "obs"]),
            list(batch_design_matrices.values()),
        )
        df.index.name = None
    elif batch_key is not None:
        df = pd.concat(list(batch_design_matrices.values()))
        df = df.reindex(adata.obs_names)
    else:
        # TODO(LLehner): is this needed?
        # TODO(LLehner): from line 126 looks like a circular assign
        df = batch_design_matrices[str((batch_var, anchor_var))]

    # normalize euclidean distances column(s)
    df = _normalize_distances(df, anchor)

    # add additional covariates to design matrix
    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        adata.obsm[design_matrix_key][covariates] = adata.obs[covariates].copy()

    # organize data frames after merging, depending if batches were used or not
    # TODO(LLehner): consider removing below commented code
    # adata.obsm[design_matrix_key + "_raw_dist"] = adata.obsm[design_matrix_key + "_raw_dist"][
    #     [value for key, value in adata.uns[design_matrix_key].items() if key not in ["metric"]]
    # ]
    # df = df[[value for key, value in adata.uns[design_matrix_key].items() if key not in ["metric"]]]

    # make sure that columns without numerical values are of type categorical
    # TODO(LLehner): they should be (I don't think merge/concat change type)
    # TODO(LLehner): but please check and uncomment out if needed
    # for cat_name in categorical_columns:
    #     if isinstance(cat_name, CategoricalDtype):
    #         continue
    #     else:
    #         df[cat_name] = pd.Categorical(df[cat_name])

    if copy:
        return df

    # save design matrix dataframe to adata.obsm
    # TODO(LLehner): remove and use _save_data instead, see below
    # https://github.com/scverse/squidpy/blob/2cf664ffd9a1654b6d921307a76f5732305a371c/squidpy/gr/_ppatterns.py#L398-L404
    # adata.obsm[design_matrix_key] = df

    # _save_data(
    #     adata, attr="uns", key=Key.uns.co_occurrence(cluster_key), data={"occ": out, "interval": interval}, time=start
    # )


def _add_metadata(
    cluster_key: str,
    groups: str | List[str] | NDArrayA,
    batch_key: str | None = None,
    covariates: str | List[str] | None = None,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """Add metadata to adata.uns."""
    # TODO maybe a dataclass or named tuple would be good idea.
    metadata: Dict[str, Any] = {}
    metadata["cluster_key"] = cluster_key

    if isinstance(groups, np.ndarray):
        metadata["groups"] = ["custom"]
    else:
        metadata["groups"] = [groups] if isinstance(groups, str) else groups

    if batch_key is not None:
        metadata["batch_key"] = batch_key

    metadata["metric"] = metric

    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        metadata["covariates"] = covariates

    return metadata


def _init_design_matrix(
    adata: AnnData,
    cluster_key: str,
    spatial_key: str,
    batch_key: Optional[str] = None,
    batch_var: Optional[str] = None,
) -> pd.DataFrame:
    """Initialize design matrix."""
    if batch_key is not None and batch_var is not None:
        df = adata[adata.obs[batch_key] == batch_var].obs[[cluster_key]].copy()
        df[batch_key] = adata.obs[batch_key]
    else:
        df = adata.obs[[cluster_key]].copy()
    # TODO(LLehner): is x and y consistent across datasets?
    # TODO(LLehner): also do we need to store it?
    # TODO(LLehner): please remove
    # df["x"] = adata.obsm[spatial_key][:, 0].copy()
    # df["y"] = adata.obsm[spatial_key][:, 1].copy()
    return df


def _normalize_distances(
    df: pd.DataFrame,
    anchor: List[str | None],
) -> pd.DataFrame:
    """Normalize distances to anchor."""
    # .values used for fit and transform to avoid:
    # TODO(LLehner): what's this warning about?
    # "FutureWarning: The feature names should match those that were passed during fit.
    # Starting version 1.2, an error will be raised."

    scaler = MinMaxScaler()
    anchor = ["custom_anchor"] if anchor is None else anchor
    # TODO(LLehner): check if correct
    if len(anchor) > 1:
        max_dist_anchor = df[anchor].columns[np.where(df[anchor].values == np.max(df[anchor].values))[1]][0]
        scaler.fit(df[[max_dist_anchor]].values)
        df[max_dist_anchor] = scaler.transform(df[[max_dist_anchor]].values)
        anchor.remove(max_dist_anchor)
        for a in anchor:
            df[a] = scaler.transform(df[[a]].values)
    else:
        df[anchor] = scaler.fit_transform(df[anchor].values)
    # TODO(LLehner): idea is to save raw dist inside the same df (design_matrix)
    # TODO(LLehner): instead of separate obsm
    # TODO(LLehner): with column name like f"{anchor}_raw"
    # adata.obsm[design_matrix_key + "_raw_dist"] = raw_dist
    return df
