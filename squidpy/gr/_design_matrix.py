from functools import reduce
from itertools import product
from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler


def gexp_distance(
    adata: AnnData,
    annotation: str = None,  # categorical
    anchor: Union[str, list, np.ndarray] = None,
    metric: str = "euclidean",
    design_matrix_key: str = "design_matrix",
    batch_key: Optional[str] = None,  # batch_key #categorical
    covariates: Optional[Union[str, list]] = None,
) -> Optional[AnnData]:
    """Build a design matrix consisting of gene expression by distance to selected anchor point(s)."""
    # list of columns which will be categorical later on
    categorical_columns = [annotation]

    # save initial metadata to adata.uns
    adata.uns[design_matrix_key] = _add_metadata(adata, annotation, anchor, metric, batch_key, covariates)

    anchor_count = 0
    if isinstance(anchor, str) or isinstance(anchor, np.ndarray):
        anchor = [anchor]
        anchor_count = 1

    # prepare batch key for iteration (Nonetype alone in product will result in neutral element)
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
            df = _init_design_matrix(adata[adata.obs[batch_key] == batch_var], True, annotation, anchor_var, batch_key)
            anchor_coord, batch_coord = _get_coordinates(adata[adata.obs[batch_key] == batch_var], anchor_var, annotation)

        else:
            df = _init_design_matrix(adata, False, annotation, anchor_var)
            anchor_coord, batch_coord = _get_coordinates(adata, anchor_var, annotation)

        tree = KDTree(anchor_coord, metric=DistanceMetric.get_metric(metric))
        mindist, _ = tree.query(batch_coord)

        if isinstance(anchor_var, np.ndarray):
            anchor_var = "custom_anchor"
        df.insert(loc=2, column=str(anchor_var), value=mindist)
        if batch_var is not None:
            df["obs"] = adata[adata.obs[batch_key] == batch_var].obs_names
        else:
            df["obs"] = adata.obs_names
        batch_design_matrices[str((batch_var, anchor_var))] = df

    # merge individual data frames
    # use merge when several anchor points were used and concat when one anchor but several slides were used
    # if a single anchor point with a single batch is used take design matrix directly
    if batch_key is None and anchor_count == 0:
        df = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=[annotation, "x", "y", "obs"]),
            list(batch_design_matrices.values()),
        )
        df.set_index("obs", inplace=True)
        df.index.name = None
    elif batch_key is not None:
        df = pd.concat(list(batch_design_matrices.values()))
        df = df.reindex(adata.obs_names)
    else:
        df = batch_design_matrices[str((batch_var, anchor_var))]

    # normalize euclidean distances column(s)
    df = _normalize_distances(df, anchor, design_matrix_key)

    # add additional covariates to design matrix
    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        for covariate in covariates:
            df[covariate] = adata.obs[covariate]
            adata.obsm[design_matrix_key + "_raw_dist"][covariate] = adata.obs[covariate]
            categorical_columns.append(covariate)

    # organize data frames after merging, depending if batches were used or not
    adata.obsm[design_matrix_key + "_raw_dist"] = adata.obsm[design_matrix_key + "_raw_dist"][[value for key, value in adata.uns[design_matrix_key].items() if key not in ["metric"]]]
    df = df[[value for key, value in adata.uns[design_matrix_key].items() if key not in ["metric"]]]

    # make sure that columns without numerical values are of type categorical
    for cat_name in categorical_columns:
        if isinstance(cat_name, CategoricalDtype):
            continue
        else:
            df[cat_name] = pd.Categorical(df[cat_name])

    # save design matrix dataframe to adata.obsm
    adata.obsm[design_matrix_key] = df

    return

def _normalize_distances(df: pd.DataFrame, anchor: list = None, design_matrix_key: str = None) -> pd.DataFrame:
    """Normalize distances to anchor."""
    # .values used for fit and transform to avoid:
    # "FutureWarning: The feature names should match those that were passed during fit.
    # Starting version 1.2, an error will be raised."
    scaler = MinMaxScaler()
    raw_dist = df.copy()
    if "custom_anchor" in df.columns:
        anchor = ["custom_anchor"]
    if len(anchor) > 1:
        max_dist_anchor = df[anchor].columns[np.where(df[anchor].values == np.max(df[anchor].values))[1]][0]
        scaler.fit(df[[max_dist_anchor]].values)
        df[max_dist_anchor] = scaler.transform(df[[max_dist_anchor]].values)
        anchor.remove(max_dist_anchor)
        for a in anchor:
            df[a] = scaler.transform(df[[a]].values)
    else:
        df[anchor] = scaler.fit_transform(df[anchor].values)
    adata.obsm[design_matrix_key + "_raw_dist"] = raw_dist
    return df


def _get_coordinates(
    adata: AnnData,
    anchor: str = None,
    annotation: str = None) -> (np.ndarray, np.ndarray):
    """Get anchor coordinates and coordinates of all observations."""
    if isinstance(anchor, np.ndarray):
        return (anchor, adata.obsm["spatial"])
    else:
        return (np.array(adata[adata.obs[annotation] == anchor].obsm["spatial"]), adata.obsm["spatial"])
