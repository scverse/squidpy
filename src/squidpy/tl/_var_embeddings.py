from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scanpy import logging as logg
from spatialdata import SpatialData

from squidpy._docs import d

__all__ = ["var_embeddings"]


@d.dedent
def var_embeddings(
    sdata: SpatialData,
    table: str,
    group: str,
    design_matrix_key: str = "design_matrix",
    n_bins: int = 100,
    include_anchor: bool = False,
) -> AnnData | pd.DataFrame:
    """
    Bin variables by previously calculated distance to an anchor point.

    Parameters
    ----------
    %(adata)s
    table
        Name of the table in `SpatialData` object.
    group
        Annotation column in design matrix, given by `design_matrix_key`, that is used as anchor.
    design_matrix_key
        Name of the design matrix saved to `.obsm`.
    n_bins
        Number of bins to use for aggregation.
    include_anchor
        Whether to include the variable counts belonging to the anchor point in the aggregation.
    Returns
    -------
    Stores binned count matrices in `sdata.tables["var_by_dist_bins"]`.
    """

    adata = sdata.tables[table]

    if design_matrix_key not in adata.obsm.keys():
        raise ValueError(f"`.obsm['{design_matrix_key}']` does not exist. Aborting.")

    logg.info("Calculating embeddings for distance aggregations by gene.")

    df = adata.obsm[design_matrix_key].copy()
    # bin the data by distance
    df["bins"] = pd.cut(df[group], bins=n_bins)
    # get median value of each interval
    df["median_value"] = df["bins"].apply(calculate_median)
    # turn categorical NaNs into float 0s
    df["median_value"] = pd.to_numeric(df["median_value"], errors="coerce").fillna(0).astype(float)
    # get count matrix and add binned distance to each .obs
    X_df = adata.to_df()
    X_df["distance"] = df["median_value"]
    # aggregate the count matrix by the bins
    aggregated_df = X_df.groupby(["distance"]).sum()
    # transpose the count matrix
    result = aggregated_df.T
    # optionally include or remove variable values for distance 0 (anchor point)
    start_bin = 0
    if not include_anchor:
        result = result.drop(result.columns[0], axis=1)
        start_bin = 1

    # rename column names for plotting
    result.columns = range(start_bin, 101)
    # create genes x genes identity matrix (required for highlighting genes in plot)
    obs = pd.DataFrame(np.eye(len(result)), columns=result.index)
    obs.replace(1.0, pd.Series(obs.columns, obs.columns), inplace=True)
    obs.replace(0.0, "other", inplace=True)
    obs = obs.astype("category")
    obs.index = result.index
    adata_new = AnnData(X=result, obs=obs, var=pd.DataFrame(index=result.columns))

    sdata.tables["var_by_dist_bins"] = adata_new


def calculate_median(interval: pd.Interval) -> Any:
    median = interval.mid

    return median
