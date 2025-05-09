from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scanpy import logging as logg
from spatialdata import SpatialData

from squidpy._docs import d
from squidpy._utils import _get_adata_from_input

__all__ = ["var_embeddings"]


@d.dedent
def var_embeddings(
    data: AnnData | SpatialData,
    group: str,
    design_matrix_key: str = "design_matrix",
    table: str | None = None,
    n_bins: int = 100,
    include_anchor: bool = False,
    return_as_adata: bool = False,
) -> AnnData | None:
    """
    Bin variables by previously calculated distance to an anchor point.

    Parameters
    ----------
    data
        AnnData or SpatialData object.
    group
        Annotation column in design matrix, given by `design_matrix_key`, that is used as anchor.
    design_matrix_key
        Name of the design matrix saved to `.obsm`.
    table
        Name of the table in `SpatialData` object.
    n_bins
        Number of bins to use for aggregation.
    include_anchor
        Whether to include the variable counts belonging to the anchor point in the aggregation.
    return_as_adata
        Only evaluated, if `data` is a SpatialData object. Whether to return the result or store it as a new table.

    Returns
    -------
    AnnData or None
        If `data` is an `AnnData` object or `return_as_adata` is True, returns the new `AnnData` object.
        If `data` is a `SpatialData` object and `return_as_adata` is False, modifies `data` in place and returns None.
    """

    adata = _get_adata_from_input(data, table)

    if design_matrix_key not in adata.obsm:
        raise KeyError(f"Design matrix key '{design_matrix_key}' not found in .obsm. Available keys are: {list(adata.obsm.keys())}")

    design_matrix = adata.obsm[design_matrix_key].copy()
    if group not in design_matrix.columns:
        raise KeyError(f"Group column '{group}' not found in design matrix. Available columns: {list(design_matrix.columns)}")
    if not pd.api.types.is_numeric_dtype(design_matrix[group]):
        raise TypeError(f"The group column '{group}' must be numeric.")


    logg.info("Calculating embeddings for distance aggregations by gene.")

    # bin the data by distance and calculate the median distance for each bin
    intervals = pd.cut(design_matrix[group], bins=n_bins)

    # Extract the interval bounds as tuples and midpoints in a single pass
    design_matrix["bins"] = [(interval.left, interval.right) if pd.notnull(interval) else (0.0, 0.0) for interval in intervals]
    design_matrix["median_value"] = [interval.mid if pd.notnull(interval) else 0.0 for interval in intervals]


    # turn categorical NaNs into float 0s
    design_matrix["median_value"] = pd.to_numeric(design_matrix["median_value"], errors="coerce").fillna(0).astype(float)

    # get count matrix and add binned distance to each .obs
    X_df = adata.to_df()
    X_df["distance"] = design_matrix["median_value"]

    # aggregate the count matrix by the bins
    aggregated_df = X_df.groupby(["distance"]).sum()

    result = aggregated_df.T

    # optionally include or remove variable values for distance 0 (anchor point)
    start_bin = 0
    if not include_anchor:
        result = result.drop(result.columns[0], axis=1)
        start_bin = 1

    # rename column names for plotting
    result.columns = range(start_bin, n_bins + 1)

    adata_new = AnnData(X=result)
    adata_new.uns[design_matrix_key] = design_matrix

    if isinstance(data, AnnData):
        return adata_new
    elif isinstance(data, SpatialData):
        if return_as_adata:
            return adata_new
        else:
            data.tables["var_by_dist_bins"] = adata_new
            return None

