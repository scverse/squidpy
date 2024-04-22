from __future__ import annotations

from typing import Any

import pandas as pd
import umap
from anndata import AnnData
from scanpy import logging as logg
from sklearn.preprocessing import StandardScaler

from squidpy._docs import d

__all__ = ["var_embeddings"]


@d.dedent
def var_embeddings(
    adata: AnnData,
    group: str,
    design_matrix_key: str = "design_matrix",
    n_bins: int = 100,
    include_anchor: bool = False,
) -> AnnData:
    """
    Cluster variables by previously calculated distance to an anchor point.

    Parameters
    ----------
    %(adata)s
    group
        Annotation column in `.obs` that is used as anchor.
    design_matrix_key
        Name of the design matrix saved to `.obsm`.
    n_bins
        Number of bins to use for aggregation.
    include_anchor
        Whether to include the variable counts belonging to the anchor point in the aggregation.
    Returns
    -------
    If ``copy = True``, returns the design_matrix with the distances to an anchor point
    Otherwise, stores design_matrix in `.obsm`.
    """
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
    X_df["distance"] = df["median_value"].copy()

    # transpose the count matrix
    X_df_T = X_df.T

    # aggregate the transposed count matrix by the distances and remove the distance row
    mth_row_values = X_df_T.iloc[-1]
    result = X_df_T.groupby(mth_row_values, axis=1).sum()
    result.drop(result.tail(1).index, inplace=True)

    # optionally include or remove variable values for distance 0 (anchor point)
    if not include_anchor:
        result = result.drop(result.columns[0], axis=1)

    #reducer = umap.UMAP()

    # scale the data and reduce dimensionality
    #scaled_exp = StandardScaler().fit_transform(result.values)
    #scaled_exp_df = pd.DataFrame(scaled_exp, index=result.index, columns=result.columns)
    #embedding = reducer.fit_transform(scaled_exp_df)

    adata.varm[f"{n_bins}_bins_distance_aggregation"] = result
    #embedding_df = pd.DataFrame(embedding, index=result.index)
    #embedding_df["var"] = result.index
    #adata.uns[f"{n_bins}_bins_distance_embeddings"] = embedding_df

    return


def calculate_median(interval: pd.Interval) -> Any:
    median = interval.mid

    return median
