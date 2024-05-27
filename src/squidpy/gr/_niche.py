from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from spatialdata import SpatialData

from squidpy._docs import d

__all__ = ["niche"]


@d.dedent
def calculate_niche(
    adata: Union[AnnData, SpatialData],
    groups: str,
    flavor: str,
    radius: Union[float, None],
    n_neighbors: Union[int, None],
    limit_to: Union[str, List, None] = None,
    table_key: Union[str, None] = None,
    spatial_key: str = "spatial",
) -> Union[AnnData, SpatialData]:
    # check whether anndata or spatialdata is provided and if spatialdata, check whether a table with the provided groups is present
    is_sdata = False
    if isinstance(adata, SpatialData):
        is_sdata = True
        if table_key is not None:
            table = adata.tables[table_key]
        else:
            if len(adata.tables) > 1:
                count = 0
                for key in adata.tables.keys():
                    if groups in table.obs:
                        count += 1
                        table_key = key
                if count > 1:
                    raise ValueError(
                        f"Multiple tables in `spatialdata` with group `{groups}` detected. Please specify which table to use in `table_key`."
                    )
                elif count == 0:
                    raise ValueError(
                        f"Group `{groups}` not found in any table in `spatialdata`. Please specify a valid group in `groups`."
                    )
                else:
                    table = adata.tables[table_key]
            else:
                ((key, table),) = adata.tables.items()
                if groups not in table.obs:
                    raise ValueError(
                        f"Group {groups} not found in table in `spatialdata`. Please specify a valid group in `groups`."
                    )
    else:
        table = adata

    # check whether to use radius or knn for neighborhood profile calculation
    if radius is None and n_neighbors is None:
        raise ValueError("Either `radius` or `n_neighbors` must be provided, but both are `None`.")
    if radius is not None and n_neighbors is not None:
        raise ValueError("Either `radius` and `n_neighbors` must be provided, but both were provided.")

    # subset adata if only observations within specified groups are to be considered
    if limit_to is not None:
        if isinstance(limit_to, str):
            limit_to = [limit_to]
        table_subset = table[table.obs[groups].isin([limit_to])]
    else:
        table_subset = table

    if flavor == "neighborhood":
        rel_nhood_profile, abs_nhood_profile = _calculate_neighborhood_profile(
            table, radius, n_neighbors, table_subset, spatial_key
        )
        nhood_table = _df_to_adata(rel_nhood_profile, table_subset.obs.index)
        sc.pp.neighbors(nhood_table, use_rep="X")
        sc.tl.leiden(nhood_table)
        table.obs["niche"] = nhood_table.obs["leiden"]
        if is_sdata:
            adata.tables[f"{flavor}_niche"] = nhood_table
        else:
            rel_nhood_profile = rel_nhood_profile.reindex(table.obs.index)
            table.obsm[f"{flavor}_niche"] = rel_nhood_profile

    return


def _calculate_neighborhood_profile(
    adata: Union[AnnData, SpatialData],
    groups: str,
    radius: float,
    n_neighbors: Union[int, None],
    subset: AnnData,
    spatial_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # reset index
    adata.obs = adata.obs.reset_index()

    if n_neighbors is not None:
        # get k-nearest neighbors for each observation
        tree = KDTree(adata[spatial_key])
        _, indices = tree.query(subset[spatial_key], k=n_neighbors)
    else:
        # get neighbors within a given radius for each observation
        tree = cKDTree(adata[spatial_key])
        indices = tree.query_ball_point(subset[spatial_key], r=radius)

    # get unique categories
    category_arr = adata.obs[groups].values
    unique_categories = np.unique(category_arr)

    # get obs x k matrix where each column is the category of the k-th neighbor
    cat_by_id = np.take(category_arr, indices)

    # in obs x k matrix convert categorical values to numerical values
    cat_indices = {category: index for index, category in enumerate(unique_categories)}
    cat_values = np.vectorize(cat_indices.get)(cat_by_id)

    # For each obs calculate absolute frequency for all (not just k) categories, given the subset of categories present in obs x k matrix
    m, k = cat_by_id.shape
    abs_freq = np.zeros((m, len(unique_categories)), dtype=int)
    np.add.at(abs_freq, (np.arange(m)[:, None], cat_values), 1)

    # normalize by n_neighbors to get relative frequency of each category
    rel_freq = abs_freq / k

    return rel_freq, abs_freq


def _df_to_adata(df: pd.DataFrame, index: pd.core.indexes.base.Index) -> AnnData:
    adata = AnnData(X=df.values)
    adata.obs.index = index
    return adata
