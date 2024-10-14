from __future__ import annotations

import math
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from spatialdata import SpatialData

from squidpy._docs import d
from squidpy._utils import NDArrayA
from squidpy.gr._utils import _save_data

__all__ = ["sliding_window"]


@d.dedent
def sliding_window(
    adata: AnnData | SpatialData,
    library_key: str | None = None,
    window_size: int = 2500,
    overlap: int = 0,
    coord_columns: tuple[str, str] = ("globalX", "globalY"),
    sliding_window_key: str = "sliding_window",
    spatial_key: str = "spatial",
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    Divide a tissue slice into regulary shaped spatially contiguous regions (windows).

    Parameters
    ----------
    %(adata)s
    window_size: int
        Size of the sliding window.
    %(library_key)s
    coord_columns: Tuple[str, str]
        Tuple of column names in `adata.obs` that specify the coordinates (x, y), e.i. ('globalX', 'globalY')
    sliding_window_key: str
        Base name for sliding window columns.
    overlap: int
        Overlap size between consecutive windows. (0 = no overlap)
    %(spatial_key)s
    copy: bool
        Whether to return a copy of the AnnData object.

    Returns
    -------
    If ``copy = True``, returns the sliding window annotation(s) as pandas dataframe
    Otherwise, stores the sliding window annotation(s) in .obs.
    """
    if window_size <= 0:
        raise ValueError("Window size must be larger than 0.")
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")

    if isinstance(adata, SpatialData):
        adata = adata.table

    x, y = coord_columns

    if x not in adata.obs or y not in adata.obs:
        adata.obs[x] = adata.obsm[spatial_key][:, 0]
        adata.obs[y] = adata.obsm[spatial_key][:, 1]

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    if library_key is None:
        libraries = [None]
        lib_mask = np.ones(len(adata.obs), dtype=bool)
        lib_data = adata.obs
    else:
        libraries = adata.obs[library_key].unique()

    # Create a DataFrame to store the sliding window assignments
    sliding_window_df = pd.DataFrame(index=adata.obs.index)

    if overlap == 0:
        sliding_window_df[sliding_window_key] = np.nan
    else:
        grid_len = math.ceil(window_size / overlap)
        num_grid_systems = grid_len**2
        for i in range(num_grid_systems):
            sliding_window_df[f"{sliding_window_key}_{i+1}"] = np.nan

    for lib in libraries:
        if library_key is not None:
            lib_mask = adata.obs[library_key] == lib
            lib_data = adata.obs[lib_mask]

        min_x, min_y = lib_data[x].min(), lib_data[y].min()

        if overlap == 0:
            xgrid, ygrid = (lib_data[x] - min_x) // window_size, (lib_data[y] - min_y) // window_size
            xrows = xgrid.max()
            sliding_window_df[sliding_window_key] = sliding_window_df[sliding_window_key].astype("object")
            sliding_window_df.loc[lib_mask, sliding_window_key] = (
                str(lib) + "_" + ((ygrid * (xrows + 1)) + xgrid).astype(int).astype(str)
            )
        else:
            for i in range(num_grid_systems):
                grid_col_name = f"{sliding_window_key}_{i+1}"
                row, col = (i // grid_len), (i % grid_len)
                x_grid_start = min_x + (overlap * col)
                y_grid_start = min_y + (overlap * row)
                xgrid, ygrid = (lib_data[x] - x_grid_start) // window_size, (lib_data[y] - y_grid_start) // window_size
                xrows = xgrid.max()
                sliding_window_df[grid_col_name] = sliding_window_df[grid_col_name].astype("object")
                sliding_window_df.loc[lib_mask, grid_col_name] = (
                    str(lib) + "_" + f"{1+i}_" + ((ygrid * (xrows + 1)) + xgrid).astype(int).astype(str)
                )

    if copy:
        return sliding_window_df
    for col_name, col_data in sliding_window_df.items():
        _save_data(adata, attr="obs", key=col_name, data=col_data)


def _sliding_window_stats(adata: AnnData, sliding_window_key: str = "sliding_window") -> None:
    """
    Calculates statistics (mean, max, min) for the sliding windows.

    Parameters
    ----------

    """
    cells_per_window = adata.obs.groupby(sliding_window_key).size()
    print(
        f"Cells per window stats: MAX: {cells_per_window.max()}, MIN: {cells_per_window.min()}, mean: {cells_per_window.mean()}, nr. of windows: {len(cells_per_window)}"
    )
