from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from spatialdata import SpatialData

from squidpy._docs import d
from squidpy.gr._utils import _save_data

__all__ = ["sliding_window"]


@d.dedent
def sliding_window(
    adata: AnnData | SpatialData,
    library_key: str | None = None,
    window_size: int | None = None,
    overlap: int = 0,
    coord_columns: tuple[str, str] = ("globalX", "globalY"),
    sliding_window_key: str = "sliding_window_assignment",
    spatial_key: str = "spatial",
    drop_partial_windows: bool = False,
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
    drop_partial_windows: bool
        If True, drop windows that are smaller than the window size at the borders.
    copy: bool
        If True, return the result, otherwise save it to the adata object.

    Returns
    -------
    If ``copy = True``, returns the sliding window annotation(s) as pandas dataframe
    Otherwise, stores the sliding window annotation(s) in .obs.
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")

    if isinstance(adata, SpatialData):
        adata = adata.table

    # we don't want to modify the original adata in case of copy=True
    if copy:
        adata = adata.copy()

    # extract coordinates of observations
    x_col, y_col = coord_columns
    if x_col in adata.obs and y_col in adata.obs:
        coords = adata.obs[[x_col, y_col]].copy()
    elif spatial_key in adata.obsm:
        coords = pd.DataFrame(
            adata.obsm[spatial_key][:, :2],
            index=adata.obs.index,
            columns=[x_col, y_col],
        )
    else:
        raise ValueError(
            f"Coordinates not found. Provide `{coord_columns}` in `adata.obs` or specify a suitable `spatial_key` in `adata.obsm`."
        )

    # infer window size if not provided
    if window_size is None:
        coord_range = max(
            coords[x_col].max() - coords[x_col].min(),
            coords[y_col].max() - coords[y_col].min(),
        )
        # mostly arbitrary choice, except that full integers usually generate windows with 1-2 cells at the borders
        window_size = max(int(coord_range // 3.95), 1)

    if window_size <= 0:
        raise ValueError("Window size must be larger than 0.")

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    libraries = [None] if library_key is None else adata.obs[library_key].unique()

    # Create a DataFrame to store the sliding window assignments
    sliding_window_df = pd.DataFrame(index=adata.obs.index)

    if sliding_window_key in adata.obs:
        logg.warning(f"Overwriting existing column '{sliding_window_key}' in adata.obs.")

    for lib in libraries:
        if lib is not None:
            lib_mask = adata.obs[library_key] == lib
            lib_coords = coords.loc[lib_mask]
        else:
            lib_mask = np.ones(len(adata), dtype=bool)
            lib_coords = coords

        min_x, max_x = lib_coords[x_col].min(), lib_coords[x_col].max()
        min_y, max_y = lib_coords[y_col].min(), lib_coords[y_col].max()

        # precalculate windows
        windows = _calculate_window_corners(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            window_size=window_size,
            overlap=overlap,
            drop_partial_windows=drop_partial_windows,
        )

        windows["window_label"] = [f"window_{i}" for i in range(len(windows))]

        # for each window, find observations that fall into it
        obs_window_map = defaultdict(list)
        for _, window in windows.iterrows():
            x_start = window["x_start"]
            x_end = window["x_end"]
            y_start = window["y_start"]
            y_end = window["y_end"]
            window_label = window["window_label"]

            mask = (
                (lib_coords[x_col] >= x_start)
                & (lib_coords[x_col] < x_end)
                & (lib_coords[y_col] >= y_start)
                & (lib_coords[y_col] < y_end)
            )
            obs_indices = lib_coords.index[mask]

            # assign the window label to the observations
            for obs_idx in obs_indices:
                obs_window_map[obs_idx].append(window_label)

        # assign observations to windows
        if overlap == 0:
            window_labels = {obs_idx: labels[0] if labels else None for obs_idx, labels in obs_window_map.items()}
            sliding_window_series = pd.Series(window_labels)
            sliding_window_df.loc[sliding_window_series.index, sliding_window_key] = sliding_window_series

            # create categorical variable for ordered windows
            sliding_window_df[sliding_window_key] = pd.Categorical(
                sliding_window_df[sliding_window_key],
                ordered=True,
                categories=sorted(
                    windows["window_label"].unique(),
                    key=lambda x: int(x.split("_")[-1]),
                ),
            )

            logg.info(f"Created column '{sliding_window_key}' which maps obs to windows.")

        else:
            # create a column for each window, indicating whether each observation belongs to it
            for idx, window in windows.iterrows():
                x_start, x_end = window["x_start"], window["x_end"]
                y_start, y_end = window["y_start"], window["y_end"]

                # initialise column with False ...
                col_name = f"{sliding_window_key}_window_{idx}"
                sliding_window_df.loc[:, col_name] = False

                obs_in_window = lib_coords[
                    (lib_coords[x_col] >= x_start)
                    & (lib_coords[x_col] < x_end)
                    & (lib_coords[y_col] >= y_start)
                    & (lib_coords[y_col] < y_end)
                ].index

                # ... and assign membership
                sliding_window_df.loc[obs_in_window, col_name] = True

                logg.info(
                    f"Created {len(windows['window_label'].unique())} columns '{sliding_window_key}_*' which map obs to overlapping windows."
                )

    if copy:
        return sliding_window_df
    for col_name, col_data in sliding_window_df.items():
        _save_data(adata, attr="obs", key=col_name, data=col_data)


def _calculate_window_corners(
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    window_size: int,
    overlap: int = 0,
    drop_partial_windows: bool = False,
) -> pd.DataFrame:
    """
    Calculate the corner points of all windows covering the area from min_x to max_x and min_y to max_y,
    with specified window_size and overlap.

    Parameters
    ----------
    min_x: float
        minimum X coordinate
    max_x: float
        maximum X coordinate
    min_y: float
        minimum Y coordinate
    max_y: float
        maximum Y coordinate
    window_size: float
        size of each window
    overlap: float
        overlap between consecutive windows (must be less than window_size)
    drop_partial_windows: bool
        if True, drop border windows that are smaller than window_size;
        if False, create smaller windows at the borders to cover the remaining space.

    Returns
    -------
    windows: pandas DataFrame with columns ['x_start', 'x_end', 'y_start', 'y_end']
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if overlap >= window_size:
        raise ValueError("Overlap must be less than the window size.")

    x_step = window_size - overlap
    y_step = window_size - overlap

    # Generate starting points
    x_starts = np.arange(min_x, max_x, x_step)
    y_starts = np.arange(min_y, max_y, y_step)

    # Create all combinations of x and y starting points
    starts = list(product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + window_size
    windows["y_end"] = windows["y_start"] + window_size

    # Adjust windows that extend beyond the bounds
    if not drop_partial_windows:
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    else:
        valid_windows = (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        windows = windows[valid_windows]

    windows = windows.reset_index(drop=True)
    windows.sort_values(
        by=["y_start", "y_end", "x_start", "x_end"],
        inplace=True,
    )
    return windows[["x_start", "x_end", "y_start", "y_end"]]
