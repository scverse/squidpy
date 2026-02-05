from __future__ import annotations

from itertools import product
from typing import Literal

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
    partial_windows: Literal["adaptive", "drop", "split"] | None = None,
    max_nr_cells: int | None = None,
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
    partial_windows: Literal["adaptive", "drop", "split"] | None
        If None, possibly small windows at the edges are kept.
        If `adaptive`, all windows might be shrunken a bit to avoid small windows at the edges.
        If `drop`, possibly small windows at the edges are removed.
        If `split`, windows are split into subwindows until not exceeding `max_nr_cells`
    max_nr_cells: int | None
        The maximum number of cells allowed after merging two windows.
        Required if `partial_windows = split`
    copy: bool
        If True, return the result, otherwise save it to the adata object.

    Returns
    -------
    If ``copy = True``, returns the sliding window annotation(s) as pandas dataframe
    Otherwise, stores the sliding window annotation(s) in .obs.
    """
    if partial_windows == "split":
        if max_nr_cells is None:
            raise ValueError("`max_nr_cells` must be set when `partial_windows == split`.")
        if window_size is not None:
            logg.warning(f"Ingoring `window_size` when using `{partial_windows}`")
        if overlap != 0:
            logg.warning("Ignoring `overlap` as it cannot be used with `split`")
    else:
        if max_nr_cells is not None:
            logg.warning("Ignoring `max_nr_cells` as `partial_windows != split`")
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
        window_size = max(int(np.floor(coord_range // 3.95)), 1)

    if partial_windows != "split":
        if window_size <= 0:
            raise ValueError("Window size must be larger than 0.")
        if overlap >= window_size:
            raise ValueError("Overlap must be less than the window size.")
        if overlap >= window_size // 2 and window_size == "adaptive":
            raise ValueError("Overlap must be less than `window_size` // 2 when using `adaptive`.")

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    libraries = [None] if library_key is None else adata.obs[library_key].unique()

    # Create a DataFrame to store the sliding window assignments
    sliding_window_df = pd.DataFrame(index=adata.obs.index)

    if sliding_window_key in adata.obs:
        logg.warning(f"Overwriting existing column '{sliding_window_key}' in adata.obs")

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
            partial_windows=partial_windows,
            lib_coords=lib_coords,
            x_col=x_col,
            y_col=y_col,
            max_nr_cells=max_nr_cells,
        )

        lib_key = f"{lib}_" if lib is not None else ""

        # assign observations to windows
        for idx, window in windows.iterrows():
            x_start = window["x_start"]
            x_end = window["x_end"]
            y_start = window["y_start"]
            y_end = window["y_end"]

            mask = _get_window_mask(
                x_col=x_col,
                y_col=y_col,
                lib_coords=lib_coords,
                x_start=x_start,
                x_end=x_end,
                y_start=y_start,
                y_end=y_end,
            )
            obs_indices = lib_coords.index[mask]

            if overlap == 0:
                sliding_window_df.loc[obs_indices, sliding_window_key] = f"{lib_key}window_{idx}"

            else:
                col_name = f"{sliding_window_key}_{lib_key}window_{idx}"
                sliding_window_df.loc[obs_indices, col_name] = True
                sliding_window_df.loc[:, col_name].fillna(False, inplace=True)

    if overlap == 0:
        # create categorical variable for ordered windows
        sliding_window_df[sliding_window_key] = pd.Categorical(
            sliding_window_df[sliding_window_key],
            ordered=True,
            categories=sorted(
                sliding_window_df[sliding_window_key].unique(),
                key=lambda x: int(x.split("_")[-1]),
            ),
        )

    sliding_window_df[x_col] = coords[x_col]
    sliding_window_df[y_col] = coords[y_col]

    if copy:
        return sliding_window_df
    for col_name, col_data in sliding_window_df.items():
        _save_data(adata, attr="obs", key=col_name, data=col_data)


def _get_window_mask(
    x_col: str,
    y_col: str,
    lib_coords: pd.DataFrame,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
) -> pd.Series:
    """
    Compute a boolean mask selecting coordinates that fall within a given window.

    Parameters
    ----------
    x_col: str
        Column name in `lib_coords` containing x-coordinates.
    y_col: str
        Column name in `lib_coords` containing y-coordinates.
    lib_coords: pd.DataFrame
        DataFrame containing spatial coordinates (e.g. `adata.obs` subset for one library).
        Coordinate values are expected to be integers.
    x_start: int
        Lower bound of the window in x-direction (inclusive).
    x_end: int
        Upper bound of the window in x-direction (inclusive).
    y_start: int
        Lower bound of the window in y-direction (inclusive).
    y_end: int
        Upper bound of the window in y-direction (inclusive).

    Returns
    -------
    pd.Series
        Boolean mask indicating which rows in `lib_coords` fall inside the specified window.
    """
    mask = (
        (lib_coords[x_col] >= x_start)
        & (lib_coords[x_col] <= x_end)
        & (lib_coords[y_col] >= y_start)
        & (lib_coords[y_col] <= y_end)
    )

    return mask


def _calculate_window_corners(
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    window_size: int,
    overlap: int = 0,
    partial_windows: Literal["adaptive", "drop", "split"] | None = None,
    lib_coords: pd.DataFrame | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    max_nr_cells: int | None = None,
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
    window_size: int
        size of each window
    lib_coords: pd.DataFrame | None
        coordinates of all samples for one library
    x_col: str | None
        the column in `lib_coords` corresponding to the x coordinates
    y_col: str | None
        the column in `lib_coords` corresponding to the y coordinates
    overlap: float
        overlap between consecutive windows (must be less than window_size)
    partial_windows: Literal["adaptive", "drop", "split"] | None
        If None, possibly small windows at the edges are kept.
        If 'adaptive', all windows might be shrunken a bit to avoid small windows at the edges.
        If 'drop', possibly small windows at the edges are removed.
        If 'split', windows are split into subwindows until not exceeding `max_nr_cells`

    Returns
    -------
    windows: pandas DataFrame with columns ['x_start', 'x_end', 'y_start', 'y_end']
    """
    # adjust x and y window size if 'adaptive'
    if partial_windows == "adaptive":
        number_x_windows = np.ceil((max_x - min_x) / window_size)
        number_y_windows = np.ceil((max_y - min_y) / window_size)

        x_window_size = np.ceil((max_x - min_x) / number_x_windows)
        y_window_size = np.ceil((max_y - min_y) / number_y_windows)
    else:
        x_window_size = window_size
        y_window_size = window_size

    # create the step sizes for each window
    x_step = x_window_size - overlap
    y_step = y_window_size - overlap

    # Generate starting points
    x_starts = np.arange(min_x, max_x, x_step)
    y_starts = np.arange(min_y, max_y, y_step)

    # Create all combinations of x and y starting points
    starts = list(product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + x_window_size
    windows["y_end"] = windows["y_start"] + y_window_size

    # Adjust windows that extend beyond the bounds
    if partial_windows is None:
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    elif partial_windows == "adaptive":
        pass
    elif partial_windows == "drop":
        valid_windows = (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        windows = windows[valid_windows]
    elif partial_windows == "split":
        # split the slide recursively into windows with at most max_nr_cells
        coord_x_sorted = lib_coords.sort_values(by=[x_col])
        coord_y_sorted = lib_coords.sort_values(by=[y_col])

        windows = _split_window(
            max_nr_cells, x_col, y_col, coord_x_sorted, coord_y_sorted, min_x, max_x, min_y, max_y
        ).sort_values(["x_start", "x_end", "y_start", "y_end"])
    else:
        raise ValueError(f"{partial_windows} is not a valid partial_windows argument.")

    windows = windows.reset_index(drop=True)
    return windows[["x_start", "x_end", "y_start", "y_end"]]


def _split_window(
    max_cells: int,
    x_col: str,
    y_col: str,
    coord_x_sorted: pd.DataFrame,
    coord_y_sorted: pd.DataFrame,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
) -> pd.DataFrame:
    """
    Recursively split a rectangular window into subwindows such that each subwindow
    contains at most `max_cells` cells and at least `max_cells` // 2 cells.

    Parameters
    ----------
    max_cells : int
        Maximum number of cells allowed per window.
    x_col : str
        Name of the column in `coord_x_sorted` and `coord_y_sorted` corresponding to
        x coordinates.
    y_col : str
        Name of the column in `coord_x_sorted` and `coord_y_sorted` corresponding to
        y coordinates.
    coord_x_sorted : pandas.DataFrame
        DataFrame containing cell coordinates, sorted by `x_col`.
    coord_y_sorted : pandas.DataFrame
        DataFrame containing cell coordinates, sorted by `y_col`.
    x_start : int
        Left (minimum) x coordinate of the current window.
    x_end : int
        Right (maximum) x coordinate of the current window.
    y_start : int
        Bottom (minimum) y coordinate of the current window.
    y_end : int
        Top (maximum) y coordinate of the current window.

    Returns
    -------
    windows: pandas DataFrame with columns ['x_start', 'x_end', 'y_start', 'y_end']
    """
    # return current window if it contains less cells than max_cells
    n_cells = _get_window_mask(x_col, y_col, coord_x_sorted, x_start, x_end, y_start, y_end).sum()

    if n_cells <= max_cells:
        return pd.DataFrame({"x_start": [x_start], "x_end": [x_end], "y_start": [y_start], "y_end": [y_end]})

    # define start and stop indices of subsetted windows
    sub_coord_x_sorted = coord_x_sorted[
        _get_window_mask(x_col, y_col, coord_x_sorted, x_start, x_end, y_start, y_end)
    ].reset_index(drop=True)

    sub_coord_y_sorted = coord_y_sorted[
        _get_window_mask(x_col, y_col, coord_y_sorted, x_start, x_end, y_start, y_end)
    ].reset_index(drop=True)

    middle_pos = len(sub_coord_x_sorted) // 2

    if (x_end - x_start) > (y_end - y_start):
        # vertical split
        x_middle = sub_coord_x_sorted[x_col].iloc[middle_pos]

        indices = ((x_start, x_middle, y_start, y_end), (x_middle, x_end, y_start, y_end))
    else:
        # horizontal split
        y_middle = sub_coord_y_sorted.loc[middle_pos, y_col]

        indices = ((x_start, x_end, y_start, y_middle), (x_start, x_end, y_middle, y_end))

    # recursively continue with either left&right or upper&lower windows pairs
    windows = []
    for x_start, x_end, y_start, y_end in indices:
        windows.append(
            _split_window(
                max_cells, x_col, y_col, sub_coord_x_sorted, sub_coord_y_sorted, x_start, x_end, y_start, y_end
            )
        )

    return pd.concat(windows)
