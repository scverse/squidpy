from __future__ import annotations

import math
import time
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
    coord_columns: tuple[str, str] = ("globalX", "globalY"),
    window_size: int | tuple[int, int] | None = None,
    spatial_key: str = "spatial",
    sliding_window_key: str = "sliding_window_assignment",
    overlap: int = 0,
    max_n_cells: int = None,
    split_line: str = "h",
    n_splits: int = None,
    drop_partial_windows: bool = False,
    square: bool = False,
    window_size_per_library_key: str = "equal",
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    Divide a tissue slice into regulary shaped spatially contiguous regions (windows).

    Parameters
    ----------
    %(adata)s
    %(library_key)s
    coord_columns: Tuple[str, str]
        Tuple of column names in `adata.obs` that specify the coordinates (x, y), e.i. ('globalX', 'globalY')
    window_size: int | Tuple[str, str]
        Size of the sliding window.
    %(spatial_key)s
    sliding_window_key: str
        Base name for sliding window columns.
    overlap: int
        Overlap size between consecutive windows. (0 = no overlap)
    max_n_cells: int
        If window_size is None, either 'n_split' or 'max_n_cells' can be set.
        max_n_cells sets an upper limit for the number of cells within each region.
    split_line: str
        If 'square' is False, this set's the orientation for rectanglular regions. `h` : Horizontal, `v`: Vertical
    n_splits: int
        This can be used to split the entire region to some splits.
    drop_partial_windows: bool
        If True, drop windows that are smaller than the window size at the borders.
    square: bool
        If True, the windows will be square.
    window_size_per_library_key: str
        If 'equal', the window size will be the same for all libraries. If 'different', the window size will be optimized
        for each library based on the number of cells in the library.
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

    assert max_n_cells is None or n_splits is None, (
        "You can specify only one from the parameters 'n_split' and 'max_n_cells' "
    )
    # we don't want to modify the original adata in case of copy=True
    if copy:
        adata = adata.copy()

    if "sliding_window_assignment_colors" in adata.uns:
        del adata.uns["sliding_window_assignment_colors"]

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

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    if library_key is None:
        library_key = 'temp_fov'
        adata.obs[library_key] = 'fov1'

    libraries = adata.obs[library_key].unique()

    fovs_x_range = [
        (coords[adata.obs[library_key] == key][x_col].max(), coords[adata.obs[library_key] == key][x_col].min())
        for key in libraries
    ]
    fovs_y_range = [
        (coords[adata.obs[library_key] == key][y_col].max(), coords[adata.obs[library_key] == key][y_col].min())
        for key in libraries
    ]
    fovs_width = [i - j for (i, j) in fovs_x_range]
    fovs_height = [i - j for (i, j) in fovs_y_range]
    fovs_n_cell = [adata[adata.obs[library_key] == key].shape[0] for key in libraries]
    fovs_area = [i * j for i, j in zip(fovs_width, fovs_height)]
    fovs_density = [i / j for i, j in zip(fovs_n_cell, fovs_area)]
    window_sizes = []
    
    if window_size is None:
        if window_size_per_library_key == "equal":
            if max_n_cells:
                n_splits = max(2, int(min(fovs_n_cell) / max_n_cells))
                min_n_cells = max(int(.2 * max_n_cells), 1)
            elif n_splits is None:
                n_splits = 2
                max_n_cells = int(min(fovs_n_cell) / n_splits)
                min_n_cells = max(int(.2 * max_n_cells), 1)
            else:
                max_n_cells = int(min(fovs_n_cell) / n_splits)
                min_n_cells = max_n_cells - 1
            
            maximum_region_area = max_n_cells / max(fovs_density)
            minimum_region_area = min_n_cells / max(fovs_density)

            window_size = _optimize_tile_size(
                min(fovs_width), min(fovs_height), minimum_region_area, maximum_region_area, square, split_line
            )
            window_sizes = [window_size] * len(libraries)
        else:
            for i, lib in enumerate(libraries):
                if max_n_cells:
                    n_splits = max(2, int(fovs_n_cell[i] / max_n_cells))
                    min_n_cells = max(int(.2 * max_n_cells), 1)
                elif n_splits is None:
                    n_splits = 2
                    max_n_cells = int(fovs_n_cell[i] / n_splits)
                    min_n_cells = max(int(.2 * max_n_cells), 1)
                else:
                    max_n_cells = int(fovs_n_cell[i]/ n_splits)
                    min_n_cells = max_n_cells - 1

                min_n_cells = int(fovs_n_cell[i] / n_splits)
                minimum_region_area = min_n_cells / max(fovs_density)
                maximum_region_area = fovs_area[i] / fovs_density[i]
                window_sizes.append(
                    _optimize_tile_size(
                        fovs_width[i], fovs_height[i], minimum_region_area, maximum_region_area, square, split_line
                    )
                )
    else:
        # assert split_line is None, logg.warning("'split'  ignored as window_size is specified for square regions")
        assert n_splits is None, logg.warning("'n_split'  ignored as window_size is specified for square regions")
        assert max_n_cells is None, logg.warning("'max_n_cells' ignored as window_size is specified")
        if isinstance(window_size, (int, float)):
            if window_size <= 0:
                raise ValueError("Window size must be larger than 0.")
            else:
                window_size = (window_size, window_size)
        elif isinstance(window_size, tuple):
            for i in window_size:
                if i <= 0:
                    raise ValueError("Window size must be larger than 0.")

        window_sizes = [window_size] * len(libraries)

    # Create a DataFrame to store the sliding window assignments
    sliding_window_df = pd.DataFrame(index=adata.obs.index)
    if sliding_window_key in adata.obs:
        logg.warning(f"Overwriting existing column '{sliding_window_key}' in adata.obs.")
    for i, lib in enumerate(libraries):
        lib_mask = adata.obs[library_key] == lib
        lib_coords = coords.loc[lib_mask]
        # precalculate windows
        windows = _calculate_window_corners(
            fovs_x_range[i],
            fovs_y_range[i],
            window_size=window_sizes[i],
            overlap=overlap,
            drop_partial_windows=drop_partial_windows,
        )
        lib_key = f"{lib}_" if lib is not None else ""

        # assign observations to windows
        for idx, window in windows.iterrows():
            x_start = window["x_start"]
            x_end = window["x_end"]
            y_start = window["y_start"]
            y_end = window["y_end"]

            if drop_partial_windows:
                # Check if the window is within the bounds
                if x_end > fovs_x_range[i][0] or y_end > fovs_y_range[i][0]:
                    continue  # Skip windows that extend beyond the region

            mask = (
                (lib_coords[x_col] >= x_start)
                & (lib_coords[x_col] <= x_end)
                & (lib_coords[y_col] >= y_start)
                & (lib_coords[y_col] <= y_end)
            )
            obs_indices = lib_coords.index[mask]

            if overlap == 0:
                mask = (
                    (lib_coords[x_col] >= x_start)
                    & (lib_coords[x_col] <= x_end)
                    & (lib_coords[y_col] >= y_start)
                    & (lib_coords[y_col] <= y_end)
                )
                obs_indices = lib_coords.index[mask]
                sliding_window_df.loc[obs_indices, sliding_window_key] = f"{lib_key}window_{idx}"
                sliding_window_df.loc[:, sliding_window_key].fillna("out_of_window_0", inplace=True)
            else:
                col_name = f"{sliding_window_key}_{lib_key}window_{idx}"
                sliding_window_df.loc[obs_indices, col_name] = True
                sliding_window_df.loc[:, col_name].fillna(False, inplace=True)

    if overlap == 0:
        # create categorical variable for ordered windows
        # Ensure the column is a string type
        sliding_window_df[sliding_window_key] = pd.Categorical(
            sliding_window_df[sliding_window_key],
            ordered=True,
            categories=sorted(
                sliding_window_df[sliding_window_key].unique(),
                key=lambda x: int(str(x).split("_")[-1]),
            ),
        )

    if copy:
        return sliding_window_df
    sliding_window_df = sliding_window_df.loc[adata.obs.index]
    if 'temp_fov' in adata.obs.columns:
        del(adata.obs['temp_fov'])
    _save_data(adata, attr="obs", key=sliding_window_key, data=sliding_window_df[sliding_window_key])


def _calculate_window_corners(
    x_range: int,
    y_range: int,
    window_size: int = None,
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
    x_window_size, y_window_size = window_size

    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if overlap >= x_window_size or overlap >= y_window_size:
        raise ValueError("Overlap must be less than the window size.")

    max_x, min_x = x_range
    max_y, min_y = y_range

    x_step = x_window_size - overlap
    y_step = y_window_size - overlap

    # Align min_x and min_y to ensure that the first window starts properly
    aligned_min_x = min_x - (min_x % x_window_size) if min_x % x_window_size != 0 else min_x
    aligned_min_y = min_y - (min_y % y_window_size) if min_y % y_window_size != 0 else min_y

    # Generate starting points starting from the aligned minimum values
    x_starts = np.arange(aligned_min_x, max_x, x_step)
    y_starts = np.arange(aligned_min_y, max_y, y_step)

    # Create all combinations of x and y starting points
    starts = list(product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + x_window_size
    windows["y_end"] = windows["y_start"] + y_window_size

    if not drop_partial_windows:
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    else:
        valid_windows = (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        windows = windows[valid_windows]
    windows = windows.reset_index(drop=True)
    return windows[["x_start", "x_end", "y_start", "y_end"]]


def _optimize_tile_size(
        L: int, 
        W: int, 
        A_min: float | None = None, 
        A_max: float | None = None, 
        square: bool = False, 
        split_line: str = "v"
    ) -> tuple:
    """
    This function optimizes the tile size for covering a rectangle of dimensions LxW.
    It returns a tuple (x, y) where x and y are the dimensions of the optimal tile.

    Parameters:
    - L (int): Length of the rectangle.
    - W (int): Width of the rectangle.
    - A_min (int, optional): Minimum allowed area of each tile. If None, no minimum area limit is applied.
    - A_max (int, optional): Maximum allowed area of each tile. If None, no maximum area limit is applied.
    - square (bool, optional): If True, tiles will be square (x = y).

    Returns:
    - tuple: (x, y) representing the optimal tile dimensions.
    """
    best_tile_size = None
    min_uncovered_area = float("inf")
    area = L * W
    if square:
        # Calculate square tiles
        max_side = min(int(math.sqrt(A_max)), int(min(L, W))) if A_max else int(min(L, W))
        min_side = int(math.sqrt(A_min)) if A_min else 1
        # Try all square tile sizes from min_side to max_side
        for side in range(min_side, max_side + 1):
            if (A_min and side * side < A_min) or (A_max and side * side > A_max):
                continue  # Skip sizes that are out of the area limits

            # Calculate number of tiles that fit in the rectangle
            num_tiles_x = max(L // side, 1)
            num_tiles_y = max(W // side, 1)
            uncovered_area = area - (num_tiles_x * num_tiles_y * side * side)

            # Track the best tile size
            if uncovered_area < min_uncovered_area:
                min_uncovered_area = uncovered_area
                best_tile_size = (side, side)
    else:
        # For non-square tiles, optimize both dimensions independently
        if split_line == "v":
            max_tile_length = A_max / W if A_max else int(L)
            max_tile_width = W
            min_tile_length = A_min / W
            min_tile_width = W
        if split_line == "h":
            max_tile_length = L
            max_tile_width = A_max / L if A_max else 0
            min_tile_width = A_min / L
            min_tile_length = L
        # Try all combinations of width and height within the bounds
        for width in range(int(min_tile_width), int(max_tile_width) + 1):
            for height in range(int(min_tile_length), int(max_tile_length) + 1):
                if (A_min and width * height < A_min) or (A_max and width * height > A_max):
                    continue  # Skip sizes that are out of the area limits
                # Calculate number of tiles that fit in the rectangle
                num_tiles_x = max(L // width, 1)
                num_tiles_y = max(W // height, 1)
                uncovered_area = area - (num_tiles_x * num_tiles_y * width * height)
                # Track the best tile size (minimizing uncovered area)
                if uncovered_area < min_uncovered_area:
                    min_uncovered_area = uncovered_area
                    best_tile_size = (height, width)
    return best_tile_size