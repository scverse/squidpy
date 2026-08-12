from __future__ import annotations

import warnings
from itertools import count, product
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from spatialdata import SpatialData

from squidpy._docs import d
from squidpy.gr._utils import _save_data, extract_adata_if_sdata

__all__ = ["sliding_window"]

# Label for cells that fall in no window (only possible for grid ``drop``/``adaptive``); kept as an
# explicit category instead of ``NaN`` so downstream code (and the ordered sort) never sees a float NaN.
UNASSIGNED = "unassigned"


@d.dedent
def sliding_window(
    adata: AnnData | SpatialData,
    library_key: str | None = None,
    window_size: int | None = None,
    overlap: int = 0,
    coord_columns: tuple[str, str] = ("globalX", "globalY"),
    sliding_window_key: str = "sliding_window_assignment",
    spatial_key: str = "spatial",
    drop_partial_windows: bool | None = None,
    copy: bool = False,
    *,
    method: Literal["grid", "split"] = "grid",
    partial_windows: Literal["keep", "drop", "adaptive"] = "keep",
    max_nr_cells: int | None = None,
    table_key: str | None = None,
) -> pd.DataFrame | None:
    """
    Divide a tissue slice into spatially contiguous regions (windows).

    Two tiling strategies are available via ``method``:

    - ``"grid"`` (default) lays a regular grid of ``window_size`` windows (optionally overlapping).
      ``partial_windows`` controls the windows at the tissue edge.
    - ``"split"`` recursively splits the cells into windows of roughly equal cell count
      (at most ``max_nr_cells`` each), ignoring ``window_size``/``overlap``.

    Parameters
    ----------
    %(adata)s
    %(library_key)s
    window_size: int | None
        Size of each grid window (``method="grid"``). Inferred from the extent when ``None``.
    overlap: int
        Overlap between consecutive grid windows (0 = no overlap). Only used for ``method="grid"``.
    coord_columns: tuple[str, str]
        Column names in ``adata.obs`` holding the ``(x, y)`` coordinates, e.g. ``('globalX', 'globalY')``.
    sliding_window_key: str
        Base name for the sliding-window column(s) written to ``.obs``.
    %(spatial_key)s
    drop_partial_windows: bool | None
        Deprecated. Use ``partial_windows`` instead. ``True`` maps to ``partial_windows="drop"``.
    copy: bool
        If ``True``, return the result; otherwise store it in ``adata.obs``.
    method: Literal["grid", "split"]
        Tiling strategy. ``"grid"`` for a regular grid, ``"split"`` for equal-cell-count windows.
    partial_windows: Literal["keep", "drop", "adaptive"]
        Edge-window handling for ``method="grid"`` (ignored for ``"split"``).
        ``"keep"`` clips edge windows to the tissue bounds; ``"drop"`` removes windows that would extend
        past the bounds (their cells become ``"unassigned"``); ``"adaptive"`` shrinks all windows slightly
        so they tile the extent evenly.
    max_nr_cells: int | None
        Maximum number of cells per window. Required for (and only used by) ``method="split"``.
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns the sliding-window annotation(s) as a :class:`pandas.DataFrame`.
    Otherwise, stores the annotation(s) in ``adata.obs`` and returns ``None``.
    """
    # --- deprecation: drop_partial_windows -> partial_windows ---
    if drop_partial_windows is not None:
        warnings.warn(
            "`drop_partial_windows` is deprecated and will be removed in a future release; "
            "use `partial_windows='drop'` (or 'keep') instead.",
            FutureWarning,
            stacklevel=2,
        )
        if partial_windows != "keep":
            raise ValueError("Pass either `drop_partial_windows` (deprecated) or `partial_windows`, not both.")
        partial_windows = "drop" if drop_partial_windows else "keep"

    # --- validate arguments ---
    if method not in ("grid", "split"):
        raise ValueError(f"`method` must be 'grid' or 'split', got {method!r}.")
    if partial_windows not in ("keep", "drop", "adaptive"):
        raise ValueError(f"`partial_windows` must be 'keep', 'drop' or 'adaptive', got {partial_windows!r}.")

    if method == "split":
        if max_nr_cells is None:
            raise ValueError("`max_nr_cells` must be set when method='split'.")
        if max_nr_cells < 1:
            raise ValueError("`max_nr_cells` must be >= 1.")
        if window_size is not None or overlap != 0 or partial_windows != "keep":
            raise ValueError(
                "`window_size`, `overlap` and `partial_windows` are not used with method='split'; leave them unset."
            )
    else:  # grid
        if max_nr_cells is not None:
            raise ValueError("`max_nr_cells` is only used with method='split'.")
        if overlap < 0:
            raise ValueError("Overlap must be non-negative.")

    adata = extract_adata_if_sdata(adata, table_key=table_key)

    # we don't want to modify the original adata in case of copy=True
    if copy:
        adata = adata.copy()

    # --- extract coordinates of observations ---
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

    # --- grid: infer + validate window size ---
    if method == "grid":
        if window_size is None:
            coord_range = max(
                coords[x_col].max() - coords[x_col].min(),
                coords[y_col].max() - coords[y_col].min(),
            )
            # mostly arbitrary choice, except that full integers usually generate windows with 1-2 cells at the borders
            window_size = max(int(np.floor(coord_range // 3.95)), 1)
        if window_size <= 0:
            raise ValueError("Window size must be larger than 0.")
        if overlap >= window_size:
            raise ValueError("Overlap must be less than the window size.")
        if partial_windows == "adaptive" and overlap >= window_size // 2:
            raise ValueError("Overlap must be less than `window_size` // 2 when partial_windows='adaptive'.")

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    libraries = [None] if library_key is None else adata.obs[library_key].unique()

    if sliding_window_key in adata.obs:
        logg.warning(f"Overwriting existing column '{sliding_window_key}' in adata.obs.")

    sliding_window_df = pd.DataFrame(index=adata.obs.index)
    # For overlapping grids we emit one boolean column per window. Collect them all and concatenate once
    # at the end: adding them one-by-one fragments the frame and is quadratic in the number of windows.
    bool_columns: dict[str, pd.Series] = {}

    for lib in libraries:
        lib_coords = coords.loc[adata.obs[library_key] == lib] if lib is not None else coords
        lib_key = f"{lib}_" if lib is not None else ""

        if method == "split":
            # each cell is assigned to exactly one window (non-overlapping by construction)
            labels = _split_cells(lib_coords, coord_columns, max_nr_cells)
            for label in np.unique(labels):
                obs_indices = lib_coords.index[labels == label]
                sliding_window_df.loc[obs_indices, sliding_window_key] = f"{lib_key}window_{label}"
            continue

        min_x, max_x = lib_coords[x_col].min(), lib_coords[x_col].max()
        min_y, max_y = lib_coords[y_col].min(), lib_coords[y_col].max()

        windows = _calculate_window_corners(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            window_size=window_size,
            overlap=overlap,
            partial_windows=partial_windows,
        )

        for idx, window in windows.iterrows():
            mask = _get_window_mask(
                coord_columns=coord_columns,
                lib_coords=lib_coords,
                x_start=window["x_start"],
                x_end=window["x_end"],
                y_start=window["y_start"],
                y_end=window["y_end"],
            )
            obs_indices = lib_coords.index[mask]
            if overlap == 0:
                sliding_window_df.loc[obs_indices, sliding_window_key] = f"{lib_key}window_{idx}"
            else:
                col_name = f"{sliding_window_key}_{lib_key}window_{idx}"
                col = bool_columns.setdefault(col_name, pd.Series(False, index=sliding_window_df.index))
                col.loc[obs_indices] = True

    if bool_columns:
        sliding_window_df = pd.concat([sliding_window_df, pd.DataFrame(bool_columns)], axis=1)

    if method == "split" or overlap == 0:
        # single categorical column: order windows by their trailing index, put unassigned cells last
        sliding_window_df[sliding_window_key] = _ordered_window_categorical(sliding_window_df[sliding_window_key])

    if copy:
        return sliding_window_df
    for col_name, col_data in sliding_window_df.items():
        _save_data(adata, attr="obs", key=col_name, data=col_data)
    return None


def _ordered_window_categorical(values: pd.Series) -> pd.Categorical:
    """Ordered categorical of window labels sorted by trailing index; unassigned cells (``NaN``) go last.

    Cells outside every window (grid ``drop``/``adaptive``) arrive as ``NaN``; they become an explicit
    ``"unassigned"`` category so the ordered sort never calls ``int(...)`` on a float ``NaN``.
    """
    filled = values.fillna(UNASSIGNED)
    present = list(pd.unique(filled))
    windows = sorted((c for c in present if c != UNASSIGNED), key=lambda s: int(str(s).split("_")[-1]))
    categories = windows + ([UNASSIGNED] if UNASSIGNED in present else [])
    return pd.Categorical(filled, ordered=True, categories=categories)


def _get_window_mask(
    coord_columns: tuple[str, str],
    lib_coords: pd.DataFrame,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
) -> pd.Series:
    """Boolean mask selecting the rows of ``lib_coords`` inside the (inclusive) window."""
    x_col, y_col = coord_columns
    return (
        (lib_coords[x_col] >= x_start)
        & (lib_coords[x_col] <= x_end)
        & (lib_coords[y_col] >= y_start)
        & (lib_coords[y_col] <= y_end)
    )


def _split_cells(coords: pd.DataFrame, coord_columns: tuple[str, str], max_cells: int) -> np.ndarray:
    """Assign each cell to a window by recursive count-based (median) splitting.

    Each window holds at most ``max_cells`` cells and, unless the whole input is smaller, at least
    ``max_cells // 2``. The split is on cell *position* (the median index of the longer axis), so windows
    are **non-overlapping by construction** — no cell can land in two windows, and every split strictly
    shrinks both halves, so it always terminates (given ``max_cells >= 1``).

    Parameters
    ----------
    coords
        Coordinates for one library (index-aligned to the cells).
    coord_columns
        ``(x_col, y_col)`` column names in ``coords``.
    max_cells
        Maximum number of cells per window.

    Returns
    -------
    Integer window label per row of ``coords`` (positional order).
    """
    x_col, y_col = coord_columns
    x = coords[x_col].to_numpy()
    y = coords[y_col].to_numpy()
    labels = np.empty(len(coords), dtype=int)
    counter = count()

    def recurse(idx: np.ndarray) -> None:
        if len(idx) <= max_cells:
            labels[idx] = next(counter)
            return
        xi, yi = x[idx], y[idx]
        # split along the axis with the larger spatial extent, at the median cell
        if (xi.max() - xi.min()) >= (yi.max() - yi.min()):
            order = idx[np.argsort(xi, kind="stable")]
        else:
            order = idx[np.argsort(yi, kind="stable")]
        mid = len(order) // 2
        recurse(order[:mid])
        recurse(order[mid:])

    recurse(np.arange(len(coords), dtype=int))
    return labels


def _calculate_window_corners(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    window_size: int,
    overlap: int = 0,
    partial_windows: Literal["keep", "drop", "adaptive"] = "keep",
) -> pd.DataFrame:
    """
    Corner points of a regular grid of windows covering ``[min_x, max_x] x [min_y, max_y]``.

    Parameters
    ----------
    min_x, max_x, min_y, max_y
        Extent to tile.
    window_size
        Size of each window.
    overlap
        Overlap between consecutive windows (must be less than ``window_size``).
    partial_windows
        Edge handling: ``"keep"`` clips edge windows to the bounds; ``"drop"`` removes windows that would
        extend past the bounds; ``"adaptive"`` shrinks all windows slightly to tile the extent evenly.

    Returns
    -------
    DataFrame with columns ``['x_start', 'x_end', 'y_start', 'y_end']``.
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if overlap >= window_size:
        raise ValueError("Overlap must be less than the window size.")

    if partial_windows == "adaptive":
        total_width = max_x - min_x
        total_height = max_y - min_y
        # number of windows per axis; clamp to >= 1 so a library smaller than one window (e.g. span
        # <= overlap, common when the global window_size is set from a larger library) yields a single
        # window instead of dividing by zero.
        number_x_windows = max(int(np.ceil((total_width - overlap) / (window_size - overlap))), 1)
        number_y_windows = max(int(np.ceil((total_height - overlap) / (window_size - overlap))), 1)
        # window size per axis (integer to avoid float drift)
        x_window_size = np.ceil((total_width + (number_x_windows - 1) * overlap) / number_x_windows)
        y_window_size = np.ceil((total_height + (number_y_windows - 1) * overlap) / number_y_windows)
    else:
        x_window_size = window_size
        y_window_size = window_size

    x_step = x_window_size - overlap
    y_step = y_window_size - overlap

    # Generate starting points. A non-positive step means one window already covers the whole span
    # (span <= overlap) -> emit a single window at the minimum rather than an empty grid.
    x_starts = np.arange(min_x, max_x, x_step) if x_step > 0 else np.array([min_x])
    y_starts = np.arange(min_y, max_y, y_step) if y_step > 0 else np.array([min_y])

    # Create all combinations of x and y starting points
    starts = list(product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + x_window_size
    windows["y_end"] = windows["y_start"] + y_window_size

    if partial_windows == "keep":
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    elif partial_windows == "adaptive":
        # the integer window size can exceed max_x/max_y -> clip, then drop degenerate corner slivers.
        # Only drop a thin window when its axis has neighbours (>1 window): a sole window covering a
        # small library is thin but not redundant, and must be kept.
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
        thin_x = (windows["x_end"] - windows["x_start"]) <= overlap
        thin_y = (windows["y_end"] - windows["y_start"]) <= overlap
        # a thin window is a redundant sliver only if its axis actually has more than one window;
        # a lone window covering a small library is thin but must be kept.
        redundant_windows = (thin_x & (len(x_starts) > 1)) | (thin_y & (len(y_starts) > 1))
        windows = windows[~redundant_windows]
    elif partial_windows == "drop":
        valid_windows = (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        windows = windows[valid_windows]
    else:
        raise ValueError(f"{partial_windows} is not a valid `partial_windows` argument.")

    windows = windows.reset_index(drop=True)
    return windows[["x_start", "x_end", "y_start", "y_end"]]
