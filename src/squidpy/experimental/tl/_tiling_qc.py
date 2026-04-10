"""QC metrics for detecting tile-boundary segmentation artifacts.

Cells cut by tile borders during segmentation have characteristic
straight edges that natural cell boundaries never produce.  This module
computes per-cell metrics that quantify this artifact:

- **max_straight_edge_ratio**: length of the longest straight contour
  segment normalised by the cell's equivalent diameter.
- **cardinal_alignment_score**: how closely that segment aligns with
  0° or 90° (axis-aligned tile borders).
- **cut_score**: product of the two, combining evidence from shape and
  orientation.

All heavy computation is done per-tile via the tiling infrastructure
in :mod:`squidpy.experimental.im._tiling`, so this scales to
100k x 100k images without materialising the full array.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import anndata as ad
import dask
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import njit
from skimage.measure import find_contours, regionprops
from spatialdata._logging import logger as logg
from spatialdata.models import TableModel

if TYPE_CHECKING:
    from dask.distributed import Client

from squidpy._utils import cpu_count
from squidpy.experimental.im._tiling import (
    build_tile_specs,
    compute_cell_info,
    compute_cell_info_multiscale,
    compute_cell_info_tiled,
    extract_labels_tile_lazy,
)

__all__ = ["calculate_tiling_qc"]

# Minimum cell area in pixels — smaller cells produce noisy contours
_MIN_CELL_AREA = 20

# Default perpendicular distance tolerance for collinearity (pixels).
# Points within this distance of the start→end line are considered
# part of the same straight segment.  0.75 px works well for
# sub-pixel contours from marching squares.
_DEFAULT_DISTANCE_TOL = 0.75

# Maximum contour points to analyse.  Longer contours are resampled
# to this length via equidistant arc-length interpolation to bound
# the O(n²) two-pointer scan.
_MAX_CONTOUR_POINTS = 500

_SCORE_COLUMNS = ["max_straight_edge_ratio", "cardinal_alignment_score", "cut_score"]
_NAN_SCORES = {col: np.nan for col in _SCORE_COLUMNS}


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _collinear_scan(
    contour: np.ndarray,
    cum_arc: np.ndarray,
    total_arc: float,
    distance_tol: float,
) -> tuple[float, float]:
    """Numba-accelerated two-pointer collinearity scan.

    For each start index, extends the end index as long as all
    intermediate points stay within ``distance_tol`` of the
    start→end line.  Returns ``(best_length, best_angle)``.
    """
    n = contour.shape[0]
    best_len = 0.0
    best_angle = 0.0

    for start in range(n - 2):
        remaining_arc = total_arc - cum_arc[start]
        if remaining_arc <= best_len:
            break

        for end in range(start + 2, n):
            d0 = contour[end, 0] - contour[start, 0]
            d1 = contour[end, 1] - contour[start, 1]
            seg_len = math.sqrt(d0 * d0 + d1 * d1)
            if seg_len < 1e-12:
                continue

            max_perp = 0.0
            for k in range(start + 1, end):
                r0 = contour[k, 0] - contour[start, 0]
                r1 = contour[k, 1] - contour[start, 1]
                perp = abs(d0 * r1 - d1 * r0) / seg_len
                if perp > max_perp:
                    max_perp = perp
                if perp > distance_tol:
                    break

            if max_perp > distance_tol:
                break

            if seg_len > best_len:
                best_len = seg_len
                best_angle = math.atan2(d0, d1)

    return best_len, best_angle


def _resample_contour(contour: np.ndarray, max_points: int) -> np.ndarray:
    """Resample a contour to at most *max_points* via arc-length interpolation.

    Fully vectorised using :func:`numpy.searchsorted` — no Python
    loops.  Preserves geometry far better than naive stride-based
    subsampling because points are placed equidistantly along the
    contour arc.
    """
    n = len(contour)
    if n <= max_points:
        return contour

    diffs = np.diff(contour, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cum_arc = np.empty(n, dtype=np.float64)
    cum_arc[0] = 0.0
    cum_arc[1:] = np.cumsum(seg_lengths)
    total = cum_arc[-1]

    if total < 1e-12:
        return contour[:max_points]

    targets = np.linspace(0.0, total, max_points)

    idx = np.searchsorted(cum_arc, targets, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    seg = cum_arc[idx + 1] - cum_arc[idx]
    safe_seg = np.where(seg < 1e-12, 1.0, seg)
    frac = np.where(seg < 1e-12, 0.0, (targets - cum_arc[idx]) / safe_seg)

    return contour[idx] + frac[:, np.newaxis] * (contour[idx + 1] - contour[idx])


def _longest_collinear_segment(
    contour: np.ndarray,
    distance_tol: float = _DEFAULT_DISTANCE_TOL,
) -> tuple[float, float]:
    """Find the longest collinear run of contour points.

    Uses a numba-compiled two-pointer scan with three contour
    rotations to handle the closure point.  Long contours are
    resampled to at most :data:`_MAX_CONTOUR_POINTS` via arc-length
    interpolation to bound worst-case runtime.

    Parameters
    ----------
    contour
        ``(N, 2)`` array of ``(row, col)`` contour coordinates.
    distance_tol
        Maximum perpendicular distance (pixels) from the start→end
        line for a point to be considered part of the straight segment.

    Returns
    -------
    run_length
        Euclidean length of the longest straight segment (pixels).
    run_angle
        Angle (radians, ``[-π, π]``) of that segment.
    """
    n = len(contour)
    if n < 3:
        return 0.0, 0.0

    pts = np.asarray(contour, dtype=np.float64)
    pts = _resample_contour(pts, _MAX_CONTOUR_POINTS)
    n = len(pts)

    # find_contours returns closed contours (first ≈ last point)
    closed = np.sqrt(((pts[0] - pts[-1]) ** 2).sum()) < 1.0

    # For closed contours, drop the duplicate last point and precompute
    # segment lengths once — rotations reuse the same distances.
    if closed and n > 6:
        core = pts[:-1]
        core_diffs = np.diff(core, axis=0)
        core_seg_lens = np.sqrt((core_diffs**2).sum(axis=1))
        rotations = [0, len(core) // 3, 2 * len(core) // 3]
    else:
        core = pts
        core_diffs = np.diff(core, axis=0)
        core_seg_lens = np.sqrt((core_diffs**2).sum(axis=1))
        rotations = [0]

    best_len = 0.0
    best_angle = 0.0

    # Scan at multiple rotations so straight segments crossing the
    # closure point are not split.
    for shift in rotations:
        if shift == 0:
            rotated = core
            sl = core_seg_lens
        else:
            rotated = np.roll(core, -shift, axis=0)
            sl = np.roll(core_seg_lens, -shift)

        cum_arc = np.empty(len(rotated), dtype=np.float64)
        cum_arc[0] = 0.0
        cum_arc[1:] = np.cumsum(sl)

        length, angle = _collinear_scan(rotated, cum_arc, cum_arc[-1], distance_tol)
        if length > best_len:
            best_len = length
            best_angle = angle

    return best_len, best_angle


def _cardinal_alignment(angle: float) -> float:
    """Score how close an angle is to a cardinal direction (0° or 90°).

    Returns a value in ``[0, 1]`` where 1 means perfectly axis-aligned
    and 0 means maximally diagonal (45°).
    """
    a = abs(angle) % np.pi
    dist = min(a, abs(a - np.pi / 2), abs(a - np.pi))

    # Map [0, π/4] → [1, 0]
    return float(1.0 - dist / (np.pi / 4))


def _straight_edge_metrics(
    contour: np.ndarray,
    cell_area: float,
    distance_tol: float = _DEFAULT_DISTANCE_TOL,
) -> tuple[float, float, float]:
    """Compute straight-edge metrics for a single cell contour.

    Parameters
    ----------
    contour
        ``(N, 2)`` contour coordinates from :func:`skimage.measure.find_contours`.
    cell_area
        Area of the cell in pixels (for normalisation).
    distance_tol
        Perpendicular distance tolerance for collinearity (pixels).

    Returns
    -------
    straight_edge_ratio
        Longest collinear segment / equivalent diameter.
    cardinal_score
        Cardinal alignment of the longest straight segment.
    cut_score
        Product of the two.
    """
    eq_diam = np.sqrt(4 * cell_area / np.pi)
    if eq_diam == 0:
        return 0.0, 0.0, 0.0

    run_length, run_angle = _longest_collinear_segment(contour, distance_tol)
    straight_ratio = run_length / eq_diam
    cardinal = _cardinal_alignment(run_angle)
    cut_score = straight_ratio * cardinal

    return float(straight_ratio), float(cardinal), float(cut_score)


# ---------------------------------------------------------------------------
# Per-tile scoring
# ---------------------------------------------------------------------------


def _score_tile(
    tile_labels: np.ndarray,
    distance_tol: float = _DEFAULT_DISTANCE_TOL,
    min_area: int = _MIN_CELL_AREA,
    downsample: int = 1,
) -> pd.DataFrame:
    """Compute tiling QC metrics for all cells in a numpy label tile.

    Parameters
    ----------
    tile_labels
        ``(H, W)`` label array (background = 0, owned cells only).
    distance_tol
        Perpendicular distance tolerance for collinearity (pixels).
    min_area
        Cells smaller than this (in pixels at analysis resolution)
        are skipped and get NaN values.
    downsample
        Factor by which to downsample each cell's bounding-box crop
        before contour extraction.  ``1`` = full resolution, ``2`` =
        half, etc.  Straight edges are scale-invariant so moderate
        downsampling (2–4x) is safe and much faster for large cells.

    Returns
    -------
    DataFrame with columns ``max_straight_edge_ratio``,
    ``cardinal_alignment_score``, ``cut_score``, indexed by cell label.
    """
    regions = regionprops(tile_labels)
    if not regions:
        return pd.DataFrame(columns=_SCORE_COLUMNS, dtype=float)

    rows: dict[int, dict[str, float]] = {}

    for region in regions:
        lid = region.label
        area = region.area

        if area < min_area * (downsample**2):
            rows[lid] = dict(_NAN_SCORES)
            continue

        # Pad with 1px of zeros so find_contours can trace cells
        # that touch the crop edge (e.g., cells filling their bbox).
        min_row, min_col, max_row, max_col = region.bbox
        crop = (tile_labels[min_row:max_row, min_col:max_col] == lid).astype(np.float32)
        crop = np.pad(crop, 1, mode="constant", constant_values=0)

        if downsample > 1:
            crop = crop[::downsample, ::downsample]

        contours = find_contours(crop, 0.5)
        if not contours:
            rows[lid] = dict(_NAN_SCORES)
            continue

        contour = max(contours, key=len)
        analysis_area = area / (downsample**2) if downsample > 1 else area
        ser, cas, cs = _straight_edge_metrics(contour, analysis_area, distance_tol)

        rows[lid] = {
            "max_straight_edge_ratio": ser,
            "cardinal_alignment_score": cas,
            "cut_score": cs,
        }

    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Centroid computation (shared logic with _feature.py)
# ---------------------------------------------------------------------------


def _compute_centroids_for_labels(
    sdata: sd.SpatialData,
    labels_key: str,
    labels_da: xr.DataArray,
    scale: str | None,
) -> dict:
    """Compute cell centroids using the most efficient strategy available."""
    if isinstance(sdata.labels[labels_key], xr.DataTree):
        logg.info("Computing centroids from coarse scale.")
        return compute_cell_info_multiscale(sdata.labels[labels_key], target_scale=scale or "scale0")

    n_pixels = labels_da.sizes.get("y", 1) * labels_da.sizes.get("x", 1)
    if n_pixels <= 4096 * 4096:
        lbl_np = labels_da.values
        if lbl_np.ndim > 2:
            lbl_np = lbl_np.squeeze()
        return compute_cell_info(lbl_np)

    logg.info("Computing centroids in tiled mode (large single-scale labels).")
    return compute_cell_info_tiled(labels_da)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_METHOD_KEY = "tiling_qc"


def calculate_tiling_qc(
    sdata: sd.SpatialData,
    labels_key: str,
    scale: str | None = None,
    tile_size: int = 2048,
    overlap_margin: int | Literal["auto"] = "auto",
    distance_tol: float = _DEFAULT_DISTANCE_TOL,
    min_area: int = _MIN_CELL_AREA,
    downsample: int = 1,
    n_jobs: int = -1,
    client: Client | None = None,
    adata_key_added: str | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Score cells for tile-boundary segmentation artifacts.

    Computes per-cell metrics that detect artificially straight edges
    caused by tiled segmentation.  Large images are processed via the
    same tiling infrastructure as
    :func:`~squidpy.experimental.im.calculate_image_features`.

    Results are stored in a QC table (default
    ``sdata.tables["{labels_key}_qc"]``).  Scores live in ``.obs``;
    the ``.X`` matrix is empty.  Algorithm parameters are recorded in
    ``.uns["tiling_qc"]``.

    Parameters
    ----------
    sdata
        SpatialData object.
    labels_key
        Key in ``sdata.labels`` with segmentation masks.
    scale
        Scale level for multi-scale labels.
    tile_size
        Side length of the tiling grid (pixels).
    overlap_margin
        Overlap around each tile.  ``"auto"`` computes the minimum from
        the largest cell's bounding box.
    distance_tol
        Maximum perpendicular distance (pixels) from the fitted line
        for a contour point to be considered part of a straight
        segment.  Default 0.75 px.
    min_area
        Cells smaller than this (pixels) are skipped (NaN scores).
    downsample
        Factor by which to downsample each cell's bounding-box crop
        before contour extraction.  Straightness is scale-invariant,
        so ``2``--``4`` is safe and much faster on large cells.
    n_jobs
        Number of threads for tile processing.  ``-1`` (default) uses
        all available CPUs.  Ignored when ``client`` is provided.
    client
        A :class:`dask.distributed.Client` for distributed execution.
        When provided, tile processing is submitted to this client,
        ``n_jobs`` is ignored, and progress is reported via the dask
        dashboard.  Workers must have access to the underlying data
        store (e.g. shared filesystem or cloud storage for zarr).
    adata_key_added
        Key under which to store the result in ``sdata.tables``.
        Defaults to ``"{labels_key}_qc"``.
    inplace
        If ``True``, store result in ``sdata.tables``.  Otherwise
        return the AnnData directly.

    Returns
    -------
    :class:`~anndata.AnnData` when ``inplace=False``, otherwise ``None``.
    The AnnData ``.obs`` contains three scores per cell:

    - ``max_straight_edge_ratio``: longest collinear boundary segment /
      equivalent diameter.
    - ``cardinal_alignment_score``: axis-alignment of that segment
      (1 = cardinal, 0 = diagonal).
    - ``cut_score``: product of the two.

    Notes
    -----
    Tile processing is parallelised via :func:`dask.compute`.  By
    default a threaded scheduler with ``n_jobs`` workers is used.
    Pass a :class:`~dask.distributed.Client` to use a distributed
    cluster instead.
    """
    if labels_key not in sdata.labels:
        raise ValueError(f"Labels key '{labels_key}' not found, valid keys: {list(sdata.labels.keys())}")

    labels_node = sdata.labels[labels_key]
    if isinstance(labels_node, xr.DataTree):
        if scale is None:
            raise ValueError("When using multi-scale labels, please specify the scale.")
        labels_da = labels_node[scale].ds["image"]
    else:
        labels_da = labels_node

    cell_info = _compute_centroids_for_labels(sdata, labels_key, labels_da, scale)
    if not cell_info:
        raise ValueError("No cells found in labels (all zeros).")

    H = int(labels_da.sizes.get("y", labels_da.shape[-2]))
    W = int(labels_da.sizes.get("x", labels_da.shape[-1]))

    specs = build_tile_specs((H, W), cell_info, tile_size=tile_size, overlap_margin=overlap_margin)
    logg.info(
        f"Tiling QC: {len(specs)} tiles ({tile_size}x{tile_size}, margin={overlap_margin}, downsample={downsample}x)."
    )

    @dask.delayed
    def _process_one(spec):
        tile_lbl = extract_labels_tile_lazy(labels_da, spec)
        return _score_tile(tile_lbl, distance_tol=distance_tol, min_area=min_area, downsample=downsample)

    tasks = [_process_one(spec) for spec in specs]

    if client is not None:
        if n_jobs != -1:
            logg.warning("`n_jobs` is ignored when a `client` is provided. Parallelism is controlled by the client.")
        results = dask.compute(*tasks, scheduler=client)
    else:
        num_workers = cpu_count() if n_jobs == -1 else n_jobs
        with ProgressBar():
            results = dask.compute(*tasks, scheduler="threads", num_workers=num_workers)

    tile_dfs = [df for df in results if not df.empty]

    if not tile_dfs:
        raise ValueError("No cells scored — labels may be empty or all below min_area.")

    combined = pd.concat(tile_dfs, axis=0).sort_index()

    if combined.index.duplicated().any():
        dups = combined.index[combined.index.duplicated()].unique().tolist()
        raise RuntimeError(f"Duplicate cell IDs across tiles — tile ownership may be broken. Duplicates: {dups}")

    n_cells = len(combined)
    adata = ad.AnnData(
        X=np.empty((n_cells, 0), dtype=np.float32),
    )
    adata.obs_names = [f"cell_{i}" for i in combined.index]

    adata.obs["region"] = pd.Categorical([labels_key] * n_cells)
    adata.obs["label_id"] = combined.index.values
    adata.uns["spatialdata_attrs"] = {
        "region": labels_key,
        "region_key": "region",
        "instance_key": "label_id",
    }

    for col in combined.columns:
        adata.obs[col] = combined[col].values

    adata.obs["centroid_y"] = np.array([cell_info[lid].centroid_y for lid in combined.index])
    adata.obs["centroid_x"] = np.array([cell_info[lid].centroid_x for lid in combined.index])

    adata.uns[_METHOD_KEY] = {
        "scale": scale,
        "tile_size": tile_size,
        "overlap_margin": overlap_margin,
        "distance_tol": distance_tol,
        "min_area": min_area,
        "downsample": downsample,
    }

    if inplace:
        table_key = adata_key_added if adata_key_added is not None else f"{labels_key}_qc"
        sdata.tables[table_key] = TableModel.parse(adata)
        return None
    return adata
