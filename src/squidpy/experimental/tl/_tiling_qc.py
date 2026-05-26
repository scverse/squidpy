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
- **smoothed_cut_score**: cut_score multiplied by the mean cut_score of
  the ``n_neighbors`` nearest spatial neighbors - amplifies boundary
  cells while suppressing isolated high-scorers.
- **is_outlier**: boolean flag gated on per-cell cut_score and/or
  spatially smoothed score exceeding their respective MAD thresholds.
- **nhood_outlier_fraction**: fraction of ``n_neighbors`` nearest
  neighbors that are smoothed-score outliers (MAD-based).  Bounded
  [0, 1]; high values precisely trace the FOV tile grid.

All heavy computation is done per-tile via the tiling infrastructure
in :mod:`squidpy.experimental.im._tiling`, so this scales to
100k x 100k images without materialising the full array.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

import anndata as ad
import dask
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import njit
from skimage.measure import find_contours, regionprops
from sklearn.neighbors import BallTree
from spatialdata._logging import logger as logg
from spatialdata.models import TableModel

from squidpy._utils import _get_n_cores
from squidpy.experimental.tl._tiling_stitch import _STITCH_COLUMNS, _STITCH_PARAM_KEYS
from squidpy.experimental.im._tiling import (
    build_tile_specs,
    compute_cell_info,
    compute_cell_info_multiscale,
    compute_cell_info_tiled,
    extract_labels_tile_lazy,
)

__all__ = ["TilingQCParams", "calculate_tiling_qc"]


@dataclass(slots=True)
class TilingQCParams:
    """Advanced tuning knobs for :func:`calculate_tiling_qc`.

    Pass an instance (or a ``Mapping`` of field names to values) as
    ``tiling_qc_params`` to override.  All fields default to the values
    below.

    Attributes
    ----------
    distance_tol
        Maximum perpendicular distance (pixels) from the fitted line for a
        contour point to be considered part of a straight segment.
    min_area
        Cells smaller than this (pixels at analysis resolution) are skipped
        and assigned NaN scores.
    max_contour_points
        Cap on contour resolution; longer contours are resampled to this
        length via arc-length interpolation before the O(n^2) two-pointer
        scan, bounding worst-case runtime on very large cells.
    """

    distance_tol: float = 0.75
    min_area: int = 20
    max_contour_points: int = 500


def _resolve_qc_params(qc_params: TilingQCParams | Mapping[str, Any] | None) -> TilingQCParams:
    """Normalise the ``tiling_qc_params`` argument to a :class:`TilingQCParams` instance."""
    if qc_params is None:
        return TilingQCParams()
    if isinstance(qc_params, TilingQCParams):
        return qc_params
    if isinstance(qc_params, Mapping):
        valid = {f.name for f in fields(TilingQCParams)}
        unknown = set(qc_params) - valid
        if unknown:
            raise ValueError(
                f"Unknown tiling_qc_params field(s): {sorted(unknown)}; expected from {sorted(valid)}."
            )
        return TilingQCParams(**qc_params)
    raise TypeError(
        f"tiling_qc_params must be TilingQCParams, Mapping, or None; got {type(qc_params).__name__}."
    )


_QC_DEFAULTS = TilingQCParams()

# Standard consistency factor sd ~ 1.4826 x MAD for normal distributions.
_MAD_TO_SD = 1.4826

_TILE_SCORE_COLUMNS = ["max_straight_edge_ratio", "cardinal_alignment_score", "cut_score"]
_POST_SCORE_COLUMNS = ["smoothed_cut_score", "is_outlier", "nhood_outlier_fraction"]
_SCORE_COLUMNS = _TILE_SCORE_COLUMNS + _POST_SCORE_COLUMNS
_NAN_TILE_SCORES = dict.fromkeys(_TILE_SCORE_COLUMNS, np.nan)


def _has_distributed_client() -> bool:
    """Return True iff a ``dask.distributed.Client`` is active in this process.

    Mirrors the public dask idiom: if a Client is in scope, ``dask.compute``
    will pick it up automatically — we only need to know whether to fall
    back to the local threaded scheduler.
    """
    try:
        # ImportError guards against partial dask installs without the distributed extra;
        # ValueError is what get_client() raises when no Client is currently active.
        from dask.distributed import get_client

        get_client()
    except (ImportError, ValueError):
        return False
    return True


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

    Fully vectorised using :func:`numpy.searchsorted` - no Python
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
    distance_tol: float = _QC_DEFAULTS.distance_tol,
    max_contour_points: int = _QC_DEFAULTS.max_contour_points,
) -> tuple[float, float]:
    """Find the longest collinear run of contour points.

    Uses a numba-compiled two-pointer scan with three contour
    rotations to handle the closure point.  Long contours are
    resampled to at most ``max_contour_points`` via arc-length
    interpolation to bound worst-case runtime.

    Parameters
    ----------
    contour
        ``(N, 2)`` array of ``(row, col)`` contour coordinates.
    distance_tol
        Maximum perpendicular distance (pixels) from the start→end
        line for a point to be considered part of the straight segment.
    max_contour_points
        Cap on contour resolution; longer contours are resampled to
        this length before the collinearity scan.

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
    pts = _resample_contour(pts, max_contour_points)
    n = len(pts)

    # find_contours returns closed contours (first ≈ last point)
    closed = np.sqrt(((pts[0] - pts[-1]) ** 2).sum()) < 1.0

    # For closed contours, drop the duplicate last point and precompute
    # segment lengths once - rotations reuse the same distances.
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
    distance_tol: float = _QC_DEFAULTS.distance_tol,
    max_contour_points: int = _QC_DEFAULTS.max_contour_points,
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

    run_length, run_angle = _longest_collinear_segment(contour, distance_tol, max_contour_points)
    straight_ratio = run_length / eq_diam
    cardinal = _cardinal_alignment(run_angle)
    cut_score = straight_ratio * cardinal

    return float(straight_ratio), float(cardinal), float(cut_score)


# ---------------------------------------------------------------------------
# Per-tile scoring
# ---------------------------------------------------------------------------


def _score_tile(
    tile_labels: np.ndarray,
    distance_tol: float = _QC_DEFAULTS.distance_tol,
    min_area: int = _QC_DEFAULTS.min_area,
    downsample: int = 1,
    max_contour_points: int = _QC_DEFAULTS.max_contour_points,
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
        return pd.DataFrame(columns=_TILE_SCORE_COLUMNS, dtype=float)

    rows: dict[int, dict[str, float]] = {}

    for region in regions:
        lid = region.label
        area = region.area

        if area < min_area * (downsample**2):
            rows[lid] = dict(_NAN_TILE_SCORES)
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
            rows[lid] = dict(_NAN_TILE_SCORES)
            continue

        contour = max(contours, key=len)
        analysis_area = area / (downsample**2) if downsample > 1 else area
        ser, cas, cs = _straight_edge_metrics(contour, analysis_area, distance_tol, max_contour_points)

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
    downsample: int = 1,
    outlier_use_cut: bool = True,
    outlier_use_smoothed: bool = True,
    nmads_cut: float = 1.5,
    nmads_smoothed: float = 3,
    n_neighbors: int = 10,
    tiling_qc_params: TilingQCParams | Mapping[str, Any] | None = None,
    n_jobs: int = -1,
    table_key_added: str | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Score cells for tile-boundary segmentation artifacts.

    Computes per-cell metrics that detect artificially straight edges
    caused by tiled segmentation.  Large images are processed via the
    cell-aware tiling infrastructure in
    ``squidpy.experimental.im._tiling``.

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
    downsample
        Factor by which to downsample each cell's bounding-box crop
        before contour extraction.  Straightness is scale-invariant,
        so ``2``--``4`` is safe and much faster on large cells.
    outlier_use_cut
        Gate ``is_outlier`` on the per-cell ``cut_score`` exceeding
        its own MAD threshold.  Requires the cell itself to have a
        straight cardinal-aligned edge.
    outlier_use_smoothed
        Gate ``is_outlier`` on the spatially smoothed score
        (``smoothed_cut_score``) exceeding its MAD threshold.
        Requires the cell to be in a spatial cluster of high-scorers.
    nmads_cut
        Number of MADs for the ``cut_score`` outlier gate.
        Threshold is ``median + nmads_cut x MAD x 1.4826``.
    nmads_smoothed
        Number of MADs for the ``smoothed_cut_score`` outlier gate.
        Threshold is ``median + nmads_smoothed x MAD x 1.4826``.
    n_neighbors
        Number of nearest spatial neighbors used to compute
        ``smoothed_cut_score`` and ``nhood_outlier_fraction``.  In a
        perfect grid each cell has 8 immediate neighbours; the default
        of 10 leaves a little wiggle room for biological irregularity
        without wasting compute on distant cells.
    tiling_qc_params
        Advanced tuning knobs as a :class:`TilingQCParams` instance or
        a ``Mapping`` of its field names to values.  See
        :class:`TilingQCParams` for each field's meaning and default.
        ``None`` (default) uses all defaults.
    n_jobs
        Number of threads for tile processing.  ``-1`` (default) uses
        all available CPUs.  Ignored when an active
        ``dask.distributed.Client`` is in scope (the client's own
        worker pool is used instead).
    table_key_added
        Key under which to store the result in ``sdata.tables``.
        Defaults to ``"{labels_key}_qc"``.
    inplace
        If ``True``, store result in ``sdata.tables``.  Otherwise
        return the AnnData directly.

    Returns
    -------
    :class:`~anndata.AnnData` when ``inplace=False``, otherwise ``None``.
    The AnnData ``.obs`` contains five scores per cell:

    - ``max_straight_edge_ratio``: longest collinear boundary segment /
      equivalent diameter.
    - ``cardinal_alignment_score``: axis-alignment of that segment
      (1 = cardinal, 0 = diagonal).
    - ``cut_score``: product of the two.
    - ``smoothed_cut_score``: ``cut_score x mean(neighbor cut_scores)``
      over the ``n_neighbors`` nearest spatial neighbors.  Amplifies
      cells on FOV boundaries while suppressing isolated high-scorers.
    - ``is_outlier``: boolean, ``True`` when the enabled outlier
      gates are satisfied (``cut_score`` and/or ``smoothed_cut_score``
      exceeding their respective MAD thresholds).
    - ``nhood_outlier_fraction``: fraction of ``n_neighbors`` nearest
      neighbors that are smoothed-score outliers (MAD-based).  Bounded
      [0, 1]; high values trace the tile grid.

    Notes
    -----
    Tile processing is parallelised via :func:`dask.compute`.  When an
    active ``dask.distributed.Client`` is in scope it is picked up
    automatically and used for execution; otherwise a local threaded
    scheduler with ``n_jobs`` workers is used.

    If you invoke this function from inside a dask worker task (e.g.,
    via ``client.submit(calculate_tiling_qc, ...)``), wrap the call in
    ``distributed.secede`` / ``distributed.rejoin`` to release the
    worker slot before the inner tile tasks are submitted; without
    that, the cluster can deadlock when all workers are busy holding
    the outer job.

    Re-running ``calculate_tiling_qc`` on a labels element whose QC
    table already carries ``stitch_*`` columns from a previous run of
    :func:`stitch_tile_cuts` produces a fresh AnnData without those
    columns.  A warning is logged (in both ``inplace=True`` and
    ``inplace=False`` modes) listing the previous stitch parameters
    and a copy-pasteable invocation so they can be re-derived from
    the new outlier set.
    """
    if labels_key not in sdata.labels:
        raise ValueError(f"Labels key '{labels_key}' not found, valid keys: {list(sdata.labels.keys())}")
    if not outlier_use_cut and not outlier_use_smoothed:
        raise ValueError("At least one outlier gate must be enabled (outlier_use_cut or outlier_use_smoothed).")
    if outlier_use_cut and nmads_cut <= 0:
        raise ValueError(f"nmads_cut must be positive, got {nmads_cut}.")
    if outlier_use_smoothed and nmads_smoothed <= 0:
        raise ValueError(f"nmads_smoothed must be positive, got {nmads_smoothed}.")
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")
    qc_params = _resolve_qc_params(tiling_qc_params)

    labels_node = sdata.labels[labels_key]
    if isinstance(labels_node, xr.DataTree):
        if scale is None:
            raise ValueError("When using multi-scale labels, please specify the scale.")
        labels_da = labels_node[scale].ds["image"]
    else:
        if scale is not None:
            logg.warning(f"`scale={scale!r}` ignored: labels at {labels_key!r} are single-scale.")
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
        return _score_tile(
            tile_lbl,
            distance_tol=qc_params.distance_tol,
            min_area=qc_params.min_area,
            downsample=downsample,
            max_contour_points=qc_params.max_contour_points,
        )

    tasks = [_process_one(spec) for spec in specs]

    if _has_distributed_client():
        if n_jobs != -1:
            logg.warning(
                "`n_jobs` is ignored when an active dask.distributed Client is in scope. "
                "Parallelism is controlled by the client."
            )
        results = dask.compute(*tasks)
    else:
        num_workers = _get_n_cores(n_jobs)
        with ProgressBar():
            results = dask.compute(*tasks, scheduler="threads", num_workers=num_workers)

    tile_dfs = [df for df in results if not df.empty]

    if not tile_dfs:
        raise ValueError("No cells scored - labels may be empty or all below min_area.")

    combined = pd.concat(tile_dfs, axis=0).sort_index()

    if combined.index.duplicated().any():
        dups = combined.index[combined.index.duplicated()].unique().tolist()
        raise RuntimeError(f"Duplicate cell IDs across tiles - tile ownership may be broken. Duplicates: {dups}")

    # --- Spatial context post-processing ---
    n_cells = len(combined)

    centroid_y = np.array([cell_info[lid].centroid_y for lid in combined.index])
    centroid_x = np.array([cell_info[lid].centroid_x for lid in combined.index])
    centroids = np.column_stack([centroid_y, centroid_x])

    if n_cells <= 1:
        combined["smoothed_cut_score"] = combined["cut_score"]
        combined["is_outlier"] = False
        combined["nhood_outlier_fraction"] = 0.0
    else:
        effective_k = min(n_neighbors, n_cells - 1)
        tree = BallTree(centroids)
        _, indices = tree.query(centroids, k=effective_k + 1)  # +1 because query includes self
        neighbor_idx = indices[:, 1:]

        cut_scores = combined["cut_score"].values.copy()
        cut_scores = np.where(np.isnan(cut_scores), 0.0, cut_scores)
        neighbor_mean = cut_scores[neighbor_idx].mean(axis=1)
        smoothed = cut_scores * neighbor_mean
        combined["smoothed_cut_score"] = smoothed

        # Build is_outlier from enabled gates (AND when both active).
        # A gate whose MAD is degenerate has no signal — treat it as a
        # no-op so it cannot poison the other gate's result.  If no gate
        # produced a meaningful filter, fall back to "no outliers".
        is_outlier = np.ones(n_cells, dtype=bool)
        gates_applied = 0

        if outlier_use_cut:
            median_c = np.median(cut_scores)
            mad_c = np.median(np.abs(cut_scores - median_c))
            if mad_c >= 1e-12:
                is_outlier &= cut_scores >= median_c + nmads_cut * mad_c * _MAD_TO_SD
                gates_applied += 1

        if outlier_use_smoothed:
            median_s = np.median(smoothed)
            mad_s = np.median(np.abs(smoothed - median_s))
            if mad_s >= 1e-12:
                is_outlier &= smoothed >= median_s + nmads_smoothed * mad_s * _MAD_TO_SD
                gates_applied += 1

        if gates_applied == 0:
            is_outlier[:] = False

        combined["is_outlier"] = is_outlier

        neighbor_outlier_frac = combined["is_outlier"].values[neighbor_idx].mean(axis=1)
        combined["nhood_outlier_fraction"] = neighbor_outlier_frac

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

    # TODO: migrate tiling QC scores to .obsm once spatialdata-plot
    # supports rendering labels colored by obsm keys.
    # See scverse/spatialdata-plot#587.
    for col in combined.columns:
        adata.obs[col] = combined[col].values

    adata.obs["centroid_y"] = centroid_y
    adata.obs["centroid_x"] = centroid_x

    adata.uns[_METHOD_KEY] = {
        "scale": scale,
        "tile_size": tile_size,
        "overlap_margin": overlap_margin,
        "downsample": downsample,
        "outlier_use_cut": outlier_use_cut,
        "outlier_use_smoothed": outlier_use_smoothed,
        "nmads_cut": nmads_cut,
        "nmads_smoothed": nmads_smoothed,
        "n_neighbors": n_neighbors,
        "tiling_qc_params": asdict(qc_params),
    }

    table_key = table_key_added if table_key_added is not None else f"{labels_key}_qc"
    _warn_if_dropping_stitch_columns(sdata, table_key, labels_key)
    if inplace:
        sdata.tables[table_key] = TableModel.parse(adata)
        return None
    return adata


def _warn_if_dropping_stitch_columns(sdata: sd.SpatialData, table_key: str, labels_key: str) -> None:
    """Warn if re-running QC would drop downstream stitch results.

    ``calculate_tiling_qc`` replaces the QC table wholesale, so any columns
    added by :func:`stitch_tile_cuts` to a previous version of this table
    are about to disappear.  We emit an actionable warning listing the
    previous stitch parameters (from ``.uns["tiling_stitch"]``) and a
    copy-pasteable invocation to restore them.
    """
    if table_key not in sdata.tables:
        return
    existing = sdata.tables[table_key]
    present = [c for c in _STITCH_COLUMNS if c in existing.obs.columns]
    if not present:
        return

    prev_params = existing.uns.get("tiling_stitch", {}) if hasattr(existing, "uns") else {}
    # `tiling_stitch` mixes top-level constructor kwargs with the nested
    # ``stitch_params`` bundle and diagnostic outputs (n_outliers, ...).
    # Filter to the allowlist + the nested bundle so the rerun string is
    # a valid Python call.
    parts = [f"labels_key={labels_key!r}"]
    parts.extend(f"{k}={v!r}" for k, v in prev_params.items() if k in _STITCH_PARAM_KEYS)
    nested = prev_params.get("stitch_params")
    if isinstance(nested, dict) and nested:
        parts.append(f"stitch_params={nested!r}")
    rerun = f"sq.experimental.tl.stitch_tile_cuts(sdata, {', '.join(parts)})"
    logg.warning(
        f"Re-running calculate_tiling_qc dropped previous stitch columns "
        f"({', '.join(present)}) from sdata.tables[{table_key!r}].  "
        f"To restore them, run: {rerun}"
    )
