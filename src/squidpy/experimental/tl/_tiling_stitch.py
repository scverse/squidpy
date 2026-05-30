"""Stitching of tile-cut cells flagged by :func:`~squidpy.experimental.tl.calculate_tiling_qc`.

When segmentation is run tile-by-tile (Cellpose, Stardist, Mesmer, ...) cells
that straddle tile boundaries get cut into 2-4 pieces with characteristic
straight, axis-aligned cut edges.  :func:`~squidpy.experimental.tl.calculate_tiling_qc` flags these
as ``is_outlier=True``.  This module pairs facing cut edges across boundaries
and assigns each candidate pair a heuristic geometric score in [0, 1].

The score is a weighted mean of five dataset-independent geometric features --
``iou``, ``endpoint_match``, ``merge_compactness``, ``merge_solidity`` and
``gap_proximity`` -- computed from the cut-edge geometry and the union mask
after closing the seam gap.  No model is fitted or shipped: the weights
default to flat-equal and are user-tunable via ``StitchParams.feature_weights``;
the features actually used, the weights applied, and the formula are recorded
in ``.uns["tiling_stitch"]``.  Users should tune ``min_confidence`` for their
data; ``0.7`` is a reasonable starting point, not a calibrated probability.

The labels element is **never** modified here -- only ``.obs`` columns are
written.  Materialising a stitched labels element is opt-in via
:func:`squidpy.experimental.im.make_stitched_labels`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import spatialdata as sd
import xarray as xr
from scipy.ndimage import binary_closing
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.morphology import disk as morph_disk
from spatialdata._logging import logger as logg

from squidpy.experimental.utils._geometry import equivalent_diameter, largest_contour
from squidpy.experimental.utils._labels import iter_chunked_regionprops, resolve_labels_array
from squidpy.experimental.utils._params import resolve_params

if TYPE_CHECKING:
    from collections.abc import Iterable

    import anndata as ad

__all__ = ["StitchParams", "assign_stitch_groups"]

# The scored geometric features and their formula.  Defined before StitchParams
# so __post_init__ can validate feature_weights keys against this tuple.
_SCORE_FEATURES: tuple[str, ...] = ("iou", "endpoint_match", "merge_compactness", "merge_solidity", "gap_proximity")


@dataclass(slots=True)
class StitchParams:
    """Advanced tuning knobs for :func:`~squidpy.experimental.tl.assign_stitch_groups`.

    Defaults work for typical 2D segmentation tiles produced by
    cellpose-like pipelines.  Pass an instance (or a ``Mapping`` of
    field names to values) as ``stitch_params`` to override.  Most fields
    are *advanced* -- the defaults rarely need changing; ``feature_weights``
    is the main knob a user might reach for.
    """

    distance_tol: float = 0.75
    """Advanced: sub-pixel tolerance for "lies on a bbox edge"."""

    min_edge_length: float = 5.0
    """Advanced: absolute floor on cut-edge length (pixels)."""

    min_edge_length_ratio: float = 0.4
    """Advanced: minimum cut-edge length relative to the cell's equivalent diameter."""

    min_edge_coverage: float = 0.5
    """Advanced: minimum fraction of parallel-axis positions covered by near-edge contour points."""

    candidate_min_iou: float = 0.2
    """Advanced: loose 1-D IoU floor at candidate enumeration."""

    close_radius: int = 3
    """Advanced: morphological closing disk radius for the union mask.  Also the
    length scale for ``gap_proximity`` (normalised by ``2 * close_radius``)."""

    feature_weights: Mapping[str, float] | None = None
    """Per-feature weights for the score, keyed by names in :data:`_SCORE_FEATURES`.

    ``None`` (default) means flat-equal weights.  A partial mapping is allowed:
    unspecified features keep weight ``1.0``.  Weights must be non-negative and
    are renormalised to sum to 1, so ``stitch_confidence`` stays in [0, 1]."""

    def __post_init__(self) -> None:
        # Coerce numeric types (accept numpy scalars cleanly) and bounds-check.
        self.distance_tol = float(self.distance_tol)
        self.min_edge_length = float(self.min_edge_length)
        self.min_edge_length_ratio = float(self.min_edge_length_ratio)
        self.min_edge_coverage = float(self.min_edge_coverage)
        self.candidate_min_iou = float(self.candidate_min_iou)
        self.close_radius = int(self.close_radius)
        if self.distance_tol < 0:
            raise ValueError(f"distance_tol must be >= 0, got {self.distance_tol}.")
        if self.min_edge_length < 0:
            raise ValueError(f"min_edge_length must be >= 0, got {self.min_edge_length}.")
        if not 0.0 <= self.min_edge_length_ratio <= 1.0:
            raise ValueError(f"min_edge_length_ratio must be in [0, 1], got {self.min_edge_length_ratio}.")
        if not 0.0 <= self.min_edge_coverage <= 1.0:
            raise ValueError(f"min_edge_coverage must be in [0, 1], got {self.min_edge_coverage}.")
        if not 0.0 <= self.candidate_min_iou <= 1.0:
            raise ValueError(f"candidate_min_iou must be in [0, 1], got {self.candidate_min_iou}.")
        if self.close_radius < 0:
            raise ValueError(f"close_radius must be >= 0, got {self.close_radius}.")
        if self.feature_weights is not None:
            if not isinstance(self.feature_weights, Mapping):
                raise TypeError(
                    f"feature_weights must be a Mapping or None, got {type(self.feature_weights).__name__}."
                )
            unknown = set(self.feature_weights) - set(_SCORE_FEATURES)
            if unknown:
                raise ValueError(
                    f"Unknown feature_weights key(s): {sorted(unknown)}; expected from {list(_SCORE_FEATURES)}."
                )
            coerced = {}
            for k, v in self.feature_weights.items():
                fv = float(v)
                if fv < 0:
                    raise ValueError(f"feature_weights[{k!r}] must be >= 0, got {fv}.")
                coerced[k] = fv
            # Store a plain dict of floats (drops numpy scalars, deterministic order).
            self.feature_weights = {k: coerced[k] for k in _SCORE_FEATURES if k in coerced}


def _resolve_stitch_params(stitch_params: StitchParams | Mapping[str, Any] | None) -> StitchParams:
    """Normalise the ``stitch_params`` argument to a :class:`StitchParams` instance."""
    return resolve_params(stitch_params, StitchParams, label="stitch_params")


def _resolve_feature_weights(feature_weights: Mapping[str, float] | None) -> dict[str, float]:
    """Return a full ``{feature: weight}`` dict over :data:`_SCORE_FEATURES`, renormalised to sum 1.

    ``None`` -> flat-equal.  A partial mapping fills unspecified features with
    weight ``1.0`` before renormalising.  Validation (unknown keys, negatives)
    happens in :meth:`StitchParams.__post_init__`; this helper assumes clean input.
    """
    base = dict.fromkeys(_SCORE_FEATURES, 1.0)
    if feature_weights:
        base.update(feature_weights)
    total = sum(base.values())
    if total <= 0:
        raise ValueError("feature_weights must have a positive sum (at least one feature with weight > 0).")
    return {f: base[f] / total for f in _SCORE_FEATURES}


_METHOD_KEY = "tiling_stitch"
_STITCH_DEFAULTS = StitchParams()

# Contract between calculate_tiling_qc and assign_stitch_groups.  _STITCH_COLUMNS
# is the obs columns stitch writes back into the QC table; _STITCH_PARAM_KEYS
# is the subset of top-level kwargs valid for re-running assign_stitch_groups
# (the advanced tuning lives in a nested ``stitch_params`` dict).
_STITCH_COLUMNS = ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence")
_STITCH_PARAM_KEYS = frozenset({"min_confidence", "max_gap", "max_group_size"})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CutEdge:
    """A candidate cut edge on a single cell's bbox.

    Attributes
    ----------
    cell_id
        Label ID of the piece carrying this edge.
    axis
        ``"h"`` (horizontal cut: edge is a horizontal line, cell sits above
        or below it) or ``"v"`` (vertical cut).
    coord
        Position of the cut line: y-coord for ``"h"``, x-coord for ``"v"``.
    extent
        ``(min, max)`` along the parallel axis -- the chord at the cut line.
    normal_dir
        ``+1`` if the cell's centroid sits at greater coord than the cut
        line, ``-1`` otherwise.  Used to enforce facing pairs.
    length
        Euclidean length of the run (``extent[1] - extent[0]``).
    """

    cell_id: int
    axis: str
    coord: float
    extent: tuple[float, float]
    normal_dir: int
    length: float


@dataclass(frozen=True)
class _StitchPair:
    """A scored candidate pairing of two cut edges across a tile boundary.

    ``confidence`` is the weighted mean of the geometric features (see
    :data:`_SCORE_FEATURES`); the individual feature components are kept for
    diagnostics and for the ``min``-based group-confidence aggregation.
    """

    cell_a: int
    cell_b: int
    axis: str
    confidence: float
    iou: float
    endpoint_match: float
    gap_proximity: float
    merge_solidity: float
    merge_compactness: float
    edge_a: _CutEdge | None = field(default=None, repr=False)
    edge_b: _CutEdge | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Stage 1: cut-edge extraction
# ---------------------------------------------------------------------------


def _read_bbox_slice(labels_da: xr.DataArray | np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Read a 2-D bbox slice from numpy or xarray, squeezing singleton dims."""
    if isinstance(labels_da, np.ndarray):
        return labels_da[y0:y1, x0:x1]
    arr = labels_da.isel(y=slice(y0, y1), x=slice(x0, x1)).values
    while arr.ndim > 2:
        arr = arr.squeeze(0)
    return arr


def _compute_outlier_bboxes(
    labels_da: xr.DataArray | np.ndarray,
    outlier_ids: Iterable[int],
    chunk_size: int = 4096,
) -> dict[int, tuple[int, int, int, int]]:
    """Compute global bboxes for the outlier subset in a single chunked pass.

    Returns mapping ``label_id -> (min_row, min_col, max_row, max_col)``.
    Works on numpy or dask-backed xarray; for xarray the array is read in
    ``chunk_size`` x ``chunk_size`` tiles so memory is bounded.
    """
    outlier_set = {int(x) for x in outlier_ids}
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    # Single chunked pass (shared with the QC reader); only outlier labels are
    # accumulated, merging bboxes across chunk boundaries for cells that span them.
    # TODO: faster path -- pre-mask each chunk with np.where(np.isin(chunk,
    # outlier_set), chunk, 0) before regionprops, so non-outlier cells are
    # skipped instead of scanned.  Worth doing if outlier fraction is < ~5%.
    for lid, region, y0, x0 in iter_chunked_regionprops(labels_da, chunk_size=chunk_size, label_subset=outlier_set):
        r0, c0, r1, c1 = region.bbox
        r0 += y0
        c0 += x0
        r1 += y0
        c1 += x0
        prev = bboxes.get(lid)
        if prev is None:
            bboxes[lid] = (r0, c0, r1, c1)
        else:
            bboxes[lid] = (min(prev[0], r0), min(prev[1], c0), max(prev[2], r1), max(prev[3], c1))
    return bboxes


def _bbox_edge_run(
    contour: np.ndarray,
    perp_axis: int,
    target: float,
    distance_tol: float = _STITCH_DEFAULTS.distance_tol,
    min_coverage: float = _STITCH_DEFAULTS.min_edge_coverage,
) -> tuple[float, float, float] | None:
    """Find the extent of contour points lying near a single bbox edge.

    A genuine cut edge has many contour points clustered at the bbox boundary,
    spanning a long parallel-axis range with high integer-position coverage.
    A naturally curved cell only touches its bbox at a single point, which
    fails either the count, length, or coverage check.

    Returns ``(ext_lo, ext_hi, length)`` if a substantial run is found.
    """
    parallel_axis = 1 - perp_axis
    near = np.abs(contour[:, perp_axis] - target) <= distance_tol
    if near.sum() < 3:
        return None
    parallel_vals = contour[near, parallel_axis]
    ext_lo = float(parallel_vals.min())
    ext_hi = float(parallel_vals.max())
    length = ext_hi - ext_lo
    if length <= 0:
        return None
    width = max(int(np.ceil(length)), 1)
    bins = np.zeros(width + 1, dtype=bool)
    bins[np.clip((parallel_vals - ext_lo).astype(int), 0, width)] = True
    coverage = float(bins.sum()) / (width + 1)
    if coverage < min_coverage:
        return None
    return ext_lo, ext_hi, length


def _extract_cut_edges(
    labels_da: xr.DataArray | np.ndarray,
    outlier_ids: Iterable[int],
    bboxes: dict[int, tuple[int, int, int, int]] | None = None,
    distance_tol: float = _STITCH_DEFAULTS.distance_tol,
    min_edge_length: float = _STITCH_DEFAULTS.min_edge_length,
    min_edge_length_ratio: float = _STITCH_DEFAULTS.min_edge_length_ratio,
    min_edge_coverage: float = _STITCH_DEFAULTS.min_edge_coverage,
) -> list[_CutEdge]:
    """Extract cardinal-aligned bbox-edge runs (cut-edge candidates) per outlier.

    For each outlier cell:
    1. Crop labels to its bbox + 1 px pad, build a binary mask.
    2. Trace its contour with :func:`skimage.measure.find_contours`.
    3. Check each of the 4 bbox-edge lines for a substantial straight run.

    A piece cut at a tile boundary always has its cut on a bbox edge -- the
    piece terminates exactly at the cut.  Curved cells only touch the bbox
    at a single contour point, which the density check rejects.

    Cells at a 4-tile corner produce 2 perpendicular edges; mid-stripe pieces
    can produce 2 parallel edges.
    """
    outlier_list = [int(x) for x in outlier_ids]
    if bboxes is None:
        bboxes = _compute_outlier_bboxes(labels_da, outlier_list)

    edges: list[_CutEdge] = []
    for lid in outlier_list:
        bbox = bboxes.get(lid)
        if bbox is None:
            continue
        min_r, min_c, max_r, max_c = bbox

        crop_arr = _read_bbox_slice(labels_da, min_r, max_r, min_c, max_c)
        mask = (crop_arr == lid).astype(np.float32)
        if not mask.any():
            continue
        mask = np.pad(mask, 1, mode="constant", constant_values=0)
        contour = largest_contour(mask)
        if contour is None:
            continue
        contour_global = contour.copy()
        contour_global[:, 0] += min_r - 1
        contour_global[:, 1] += min_c - 1

        # Local centroid from the mask (avoids a second regionprops call).
        ys, xs = np.where(mask)
        cy = float(ys.mean()) + min_r - 1
        cx = float(xs.mean()) + min_c - 1
        area = float(mask.sum())
        eq_diameter = equivalent_diameter(area)
        min_len = max(min_edge_length, min_edge_length_ratio * eq_diameter)

        # find_contours places level set 0.5 outside the integer pixel boundary.
        bbox_targets = [
            ("h", float(min_r) - 0.5),
            ("h", float(max_r) - 0.5),
            ("v", float(min_c) - 0.5),
            ("v", float(max_c) - 0.5),
        ]
        for axis, target in bbox_targets:
            perp_axis = 0 if axis == "h" else 1
            run = _bbox_edge_run(contour_global, perp_axis, target, distance_tol, min_edge_coverage)
            if run is None:
                continue
            ext_lo, ext_hi, length = run
            if length < min_len:
                continue
            cell_coord = cy if axis == "h" else cx
            normal = 1 if cell_coord > target else -1
            edges.append(
                _CutEdge(
                    cell_id=lid,
                    axis=axis,
                    coord=target,
                    extent=(ext_lo, ext_hi),
                    normal_dir=normal,
                    length=float(length),
                )
            )

    return edges


# ---------------------------------------------------------------------------
# Stage 2: pair candidate enumeration + features
# ---------------------------------------------------------------------------


def _extent_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _merge_shape_features(
    labels_da: xr.DataArray | np.ndarray,
    cell_ids: Iterable[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    close_radius: int = _STITCH_DEFAULTS.close_radius,
) -> dict[str, float]:
    """Materialise the union of given pieces, close the gap, and return shape stats.

    Solidity (area / convex_hull_area) and compactness (4*pi*A / P^2) drop
    sharply when two unrelated cells are joined -- the union is concave at the
    join.  ``merge_compactness`` is typically the strongest single
    discriminator between true cuts and false merges.
    """
    cell_list = [int(c) for c in cell_ids]
    if not cell_list:
        return {"merge_solidity": 0.0, "merge_compactness": 0.0}

    # Union bbox + padding to give morphological closing room.
    rs = [bboxes[c][0] for c in cell_list if c in bboxes]
    cs = [bboxes[c][1] for c in cell_list if c in bboxes]
    re = [bboxes[c][2] for c in cell_list if c in bboxes]
    ce = [bboxes[c][3] for c in cell_list if c in bboxes]
    if not rs:
        return {"merge_solidity": 0.0, "merge_compactness": 0.0}
    pad = close_radius + 2
    H = labels_da.shape[-2] if hasattr(labels_da, "shape") else int(labels_da.sizes["y"])
    W = labels_da.shape[-1] if hasattr(labels_da, "shape") else int(labels_da.sizes["x"])
    r0 = max(min(rs) - pad, 0)
    c0 = max(min(cs) - pad, 0)
    r1 = min(max(re) + pad, H)
    c1 = min(max(ce) + pad, W)

    crop = _read_bbox_slice(labels_da, r0, r1, c0, c1)
    mask = np.isin(crop, cell_list)
    if not mask.any():
        return {"merge_solidity": 0.0, "merge_compactness": 0.0}

    closed = binary_closing(mask, structure=morph_disk(close_radius))
    cc = cc_label(closed, connectivity=2)
    if cc.max() == 0:
        return {"merge_solidity": 0.0, "merge_compactness": 0.0}
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    biggest = int(sizes.argmax())
    region = regionprops((cc == biggest).astype(np.uint8))[0]
    perimeter = max(region.perimeter, 1.0)
    compactness = float(min(4 * np.pi * region.area / (perimeter * perimeter), 1.0))
    # Clamp solidity to 1.0: skimage can return area/convex_area slightly >1 for
    # thin/degenerate rasterised regions, which would push the score out of [0, 1].
    solidity = float(min(region.solidity, 1.0))
    return {"merge_solidity": solidity, "merge_compactness": compactness}


def _pair_geometry_features(
    e: _CutEdge,
    c: _CutEdge,
    max_gap: float,
    candidate_min_iou: float = _STITCH_DEFAULTS.candidate_min_iou,
) -> dict[str, float] | None:
    """Compute geometry-only features for a candidate pair, returning ``None``
    if the pair fails the basic facing/overlap/IoU filters.
    """
    if c.normal_dir == e.normal_dir:
        return None
    # Facing: cell with +1 normal must sit at greater coord than cell with -1.
    if (e.coord - c.coord) * e.normal_dir < -1e-6:
        return None
    overlap = _extent_overlap(e.extent, c.extent)
    if overlap <= 0:
        return None
    union = e.length + c.length - overlap
    iou = overlap / union if union > 0 else 0.0
    if iou < candidate_min_iou:
        return None
    gap = abs(e.coord - c.coord)
    if gap > max_gap:
        return None
    endpoint_dist = abs(e.extent[0] - c.extent[0]) + abs(e.extent[1] - c.extent[1])
    max_len = max(e.length, c.length)
    endpoint_match = max(0.0, 1.0 - endpoint_dist / max_len) if max_len > 0 else 0.0
    # Return the raw perpendicular gap; gap_proximity is derived later against
    # the closing reach (2*close_radius), NOT against max_gap (a search radius).
    return {
        "iou": float(iou),
        "endpoint_match": float(endpoint_match),
        "gap": float(gap),
    }


def _enumerate_pair_candidates(
    edges: list[_CutEdge],
    max_gap: float,
    candidate_min_iou: float = _STITCH_DEFAULTS.candidate_min_iou,
) -> list[tuple[_CutEdge, _CutEdge, dict[str, float]]]:
    """Find all (e, c) pairs of facing cut edges with their geometry features.

    Returns one entry per surviving candidate.  No selection / scoring yet.
    """
    out: list[tuple[_CutEdge, _CutEdge, dict[str, float]]] = []
    by_axis: dict[str, list[_CutEdge]] = {"h": [], "v": []}
    for e in edges:
        by_axis[e.axis].append(e)

    for axis_edges in by_axis.values():
        axis_edges.sort(key=lambda e: e.coord)
        coords = np.array([e.coord for e in axis_edges])
        for i, e in enumerate(axis_edges):
            lo = int(np.searchsorted(coords, e.coord - max_gap, side="left"))
            hi = int(np.searchsorted(coords, e.coord + max_gap, side="right"))
            for j in range(lo, hi):
                if j <= i:
                    continue  # symmetry: emit each unordered pair once
                c = axis_edges[j]
                if c.cell_id == e.cell_id:
                    continue
                feats = _pair_geometry_features(e, c, max_gap, candidate_min_iou=candidate_min_iou)
                if feats is None:
                    continue
                out.append((e, c, feats))
    return out


# ---------------------------------------------------------------------------
# Stage 4: scoring (weighted mean of geometry + shape features)
# ---------------------------------------------------------------------------


def _gap_proximity(gap: float, close_radius: int) -> float:
    """Map the raw perpendicular gap to [0, 1] against the closing reach.

    Normalised by ``2 * close_radius`` -- the scale at which morphological
    closing could actually bridge the seam -- so the feature is independent of
    the ``max_gap`` search radius and only reaches 0 when the gap genuinely
    exceeds what closing can join.  When closing is disabled (``close_radius=0``)
    the feature is inactive and returns ``1.0`` rather than collapsing the score.
    """
    reach = 2 * close_radius
    # gap<=0 (touching/overlapping) or reach<=0 (closing disabled, close_radius=0)
    # -> the feature is inactive (neutral 1.0), never a silent score cliff.
    if gap <= 0 or reach <= 0:
        return 1.0
    return max(0.0, 1.0 - gap / reach)


def _score_pair_features(features: dict[str, float], weights: dict[str, float]) -> float:
    """Return the heuristic stitch score in [0, 1].

    Weighted mean of the features in :data:`_SCORE_FEATURES` (``weights`` are
    pre-normalised to sum 1).  The score is dataset-independent and not a
    calibrated probability -- users pick ``min_confidence`` based on their
    false-merge tolerance.
    """
    return float(sum(weights[name] * features[name] for name in _SCORE_FEATURES))


def _score_pairs(
    candidates: list[tuple[_CutEdge, _CutEdge, dict[str, float]]],
    labels_da: xr.DataArray | np.ndarray,
    bboxes: dict[int, tuple[int, int, int, int]],
    weights: dict[str, float],
    close_radius: int = _STITCH_DEFAULTS.close_radius,
) -> list[_StitchPair]:
    """Compute shape features per candidate and score every pair (no filtering).

    Returns all scored pairs (one per ``(cell_a, cell_b, axis)``, keeping max
    confidence on duplicates); the ``min_confidence`` cut is applied by the
    caller so diagnostics can also see below-threshold pairs.
    """
    scored: list[_StitchPair] = []
    for e, c, geom in candidates:
        shape = _merge_shape_features(labels_da, [e.cell_id, c.cell_id], bboxes, close_radius=close_radius)
        feats = {**geom, **shape, "gap_proximity": _gap_proximity(geom["gap"], close_radius)}
        confidence = _score_pair_features(feats, weights)
        # Canonicalise so cell_a < cell_b for deterministic union-find.
        if e.cell_id < c.cell_id:
            ea, eb = e, c
        else:
            ea, eb = c, e
        scored.append(
            _StitchPair(
                cell_a=ea.cell_id,
                cell_b=eb.cell_id,
                axis=e.axis,
                confidence=confidence,
                iou=feats["iou"],
                endpoint_match=feats["endpoint_match"],
                gap_proximity=feats["gap_proximity"],
                merge_solidity=feats["merge_solidity"],
                merge_compactness=feats["merge_compactness"],
                edge_a=ea,
                edge_b=eb,
            )
        )

    # Deduplicate to one entry per (cell_a, cell_b, axis), keeping max confidence.
    by_pair: dict[tuple[int, int, str], _StitchPair] = {}
    for p in scored:
        k = (p.cell_a, p.cell_b, p.axis)
        if k not in by_pair or by_pair[k].confidence < p.confidence:
            by_pair[k] = p
    return sorted(by_pair.values(), key=lambda p: (-p.confidence, p.cell_a, p.cell_b))


# ---------------------------------------------------------------------------
# Stage 5: group assembly via union-find + validation
# ---------------------------------------------------------------------------


def _validate_group_geometry(
    pairs_in_group: list[_StitchPair],
    size: int,
    max_gap: float,
) -> bool:
    """Geometric sanity check for groups of size >= 3.

    Two cases:

    - **Corner group** (size 4, both axes present): the cut edges' endpoints
      must converge near a single junction point (one ``h`` cut crossing one
      ``v`` cut defines the junction).  If the spread of edge extents from
      the junction is greater than ``max_gap``, the group is implausible.

    - **Chain group** (size 3 or 4, all pairs share one axis): legitimate
      same-axis chains (e.g., a cell split by 3 horizontal seams into 4
      vertically-stacked pieces) have pairs at N-1 *distinct* seam
      coordinates.  Multiple pairs at the same seam coord would imply
      geometrically impossible "two cuts at the same seam" pairings -- a
      signature of a false-positive cluster -- so we reject.
    """
    h_pairs = [p for p in pairs_in_group if p.axis == "h"]
    v_pairs = [p for p in pairs_in_group if p.axis == "v"]

    # Chain case: only one axis present and size >= 3.
    if not h_pairs or not v_pairs:
        if size < 3:
            return True  # 2-piece groups are trivially valid on one axis
        # Each pair's seam coord is roughly midway between its two edges.
        seam_coords = [round((p.edge_a.coord + p.edge_b.coord) / 2.0, 1) for p in pairs_in_group]
        # Allow a max_gap-sized tolerance for "distinct" seams.
        sorted_coords = sorted(seam_coords)
        for prev, cur in zip(sorted_coords, sorted_coords[1:], strict=False):
            if cur - prev <= max_gap:
                return False
        return True

    # Mixed-axis case: only validate the 4-piece corner pattern.  3-piece
    # L-shapes (one h pair + one v pair sharing a corner cell) are
    # geometrically valid and don't have a junction to converge on.
    if size != 4:
        return True

    # Corner case: both axes present, size 4.  Junction y/x is the mean of edge coords.
    h_edges = [p.edge_a for p in h_pairs] + [p.edge_b for p in h_pairs]
    v_edges = [p.edge_a for p in v_pairs] + [p.edge_b for p in v_pairs]
    junction_y = float(np.mean([e.coord for e in h_edges]))
    junction_x = float(np.mean([e.coord for e in v_edges]))
    for e in h_edges:
        if min(abs(e.extent[0] - junction_x), abs(e.extent[1] - junction_x)) > max_gap:
            return False
    for e in v_edges:
        if min(abs(e.extent[0] - junction_y), abs(e.extent[1] - junction_y)) > max_gap:
            return False
    return True


def _assemble_groups(
    pairs: list[_StitchPair],
    candidate_ids: Iterable[int],
    max_group_size: int,
    max_gap: float,
) -> tuple[dict[int, int], dict[int, float]]:
    """Build stitch groups via union-find with size + corner validation.

    Returns
    -------
    groups
        ``cell_id -> group_id`` (group_id == own cell_id for unstitched).
    confidences
        ``cell_id -> stitch_confidence`` -- min over pairwise confidences in
        the cell's group; ``1.0`` for confirmed-solo (no surviving pair).
    """
    # Build undirected connected components via scipy.  Cells map to a
    # contiguous [0, n) index space; pairs become symmetric edges in a CSR
    # adjacency matrix.  We then re-key components by the smallest cell_id
    # they contain so the group root is deterministic.
    candidate_list = sorted({int(c) for c in candidate_ids})
    if not candidate_list:
        return {}, {}
    id_to_idx = {cid: i for i, cid in enumerate(candidate_list)}
    n = len(candidate_list)

    valid_pairs = [p for p in pairs if p.cell_a in id_to_idx and p.cell_b in id_to_idx]
    if valid_pairs:
        rows = [id_to_idx[p.cell_a] for p in valid_pairs]
        cols = [id_to_idx[p.cell_b] for p in valid_pairs]
        adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
        _, comp_labels = connected_components(adj, directed=False)
    else:
        comp_labels = np.arange(n)

    cells_by_comp: dict[int, list[int]] = {}
    for i, comp in enumerate(comp_labels):
        cells_by_comp.setdefault(int(comp), []).append(candidate_list[i])

    members: dict[int, list[int]] = {}
    root_of_cell: dict[int, int] = {}
    for comp_members in cells_by_comp.values():
        comp_members.sort()
        root = comp_members[0]
        members[root] = comp_members
        for cid in comp_members:
            root_of_cell[cid] = root

    pairs_by_group: dict[int, list[_StitchPair]] = {}
    for p in valid_pairs:
        pairs_by_group.setdefault(root_of_cell[p.cell_a], []).append(p)

    groups: dict[int, int] = {}
    confidences: dict[int, float] = {}

    for root, mem in members.items():
        size = len(mem)
        group_pairs = pairs_by_group.get(root, [])

        # Size cap: collapse oversized groups back to singletons.
        if size > max_group_size:
            for m in mem:
                groups[m] = m
                confidences[m] = 1.0
            continue

        # Geometric validation for 3+ piece groups: corner-junction for
        # mixed-axis 4-groups, chain (distinct seam coords) for same-axis 3+.
        if size >= 3 and not _validate_group_geometry(group_pairs, size, max_gap):
            for m in mem:
                groups[m] = m
                confidences[m] = 1.0
            continue

        if size == 1:
            groups[mem[0]] = mem[0]
            confidences[mem[0]] = 1.0
            continue

        # Group confidence = min over pairwise confidences (weakest link).
        group_conf = float(min(p.confidence for p in group_pairs))
        for m in mem:
            groups[m] = root
            confidences[m] = group_conf

    return groups, confidences


def _build_diagnostics(
    all_pairs: list[_StitchPair],
    groups: dict[int, int],
    group_sizes: dict[int, int],
    min_confidence: float,
) -> dict[str, np.ndarray]:
    """Per-pair diagnostics for ``save_diagnostics``, as a zarr-safe dict of arrays.

    One entry per scored candidate (including below-threshold ones), with each
    feature, the confidence, the assigned ``group_id``, and a ``status``:

    - ``"accepted"``       -- passed the confidence cut and landed in a multi-piece group;
    - ``"below_threshold"`` -- confidence < ``min_confidence``;
    - ``"collapsed_group"`` -- passed the cut but its group was collapsed to a
      singleton by geometry validation or the size cap.

    Returned as a ``dict`` of equal-length :class:`numpy.ndarray` (rather than a
    DataFrame) so it round-trips cleanly through zarr/h5ad-backed ``.uns``.
    """
    n = len(all_pairs)
    out: dict[str, np.ndarray] = {
        "cell_a": np.empty(n, dtype=np.int64),
        "cell_b": np.empty(n, dtype=np.int64),
        "axis": np.empty(n, dtype="<U1"),
        **{f: np.empty(n, dtype=np.float64) for f in _SCORE_FEATURES},
        "confidence": np.empty(n, dtype=np.float64),
        "group_id": np.empty(n, dtype=np.int64),
        "status": np.empty(n, dtype="<U16"),
    }
    for i, p in enumerate(all_pairs):
        root = groups.get(p.cell_a, p.cell_a)
        if p.confidence < min_confidence:
            status = "below_threshold"
        elif group_sizes.get(root, 1) > 1:
            status = "accepted"
        else:
            status = "collapsed_group"
        out["cell_a"][i] = int(p.cell_a)
        out["cell_b"][i] = int(p.cell_b)
        out["axis"][i] = p.axis
        for f in _SCORE_FEATURES:
            out[f][i] = float(getattr(p, f))
        out["confidence"][i] = float(p.confidence)
        out["group_id"][i] = int(root)
        out["status"][i] = status
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assign_stitch_groups(
    sdata: sd.SpatialData,
    labels_key: str,
    qc_table_key: str | None = None,
    min_confidence: float = 0.7,
    max_gap: float = 3.0,
    max_group_size: int = 4,
    stitch_params: StitchParams | Mapping[str, Any] | None = None,
    save_diagnostics: bool = False,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Assign tile-cut cell pieces to stitch groups.

    Reads ``is_outlier=True`` cells flagged by
    :func:`~squidpy.experimental.tl.calculate_tiling_qc`, pairs facing cut
    edges across tile boundaries, scores each pair via a transparent geometric
    composite, and assembles high-confidence pairs into stitch groups via
    union-find.  This only *annotates* which pieces belong together -- it does
    **not** modify the labels element.  Materialising a stitched labels element
    is opt-in via :func:`squidpy.experimental.im.make_stitched_labels`.

    The score per pair is a weighted mean of five geometric features in [0, 1]:
    ``iou`` (1-D extent overlap), ``endpoint_match`` (chord endpoints coincide),
    ``merge_compactness`` (``4*pi*A / P^2`` of the closed union mask),
    ``merge_solidity`` (union area / convex hull area), and ``gap_proximity``
    (seam gap relative to the morphological closing reach).  Weights default to
    flat-equal and are tunable via ``StitchParams.feature_weights``.  No
    coefficients are fitted or shipped; the features, weights, and formula are
    recorded in ``.uns["tiling_stitch"]`` so a run is re-derivable from its own
    metadata.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` with a labels element and a QC
        table from :func:`~squidpy.experimental.tl.calculate_tiling_qc`.
    labels_key
        Key in ``sdata.labels``.
    qc_table_key
        Key of the QC table.  Defaults to ``"{labels_key}_qc"``.
    min_confidence
        Threshold on ``stitch_confidence``.  ``0.7`` (default) is a starting
        point; raise it for stricter precision, lower for recall.  Tune for
        your data -- the score is heuristic, not a calibrated probability.
    max_gap
        Maximum perpendicular distance (px) between facing cut edges for a pair
        to be *considered* a candidate.  This is a search radius only; it does
        not scale the score.
    max_group_size
        Cap on group size; oversized groups (likely false merges) collapse
        to singletons.
    stitch_params
        Advanced tuning knobs as a :class:`StitchParams` instance or a
        ``Mapping`` of its field names to values.  See :class:`StitchParams`
        for each field's meaning and default.  ``None`` (default) uses
        all defaults.
    save_diagnostics
        If ``True``, write a per-pair diagnostics table (every scored candidate:
        its feature values, confidence, assigned ``group_id``, and a ``status`` of
        ``"accepted"`` / ``"below_threshold"`` / ``"collapsed_group"``) to
        ``.uns["tiling_stitch"]["diagnostics"]`` as a dict of equal-length arrays.
        Useful for tuning ``min_confidence``; off by default to keep ``.uns`` lean.
    inplace
        If ``True``, write back into ``sdata.tables[qc_table_key]``.
        Otherwise return the modified AnnData.

    Returns
    -------
    The QC :class:`~anndata.AnnData` with four new ``.obs`` columns when
    ``inplace=False``, otherwise ``None``.
    """
    if labels_key not in sdata.labels:
        raise ValueError(f"Labels key '{labels_key}' not found in sdata.labels.")
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}.")
    if max_gap < 0:
        raise ValueError(f"max_gap must be non-negative, got {max_gap}.")
    if max_group_size < 1:
        raise ValueError(f"max_group_size must be >= 1, got {max_group_size}.")
    params = _resolve_stitch_params(stitch_params)
    weights = _resolve_feature_weights(params.feature_weights)

    table_key = qc_table_key if qc_table_key is not None else f"{labels_key}_qc"
    if table_key not in sdata.tables:
        raise ValueError(f"QC table '{table_key}' not found.  Run calculate_tiling_qc first.")
    adata = sdata.tables[table_key].copy()

    if "is_outlier" not in adata.obs.columns:
        raise ValueError(f"QC table '{table_key}' is missing 'is_outlier'; re-run calculate_tiling_qc.")
    if "label_id" not in adata.obs.columns:
        raise ValueError(f"QC table '{table_key}' is missing 'label_id'.")

    existing = [c for c in _STITCH_COLUMNS if c in adata.obs.columns]
    if existing:
        logg.warning(f"Overwriting existing stitch columns: {existing}.")
        adata.obs.drop(columns=existing, inplace=True)

    # Resolve which labels DataArray was used at QC time (multi-scale aware).
    qc_params = adata.uns.get("tiling_qc", {})
    scale = qc_params.get("scale")
    labels_da = resolve_labels_array(sdata, labels_key, scale)

    label_ids = adata.obs["label_id"].astype(int).to_numpy()
    is_outlier = adata.obs["is_outlier"].to_numpy(dtype=bool)
    outlier_ids = label_ids[is_outlier].tolist()

    n_outliers = len(outlier_ids)
    logg.info(f"Stitching {n_outliers} outlier cells (out of {len(label_ids)} total).")

    if n_outliers == 0:
        logg.warning("No outliers flagged; nothing to stitch.")
        groups: dict[int, int] = {}
        confidences: dict[int, float] = {}
        edges: list[_CutEdge] = []
        all_pairs: list[_StitchPair] = []
        pairs: list[_StitchPair] = []
    else:
        bboxes = _compute_outlier_bboxes(labels_da, outlier_ids)
        missing = [lid for lid in outlier_ids if lid not in bboxes]
        if missing:
            logg.warning(
                f"{len(missing)} outlier label_id(s) flagged in the QC table do not appear "
                f"in '{labels_key}' (e.g. {missing[:5]}); they will not be stitched."
            )
        edges = _extract_cut_edges(
            labels_da,
            outlier_ids,
            bboxes=bboxes,
            distance_tol=params.distance_tol,
            min_edge_length=params.min_edge_length,
            min_edge_length_ratio=params.min_edge_length_ratio,
            min_edge_coverage=params.min_edge_coverage,
        )
        candidates = _enumerate_pair_candidates(edges, max_gap=max_gap, candidate_min_iou=params.candidate_min_iou)
        # Score every candidate, then apply the confidence cut.  Keeping the
        # full list lets save_diagnostics expose below-threshold pairs too.
        all_pairs = _score_pairs(candidates, labels_da, bboxes, weights=weights, close_radius=params.close_radius)
        pairs = [p for p in all_pairs if p.confidence >= min_confidence]
        groups, confidences = _assemble_groups(pairs, outlier_ids, max_group_size=max_group_size, max_gap=max_gap)

    # True candidate count (pre-threshold) for the audit block; then release the
    # below-threshold pairs unless diagnostics needs them.
    n_candidates = len(all_pairs)
    if not save_diagnostics:
        all_pairs = []

    # Write .obs columns with three states distinguished by stitch_confidence:
    # - non-outlier cell      -> own label_id, False, 1, NaN  (not evaluated)
    # - outlier solo          -> own label_id, False, 1, 1.0  (checked, no partner)
    # - outlier stitched      -> shared root,  True,  n, composite score
    n = len(label_ids)
    stitch_group_id = label_ids.copy()
    is_stitched = np.zeros(n, dtype=bool)
    n_pieces = np.ones(n, dtype=np.int32)
    stitch_confidence = np.full(n, np.nan, dtype=np.float64)

    group_sizes: dict[int, int] = {}
    if outlier_ids:
        for root in groups.values():
            group_sizes[root] = group_sizes.get(root, 0) + 1

        id_to_idx = {int(lid): i for i, lid in enumerate(label_ids)}
        for cid, root in groups.items():
            i = id_to_idx[int(cid)]
            stitch_group_id[i] = int(root)
            size = group_sizes[root]
            n_pieces[i] = size
            is_stitched[i] = size > 1
            stitch_confidence[i] = float(confidences.get(cid, 1.0))

    adata.obs["stitch_group_id"] = stitch_group_id
    adata.obs["is_stitched"] = is_stitched
    adata.obs["n_pieces"] = n_pieces
    adata.obs["stitch_confidence"] = stitch_confidence

    n_groups = sum(1 for s in group_sizes.values() if s > 1)
    n_stitched = int(is_stitched.sum())
    # Use string keys so the dict round-trips through zarr-backed .uns cleanly.
    pieces_dist: dict[str, int] = {}
    for s in group_sizes.values():
        if s > 1:
            key = str(int(s))
            pieces_dist[key] = pieces_dist.get(key, 0) + 1

    # asdict(params) may carry feature_weights=None; drop it so no None is nested
    # in .uns (not reliably zarr-serialisable). The resolved, renormalised weights
    # are recorded separately under "feature_weights" for reproducibility.
    stitch_params_dump = {k: v for k, v in asdict(params).items() if v is not None}
    adata.uns[_METHOD_KEY] = {
        "min_confidence": float(min_confidence),
        "max_gap": float(max_gap),
        "max_group_size": int(max_group_size),
        "stitch_params": stitch_params_dump,
        "n_outliers": int(n_outliers),
        "n_candidate_pairs": int(n_candidates),
        "n_stitched_groups": int(n_groups),
        "n_stitched_cells": int(n_stitched),
        "n_pieces_distribution": pieces_dist,
        "score_features": list(_SCORE_FEATURES),
        "feature_weights": {k: float(v) for k, v in weights.items()},
    }

    if save_diagnostics:
        adata.uns[_METHOD_KEY]["diagnostics"] = _build_diagnostics(all_pairs, groups, group_sizes, min_confidence)

    if not inplace:
        return adata
    sdata.tables[table_key] = adata
    return None
