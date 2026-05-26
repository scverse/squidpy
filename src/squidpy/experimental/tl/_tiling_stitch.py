"""Stitching of tile-cut cells flagged by :func:`calculate_tiling_qc`.

When segmentation is run tile-by-tile (Cellpose, Stardist, Mesmer, ...) cells
that straddle tile boundaries get cut into 2-4 pieces with characteristic
straight, axis-aligned cut edges.  :func:`calculate_tiling_qc` flags these
as ``is_outlier=True``.  This module pairs facing cut edges across boundaries
and assigns each candidate pair a heuristic geometric score in [0, 1].

The score is the arithmetic mean of four dataset-independent features --
``iou``, ``endpoint_match``, ``merge_compactness``, ``merge_solidity`` --
computed from the cut-edge geometry and the union mask after closing the
seam gap.  No model is fitted or shipped; the formula is documented inline
and recorded in ``.uns["tiling_stitch"]["score_formula"]``.  Users should
tune ``min_confidence`` for their data; ``0.7`` is a reasonable default.

The labels element is **never** modified here -- only ``.obs`` columns are
written.  Materialising a stitched labels element is opt-in via
:func:`squidpy.experimental.im.make_stitched_labels`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import spatialdata as sd
import xarray as xr
from scipy.ndimage import binary_closing
from skimage.measure import find_contours, regionprops
from skimage.measure import label as cc_label
from skimage.morphology import disk as morph_disk
from spatialdata._logging import logger as logg

from squidpy.experimental.utils._labels import resolve_labels_array

if TYPE_CHECKING:
    from collections.abc import Iterable

    import anndata as ad

__all__ = ["stitch_tile_cuts"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sub-pixel tolerance for "lies on a bbox edge".  ``find_contours`` returns
# half-pixel coordinates, so a real edge run sits within ~0.5 px of the line.
_DEFAULT_DISTANCE_TOL = 0.75

# Cut-edge length must exceed both an absolute floor (filters tiny cells) and
# this fraction of the cell's equivalent diameter (filters arc-tops on
# naturally curved cells, where the bbox-edge contact is a single pixel).
_DEFAULT_MIN_EDGE_LENGTH = 5.0
_DEFAULT_MIN_EDGE_LENGTH_RATIO = 0.4

# Density check: of the integer parallel-axis positions within a candidate
# run, what fraction has at least one near-edge contour point?  A single-
# point arc-top fails this; a real chord across the cut passes trivially.
_DEFAULT_MIN_EDGE_COVERAGE = 0.5

# Loose IoU floor used at candidate enumeration -- selection happens at the
# calibrated score stage.  Keeping this loose lets the model see borderline
# negatives during scoring rather than excluding them upstream.
_DEFAULT_CANDIDATE_MIN_IOU = 0.2

# Morphological closing radius used to bridge the gap when materialising the
# union mask for shape-quality features.  Larger than ``max_gap`` to be
# robust to small cells where the gap is a meaningful fraction of cell size.
_DEFAULT_CLOSE_RADIUS = 3

_METHOD_KEY = "tiling_stitch"
_STITCH_COLUMNS = ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence")

# Features combined into ``stitch_confidence`` (arithmetic mean).  All four
# are dataset-independent geometry / shape signals in [0, 1].  ``gap_score``
# is also computed but only used as a hard filter (already inside
# ``max_gap`` by construction); it does not enter the score.
_SCORE_FEATURES: tuple[str, ...] = (
    "iou",
    "endpoint_match",
    "merge_compactness",
    "merge_solidity",
)
_SCORE_FORMULA = "arithmetic_mean(iou, endpoint_match, merge_compactness, merge_solidity)"

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

    Confidence is the calibrated logistic-regression probability; feature
    components are kept for diagnostics and for ``min``-based group
    confidence aggregation.
    """

    cell_a: int
    cell_b: int
    axis: str
    confidence: float
    iou: float
    endpoint_match: float
    gap_score: float
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

    if isinstance(labels_da, np.ndarray):
        for region in regionprops(labels_da):
            if region.label in outlier_set:
                bboxes[region.label] = region.bbox
        return bboxes

    H = int(labels_da.sizes.get("y", labels_da.shape[-2]))
    W = int(labels_da.sizes.get("x", labels_da.shape[-1]))
    # TODO: faster path -- pre-mask each chunk with np.where(np.isin(chunk,
    # outlier_set), chunk, 0) before regionprops, so non-outlier cells are
    # skipped instead of scanned.  Worth doing if outlier fraction is < ~5%.
    for y0 in range(0, H, chunk_size):
        y1 = min(y0 + chunk_size, H)
        for x0 in range(0, W, chunk_size):
            x1 = min(x0 + chunk_size, W)
            chunk = _read_bbox_slice(labels_da, y0, y1, x0, x1)
            for region in regionprops(chunk):
                lid = int(region.label)
                if lid not in outlier_set:
                    continue
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
    distance_tol: float = _DEFAULT_DISTANCE_TOL,
    min_coverage: float = _DEFAULT_MIN_EDGE_COVERAGE,
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
    distance_tol: float = _DEFAULT_DISTANCE_TOL,
    min_edge_length: float = _DEFAULT_MIN_EDGE_LENGTH,
    min_edge_length_ratio: float = _DEFAULT_MIN_EDGE_LENGTH_RATIO,
    min_edge_coverage: float = _DEFAULT_MIN_EDGE_COVERAGE,
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
        contours = find_contours(mask, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        contour_global = contour.copy()
        contour_global[:, 0] += min_r - 1
        contour_global[:, 1] += min_c - 1

        # Local centroid from the mask (avoids a second regionprops call).
        ys, xs = np.where(mask)
        cy = float(ys.mean()) + min_r - 1
        cx = float(xs.mean()) + min_c - 1
        area = float(mask.sum())
        eq_diameter = float(np.sqrt(4 * area / np.pi))
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
    close_radius: int = _DEFAULT_CLOSE_RADIUS,
) -> dict[str, float]:
    """Materialise the union of given pieces, close the gap, and return shape stats.

    Solidity (area / convex_hull_area) and compactness (4*pi*A / P^2) drop
    sharply when two unrelated cells are joined -- the union is concave at the
    join.  ``merge_compactness`` is the strongest single feature in the
    calibration model.
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
    return {"merge_solidity": float(region.solidity), "merge_compactness": compactness}


def _pair_geometry_features(
    e: _CutEdge,
    c: _CutEdge,
    max_gap: float,
    candidate_min_iou: float = _DEFAULT_CANDIDATE_MIN_IOU,
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
    gap_score = 1.0 - gap / max_gap
    return {
        "iou": float(iou),
        "endpoint_match": float(endpoint_match),
        "gap_score": float(gap_score),
    }


def _enumerate_pair_candidates(
    edges: list[_CutEdge],
    max_gap: float,
    candidate_min_iou: float = _DEFAULT_CANDIDATE_MIN_IOU,
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
# Stage 4: scoring (arithmetic mean of geometry + shape features)
# ---------------------------------------------------------------------------


def _score_pair_features(features: dict[str, float]) -> float:
    """Return the heuristic stitch score in [0, 1].

    Arithmetic mean of the four features in :data:`_SCORE_FEATURES`.  The
    score is dataset-independent and not a calibrated probability -- users
    pick ``min_confidence`` based on their false-merge tolerance.
    """
    return float(sum(features[name] for name in _SCORE_FEATURES) / len(_SCORE_FEATURES))


def _score_pairs(
    candidates: list[tuple[_CutEdge, _CutEdge, dict[str, float]]],
    labels_da: xr.DataArray | np.ndarray,
    bboxes: dict[int, tuple[int, int, int, int]],
    min_confidence: float,
    close_radius: int = _DEFAULT_CLOSE_RADIUS,
) -> list[_StitchPair]:
    """Compute shape features per candidate, score, and apply confidence filter.

    Greedy: each cell keeps its highest-confidence pairing per axis.  Mid-stripe
    cells (two parallel cuts) retain one pair per axis.
    """
    scored: list[_StitchPair] = []
    for e, c, geom in candidates:
        shape = _merge_shape_features(labels_da, [e.cell_id, c.cell_id], bboxes, close_radius=close_radius)
        feats = {**geom, **shape}
        confidence = _score_pair_features(feats)
        if confidence < min_confidence:
            continue
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
                gap_score=feats["gap_score"],
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


class _UnionFind:
    """Union-find with smallest-label-as-root for deterministic group IDs."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _validate_corner_junction(
    pairs_in_group: list[_StitchPair],
    max_gap: float,
) -> bool:
    """For 4-piece groups, require the cut edges' endpoints to converge near
    a single junction point.

    Two perpendicular cuts (one h, one v) define the junction; the four
    pieces' edges should share endpoints there.  If the spread is greater
    than ``max_gap``, the group geometry is implausible.
    """
    h_edges = [p.edge_a for p in pairs_in_group if p.axis == "h"] + [p.edge_b for p in pairs_in_group if p.axis == "h"]
    v_edges = [p.edge_a for p in pairs_in_group if p.axis == "v"] + [p.edge_b for p in pairs_in_group if p.axis == "v"]
    if not h_edges or not v_edges:
        return True  # not a corner; nothing to validate

    # Junction y is roughly the mean h-edge coord; junction x is mean v-edge coord.
    junction_y = float(np.mean([e.coord for e in h_edges if e is not None]))
    junction_x = float(np.mean([e.coord for e in v_edges if e is not None]))

    # Each h edge's extent should reach to junction_x within max_gap; each v
    # edge's extent should reach to junction_y within max_gap.
    for e in h_edges:
        if e is None:
            continue
        if min(abs(e.extent[0] - junction_x), abs(e.extent[1] - junction_x)) > max_gap:
            return False
    for e in v_edges:
        if e is None:
            continue
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
    uf = _UnionFind()
    candidate_list = [int(c) for c in candidate_ids]
    for cid in candidate_list:
        uf.find(cid)
    for p in pairs:
        uf.union(p.cell_a, p.cell_b)

    # Collect group members + the pairs internal to each group.
    members: dict[int, list[int]] = {}
    for cid in candidate_list:
        members.setdefault(uf.find(cid), []).append(cid)
    pairs_by_group: dict[int, list[_StitchPair]] = {}
    for p in pairs:
        root = uf.find(p.cell_a)
        pairs_by_group.setdefault(root, []).append(p)

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

        # Corner validation for 4-piece groups.
        if size == 4 and not _validate_corner_junction(group_pairs, max_gap):
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def stitch_tile_cuts(
    sdata: sd.SpatialData,
    labels_key: str,
    qc_table_key: str | None = None,
    min_confidence: float = 0.7,
    max_gap: float = 3.0,
    max_group_size: int = 4,
    distance_tol: float = _DEFAULT_DISTANCE_TOL,
    min_edge_length: float = _DEFAULT_MIN_EDGE_LENGTH,
    min_edge_length_ratio: float = _DEFAULT_MIN_EDGE_LENGTH_RATIO,
    min_edge_coverage: float = _DEFAULT_MIN_EDGE_COVERAGE,
    candidate_min_iou: float = _DEFAULT_CANDIDATE_MIN_IOU,
    close_radius: int = _DEFAULT_CLOSE_RADIUS,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Stitch tile-cut cells flagged by :func:`calculate_tiling_qc`.

    Reads ``is_outlier=True`` cells from the QC table, pairs facing cut edges
    across tile boundaries, scores each pair via a transparent geometric
    composite, and assembles high-confidence pairs into stitch groups via
    union-find.

    The score per pair is the arithmetic mean of four features in [0, 1]:
    ``iou`` (1-D extent overlap), ``endpoint_match`` (chord endpoints
    coincide), ``merge_compactness`` (``4*pi*A / P^2`` of the closed union
    mask), and ``merge_solidity`` (union area / convex hull area).  No
    coefficients are fitted or shipped; the formula is recorded in
    ``.uns["tiling_stitch"]`` so users can audit and re-derive offline.

    The labels element is **never modified** -- only ``.obs`` columns are
    written.  Materialising a stitched labels element is opt-in via
    :func:`squidpy.experimental.im.make_stitched_labels`.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` with a labels element and a QC
        table from :func:`calculate_tiling_qc`.
    labels_key
        Key in ``sdata.labels``.
    qc_table_key
        Key of the QC table.  Defaults to ``"{labels_key}_qc"``.
    min_confidence
        Threshold on ``stitch_confidence`` (arithmetic mean of the four
        geometric features).  ``0.7`` (default) is a starting point;
        raise it for stricter precision, lower for recall.  Tune for
        your data -- the score is heuristic, not a calibrated probability.
    max_gap
        Maximum perpendicular distance (px) between facing cut edges to be
        considered a candidate pair.
    max_group_size
        Cap on group size; oversized groups (likely false merges) collapse
        to singletons.
    distance_tol
        Sub-pixel tolerance for "lies on a bbox edge" when extracting cut
        edges from each outlier's contour.  Default 0.75 px is tuned for
        ``find_contours``'s half-pixel coordinates.
    min_edge_length
        Absolute floor on cut-edge length (pixels).  Filters out tiny
        cells whose bbox-edge contact is sub-pixel noise.
    min_edge_length_ratio
        Minimum cut-edge length relative to the cell's equivalent
        diameter.  Filters arc-tops on naturally curved cells where the
        bbox-edge contact is a single point.
    min_edge_coverage
        Minimum fraction of integer parallel-axis positions in a
        candidate run that must have at least one near-edge contour
        point.  Filters single-point arc-tops; a real chord passes
        trivially.
    candidate_min_iou
        Loose 1-D IoU floor at candidate enumeration.  Confidence-based
        selection happens later via ``min_confidence``; keep this loose.
    close_radius
        Morphological closing disk radius used when materialising the
        union mask for the shape-quality features.  Should be larger
        than ``max_gap`` so small cells whose gap is a meaningful
        fraction of the cell don't drop the closed-mask features.
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
            distance_tol=distance_tol,
            min_edge_length=min_edge_length,
            min_edge_length_ratio=min_edge_length_ratio,
            min_edge_coverage=min_edge_coverage,
        )
        candidates = _enumerate_pair_candidates(edges, max_gap=max_gap, candidate_min_iou=candidate_min_iou)
        pairs = _score_pairs(
            candidates,
            labels_da,
            bboxes,
            min_confidence=min_confidence,
            close_radius=close_radius,
        )
        groups, confidences = _assemble_groups(pairs, outlier_ids, max_group_size=max_group_size, max_gap=max_gap)

    # Write .obs columns with three states distinguished by stitch_confidence:
    # - non-outlier cell      -> own label_id, False, 1, NaN  (not evaluated)
    # - outlier solo          -> own label_id, False, 1, 1.0  (checked, no partner)
    # - outlier stitched      -> shared root,  True,  n, calibrated P
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

    adata.uns[_METHOD_KEY] = {
        "min_confidence": float(min_confidence),
        "max_gap": float(max_gap),
        "max_group_size": int(max_group_size),
        "distance_tol": float(distance_tol),
        "min_edge_length": float(min_edge_length),
        "min_edge_length_ratio": float(min_edge_length_ratio),
        "min_edge_coverage": float(min_edge_coverage),
        "candidate_min_iou": float(candidate_min_iou),
        "close_radius": int(close_radius),
        "n_outliers": int(n_outliers),
        "n_candidate_pairs": int(len(pairs)),
        "n_stitched_groups": int(n_groups),
        "n_stitched_cells": int(n_stitched),
        "n_pieces_distribution": pieces_dist,
        "score_features": list(_SCORE_FEATURES),
        "score_formula": _SCORE_FORMULA,
    }

    if not inplace:
        return adata
    sdata.tables[table_key] = adata
    return None
