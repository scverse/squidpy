"""Stitching of tile-cut cells flagged by :func:`~squidpy.experimental.tl.calculate_tiling_qc`.

When segmentation is run tile-by-tile (Cellpose, Stardist, Mesmer, ...) cells
that straddle tile boundaries get cut into 2-4 pieces with characteristic
straight, axis-aligned cut edges.  :func:`~squidpy.experimental.tl.calculate_tiling_qc`
flags these (``is_seam_cut`` when seam detection is enabled, else ``is_outlier``).
This module pairs facing cut edges across boundaries and assigns each candidate
pair a heuristic geometric score in [0, 1].

The score is the flat (unweighted) mean of four dataset-independent geometric
features -- ``iou``, ``endpoint_match``, ``merge_compactness`` and
``merge_solidity`` -- computed from the cut-edge geometry and the union mask
after closing the seam gap.  No model is fitted or shipped; the features are
recorded in ``.uns["tiling_stitch"]``.  Users should tune ``min_confidence``
for their data; ``0.6`` is a reasonable starting point, not a calibrated
probability.

Everything is scale-invariant: candidate enumeration is rank-based (k nearest
facing edges, no absolute-pixel search radius), cut-edge lengths are relative to
the data's median cell diameter ``D`` (read from
``.uns["tiling_qc"]["seam_diameter"]``), and how far apart two halves of one cut
may lie is bounded by the width of the seam band they sit on -- measured by
``calculate_tiling_qc``, not assumed.  Cut edges are extracted only on
and facing the seam bands detected by ``calculate_tiling_qc`` -- so pairing no
longer merges touching interior cells, and both halves of a genuine cut are
recovered even when a 1-px segmentation fringe pushes the flat cut inward.

The labels element is **never** modified here -- only ``.obs`` columns are
written.  Materialising a stitched labels element is opt-in via
:func:`!make_stitched_labels`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import spatialdata as sd
import xarray as xr
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from spatialdata._logging import logger as logg

from squidpy.experimental.tl._seam import SeamDetectionParams, SeamScale, cell_flat_edges, seam_offset
from squidpy.experimental.utils._labels import iter_chunked_regionprops, resolve_labels_array
from squidpy.experimental.utils._params import resolve_params

if TYPE_CHECKING:
    from collections.abc import Iterable

    import anndata as ad

__all__ = ["StitchParams", "assign_stitch_groups"]

# The geometric features whose flat mean is the stitch score.  `gap_proximity` was intentionally
# dropped: with a per-pair `close_radius` that bridges each pair's own seam gap, it penalised
# wide-but-genuine seams and lowered recall without helping precision (validated on ground truth).
_SCORE_FEATURES: tuple[str, ...] = ("iou", "endpoint_match", "merge_compactness", "merge_solidity")
# The subset computed by the expensive merge-union step; the rest are cheap
# geometry features known before it, which drives the scoring early-prune.
_SHAPE_FEATURES: tuple[str, ...] = ("merge_compactness", "merge_solidity")


@dataclass(slots=True)
class StitchParams:
    """Advanced tuning knobs for :func:`~squidpy.experimental.tl.assign_stitch_groups`.

    Defaults work for typical 2D segmentation tiles produced by
    cellpose-like pipelines.  Pass an instance (or a ``Mapping`` of
    field names to values) as ``stitch_params`` to override.  These are
    advanced knobs -- the defaults rarely need changing.
    """

    candidate_min_iou: float = 0.2
    """Loose 1-D along-seam IoU floor for a facing edge to be a pair candidate."""

    k_neighbors: int = 5
    """Rank-based candidate cap -- each cut edge is paired only with its ``k`` nearest
    *facing* edges (by perpendicular gap).  Replaces an absolute ``max_gap`` pixel
    threshold, so the search adapts to the dataset's own seam-gap width."""

    close_radius_min: int = 2
    """Floor for the per-pair morphological closing radius.  The effective radius is
    ``max(close_radius_min, ceil(gap / 2) + 1)`` so closing always bridges that pair's
    own seam gap before the union's solidity/compactness are measured."""

    def __post_init__(self) -> None:
        # Coerce numeric types (accept numpy scalars cleanly) and bounds-check.
        self.candidate_min_iou = float(self.candidate_min_iou)
        self.k_neighbors = int(self.k_neighbors)
        self.close_radius_min = int(self.close_radius_min)
        if not 0.0 <= self.candidate_min_iou <= 1.0:
            raise ValueError(f"candidate_min_iou must be in [0, 1], got {self.candidate_min_iou}.")
        if self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {self.k_neighbors}.")
        if self.close_radius_min < 0:
            raise ValueError(f"close_radius_min must be >= 0, got {self.close_radius_min}.")


def _resolve_stitch_params(stitch_params: StitchParams | Mapping[str, Any] | None) -> StitchParams:
    """Normalise the ``stitch_params`` argument to a :class:`StitchParams` instance."""
    return resolve_params(stitch_params, StitchParams, label="stitch_params")


_METHOD_KEY = "tiling_stitch"
_STITCH_DEFAULTS = StitchParams()

# Contract between calculate_tiling_qc and assign_stitch_groups.  _STITCH_COLUMNS
# is the obs columns stitch writes back into the QC table; _STITCH_PARAM_KEYS
# is the subset of top-level kwargs valid for re-running assign_stitch_groups
# (the advanced tuning lives in a nested ``stitch_params`` dict).
_STITCH_COLUMNS = ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence")
_STITCH_PARAM_KEYS = frozenset({"min_confidence", "max_group_size"})


# Dataclasses


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

    ``confidence`` is the flat mean of the geometric features (see
    :data:`_SCORE_FEATURES`); the individual feature components are kept for
    diagnostics and for the ``min``-based group-confidence aggregation.
    """

    cell_a: int
    cell_b: int
    axis: str
    confidence: float
    iou: float
    endpoint_match: float
    merge_solidity: float
    merge_compactness: float
    edge_a: _CutEdge | None = field(default=None, repr=False)
    edge_b: _CutEdge | None = field(default=None, repr=False)


# Cut-edge extraction


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


def _extract_cut_edges(
    labels_da: xr.DataArray | np.ndarray,
    outlier_ids: Iterable[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    seams: dict[str, list[tuple[float, float, int]]],
    scale: SeamScale,
) -> tuple[list[_CutEdge], dict[int, np.ndarray]]:
    """Extract cut edges on and facing the detected seam bands, per outlier cell.

    For each cut cell we take its dominant flat boundary line on each side
    (:func:`~squidpy.experimental.tl._seam.cell_flat_edges` -- fringe-robust: the
    flat line is found at *any* coordinate, not pinned to the bbox extreme, so a
    1-px segmentation fringe no longer drops the edge) and keep only edges that
    lie on a detected seam band and face it.  This mirrors the ``is_seam_cut``
    flagging in :func:`~squidpy.experimental.tl.calculate_tiling_qc`, so the two
    stages agree on which edges are seam cuts.

    ``scale`` is the detection scale rehydrated from ``.uns["tiling_qc"]``, so the
    thresholds here are literally the ones the cells were flagged with.

    Returns
    -------
    The list of cut edges and, as a by-product of the per-cell crop already read
    here, a ``{label_id -> boolean bbox mask}`` dict that lets the scoring pass
    reconstruct merge unions in memory without re-reading the labels array.
    """
    probe_depth = scale.probe_depth

    edges: list[_CutEdge] = []
    outlier_crops: dict[int, np.ndarray] = {}
    for lid in [int(x) for x in outlier_ids]:
        bbox = bboxes.get(lid)
        if bbox is None:
            continue
        min_r, min_c, max_r, max_c = bbox
        # Read the bbox padded by the probe depth: cell_flat_edges probes background
        # *beyond* the cut edge to measure the gap, so it needs the neighbourhood.
        pad = probe_depth + 2
        r0 = max(0, min_r - pad)
        c0 = max(0, min_c - pad)
        crop = _read_bbox_slice(labels_da, r0, max_r + pad, c0, max_c + pad)
        h = max_r - min_r
        w = max_c - min_c
        by0 = min_r - r0
        bx0 = min_c - c0
        # Slice to the cell's own bbox before comparing: the crop is padded by the probe depth,
        # so comparing first would test several times the pixels that are kept.
        cell_mask = crop[by0 : by0 + h, bx0 : bx0 + w] == lid  # boolean bbox mask; reused by scoring
        if not cell_mask.any():
            continue
        outlier_crops[lid] = cell_mask

        flat = cell_flat_edges(
            cell_mask,
            crop != 0,  # occupancy: this crop is read unmasked, so neighbours are visible
            (by0, bx0, by0 + h, bx0 + w),
            (r0, c0),
            scale.min_len,
            scale.flat_tol,
            probe_depth,
        )
        for e in flat:
            # Keep only edges lying on a detected seam band and facing it -- the *same*
            # predicate `calculate_tiling_qc` flags cells with, so the two stages agree by
            # construction rather than by comment.
            if seam_offset(e, seams.get(e["axis"], []), scale) is None:
                continue
            edges.append(
                _CutEdge(
                    cell_id=lid,
                    axis=e["axis"],
                    coord=e["coord"],
                    extent=e["extent"],
                    normal_dir=e["side"],
                    length=float(e["span"]),
                )
            )

    return edges, outlier_crops


# Pair candidate enumeration + features


def _extent_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _merge_shape_features(
    cell_a: int,
    cell_b: int,
    bboxes: dict[int, tuple[int, int, int, int]],
    outlier_crops: dict[int, np.ndarray],
    close_radius: int,
    *,
    H: int,
    W: int,
) -> dict[str, float]:
    """Reconstruct the union of two pieces, close the gap, and return shape stats.

    Solidity (area / convex_hull_area) and compactness (4*pi*A / P^2) drop
    sharply when two unrelated cells are joined -- the union is concave at the
    join.  ``merge_compactness`` is typically the strongest single
    discriminator between true cuts and false merges.

    The union mask is assembled in memory from the per-cell boolean crops
    already collected by :func:`_extract_cut_edges`, so this never re-reads the
    (possibly dask-backed) labels array -- which was the hot-loop cost, as the
    old version fetched a crop once per candidate pair.
    """
    zero = {"merge_solidity": 0.0, "merge_compactness": 0.0}
    if cell_a not in bboxes or cell_b not in bboxes:
        return zero
    if cell_a not in outlier_crops or cell_b not in outlier_crops:
        return zero

    r0a, c0a, r1a, c1a = bboxes[cell_a]
    r0b, c0b, r1b, c1b = bboxes[cell_b]
    # Padded + border-clamped union bbox. Identical bounds to the old single
    # `np.isin` crop, so the reconstructed mask matches it pixel-for-pixel.
    pad = close_radius + 2
    r0 = max(min(r0a, r0b) - pad, 0)
    c0 = max(min(c0a, c0b) - pad, 0)
    r1 = min(max(r1a, r1b) + pad, H)
    c1 = min(max(c1a, c1b) + pad, W)

    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    # Place each cell's pre-fetched bbox mask at its offset within the union.
    mask[r0a - r0 : r1a - r0, c0a - c0 : c1a - c0] |= outlier_crops[cell_a]
    mask[r0b - r0 : r1b - r0, c0b - c0 : c1b - c0] |= outlier_crops[cell_b]
    if not mask.any():
        return zero

    # Closing by a disk of radius r, via distance transforms: pixel-identical to
    # `binary_closing(mask, disk(r))` but O(N) instead of O(N * r^2).  The radius tracks each
    # pair's own seam gap, so the explicit-footprint form got dramatically more expensive
    # exactly on the wide-gap data this feature targets.
    closed = distance_transform_edt(distance_transform_edt(~mask) <= close_radius) > close_radius
    cc = cc_label(closed, connectivity=2)
    if cc.max() == 0:
        return zero
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
    candidate_min_iou: float = _STITCH_DEFAULTS.candidate_min_iou,
) -> dict[str, float] | None:
    """Compute geometry-only features for a candidate pair, returning ``None``
    if the pair fails the basic facing/overlap/IoU filters.

    No absolute-pixel gap cutoff is applied here -- the raw perpendicular ``gap``
    is returned so the (rank-based) enumerator can pick each edge's nearest
    facing partners, and the diameter-relative plausibility guard is applied in
    :func:`_score_pairs`.
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
    endpoint_dist = abs(e.extent[0] - c.extent[0]) + abs(e.extent[1] - c.extent[1])
    max_len = max(e.length, c.length)
    endpoint_match = max(0.0, 1.0 - endpoint_dist / max_len) if max_len > 0 else 0.0
    return {
        "iou": float(iou),
        "endpoint_match": float(endpoint_match),
        "gap": float(gap),
    }


def _enumerate_pair_candidates(
    edges: list[_CutEdge],
    k_neighbors: int = _STITCH_DEFAULTS.k_neighbors,
    candidate_min_iou: float = _STITCH_DEFAULTS.candidate_min_iou,
) -> list[tuple[_CutEdge, _CutEdge, dict[str, float]]]:
    """Find candidate pairs of facing cut edges, rank-based (no absolute max_gap).

    For each edge, keep only its ``k_neighbors`` nearest *facing + overlapping*
    edges by perpendicular gap.  A rank-based cap adapts to the dataset's own
    seam-gap width -- unlike an absolute pixel radius, which fails when the
    inter-FOV gap is wider than a hand-tuned constant.  Each unordered
    ``(cell_a, cell_b, axis)`` pair is emitted once.  No scoring yet.
    """
    out: dict[tuple[int, int, str], tuple[_CutEdge, _CutEdge, dict[str, float]]] = {}
    by_axis: dict[str, list[_CutEdge]] = {"h": [], "v": []}
    for e in edges:
        by_axis[e.axis].append(e)

    for axis_edges in by_axis.values():
        for e in axis_edges:
            facing: list[tuple[float, _CutEdge, dict[str, float]]] = []
            for c in axis_edges:
                if c is e or c.cell_id == e.cell_id:
                    continue
                feats = _pair_geometry_features(e, c, candidate_min_iou=candidate_min_iou)
                if feats is None:
                    continue
                facing.append((feats["gap"], c, feats))
            facing.sort(key=lambda t: t[0])
            for _g, c, feats in facing[:k_neighbors]:
                key = (min(e.cell_id, c.cell_id), max(e.cell_id, c.cell_id), e.axis)
                if key not in out:
                    out[key] = (e, c, feats)
    return list(out.values())


# Scoring


def _score_pair_features(features: dict[str, float]) -> float:
    """Return the heuristic stitch score in [0, 1].

    Flat (unweighted) mean of the four features in :data:`_SCORE_FEATURES`.
    The score is dataset-independent and not a calibrated probability -- users
    pick ``min_confidence`` based on their false-merge tolerance.
    """
    return float(sum(features[name] for name in _SCORE_FEATURES) / len(_SCORE_FEATURES))


def _max_achievable_score(known_features: dict[str, float]) -> float:
    """Upper bound on the stitch score from the cheap geometry features alone.

    The deferred shape features (:data:`_SHAPE_FEATURES`) are each in ``[0, 1]``,
    so assume their best case. Built on :func:`_score_pair_features` so the bound
    can never drift from the real score if the feature set or weighting changes.
    """
    return _score_pair_features({**known_features, **dict.fromkeys(_SHAPE_FEATURES, 1.0)})


def _seam_span(seams: dict[str, list[tuple[float, float, int]]], axis: str, coord_a: float, coord_b: float) -> float:
    """Widest plausible separation between two halves of one cut, from the measured seam band.

    Both halves sit on the inner edges of the same band, so their perpendicular separation
    cannot exceed that band's width.  Tying the bound to the detected seam rather than to a
    multiple of the cell diameter also bounds the per-pair closing radius, which otherwise
    grows with the gap until it bridges pieces that were never one cell.
    """
    bands = seams.get(axis, [])
    if not bands:
        return 0.0
    mid = (coord_a + coord_b) / 2.0
    _centre, half, _cnt = min(bands, key=lambda b: abs(mid - b[0]))
    return 2.0 * half


def _score_pairs(
    candidates: list[tuple[_CutEdge, _CutEdge, dict[str, float]]],
    bboxes: dict[int, tuple[int, int, int, int]],
    outlier_crops: dict[int, np.ndarray],
    min_confidence: float,
    seams: dict[str, list[tuple[float, float, int]]],
    *,
    close_radius_min: int = _STITCH_DEFAULTS.close_radius_min,
    H: int,
    W: int,
) -> list[_StitchPair]:
    """Compute shape features per candidate, score, and keep pairs >= min_confidence.

    The morphological closing radius is chosen *per pair* to bridge that pair's
    own seam gap (``max(close_radius_min, ceil(gap / 2) + 1)``), and candidates
    separated by more than the width of the seam band they lie on are discarded
    (:func:`_seam_span`), which also bounds that radius.  One entry per
    ``(cell_a, cell_b, axis)`` (keeping max confidence on duplicates).
    """
    scored: list[_StitchPair] = []
    for e, c, geom in candidates:
        gap = geom["gap"]
        if gap > _seam_span(seams, e.axis, e.coord, c.coord):  # wider than its own seam: not one cut
            continue
        # Per-pair closing radius bridges this pair's own gap; the gap is already bounded by
        # the seam band's width above, so the radius needs no separate cap.
        close_radius = max(close_radius_min, int(np.ceil(gap / 2.0)) + 1)
        # Skip the costly union reconstruction when even the best case for the
        # deferred shape features can't reach min_confidence.
        if _max_achievable_score(geom) < min_confidence:
            continue
        shape = _merge_shape_features(e.cell_id, c.cell_id, bboxes, outlier_crops, close_radius=close_radius, H=H, W=W)
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
                merge_solidity=feats["merge_solidity"],
                merge_compactness=feats["merge_compactness"],
                edge_a=ea,
                edge_b=eb,
            )
        )

    return sorted(scored, key=lambda p: (-p.confidence, p.cell_a, p.cell_b))


# Group assembly (union-find + validation)


def _validate_group_geometry(
    pairs_in_group: list[_StitchPair],
    size: int,
    gap_tol: float,
) -> bool:
    """Geometric sanity check for groups of size >= 3.

    Two cases:

    - **Corner group** (size 4, both axes present): the cut edges' endpoints
      must converge near a single junction point (one ``h`` cut crossing one
      ``v`` cut defines the junction).  If the spread of edge extents from
      the junction is greater than ``gap_tol``, the group is implausible.

    - **Chain group** (size 3 or 4, all pairs share one axis): legitimate
      same-axis chains (e.g., a cell split by 3 horizontal seams into 4
      vertically-stacked pieces) have pairs at N-1 *distinct* seam
      coordinates.  Multiple pairs at the same seam coord would imply
      geometrically impossible "two cuts at the same seam" pairings -- a
      signature of a false-positive cluster -- so we reject.

    ``gap_tol`` is the width of the widest detected seam band, used consistently
    with the per-pair bound applied during candidate scoring.
    """
    h_pairs = [p for p in pairs_in_group if p.axis == "h"]
    v_pairs = [p for p in pairs_in_group if p.axis == "v"]

    # Chain case: only one axis present and size >= 3.
    if not h_pairs or not v_pairs:
        if size < 3:
            return True  # 2-piece groups are trivially valid on one axis
        # Each pair's seam coord is roughly midway between its two edges.
        seam_coords = [round((p.edge_a.coord + p.edge_b.coord) / 2.0, 1) for p in pairs_in_group]
        # Allow a gap_tol-sized tolerance for "distinct" seams.
        sorted_coords = sorted(seam_coords)
        for prev, cur in zip(sorted_coords, sorted_coords[1:], strict=False):
            if cur - prev <= gap_tol:
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
        if min(abs(e.extent[0] - junction_x), abs(e.extent[1] - junction_x)) > gap_tol:
            return False
    for e in v_edges:
        if min(abs(e.extent[0] - junction_y), abs(e.extent[1] - junction_y)) > gap_tol:
            return False
    return True


def _assemble_groups(
    pairs: list[_StitchPair],
    candidate_ids: Iterable[int],
    max_group_size: int,
    gap_tol: float,
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
        if size >= 3 and not _validate_group_geometry(group_pairs, size, gap_tol):
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


# Public entry point


def assign_stitch_groups(
    sdata: sd.SpatialData,
    labels_key: str,
    qc_table_key: str | None = None,
    min_confidence: float = 0.6,
    max_group_size: int = 4,
    candidates: Literal["auto", "is_seam_cut", "is_outlier"] = "auto",
    stitch_params: StitchParams | Mapping[str, Any] | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Assign tile-cut cell pieces to stitch groups.

    Reads the cells flagged by :func:`~squidpy.experimental.tl.calculate_tiling_qc`
    (``is_seam_cut`` by default), extracts each piece's cut edges *on and facing*
    the detected FOV seam bands, pairs facing edges across boundaries (rank-based:
    each edge with its ``k`` nearest facing partners -- no absolute-pixel search
    radius), scores each pair via a transparent geometric composite, and assembles
    high-confidence pairs into stitch groups via union-find.  This only *annotates*
    which pieces belong together -- it does **not** modify the labels element.
    Materialising a stitched labels element is opt-in via :func:`!make_stitched_labels`.

    The score per pair is the flat (unweighted) mean of four geometric features
    in [0, 1]: ``iou`` (1-D extent overlap), ``endpoint_match`` (chord endpoints
    coincide), ``merge_compactness`` (``4*pi*A / P^2`` of the closed union mask)
    and ``merge_solidity`` (union area / convex hull area).  No coefficients are
    fitted or shipped; the features are recorded in ``.uns["tiling_stitch"]``.

    **Requires seam detection.**  Run ``calculate_tiling_qc(..., detect_seams=True)``
    first: the seam bands and data length scale ``D`` it records in
    ``.uns["tiling_qc"]`` localise cut-edge extraction to real seams, so pairing
    no longer merges touching interior cells.

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
        Threshold on ``stitch_confidence``.  ``0.6`` (default) is a starting
        point; raise it for stricter precision, lower for recall.  Tune for
        your data -- the score is heuristic, not a calibrated probability.
    max_group_size
        Cap on group size; oversized groups (likely false merges) collapse
        to singletons.
    candidates
        Which QC column gates the cells considered for stitching.  ``"auto"``
        (default) uses ``is_seam_cut`` when present -- these are localised to
        detected FOV seams -- and otherwise falls back to ``is_outlier``.  Set
        explicitly to ``"is_seam_cut"`` or ``"is_outlier"`` to force one.
    stitch_params
        Advanced tuning knobs as a :class:`StitchParams` instance or a
        ``Mapping`` of its field names to values.  See :class:`StitchParams`
        for each field's meaning and default.  ``None`` (default) uses
        all defaults.
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
    if max_group_size < 1:
        raise ValueError(f"max_group_size must be >= 1, got {max_group_size}.")
    params = _resolve_stitch_params(stitch_params)

    table_key = qc_table_key if qc_table_key is not None else f"{labels_key}_qc"
    if table_key not in sdata.tables:
        raise ValueError(f"QC table '{table_key}' not found.  Run calculate_tiling_qc first.")
    adata = sdata.tables[table_key].copy()

    if "label_id" not in adata.obs.columns:
        raise ValueError(f"QC table '{table_key}' is missing 'label_id'.")
    # Candidate gate: prefer the seam-aware `is_seam_cut` flag (localised to detected FOV seams,
    # so pairing no longer merges touching interior cells); fall back to the MAD `is_outlier`.
    # Seam data is mandatory below, so `is_seam_cut` is always present under "auto".
    gate_col = "is_seam_cut" if candidates == "auto" else candidates
    if gate_col not in adata.obs.columns:
        raise ValueError(
            f"QC table '{table_key}' is missing '{gate_col}'; re-run calculate_tiling_qc "
            f"(with detect_seams=True for 'is_seam_cut')."
        )

    existing = [c for c in _STITCH_COLUMNS if c in adata.obs.columns]
    if existing:
        logg.warning(f"Overwriting existing stitch columns: {existing}.")
        adata.obs.drop(columns=existing, inplace=True)

    # Seam contract from calculate_tiling_qc(detect_seams=True): the seam bands localise
    # cut-edge extraction and `seam_diameter` sets every length scale.  The extraction
    # fractions mirror detection (read from seam_params) so both stages agree on seam cuts.
    qc_params = adata.uns.get("tiling_qc", {})
    scale = qc_params.get("scale")
    seams_uns = qc_params.get("seams")
    diameter = qc_params.get("seam_diameter")
    if not seams_uns or diameter is None:
        raise ValueError(
            f"QC table '{table_key}' has no seam detection results; assign_stitch_groups requires "
            f"seam-localised cut edges.  Re-run calculate_tiling_qc(..., detect_seams=True)."
        )
    diameter = float(diameter)
    seams = {
        axis: [(float(b["coord"]), float(b["half_width"]), int(b["n_edges"])) for b in seams_uns.get(axis, [])]
        for axis in ("v", "h")
    }
    # Rehydrate the exact parameters detection ran with, and resolve them against the same D.
    seam_scale = resolve_params(qc_params.get("seam_params"), SeamDetectionParams, label="seam_params")._resolve(
        diameter
    )

    labels_da = resolve_labels_array(sdata, labels_key, scale)

    label_ids = adata.obs["label_id"].astype(int).to_numpy()
    is_outlier = adata.obs[gate_col].to_numpy(dtype=bool)
    outlier_ids = label_ids[is_outlier].tolist()

    n_outliers = len(outlier_ids)
    logg.info(f"Stitching {n_outliers} candidate cells ('{gate_col}', out of {len(label_ids)} total).")

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
        edges, outlier_crops = _extract_cut_edges(labels_da, outlier_ids, bboxes, seams, seam_scale)
        H, W = labels_da.shape[-2], labels_da.shape[-1]
        cand = _enumerate_pair_candidates(
            edges, k_neighbors=params.k_neighbors, candidate_min_iou=params.candidate_min_iou
        )
        pairs = _score_pairs(
            cand,
            bboxes,
            outlier_crops,
            min_confidence,
            seams,
            close_radius_min=params.close_radius_min,
            H=H,
            W=W,
        )
        # Tolerance for "distinct seams" / corner convergence: the widest detected band.
        gap_tol = max(
            (2.0 * half for bands in seams.values() for _c, half, _n in bands),
            default=0.0,
        )
        groups, confidences = _assemble_groups(pairs, outlier_ids, max_group_size=max_group_size, gap_tol=gap_tol)

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

    adata.uns[_METHOD_KEY] = {
        "min_confidence": float(min_confidence),
        "max_group_size": int(max_group_size),
        "candidate_gate": gate_col,
        "seam_diameter": float(diameter),
        "stitch_params": asdict(params),
        "n_outliers": int(n_outliers),
        "n_candidate_pairs": int(len(pairs)),
        "n_stitched_groups": int(n_groups),
        "n_stitched_cells": int(n_stitched),
        "n_pieces_distribution": pieces_dist,
        "score_features": list(_SCORE_FEATURES),
    }

    if not inplace:
        return adata
    sdata.tables[table_key] = adata
    return None
