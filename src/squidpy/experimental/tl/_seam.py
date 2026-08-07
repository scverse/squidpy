"""Emergent-seam detection for tile-boundary cut cells.

The MAD-based ``is_outlier`` gate in :func:`~squidpy.experimental.tl.calculate_tiling_qc`
flags cells whose boundary is unusually straight.  In dense tissue with a wide inter-FOV
gap -- and when a cut leaves only one segmentable half -- that signal is swamped: most real
cut cells are missed and interior cells are flagged instead.

This module adds a complementary, geometry-only detector that needs neither the FOV size
nor tile overlap, and works on single-sided cuts:

1. For every cell, collect **all** long, cardinal (axis-aligned) flat boundary runs.  Their
   coordinates are the seam-cut candidates.
2. Seam lines **emerge** as the coordinates where many such runs align (a consensus over
   many cells), with a wide inter-FOV gap spreading a seam into a band whose width is read
   off from the peak spread -- recovering the seam grid from the data alone.
3. A cell is flagged ``is_seam_cut`` iff any of its cardinal edges lies in a detected seam
   band and faces it.

**No absolute-pixel thresholds.**  Every length scale is expressed as a ratio and resolved
at runtime against the data's own length scale ``D`` (the median cell equivalent diameter)
and the *observed* inter-cell gap distribution, so the same defaults transfer across
resolutions, cell sizes and FOV pitches.  See :class:`SeamDetectionParams`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks


@dataclass(slots=True, frozen=True)
class SeamDetectionParams:
    """Scale-invariant tuning knobs for emergent-seam cut-cell detection.

    Every knob is a **dimensionless ratio** (or a relative multiple), resolved at runtime
    against the data's length scale ``D`` (median cell equivalent diameter) and the observed
    background -- so the defaults transfer across datasets without re-tuning absolute pixels.
    """

    edge_len_frac: float = 0.5
    """Minimum cardinal flat-run length, as a fraction of the median cell diameter ``D``.
    A genuine tile cut leaves a straight edge spanning a good fraction of the cell; short
    facets fall below this and are ignored, which keeps precision up."""

    flat_tol: float = 1.5
    """Flatness tolerance (px): max deviation from a single coordinate for a run to be 'flat'.
    A rasterisation / pixel-grid constant (~1px), independent of cell size or resolution --
    scaling it with the cell size would accept curved edges on large cells."""

    probe_frac: float = 0.6
    """How far to probe for background beyond an edge, as a fraction of ``D``."""

    gap_membrane_mult: float = 2.5
    """An edge counts as a seam-candidate (wide gap) if the background beyond it exceeds this
    multiple of the *observed* typical inter-cell membrane width (data-derived, not a constant)."""

    bin_width: float = 2.0
    """Seam histogram bin width (px): a pixel-grid resolution constant, independent of cell size."""

    bg_multiple: float = 4.0
    """A seam peak must exceed this multiple of the robust background histogram level (relative)."""

    min_seam_edge_frac: float = 0.01
    """A seam band must collect at least this fraction of the axis' wide-gap edges (low floor)."""

    seam_strength_frac: float = 0.25
    """A seam peak must be at least this fraction of the *strongest* peak on its axis.  Real seams
    collect many aligned edges; a stray straight membrane makes only a small peak, so this
    (data-relative, scale-invariant) test rejects it whether there is one seam or many."""

    cluster_frac: float = 1.4
    """Histogram peaks within this multiple of ``D`` merge into one seam band (spans the gap)."""

    band_margin_frac: float = 0.15
    """Extra seam-band half-width, as a fraction of ``D``."""

    flag_tol_frac: float = 0.25
    """Slack added to the band half-width when testing edge membership, as a fraction of ``D``."""

    face_slack_frac: float = 0.15
    """Slack allowing an edge slightly past the band centre to still face the seam, fraction of ``D``."""

    def __post_init__(self) -> None:
        for name in (
            "edge_len_frac",
            "flat_tol",
            "probe_frac",
            "gap_membrane_mult",
            "bin_width",
            "bg_multiple",
            "min_seam_edge_frac",
            "seam_strength_frac",
            "cluster_frac",
            "band_margin_frac",
            "flag_tol_frac",
            "face_slack_frac",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.edge_len_frac <= 0:
            raise ValueError(f"edge_len_frac must be > 0, got {self.edge_len_frac}.")
        if not 0.0 <= self.min_seam_edge_frac <= 1.0:
            raise ValueError(f"min_seam_edge_frac must be in [0, 1], got {self.min_seam_edge_frac}.")
        if self.probe_frac <= 0 or self.gap_membrane_mult <= 0:
            raise ValueError("probe_frac and gap_membrane_mult must be > 0.")


_SEAM_DEFAULTS = SeamDetectionParams()


@dataclass(frozen=True)
class SeamScale:
    """Pixel thresholds resolved from the data's own length scale ``D`` and gap statistics."""

    diameter: float  # median cell equivalent diameter (px)
    membrane: float  # observed typical inter-cell gap / membrane width (px)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return ``(start, length)`` of the longest run of ``True`` in a 1-D boolean array."""
    best_s = best_len = 0
    s = None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s > best_len:
                best_s, best_len = s, i - s
            s = None
    if s is not None and len(mask) - s > best_len:
        best_s, best_len = s, len(mask) - s
    return best_s, best_len


def _dominant_flat_line(extreme: np.ndarray, present: np.ndarray, flat_tol: float) -> tuple[float, int, int]:
    """Longest flat run on one side of a cell, at *any* coordinate (not just the extreme).

    A cut leaves a straight boundary that is a plateau in the per-index extreme coordinate --
    but that plateau need not be at the cell's outermost point (a wide cell can have a partial
    cut plus other geometry).  Scan the distinct extreme values and return the one whose
    within-``flat_tol`` run of consecutive present indices is longest: ``(coord, start, length)``.
    """
    vals = extreme.copy()
    best_coord, best_s, best_len = 0.0, 0, 0
    for c in np.unique(np.round(vals[present])):
        on = present & (np.abs(vals - c) <= flat_tol)
        s, ln = _longest_true_run(on)
        if ln > best_len:
            best_coord, best_s, best_len = float(c), s, ln
    return best_coord, best_s, best_len


def _probe_gap(
    tile_labels: np.ndarray, axis: str, coord: int, side: int, run_lo: int, run_hi: int, probe_depth: int
) -> float:
    """Median count of consecutive background pixels just outside a flat edge (tile-local coords)."""
    height, width = tile_labels.shape
    step = 1 if side == -1 else -1
    idxs = np.linspace(run_lo, run_hi - 1, min(9, max(1, run_hi - run_lo))).astype(int)
    depths = []
    for t in idxs:
        d = 0
        for k in range(1, probe_depth + 1):
            y, x = (t, coord + step * k) if axis == "v" else (coord + step * k, t)
            if not (0 <= y < height and 0 <= x < width) or tile_labels[y, x] != 0:
                break
            d += 1
        depths.append(d)
    return float(np.median(depths)) if depths else 0.0


def cell_flat_edges(
    mask: np.ndarray,
    tile_labels: np.ndarray,
    bbox: tuple[int, int, int, int],
    origin: tuple[int, int],
    min_len: int,
    flat_tol: float,
    probe_depth: int,
) -> list[dict[str, Any]]:
    """All cardinal flat boundary runs of one cell (multiple per cell; global coordinates).

    Returns a list of ``{"axis": "v"|"h", "coord": float, "span": int, "side": +1|-1, "gap": float}``
    -- one per side (right/left/bottom/top) whose longest flat run reaches ``min_len``.  ``gap`` is
    the background depth just beyond the edge (used later to separate seam cuts from touching facets).
    Length filtering against the data scale and gap thresholding are applied by the caller.
    """
    y0, x0, _, _ = bbox
    oy, ox = origin
    height, width = mask.shape
    rows = mask.any(1)
    cols = mask.any(0)
    rightmost = np.where(rows, width - 1 - mask[:, ::-1].argmax(1), np.nan).astype(float)
    leftmost = np.where(rows, mask.argmax(1), np.nan).astype(float)
    bottommost = np.where(cols, height - 1 - mask[::-1, :].argmax(0), np.nan).astype(float)
    topmost = np.where(cols, mask.argmax(0), np.nan).astype(float)

    out: list[dict[str, Any]] = []
    for axis, extreme, present, side in [
        ("v", rightmost, rows, -1),
        ("v", leftmost, rows, +1),
        ("h", bottommost, cols, -1),
        ("h", topmost, cols, +1),
    ]:
        if int(present.sum()) < min_len:
            continue
        c, s, ln = _dominant_flat_line(extreme, present, flat_tol)
        if ln < min_len:
            continue
        c_round = int(round(c))
        if axis == "v":
            perp_local, run_lo, run_hi = x0 + c_round, y0 + s, y0 + s + ln
            coord_global = ox + perp_local
        else:
            perp_local, run_lo, run_hi = y0 + c_round, x0 + s, x0 + s + ln
            coord_global = oy + perp_local
        gap = _probe_gap(tile_labels, axis, perp_local, side, run_lo, run_hi, probe_depth)
        out.append({"axis": axis, "coord": float(coord_global), "span": int(ln), "side": int(side), "gap": gap})
    return out


def detect_seams(
    edges: list[dict[str, Any]],
    extent_x: int,
    extent_y: int,
    scale: SeamScale,
    params: SeamDetectionParams = _SEAM_DEFAULTS,
) -> dict[str, list[tuple[float, float, int]]]:
    """Locate seam bands per axis from wide-gap cut-edges (all thresholds derived from ``scale``).

    Returns ``{"v": [(centre, half_width, count), ...], "h": [...]}``.
    """
    d = scale.diameter
    bin_w = params.bin_width
    cluster_gap = params.cluster_frac * d
    band_margin = params.band_margin_frac * d
    gap_thresh = params.gap_membrane_mult * scale.membrane

    out: dict[str, list[tuple[float, float, int]]] = {}
    for axis, extent in (("v", extent_x), ("h", extent_y)):
        coords = np.array([e["coord"] for e in edges if e["axis"] == axis and e["gap"] >= gap_thresh])
        if coords.size == 0:
            out[axis] = []
            continue
        # Floor is a fraction of the wide-gap edges ON THIS AXIS (the ones that can form a seam),
        # not of all edges -- so it stays comparable to a real seam's edge count at any scale.
        min_count = max(3.0, params.min_seam_edge_frac * coords.size)
        bins = np.arange(0, extent + bin_w, bin_w)
        hist, _ = np.histogram(coords, bins=bins)
        # Smooth only for sub-pixel wobble (small window); the CLUSTER step below is what
        # widens a seam across the inter-FOV gap -- over-smoothing here collapses the gap's
        # twin peaks into one narrow band and loses the far-side cells.
        win = max(1, int(round((0.12 * d) / bin_w)))
        smooth = np.convolve(hist, np.ones(2 * win + 1), mode="same")
        bg = np.median(smooth[smooth > 0]) if (smooth > 0).any() else 0.0
        height = max(min_count, params.bg_multiple * bg)
        peaks, _ = find_peaks(smooth, height=height, distance=1)
        if peaks.size == 0:
            out[axis] = []
            continue
        # keep only peaks strong relative to the strongest on this axis (rejects stray membranes)
        peak_heights = smooth[peaks]
        peaks = peaks[peak_heights >= params.seam_strength_frac * peak_heights.max()]
        if peaks.size == 0:
            out[axis] = []
            continue
        centres = bins[peaks] + bin_w / 2
        counts = smooth[peaks].astype(float)
        order = np.argsort(centres)
        centres, counts = centres[order], counts[order]
        clusters, cur = [], [0]
        for i in range(1, len(centres)):
            if centres[i] - centres[cur[-1]] <= cluster_gap:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)
        bands = []
        for cl in clusters:
            cc, ww = centres[cl], counts[cl]
            centre = float(np.average(cc, weights=ww))
            half = float((cc.max() - cc.min()) / 2 + band_margin)
            bands.append((centre, half, int(ww.sum())))
        out[axis] = bands
    return out


def flag_cells_on_seams(
    edges_by_cell: dict[int, list[dict[str, Any]]],
    seams: dict[str, list[tuple[float, float, int]]],
    scale: SeamScale,
    params: SeamDetectionParams = _SEAM_DEFAULTS,
) -> dict[int, float]:
    """Return ``{cell_id: seam_dist}`` for every cell with a cardinal edge on and facing a seam.

    Once a seam is detected, membership does **not** require a wide gap -- a genuine two-sided
    cut whose other half sits close still counts, because its edge lies on the seam consensus.
    """
    flag_tol = params.flag_tol_frac * scale.diameter
    face_slack = params.face_slack_frac * scale.diameter
    flagged: dict[int, float] = {}
    for cid, edges in edges_by_cell.items():
        best = None
        for e in edges:
            for centre, half, _cnt in seams[e["axis"]]:
                signed = centre - e["coord"]
                if abs(signed) > half + flag_tol:
                    continue
                faces = (signed >= -face_slack) if e["side"] == -1 else (signed <= face_slack)
                if faces and (best is None or abs(signed) < best):
                    best = abs(signed)
        if best is not None:
            flagged[cid] = best
    return flagged


def estimate_membrane_width(gaps: np.ndarray) -> float:
    """Typical inter-cell membrane width from the distribution of edge background depths.

    Touching cells share a thin membrane (small gap); true seams / borders show large gaps.
    The lower mode of the gap distribution is the membrane width.  Returns >= 1px.
    """
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size == 0:
        return 1.0
    # the membrane is the typical *small* gap: use the median of the lower half.
    lower = gaps[gaps <= np.median(gaps)]
    return float(max(1.0, np.median(lower) if lower.size else np.median(gaps)))
