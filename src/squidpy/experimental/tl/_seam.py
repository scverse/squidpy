"""Emergent-seam detection for tile-boundary cut cells.

The MAD-based ``is_outlier`` gate in :func:`~squidpy.experimental.tl.calculate_tiling_qc`
flags cells whose boundary is unusually straight.  In dense tissue with a wide inter-FOV
gap -- and when a cut leaves only one segmentable half -- that signal is swamped: most
real cut cells are missed and interior cells are flagged instead.

This module adds a complementary, geometry-only detector that needs neither the FOV size
nor tile overlap, and works on single-sided cuts:

1. For every cell, find its longest **cardinal** (axis-aligned) flat boundary run, keeping
   only runs with a **wide background gap beyond them** (a real seam cut faces a gap; a
   dense-tissue facet faces a neighbouring cell one pixel away).
2. Seam lines **emerge** as the coordinates where many such edges pile up.  A wide gap
   spreads a seam's edges into a band, so nearby peaks are clustered and the band width is
   read off from the spread -- recovering the seam geometry from the data alone.
3. A cell is flagged ``is_seam_cut`` iff its flat edge lies in a detected seam band and
   faces it.

See :class:`SeamDetectionParams` for the tuning knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks


@dataclass(slots=True, frozen=True)
class SeamDetectionParams:
    """Tuning knobs for emergent-seam cut-cell detection.

    Defaults work for typical CosMx-style per-FOV segmentation.  All lengths are in pixels
    at the analysis resolution.  Frozen so validation in ``__post_init__`` cannot be
    silently bypassed.
    """

    min_edge_len: int = 6
    """Minimum length (px) of a cardinal flat boundary run to count as a candidate edge."""

    flat_tol: float = 1.5
    """Max deviation (px) from a single coordinate for the run to stay 'flat' (rasterisation slack)."""

    min_edge_frac: float = 0.4
    """Flat run must cover at least this fraction of the cell's extent along the run axis."""

    min_bg_depth: float = 3.0
    """Minimum background depth (px) just beyond the edge -- separates a seam gap from a 1px membrane."""

    probe_depth: int = 8
    """How far (px) to probe for background beyond an edge.  Kept small so it stays within the tile margin."""

    bin_width: float = 2.0
    """Histogram bin width (px) for locating seam coordinates."""

    min_seam_count: int = 12
    """A seam band must collect at least this many cut-edges (absolute floor)."""

    bg_multiple: float = 4.0
    """A seam peak must exceed this multiple of the robust background histogram level."""

    cluster_gap: float = 22.0
    """Peaks within this distance (px) merge into one seam band (spans the inter-FOV gap)."""

    band_margin: float = 3.0
    """Extra half-width (px) added to each seam band."""

    flag_tol: float = 4.0
    """Slack (px) added to the band half-width when testing edge membership."""

    face_slack: float = 3.0
    """Slack (px) allowing an edge slightly past the band centre to still count as facing the seam."""

    def __post_init__(self) -> None:
        for name in ("min_edge_len", "probe_depth", "min_seam_count"):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in (
            "flat_tol",
            "min_edge_frac",
            "min_bg_depth",
            "bin_width",
            "bg_multiple",
            "cluster_gap",
            "band_margin",
            "flag_tol",
            "face_slack",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.min_edge_len < 3:
            raise ValueError(f"min_edge_len must be >= 3, got {self.min_edge_len}.")
        if not 0.0 <= self.min_edge_frac <= 1.0:
            raise ValueError(f"min_edge_frac must be in [0, 1], got {self.min_edge_frac}.")
        if self.min_bg_depth < 0 or self.probe_depth < 1:
            raise ValueError("min_bg_depth must be >= 0 and probe_depth >= 1.")


_SEAM_DEFAULTS = SeamDetectionParams()


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


def _bg_depth(
    tile_labels: np.ndarray, axis: str, coord: int, side: int, run_lo: int, run_hi: int, probe_depth: int
) -> float:
    """Median count of consecutive background pixels just outside a flat edge (tile-local coords)."""
    H, W = tile_labels.shape
    step = 1 if side == -1 else -1
    idxs = np.linspace(run_lo, run_hi - 1, min(9, max(1, run_hi - run_lo))).astype(int)
    depths = []
    for t in idxs:
        d = 0
        for k in range(1, probe_depth + 1):
            y, x = (t, coord + step * k) if axis == "v" else (coord + step * k, t)
            if not (0 <= y < H and 0 <= x < W) or tile_labels[y, x] != 0:
                break
            d += 1
        depths.append(d)
    return float(np.median(depths)) if depths else 0.0


def cell_flat_edge(
    mask: np.ndarray,
    tile_labels: np.ndarray,
    bbox: tuple[int, int, int, int],
    origin: tuple[int, int],
    params: SeamDetectionParams = _SEAM_DEFAULTS,
) -> dict[str, Any] | None:
    """Dominant cardinal flat cut-edge of one cell, or ``None``.

    Parameters
    ----------
    mask
        Boolean cell mask in its (tile-local) bounding box.
    tile_labels
        The full tile label array (for background probing).
    bbox
        ``(min_row, min_col, max_row, max_col)`` of the cell in tile-local coords.
    origin
        ``(y0, x0)`` global offset of the tile, to return a global edge coordinate.
    params
        :class:`SeamDetectionParams`.

    Returns
    -------
    ``{"axis": "v"|"h", "coord": float(global), "span": int, "side": +1|-1}`` for the longest
    qualifying edge, or ``None`` if the cell has no cardinal flat edge with a wide gap beyond.
    """
    y0, x0, _, _ = bbox
    oy, ox = origin
    Hc, Wc = mask.shape
    rows_present = mask.any(1)
    cols_present = mask.any(0)
    rightmost = np.where(rows_present, Wc - 1 - mask[:, ::-1].argmax(1), np.nan).astype(float)
    leftmost = np.where(rows_present, mask.argmax(1), np.nan).astype(float)
    bottommost = np.where(cols_present, Hc - 1 - mask[::-1, :].argmax(0), np.nan).astype(float)
    topmost = np.where(cols_present, mask.argmax(0), np.nan).astype(float)

    # axis, extreme array (perp coord within bbox), presence along parallel axis, side
    candidates = [
        ("v", rightmost, rows_present, -1),
        ("v", leftmost, rows_present, +1),
        ("h", bottommost, cols_present, -1),
        ("h", topmost, cols_present, +1),
    ]
    best: dict[str, Any] | None = None
    for axis, extreme, present, side in candidates:
        n_present = int(present.sum())
        if n_present < params.min_edge_len:
            continue
        for c in (np.nanmax(extreme[present]), np.nanmin(extreme[present])):
            on = present & (np.abs(extreme - c) <= params.flat_tol)
            s, ln = _longest_true_run(on)
            if ln < params.min_edge_len or ln / n_present < params.min_edge_frac:
                continue
            c_round = int(round(c))
            if axis == "v":  # perp axis = x (columns); parallel axis = y (rows)
                perp_local, run_lo, run_hi = x0 + c_round, y0 + s, y0 + s + ln
                coord_global = ox + perp_local
            else:  # perp axis = y (rows); parallel axis = x (columns)
                perp_local, run_lo, run_hi = y0 + c_round, x0 + s, x0 + s + ln
                coord_global = oy + perp_local
            bg = _bg_depth(tile_labels, axis, perp_local, side, run_lo, run_hi, params.probe_depth)
            if bg < params.min_bg_depth:
                continue
            edge = {"axis": axis, "coord": float(coord_global), "span": int(ln), "side": int(side)}
            if best is None or edge["span"] > best["span"]:
                best = edge
    return best


def detect_seams(
    coords_v: np.ndarray,
    coords_h: np.ndarray,
    extent_x: int,
    extent_y: int,
    params: SeamDetectionParams = _SEAM_DEFAULTS,
) -> dict[str, list[tuple[float, float, int]]]:
    """Locate seam bands per axis from the coordinates of cut-edges.

    Returns ``{"v": [(centre, half_width, count), ...], "h": [...]}`` (v = vertical seams at
    x-coordinates, h = horizontal seams at y-coordinates).
    """
    out: dict[str, list[tuple[float, float, int]]] = {}
    for axis, coords, extent in (("v", coords_v, extent_x), ("h", coords_h, extent_y)):
        if coords.size == 0:
            out[axis] = []
            continue
        bins = np.arange(0, extent + params.bin_width, params.bin_width)
        hist, _ = np.histogram(coords, bins=bins)
        win = max(1, int(round(2.0 / params.bin_width)))
        smooth = np.convolve(hist, np.ones(2 * win + 1), mode="same")
        bg = np.median(smooth[smooth > 0]) if (smooth > 0).any() else 0.0
        height = max(params.min_seam_count, params.bg_multiple * bg)
        peaks, _ = find_peaks(smooth, height=height, distance=1)
        if peaks.size == 0:
            out[axis] = []
            continue
        centres = bins[peaks] + params.bin_width / 2
        counts = smooth[peaks].astype(float)
        order = np.argsort(centres)
        centres, counts = centres[order], counts[order]
        clusters, cur = [], [0]
        for i in range(1, len(centres)):
            if centres[i] - centres[cur[-1]] <= params.cluster_gap:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)
        seams = []
        for cl in clusters:
            c, w = centres[cl], counts[cl]
            centre = float(np.average(c, weights=w))
            half = float((c.max() - c.min()) / 2 + params.band_margin)
            seams.append((centre, half, int(w.sum())))
        out[axis] = seams
    return out


def flag_seam_cells(
    edge_axis: np.ndarray,
    edge_coord: np.ndarray,
    edge_side: np.ndarray,
    edge_span: np.ndarray,
    seams: dict[str, list[tuple[float, float, int]]],
    params: SeamDetectionParams = _SEAM_DEFAULTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag cells whose flat edge lies in a seam band and faces it.

    ``edge_axis`` uses 1 for vertical, 2 for horizontal, 0 for 'no edge'.  Returns
    ``(is_seam_cut, seam_dist)`` arrays; ``seam_dist`` is NaN where not flagged.
    """
    n = edge_axis.shape[0]
    is_cut = np.zeros(n, bool)
    seam_dist = np.full(n, np.nan)
    for i in range(n):
        ax = edge_axis[i]
        if ax == 0 or edge_span[i] < params.min_edge_len:
            continue
        for centre, half, _cnt in seams["v" if ax == 1 else "h"]:
            signed = centre - edge_coord[i]
            if abs(signed) > half + params.flag_tol:
                continue
            faces = (signed >= -params.face_slack) if edge_side[i] == -1 else (signed <= params.face_slack)
            if faces:
                is_cut[i] = True
                seam_dist[i] = abs(signed)
                break
    return is_cut, seam_dist
