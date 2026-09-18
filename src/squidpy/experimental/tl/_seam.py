"""Emergent-seam detection for tile-boundary cut cells.

The MAD-based ``is_outlier`` gate in :func:`~squidpy.experimental.tl.calculate_tiling_qc`
flags cells whose boundary is unusually straight.  In dense tissue with a wide inter-FOV
gap -- and when a cut leaves only one segmentable half -- that signal is swamped: most real
cut cells are missed and interior cells are flagged instead.

This module adds a complementary, geometry-only detector that needs neither the FOV size
nor tile overlap, and works on single-sided cuts:

1. For every cell, collect **all** long, cardinal (axis-aligned) flat boundary runs, together
   with the depth of background lying just beyond each one.
2. Seam lines **emerge** as the coordinates where many such runs align -- a consensus over
   many cells -- with a wide inter-FOV gap spreading a seam into a band.  A peak counts as a
   seam when its aligned-edge count is too large to arise from edges scattered uniformly
   along the axis, so a weakly-populated seam is judged on its own evidence rather than
   against the strongest peak present.
3. A cell is flagged ``is_seam_cut`` iff any of its cardinal edges lies in a detected seam
   band and faces it.  Membership does not require a wide gap: once the seam's location is
   known, a close-gap or single-sided cut on that line counts too.

**Which edges vote** depends on what the data supports.  In packed tissue an ordinary facet
faces a neighbour one membrane away, so a wide background gap is rare and isolates the seam;
in sparse tissue nearly every edge faces open background and the gap says nothing, so all
edges vote and the alignment consensus carries detection alone.  The split between the two
gap classes and the choice of channel are both read off the observed gap distribution
(:func:`gap_channel`), never set by hand.

**No absolute-pixel thresholds.**  Every length scale is expressed as a ratio and resolved
at runtime against the data's own length scale ``D`` (the median cell equivalent diameter)
and the observed gap distribution, so the same defaults transfer across resolutions, cell
sizes and FOV pitches.  See :class:`SeamDetectionParams`.

**Gaps must be probed on unmasked pixels.**  The caller measures background beyond an edge on
an occupancy array that includes cells belonging to *other* processing tiles.  Probing a
tile-masked array instead makes every neighbour dropped by masking look like open background,
which manufactures a seam along each processing tile border -- a detection result that would
depend on ``tile_size`` rather than on the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import poisson


@dataclass(slots=True, frozen=True)
class SeamDetectionParams:
    """Scale-invariant tuning knobs for emergent-seam cut-cell detection.

    No knob is an absolute pixel length.  Each is either a **dimensionless ratio** resolved at
    runtime against the data's length scale ``D`` (median cell equivalent diameter), a
    **statistical criterion** (:attr:`alpha`, :attr:`gap_selectivity_max`) evaluated against
    the data's own distributions, or a **pixel-grid constant** (:attr:`flat_tol`,
    :attr:`bin_width`) that describes the raster rather than the tissue -- so the defaults
    transfer across resolutions, cell sizes and FOV pitches.

    The thresholds that matter most are derived rather than set: the split between membrane
    gaps and open background comes from the observed gap distribution, and the height a
    histogram peak must reach comes from a null built on the edge count and axis extent.
    """

    edge_len_frac: float = 0.25
    """Minimum cardinal flat-run length, as a fraction of the median cell diameter ``D``.

    A cut rarely bisects a cell through its widest point, so requiring the straight edge to
    span half the cell (the previous default) discards most real cuts.  Measured on real
    CosMx breast tissue against seam-adjacency ground truth, lowering this from 0.5 to 0.25
    raises recall 0.38 -> 0.54 at precision 0.98 -> 0.89 (F1 0.54 -> 0.67); on both synthetic
    fixtures F1 moves by less than 0.01 either way, and the no-seam control stays empty at
    every value.  Raising it trades recall for precision."""

    flat_tol: float = 1.5
    """Flatness tolerance (px): max deviation from a single coordinate for a run to be 'flat'.
    A rasterisation / pixel-grid constant (~1px), independent of cell size or resolution --
    scaling it with the cell size would accept curved edges on large cells."""

    probe_frac: float = 0.6
    """How far to probe for background beyond an edge, as a fraction of ``D``."""

    gap_selectivity_max: float = 0.25
    """Use the wide-gap edge channel only while at most this fraction of edges passes the
    (data-derived) gap split -- i.e. only while the split actually discriminates.

    In packed tissue an ordinary facet faces a neighbour one membrane away, so a wide gap is
    rare and the split isolates the seam: measured 3-5% on real CosMx breast tissue and 5.5%
    on the dense synthetic.  In sparse tissue almost every edge faces open background (45-48%
    measured), the gap says nothing about seams, and detection falls back to all edges.  The
    decision boundary sits an order of magnitude from either regime."""

    bin_width: float = 2.0
    """Seam histogram bin width (px): a pixel-grid resolution constant, independent of cell size."""

    alpha: float = 0.01
    """Significance level for keeping a histogram peak, Bonferroni-corrected over the bins.

    A peak is a seam when its aligned-edge count is too large to come from edges scattered
    uniformly along the axis.  The null is computed from the data (edge count and axis
    extent), so this transfers across datasets in a way a hand-set height ratio does not, and
    a weakly-populated seam is judged on its own significance rather than against the
    strongest peak on its axis."""

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
            "gap_selectivity_max",
            "bin_width",
            "alpha",
            "cluster_frac",
            "band_margin_frac",
            "flag_tol_frac",
            "face_slack_frac",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.edge_len_frac <= 0:
            raise ValueError(f"edge_len_frac must be > 0, got {self.edge_len_frac}.")
        if not 0.0 <= self.gap_selectivity_max <= 1.0:
            raise ValueError(f"gap_selectivity_max must be in [0, 1], got {self.gap_selectivity_max}.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}.")
        if self.probe_frac <= 0:
            raise ValueError(f"probe_frac must be > 0, got {self.probe_frac}.")


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
    occupancy: np.ndarray, axis: str, coord: int, side: int, run_lo: int, run_hi: int, probe_depth: int
) -> float:
    """Median count of consecutive background pixels just outside a flat edge (tile-local coords).

    ``occupancy`` must mark **every** cell in the neighbourhood, not only the ones owned by the
    current processing tile -- otherwise a neighbour dropped by tile masking reads as background
    and an ordinary membrane is mistaken for a seam gap.
    """
    height, width = occupancy.shape
    step = 1 if side == -1 else -1
    idxs = np.linspace(run_lo, run_hi - 1, min(9, max(1, run_hi - run_lo))).astype(int)
    depths = []
    for t in idxs:
        d = 0
        for k in range(1, probe_depth + 1):
            y, x = (t, coord + step * k) if axis == "v" else (coord + step * k, t)
            if not (0 <= y < height and 0 <= x < width) or occupancy[y, x]:
                break
            d += 1
        depths.append(d)
    return float(np.median(depths)) if depths else 0.0


def cell_flat_edges(
    mask: np.ndarray,
    occupancy: np.ndarray,
    bbox: tuple[int, int, int, int],
    origin: tuple[int, int],
    min_len: int,
    flat_tol: float,
    probe_depth: int,
) -> list[dict[str, Any]]:
    """All cardinal flat boundary runs of one cell (multiple per cell; global coordinates).

    ``occupancy`` is the boolean occupancy of the surrounding crop **including cells not owned
    by the current tile** -- see :func:`_probe_gap`.

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
        if axis == "v":  # perp = x (column); parallel/along-seam = y (row)
            perp_local, run_lo, run_hi = x0 + c_round, y0 + s, y0 + s + ln
            coord_global, ext_lo, ext_hi = ox + perp_local, oy + run_lo, oy + run_hi
        else:  # perp = y (row); parallel/along-seam = x (column)
            perp_local, run_lo, run_hi = y0 + c_round, x0 + s, x0 + s + ln
            coord_global, ext_lo, ext_hi = oy + perp_local, ox + run_lo, ox + run_hi
        gap = _probe_gap(occupancy, axis, perp_local, side, run_lo, run_hi, probe_depth)
        out.append(
            {
                "axis": axis,
                "coord": float(coord_global),
                "span": int(ln),
                "side": int(side),
                "gap": gap,
                "extent": (float(ext_lo), float(ext_hi)),
            }
        )
    return out


def _otsu_gap_split(gaps: np.ndarray) -> float:
    """Otsu split of the observed edge-gap distribution (returns the threshold in pixels).

    The two classes are physical: an edge either faces a neighbour across a thin membrane, or
    it faces open background.  Taking the split from the data replaces a hand-set multiple of
    an estimated membrane width, and adapts to imaging resolution and segmentation style.
    """
    g = gaps[np.isfinite(gaps)]
    if g.size == 0:
        return float("inf")
    counts = np.bincount(np.rint(g).astype(int))
    if counts.size < 2:
        return float(counts.size)
    total = counts.sum()
    values = np.arange(counts.size, dtype=float)
    w0 = np.cumsum(counts)[:-1] / total
    m0 = np.cumsum(values * counts)[:-1] / total
    mu = float((values * counts).sum() / total)
    w1 = 1.0 - w0
    ok = (w0 > 0) & (w1 > 0)
    between = np.zeros_like(w0)
    between[ok] = (mu * w0[ok] - m0[ok]) ** 2 / (w0[ok] * w1[ok])
    return float(np.argmax(between)) + 0.5


def gap_channel(edges: list[dict[str, Any]], params: SeamDetectionParams = _SEAM_DEFAULTS) -> tuple[float, float, bool]:
    """Choose the edge channel the data supports.

    Returns ``(threshold, selectivity, use_wide_gap)`` -- the data-derived gap split, the
    fraction of edges above it, and whether that split is selective enough to filter on.
    See :attr:`SeamDetectionParams.gap_selectivity_max`.
    """
    gaps = np.array([e["gap"] for e in edges], dtype=float)
    if gaps.size == 0:
        return float("inf"), 0.0, False
    thresh = _otsu_gap_split(gaps)
    selectivity = float(np.mean(gaps >= thresh))
    return thresh, selectivity, selectivity <= params.gap_selectivity_max


def _min_significant_count(n_edges: int, n_bins: int, window: int, alpha: float) -> float:
    """Smallest smoothed bin count not explainable by uniformly scattered edges.

    Under the null the edges are spread uniformly along the axis, so the count in a
    ``2*window+1`` bin neighbourhood is Poisson with mean ``n_edges*(2*window+1)/n_bins``.
    The threshold is that distribution's upper tail at ``alpha``, Bonferroni-corrected over
    the bins tested.  Both inputs come from the data, so the test transfers across datasets
    where a fixed height ratio would not.
    """
    if n_edges == 0 or n_bins == 0:
        return float("inf")
    lam = n_edges * (2 * window + 1) / n_bins
    return float(max(3.0, poisson.isf(alpha / n_bins, lam) + 1))


def detect_seams(
    edges: list[dict[str, Any]],
    extent_x: int,
    extent_y: int,
    diameter: float,
    params: SeamDetectionParams = _SEAM_DEFAULTS,
) -> dict[str, list[tuple[float, float, int]]]:
    """Locate seam bands per axis from the aligned edges the data supports.

    Two channels carry the seam signal in different tissue regimes, and the selectivity of the
    gap split decides which one is used (:func:`gap_channel`):

    * **wide-gap edges** -- in packed tissue an ordinary facet faces a neighbour one membrane
      away, so a wide gap is rare and isolates the seam cleanly.
    * **all edges** -- in sparse tissue nearly every edge faces open background, so the gap
      carries no information about seams and the alignment consensus carries detection alone.

    A peak becomes a seam when its aligned-edge count is too large to arise from edges
    scattered uniformly along the axis (:func:`_min_significant_count`), so a weakly-populated
    seam is judged on its own significance rather than against the strongest peak on its axis.

    Returns ``{"v": [(centre, half_width, count), ...], "h": [...]}``.
    """
    d = diameter
    bin_w = params.bin_width
    cluster_gap = params.cluster_frac * d
    band_margin = params.band_margin_frac * d
    gap_thresh, _selectivity, use_wide_gap = gap_channel(edges, params)

    out: dict[str, list[tuple[float, float, int]]] = {}
    for axis, extent in (("v", extent_x), ("h", extent_y)):
        coords = np.array(
            [e["coord"] for e in edges if e["axis"] == axis and (not use_wide_gap or e["gap"] >= gap_thresh)]
        )
        if coords.size == 0:
            out[axis] = []
            continue
        bins = np.arange(0, extent + bin_w, bin_w)
        hist, _ = np.histogram(coords, bins=bins)
        # Smooth only for sub-pixel wobble (small window); the CLUSTER step below is what
        # widens a seam across the inter-FOV gap -- over-smoothing here collapses the gap's
        # twin peaks into one narrow band and loses the far-side cells.
        win = max(1, int(round((0.12 * d) / bin_w)))
        smooth = np.convolve(hist, np.ones(2 * win + 1), mode="same")
        height = _min_significant_count(coords.size, hist.size, win, params.alpha)
        peaks, _ = find_peaks(smooth, height=height, distance=1)
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
    diameter: float,
    params: SeamDetectionParams = _SEAM_DEFAULTS,
) -> dict[int, float]:
    """Return ``{cell_id: seam_dist}`` for every cell with a cardinal edge on and facing a seam.

    Once a seam is detected, membership does **not** require a wide gap -- a genuine two-sided
    cut whose other half sits close still counts, because its edge lies on the seam consensus.
    """
    flag_tol = params.flag_tol_frac * diameter
    face_slack = params.face_slack_frac * diameter
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
