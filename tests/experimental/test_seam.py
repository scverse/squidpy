"""Tests for emergent-seam cut-cell detection in calculate_tiling_qc."""

from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.tl import SeamDetectionParams, calculate_tiling_qc
from squidpy.experimental.tl._seam import (
    _min_significant_count,
    _otsu_gap_split,
    cell_flat_edges,
    detect_seams,
    flag_cells_on_seams,
    gap_channel,
)


def _assert_seams_match(seams, truth, context, tol=8):
    """Assert each axis reports exactly the true seams, each within ``tol`` px."""
    for axis in ("v", "h"):
        coords = sorted(b["coord"] for b in seams[axis])
        assert len(coords) == len(truth), (
            f"axis {axis} {context}: expected {list(truth)}, got {[round(c, 1) for c in coords]}"
        )
        for want, got in zip(sorted(truth), coords, strict=True):
            assert abs(got - want) <= tol, f"axis {axis} {context}: seam {got:.1f} too far from true {want}"


def _recall(pred_ids: set[int], truth_ids: frozenset[int]) -> float:
    return len(pred_ids & truth_ids) / len(truth_ids) if truth_ids else 0.0


def _precision(pred_ids: set[int], truth_ids: frozenset[int]) -> float:
    return len(pred_ids & truth_ids) / len(pred_ids) if pred_ids else 0.0


class TestSeamDetectionIntegration:
    def test_seam_columns_added(self, sdata_dense_seam):
        sdata, _ = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=True, inplace=False)
        assert "is_seam_cut" in adata.obs
        assert adata.obs["is_seam_cut"].dtype == bool
        flagged = adata.obs["is_seam_cut"].to_numpy()
        assert np.isfinite(adata.obs["seam_dist"].to_numpy()[flagged]).all()
        assert np.isnan(adata.obs["seam_dist"].to_numpy()[~flagged]).all()

    def test_detect_seams_false_omits_columns(self, sdata_dense_seam):
        sdata, _ = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=False, inplace=False)
        assert "is_seam_cut" not in adata.obs
        assert "seam_dist" not in adata.obs
        assert adata.uns["tiling_qc"]["seams"] == {"v": [], "h": []}

    def test_seams_detected_near_true_border(self, sdata_dense_seam):
        sdata, gt = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=True, inplace=False)
        seams = adata.uns["tiling_qc"]["seams"]
        found = [b["coord"] for b in seams["v"]] + [b["coord"] for b in seams["h"]]
        assert found, "no seams detected"
        for s in gt.seam_coords:
            assert min(abs(c - s) for c in found) <= 8

    def test_seam_recall_beats_outlier_on_dense_wide_gap(self, sdata_dense_seam):
        sdata, gt = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=True, inplace=False)
        lid = adata.obs["label_id"].to_numpy()
        seam_ids = set(lid[adata.obs["is_seam_cut"].to_numpy()].tolist())
        outlier_ids = set(lid[adata.obs["is_outlier"].to_numpy()].tolist())
        # emergent-seam detection recovers most cuts where the MAD gate collapses
        assert _recall(seam_ids, gt.cut_cell_ids) >= 0.7
        assert _recall(seam_ids, gt.cut_cell_ids) > _recall(outlier_ids, gt.cut_cell_ids) + 0.3
        assert _precision(seam_ids, gt.cut_cell_ids) >= 0.5

    def test_seam_params_recorded_in_uns(self, sdata_dense_seam):
        sdata, _ = sdata_dense_seam
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", detect_seams=True, seam_params={"alpha": 0.005}, inplace=False
        )
        assert adata.uns["tiling_qc"]["detect_seams"] is True
        assert adata.uns["tiling_qc"]["seam_params"]["alpha"] == 0.005

    def test_no_helper_columns_leak(self, sdata_dense_seam):
        sdata, _ = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=True, inplace=False)
        assert not any(c.startswith("_seam") for c in adata.obs.columns)


class TestSeamDetectionParams:
    def test_defaults_are_dimensionless_or_pixel_constants(self):
        p = SeamDetectionParams()
        # length thresholds are fractions of the cell diameter (scale-invariant)
        assert 0 < p.edge_len_frac <= 1
        # flat_tol / bin_width are pixel-grid constants (do not scale with cell size)
        assert p.flat_tol >= 1 and p.bin_width >= 1

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"edge_len_frac": 0.0}, "edge_len_frac"),
            ({"gap_selectivity_max": 1.5}, "gap_selectivity_max"),
            ({"alpha": 0.0}, "alpha"),
            ({"probe_frac": 0.0}, "probe_frac"),
        ],
    )
    def test_invalid_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SeamDetectionParams(**kwargs)


class TestSeamUnits:
    def test_flat_edge_detects_cut_with_gap(self):
        # a cell filling the tile height, cut flat at col 10, wide background to the right
        tile = np.zeros((20, 30), np.int32)
        tile[0:20, 0:10] = 5
        from skimage.measure import regionprops

        rp = regionprops(tile)[0]
        edges = cell_flat_edges(rp.image, tile != 0, rp.bbox, (0, 0), min_len=5, flat_tol=1.5, probe_depth=8)
        vs = [e for e in edges if e["axis"] == "v" and e["side"] == -1]
        assert vs and abs(vs[0]["coord"] - 9) <= 1 and vs[0]["gap"] >= 3

    def test_flat_edge_gap_small_for_touching_neighbour(self):
        # the flat side faces a neighbour ~1px away -> small gap (excluded from seam *detection*)
        tile = np.zeros((20, 30), np.int32)
        tile[0:20, 0:10] = 5
        tile[0:20, 11:21] = 6
        from skimage.measure import regionprops

        rp = next(r for r in regionprops(tile) if r.label == 5)
        edges = cell_flat_edges(rp.image, tile != 0, rp.bbox, (0, 0), min_len=5, flat_tol=1.5, probe_depth=8)
        vs = [e for e in edges if e["axis"] == "v" and e["side"] == -1]
        assert vs and vs[0]["gap"] <= 2  # membrane, not a seam gap

    def test_otsu_splits_membranes_from_open_background(self):
        # mostly 1px membranes plus a few wide gaps -> the split lands between the two modes
        gaps = np.array([1, 1, 1, 1, 2, 8, 9, 10], dtype=float)
        thr = _otsu_gap_split(gaps)
        assert 2.0 < thr < 8.0

    def test_gap_channel_filters_when_selective_and_falls_back_when_not(self):
        # packed tissue: a few wide gaps among many membranes -> the split discriminates
        packed = [{"gap": g} for g in [1] * 90 + [9] * 10]
        _thr, selectivity, use_wide = gap_channel(packed)
        assert use_wide and selectivity <= 0.25
        # sparse tissue: most edges face open background -> the gap says nothing, use all edges
        sparse = [{"gap": g} for g in [0] * 50 + [8] * 50]
        _thr, selectivity, use_wide = gap_channel(sparse)
        assert not use_wide and selectivity > 0.25

    def test_significance_threshold_scales_with_the_null_rate(self):
        # denser scatter -> a larger count is needed before a peak is surprising
        sparse_null = _min_significant_count(n_edges=100, n_bins=300, window=1, alpha=0.01)
        dense_null = _min_significant_count(n_edges=3000, n_bins=300, window=1, alpha=0.01)
        assert dense_null > sparse_null >= 3

    def test_detect_and_flag_roundtrip(self):
        rng = np.random.default_rng(0)
        # a seam at x=100 (wide gaps) over scattered facet background (small gaps)
        edges = []
        for c in 100.0 + rng.normal(0, 1, 30):
            edges.append({"axis": "v", "coord": float(c), "span": 20, "side": -1, "gap": 9.0, "cell_id": -1})
        for c in rng.uniform(0, 200, 80):
            edges.append({"axis": "v", "coord": float(c), "span": 20, "side": -1, "gap": 5.0, "cell_id": -1})
        scale = SeamDetectionParams()._resolve(20.0)
        seams = detect_seams(edges, 200, 200, scale)
        assert len(seams["v"]) == 1 and abs(seams["v"][0][0] - 100) <= 4
        # a left-body cell whose right edge lands on the seam is flagged; one far away is not
        ebc = {
            1: [{"axis": "v", "coord": 100.0, "span": 20, "side": -1, "gap": 9.0}],
            2: [{"axis": "v", "coord": 20.0, "span": 20, "side": -1, "gap": 9.0}],
        }
        flagged = flag_cells_on_seams(ebc, seams, scale)
        assert 1 in flagged and 2 not in flagged


class TestTileGridIndependence:
    """Seam detection must depend on the data only, never on the QC tiling used to process it.

    ``tile_size`` is a throughput knob: it controls how the labels raster is chopped up for
    parallel scoring.  Two runs of the same data at different ``tile_size`` must therefore
    agree on where the FOV seams are.
    """

    @pytest.mark.parametrize("tile_size", [128, 200, 420])
    def test_seams_match_truth_at_any_tile_size(self, sdata_dense_seam, tile_size):
        sdata, gt = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", tile_size=tile_size, detect_seams=True, inplace=False)
        _assert_seams_match(adata.uns["tiling_qc"]["seams"], gt.seam_coords, f"at tile_size={tile_size}")

    def test_no_seam_lands_on_the_processing_tile_grid(self, sdata_dense_seam):
        """The QC tile borders are not seams; detecting one there is a processing artifact."""
        sdata, gt = sdata_dense_seam
        tile_size = 200
        adata = calculate_tiling_qc(sdata, labels_key="labels", tile_size=tile_size, detect_seams=True, inplace=False)
        d = adata.uns["tiling_qc"]["seam_diameter"]
        extent = sdata.labels["labels"].shape[-1]
        borders = [tile_size * k for k in range(1, 1 + extent // tile_size)]
        seams = adata.uns["tiling_qc"]["seams"]
        for axis in ("v", "h"):
            for band in seams[axis]:
                c = band["coord"]
                near_true = min(abs(c - t) for t in gt.seam_coords)
                near_border = min(abs(c - b) for b in borders)
                assert near_true <= d or near_border > d, (
                    f"axis {axis}: seam at {c:.1f} sits on the tile grid {borders} "
                    f"but not on a true seam {list(gt.seam_coords)}"
                )


class TestSparseTissueDetection:
    """Detection must not depend on tissue density.

    In sparse tissue nearly every cell edge faces open background, so a wide-gap pre-filter
    keeps almost every edge and discriminates nothing.  The seam must then be found from the
    alignment consensus alone -- many cells sharing one edge coordinate -- which is the
    evidence that defines a seam in the first place.
    """

    @pytest.mark.parametrize("tile_size", [150, 200])
    def test_seams_found_in_sparse_tissue(self, sdata_tile_boundary, tile_size):
        sdata, gt = sdata_tile_boundary
        adata = calculate_tiling_qc(sdata, labels_key="labels", tile_size=tile_size, detect_seams=True, inplace=False)
        seams = adata.uns["tiling_qc"]["seams"]
        for axis, truth in (("v", gt.tile_borders_x), ("h", gt.tile_borders_y)):
            coords = sorted(b["coord"] for b in seams[axis])
            for want in truth:
                assert any(abs(c - want) <= 8 for c in coords), (
                    f"axis {axis} at tile_size={tile_size}: true seam {want} not found "
                    f"in {[round(c, 1) for c in coords]}"
                )

    def test_seams_match_truth_at_any_overlap_margin(self, sdata_dense_seam):
        """Gap probing reaches beyond a cell's own edge, so the crop must leave room for it.

        ``overlap_margin`` is sized to *contain* each owned cell; a cell sitting at the crop
        edge would have its probe clipped and under-report its gap.  Detection must not
        depend on that either.
        """
        sdata, gt = sdata_dense_seam
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, overlap_margin=2, detect_seams=True, inplace=False
        )
        _assert_seams_match(adata.uns["tiling_qc"]["seams"], gt.seam_coords, "with overlap_margin=2")
