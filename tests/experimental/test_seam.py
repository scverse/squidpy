"""Tests for emergent-seam cut-cell detection in calculate_tiling_qc."""

from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.tl import SeamDetectionParams, calculate_tiling_qc
from squidpy.experimental.tl._seam import (
    SeamScale,
    cell_flat_edges,
    detect_seams,
    estimate_membrane_width,
    flag_cells_on_seams,
)


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
            sdata, labels_key="labels", detect_seams=True, seam_params={"gap_membrane_mult": 3.0}, inplace=False
        )
        assert adata.uns["tiling_qc"]["detect_seams"] is True
        assert adata.uns["tiling_qc"]["seam_params"]["gap_membrane_mult"] == 3.0

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
            ({"min_seam_edge_frac": 1.5}, "min_seam_edge_frac"),
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
        edges = cell_flat_edges(rp.image, tile, rp.bbox, (0, 0), min_len=5, flat_tol=1.5, probe_depth=8)
        vs = [e for e in edges if e["axis"] == "v" and e["side"] == -1]
        assert vs and abs(vs[0]["coord"] - 9) <= 1 and vs[0]["gap"] >= 3

    def test_flat_edge_gap_small_for_touching_neighbour(self):
        # the flat side faces a neighbour ~1px away -> small gap (excluded from seam *detection*)
        tile = np.zeros((20, 30), np.int32)
        tile[0:20, 0:10] = 5
        tile[0:20, 11:21] = 6
        from skimage.measure import regionprops

        rp = next(r for r in regionprops(tile) if r.label == 5)
        edges = cell_flat_edges(rp.image, tile, rp.bbox, (0, 0), min_len=5, flat_tol=1.5, probe_depth=8)
        vs = [e for e in edges if e["axis"] == "v" and e["side"] == -1]
        assert vs and vs[0]["gap"] <= 2  # membrane, not a seam gap

    def test_membrane_estimate(self):
        # mostly 1px membranes plus a few wide gaps -> membrane ~1
        gaps = np.array([1, 1, 1, 1, 2, 8, 9, 10], dtype=float)
        assert estimate_membrane_width(gaps) <= 2.0

    def test_detect_and_flag_roundtrip(self):
        rng = np.random.default_rng(0)
        # a seam at x=100 (wide gaps) over scattered facet background (small gaps)
        edges = []
        for c in 100.0 + rng.normal(0, 1, 30):
            edges.append({"axis": "v", "coord": float(c), "span": 20, "side": -1, "gap": 9.0, "cell_id": -1})
        for c in rng.uniform(0, 200, 80):
            edges.append({"axis": "v", "coord": float(c), "span": 20, "side": -1, "gap": 5.0, "cell_id": -1})
        scale = SeamScale(diameter=20.0, membrane=1.0)
        seams = detect_seams(edges, 200, 200, scale, SeamDetectionParams())
        assert len(seams["v"]) == 1 and abs(seams["v"][0][0] - 100) <= 4
        # a left-body cell whose right edge lands on the seam is flagged; one far away is not
        ebc = {
            1: [{"axis": "v", "coord": 100.0, "span": 20, "side": -1, "gap": 9.0}],
            2: [{"axis": "v", "coord": 20.0, "span": 20, "side": -1, "gap": 9.0}],
        }
        flagged = flag_cells_on_seams(ebc, seams, scale, SeamDetectionParams())
        assert 1 in flagged and 2 not in flagged
