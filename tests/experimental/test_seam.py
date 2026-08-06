"""Tests for emergent-seam cut-cell detection in calculate_tiling_qc."""

from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.tl import SeamDetectionParams, calculate_tiling_qc
from squidpy.experimental.tl._seam import cell_flat_edge, detect_seams, flag_seam_cells


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
        assert "seam_dist" in adata.obs
        # seam_dist is finite exactly where flagged
        flagged = adata.obs["is_seam_cut"].to_numpy()
        assert np.isfinite(adata.obs["seam_dist"].to_numpy()[flagged]).all()
        assert np.isnan(adata.obs["seam_dist"].to_numpy()[~flagged]).all()

    def test_detect_seams_false_omits_columns(self, sdata_dense_seam):
        sdata, _ = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=False, inplace=False)
        assert "is_seam_cut" not in adata.obs
        assert "seam_dist" not in adata.obs
        assert adata.uns["tiling_qc"]["seams"] == {"v": [], "h": []}
        # no internal helper columns leak into the table
        assert not any(c.startswith("_seam_edge") for c in adata.obs.columns)

    def test_seams_detected_near_true_border(self, sdata_dense_seam):
        sdata, gt = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=True, inplace=False)
        seams = adata.uns["tiling_qc"]["seams"]
        found = [b["coord"] for b in seams["v"]] + [b["coord"] for b in seams["h"]]
        assert found, "no seams detected"
        # every true seam coordinate is recovered within a few pixels
        for s in gt.seam_coords:
            assert min(abs(c - s) for c in found) <= 6

    def test_seam_recall_beats_outlier_on_dense_wide_gap(self, sdata_dense_seam):
        sdata, gt = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=True, inplace=False)
        lid = adata.obs["label_id"].to_numpy()
        seam_ids = set(lid[adata.obs["is_seam_cut"].to_numpy()].tolist())
        outlier_ids = set(lid[adata.obs["is_outlier"].to_numpy()].tolist())
        # the emergent-seam detector recovers most cuts where the MAD gate collapses
        assert _recall(seam_ids, gt.cut_cell_ids) >= 0.7
        assert _recall(seam_ids, gt.cut_cell_ids) > _recall(outlier_ids, gt.cut_cell_ids) + 0.3
        assert _precision(seam_ids, gt.cut_cell_ids) >= 0.5

    def test_no_seam_columns_are_not_helper_columns(self, sdata_dense_seam):
        sdata, _ = sdata_dense_seam
        adata = calculate_tiling_qc(sdata, labels_key="labels", detect_seams=True, inplace=False)
        assert not any(c.startswith("_seam_edge") for c in adata.obs.columns)

    def test_seam_params_recorded_in_uns(self, sdata_dense_seam):
        sdata, _ = sdata_dense_seam
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", detect_seams=True, seam_params={"min_bg_depth": 4.0}, inplace=False
        )
        assert adata.uns["tiling_qc"]["detect_seams"] is True
        assert adata.uns["tiling_qc"]["seam_params"]["min_bg_depth"] == 4.0


class TestSeamDetectionParams:
    def test_defaults(self):
        p = SeamDetectionParams()
        assert p.min_edge_len == 6 and p.min_bg_depth == 3.0

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"min_edge_len": 2}, "min_edge_len"),
            ({"min_edge_frac": 1.5}, "min_edge_frac"),
            ({"probe_depth": 0}, "probe_depth"),
        ],
    )
    def test_invalid_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SeamDetectionParams(**kwargs)


class TestSeamUnits:
    def test_flat_edge_detects_cut_with_gap(self):
        # a 20x20 cell occupying the left half, cut flat at col 10, wide background to the right
        tile = np.zeros((20, 30), np.int32)
        tile[3:17, 0:10] = 5  # cell 5, flat right edge at x=9, background beyond (cols 10..29)
        from skimage.measure import regionprops

        rp = regionprops(tile)[0]
        edge = cell_flat_edge(rp.image, tile, rp.bbox, (0, 0), SeamDetectionParams(min_edge_len=5))
        assert edge is not None
        assert edge["axis"] == "v" and edge["side"] == -1
        assert abs(edge["coord"] - 9) <= 1

    def test_flat_edge_rejects_facet_touching_neighbour(self):
        # cell fills the tile height (top/bottom edges touch the border -> not cut candidates);
        # its only flat side faces a neighbour 1px away, so background depth is below min_bg_depth.
        tile = np.zeros((20, 30), np.int32)
        tile[0:20, 0:10] = 5
        tile[0:20, 11:21] = 6  # neighbour 1px away
        from skimage.measure import regionprops

        rp = next(r for r in regionprops(tile) if r.label == 5)
        edge = cell_flat_edge(rp.image, tile, rp.bbox, (0, 0), SeamDetectionParams(min_edge_len=5, min_bg_depth=3.0))
        assert edge is None

    def test_detect_and_flag_roundtrip(self):
        # 30 vertical edges piled at x=100 (a seam) over a background of scattered facet edges
        rng = np.random.default_rng(0)
        coords_v = np.concatenate(
            [
                100.0 + rng.normal(0, 1, 30),  # the seam
                rng.uniform(0, 200, 80),  # scattered facet background
            ]
        )
        seams = detect_seams(coords_v, np.array([]), 200, 200, SeamDetectionParams(min_seam_count=10))
        assert len(seams["v"]) == 1
        assert abs(seams["v"][0][0] - 100) <= 3
        axis = np.array([0, 1], np.int8)
        coord = np.array([np.nan, 100.0])
        side = np.array([0, -1], np.int8)
        span = np.array([0, 10])
        is_cut, _ = flag_seam_cells(axis, coord, side, span, seams)
        assert is_cut[1] and not is_cut[0]
