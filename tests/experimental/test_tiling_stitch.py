"""Tests for tile-cut cell stitching."""

from __future__ import annotations

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel

import squidpy as sq
from tests.conftest import DPI, PlotTester, PlotTesterMeta


def _run_qc_and_stitch(sdata, **stitch_kwargs):
    sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200, nmads_cut=1.0, nmads_smoothed=1.5)
    sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", **stitch_kwargs)
    return sdata.tables["labels_qc"]


class TestAssignStitchGroups:
    """Tests for sq.experimental.tl.assign_stitch_groups using the tile-boundary fixture."""

    def test_columns_present(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata)
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in adata.obs.columns

    def test_confidence_convention(self, sdata_tile_boundary):
        # NaN = not evaluated (non-candidate), 1.0 = solo candidate, composite = stitched.
        # The candidate gate defaults to `is_seam_cut` when present, else `is_outlier`.
        sdata, _ = sdata_tile_boundary
        obs = _run_qc_and_stitch(sdata, min_confidence=0.5).obs
        gate = "is_seam_cut" if "is_seam_cut" in obs.columns else "is_outlier"

        non_cands = ~obs[gate].astype(bool)
        assert non_cands.sum() > 0
        assert obs.loc[non_cands, "stitch_confidence"].isna().all()
        assert (obs.loc[non_cands, "stitch_group_id"] == obs.loc[non_cands, "label_id"]).all()
        assert (obs.loc[non_cands, "n_pieces"] == 1).all()

        solo = obs[gate].astype(bool) & ~obs["is_stitched"].astype(bool)
        if solo.sum() > 0:
            assert (obs.loc[solo, "stitch_confidence"] == 1.0).all()

        stitched = obs["is_stitched"].astype(bool)
        if stitched.sum() > 0:
            confs = obs.loc[stitched, "stitch_confidence"]
            assert ((confs >= 0.5) & (confs <= 1.0)).all()
            assert obs.loc[stitched, "n_pieces"].between(2, 4).all()

    def test_group_id_shared_within_group(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        stitched = adata.obs[adata.obs["is_stitched"].astype(bool)]
        for _gid, members in stitched.groupby("stitch_group_id"):
            assert len(members) == members["n_pieces"].iloc[0]

    def test_stitched_group_is_made_of_cut_pieces(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        stitched = adata.obs[adata.obs["is_stitched"].astype(bool)]
        found = any(
            len(set(m["label_id"].astype(int))) >= 2 and set(m["label_id"].astype(int)) <= set(gt.cut_cell_ids)
            for _gid, m in stitched.groupby("stitch_group_id")
        )
        assert found

    def test_no_intact_cells_stitched_at_high_threshold(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.9)
        intact = adata.obs["label_id"].isin(gt.intact_cell_ids)
        n_false = int((intact & adata.obs["is_stitched"].astype(bool)).sum())
        assert n_false <= 5

    def test_recovery_meets_quantitative_bounds(self, sdata_tile_boundary):
        """Quantitative floor from the validation sweep (deterministic fixture).

        At ``min_confidence=0.5`` the sweep recovers ~64% of cut pieces with zero
        intact false-merges; assert a conservative recall floor and a near-zero
        false-merge bound (small tolerance for skimage version drift).
        """
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        lid = adata.obs["label_id"].astype(int)
        stitched = adata.obs["is_stitched"].astype(bool)
        n_cut_stitched = int((lid.isin(gt.cut_cell_ids) & stitched).sum())
        n_false = int((lid.isin(gt.intact_cell_ids) & stitched).sum())
        recall = n_cut_stitched / max(len(gt.cut_cell_ids), 1)
        assert recall >= 0.5, f"recall {recall:.2f} below 0.5 floor"
        assert n_false <= 2, f"too many intact false merges: {n_false}"

    def test_uns_records_params_and_features(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        meta = _run_qc_and_stitch(sdata, min_confidence=0.7).uns["tiling_stitch"]
        assert meta["min_confidence"] == 0.7
        assert isinstance(meta["stitch_params"], dict)
        assert "model_coefficients" not in meta and "model_intercept" not in meta
        assert set(meta["score_features"]) == {
            "iou",
            "endpoint_match",
            "merge_compactness",
            "merge_solidity",
        }

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"labels_key": "labels"}, "QC table"),
            ({"labels_key": "bogus"}, "not found in sdata.labels"),
            ({"labels_key": "labels", "min_confidence": 1.5}, "min_confidence"),
        ],
        ids=["missing_qc_table", "missing_labels_key", "invalid_min_confidence"],
    )
    def test_invalid_input_raises(self, sdata_tile_boundary, kwargs, match):
        sdata, _ = sdata_tile_boundary
        with pytest.raises(ValueError, match=match):
            sq.experimental.tl.assign_stitch_groups(sdata, **kwargs)

    def test_rerun_overwrites_without_growing_columns(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        n_before = len(sdata.tables["labels_qc"].obs.columns)
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels")
        assert len(sdata.tables["labels_qc"].obs.columns) == n_before

    def test_inplace_false_returns_without_writing(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200)
        n_before = len(sdata.tables["labels_qc"].obs.columns)
        result = sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", inplace=False)
        assert result is not None and "stitch_group_id" in result.obs.columns
        assert len(sdata.tables["labels_qc"].obs.columns) == n_before

    def test_qc_rerun_removes_stitch_columns(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200)
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col not in sdata.tables["labels_qc"].obs.columns

    def test_runs_on_multiscale(self):
        from tests.experimental.conftest import make_tile_boundary_sdata

        base, _ = make_tile_boundary_sdata()
        arr = np.asarray(base.labels["labels"].values)
        ms = Labels2DModel.parse(
            xr.DataArray(da.from_array(arr, chunks=(200, 200)), dims=("y", "x")), scale_factors=[2]
        )
        sdata = SpatialData(images={"image": base.images["image"]}, labels={"labels": ms})
        sq.experimental.tl.calculate_tiling_qc(
            sdata, labels_key="labels", scale="scale0", tile_size=200, nmads_cut=1.0, nmads_smoothed=1.5
        )
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels")
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in sdata.tables["labels_qc"].obs.columns

    def test_obs_and_uns_survive_zarr_roundtrip(self, sdata_tile_boundary, tmp_path):
        from spatialdata import read_zarr

        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata, min_confidence=0.5)
        sdata.write(tmp_path / "roundtrip.zarr")
        a2 = read_zarr(tmp_path / "roundtrip.zarr").tables["labels_qc"]
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in a2.obs.columns
        assert "tiling_stitch" in a2.uns


class TestPairingContract:
    """Function-level tests of the seam-restricted, rank-based pairing pipeline.

    These exercise the stitcher directly on a hand-built two-sided cut (given the
    seam bands + data scale that ``calculate_tiling_qc`` would supply), isolating
    the pairing/scoring logic from detection.  A dense synthetic fixture is a poor
    end-to-end vehicle here -- its uniform geometry makes seam *detection*
    over-flag -- so the two-sided merge contract is locked in deterministically at
    this level instead.
    """

    @staticmethod
    def _two_sided_labels():
        # One cell cut into a left half (id 1) and a right half (id 2) across a
        # vertical seam at x~101, plus an off-seam distractor (id 3).  bboxes use
        # the skimage convention (max exclusive), as _compute_outlier_bboxes returns.
        H, W = 60, 220
        arr = np.zeros((H, W), dtype=np.int32)
        arr[20:40, 80:99] = 1  # left half:  cols 80..98
        arr[20:40, 104:123] = 2  # right half: cols 104..122 (6 px seam gap)
        arr[20:40, 160:180] = 3  # distractor, far from the seam
        bboxes = {1: (20, 80, 40, 99), 2: (20, 104, 40, 123), 3: (20, 160, 40, 180)}
        seams = {"v": [(101.0, 6.0, 50)], "h": []}
        diameter = 20.0
        return arr, bboxes, seams, diameter, H, W

    def test_facing_halves_merge_off_seam_cell_ignored(self):
        from squidpy.experimental.tl import _tiling_stitch as ts

        arr, bboxes, seams, diameter, H, W = self._two_sided_labels()
        edges, crops = ts._extract_cut_edges(arr, [1, 2, 3], bboxes, seams, diameter)
        # Both halves put a cut edge on the seam; the off-seam distractor does not.
        assert {e.cell_id for e in edges} == {1, 2}

        cands = ts._enumerate_pair_candidates(edges, k_neighbors=5, candidate_min_iou=0.2)
        pairs = ts._score_pairs(cands, bboxes, crops, 0.6, diameter, seams, close_radius_min=2, H=H, W=W)
        merged = [p for p in pairs if {p.cell_a, p.cell_b} == {1, 2}]
        assert len(merged) == 1
        assert merged[0].confidence >= 0.6
        assert 3 not in {p.cell_a for p in pairs} | {p.cell_b for p in pairs}

    def test_enumeration_is_rank_based_not_absolute_gap(self):
        # The two halves sit 6 px apart -- far beyond the old 3 px max_gap default.
        # Rank-based (k-NN) enumeration must still surface them as a candidate.
        from squidpy.experimental.tl import _tiling_stitch as ts

        arr, bboxes, seams, diameter, _H, _W = self._two_sided_labels()
        edges, _ = ts._extract_cut_edges(arr, [1, 2], bboxes, seams, diameter)
        cands = ts._enumerate_pair_candidates(edges, k_neighbors=5, candidate_min_iou=0.2)
        pair_ids = {(min(e.cell_id, c.cell_id), max(e.cell_id, c.cell_id)) for e, c, _ in cands}
        assert (1, 2) in pair_ids

    def test_pieces_farther_apart_than_the_seam_are_not_merged(self):
        """A cut's two halves cannot be separated by more than the seam band they lie on.

        The closing radius is scaled to each pair's own gap, so without a bound tied to the
        measured seam any two aligned blobs get bridged by a disk large enough to join them
        and then score as one compact, solid cell.
        """
        from squidpy.experimental.tl import _tiling_stitch as ts

        H, W = 60, 260
        arr = np.zeros((H, W), dtype=np.int32)
        arr[20:40, 80:99] = 1  # left piece,  cols 80..98
        arr[20:40, 127:146] = 2  # right piece, cols 127..145 -> 29 px apart
        bboxes = {1: (20, 80, 40, 99), 2: (20, 127, 40, 146)}
        # a ~20 px wide seam band covering both edges: the pieces are still farther apart
        # from each other than the seam itself is wide, so they are not one cut cell.
        seams = {"v": [(112.5, 10.0, 50)], "h": []}
        diameter = 20.0

        edges, crops = ts._extract_cut_edges(arr, [1, 2], bboxes, seams, diameter)
        cands = ts._enumerate_pair_candidates(edges, k_neighbors=5, candidate_min_iou=0.2)
        pairs = ts._score_pairs(cands, bboxes, crops, 0.6, diameter, seams, close_radius_min=2, H=H, W=W)
        assert not [p for p in pairs if {p.cell_a, p.cell_b} == {1, 2}], (
            "pieces 29 px apart across a 20 px seam were merged"
        )


class TestStitchVisual(PlotTester, metaclass=PlotTesterMeta):
    _ZOOM = (150, 250, 250, 350)
    _SEAM_Y = 200

    def test_plot_seam_group_recolor(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        sq.experimental.tl.calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, nmads_cut=1.0, nmads_smoothed=1.5
        )
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", min_confidence=0.5)
        adata = sdata.tables["labels_qc"]

        labels = np.asarray(sdata.labels["labels"].values)
        lut = np.arange(int(labels.max()) + 1)
        lut[adata.obs["label_id"].astype(int).to_numpy()] = adata.obs["stitch_group_id"].astype(int).to_numpy()
        regrouped = lut[labels]

        rng = np.random.default_rng(0)
        colors = rng.random((int(labels.max()) + 1, 3))
        colors[0] = 0.0

        y0, y1, x0, x1 = self._ZOOM
        before = colors[labels][y0:y1, x0:x1]  # coloured by label_id (cut pieces differ)
        after = colors[regrouped][y0:y1, x0:x1]  # coloured by stitch_group_id (pieces share a colour)
        seam = self._SEAM_Y - y0
        for panel in (before, after):
            panel[seam, ::4] = 1.0  # dashed seam marker, drawn into the array (no mpl line AA)
        sep = np.ones((before.shape[0], 4, 3))  # white column between the two panels
        combined = np.concatenate([before, sep, after], axis=1)

        # Render 1:1 (figsize * DPI == array shape) on a full-figure axis. No
        # upscaling -> no nearest-neighbour resampling, no text, no line AA, so the
        # PNG is pixel-identical across platforms/matplotlib versions (the earlier
        # tight_layout + upscaled imshow drifted by RMS ~53/28 between Linux/macOS).
        h, w = combined.shape[:2]
        fig = plt.figure(figsize=(w / DPI, h / DPI))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(combined, interpolation="nearest")
        ax.set_axis_off()
