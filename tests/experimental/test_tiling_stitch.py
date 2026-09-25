"""Tests for tile-cut cell stitching."""

from __future__ import annotations

import logging

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


def _qc_rerun_hint(sdata, **qc_kwargs) -> str:
    """Re-run QC and return the restore call it logs (the spatialdata logger does not propagate to caplog)."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("spatialdata._logging")
    logger.addHandler(handler)
    try:
        sq.experimental.tl.calculate_tiling_qc(sdata, **qc_kwargs)
    finally:
        logger.removeHandler(handler)
    (hint,) = [r.getMessage().split("To restore them, run: ")[1] for r in records if "To restore" in r.getMessage()]
    return hint


class TestAssignStitchGroups:
    """Tests for sq.experimental.tl.assign_stitch_groups using the tile-boundary fixture."""

    def test_columns_present(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata)
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in adata.obs.columns

    def test_confidence_convention(self, sdata_tile_boundary):
        # NaN = not evaluated (non-outlier), 1.0 = solo outlier, composite = stitched.
        sdata, _ = sdata_tile_boundary
        obs = _run_qc_and_stitch(sdata, min_confidence=0.5).obs

        non_outliers = ~obs["is_outlier"].astype(bool)
        assert non_outliers.sum() > 0
        assert obs.loc[non_outliers, "stitch_confidence"].isna().all()
        assert (obs.loc[non_outliers, "stitch_group_id"] == obs.loc[non_outliers, "label_id"]).all()
        assert (obs.loc[non_outliers, "n_pieces"] == 1).all()

        solo = obs["is_outlier"].astype(bool) & ~obs["is_stitched"].astype(bool)
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
        meta = _run_qc_and_stitch(sdata, min_confidence=0.7, max_gap=4.0).uns["tiling_stitch"]
        assert meta["min_confidence"] == 0.7
        assert meta["max_gap"] == 4.0
        assert meta["close_radius"] == 3
        assert "model_coefficients" not in meta and "model_intercept" not in meta
        assert set(meta["score_features"]) == {
            "iou",
            "endpoint_match",
            "merge_compactness",
            "merge_solidity",
            "gap_proximity",
        }

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"labels_key": "labels"}, "QC table"),
            ({"labels_key": "bogus"}, "not found in sdata.labels"),
            ({"labels_key": "labels", "min_confidence": 1.5}, "min_confidence"),
            ({"labels_key": "labels", "min_edge_coverage": 1.5}, "min_edge_coverage"),
        ],
        ids=["missing_qc_table", "missing_labels_key", "invalid_min_confidence", "invalid_min_edge_coverage"],
    )
    def test_invalid_input_raises(self, sdata_tile_boundary, kwargs, match):
        sdata, _ = sdata_tile_boundary
        with pytest.raises(ValueError, match=match):
            sq.experimental.tl.assign_stitch_groups(sdata, **kwargs)

    @pytest.mark.parametrize("table_key", [None, "my_qc"], ids=["default_table", "custom_table"])
    def test_qc_rerun_hint_is_runnable(self, sdata_tile_boundary, table_key):
        # re-running QC drops the stitch columns and logs the call that restores them
        sdata, _ = sdata_tile_boundary
        qc = {"labels_key": "labels", "tile_size": 200, "table_key_added": table_key}
        sq.experimental.tl.calculate_tiling_qc(sdata, nmads_cut=1.0, nmads_smoothed=1.5, **qc)
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", qc_table_key=table_key, distance_tol=1.0)
        hint = _qc_rerun_hint(sdata, **qc)
        assert "distance_tol=1.0" in hint
        eval(hint, {"sq": sq, "sdata": sdata})
        assert "stitch_group_id" in sdata.tables[table_key or "labels_qc"].obs

    def test_qc_rerun_hint_reads_nested_stitch_params(self, sdata_tile_boundary):
        # tables saved before the knobs were flattened keep them under `stitch_params`
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        meta = sdata.tables["labels_qc"].uns["tiling_stitch"]
        meta["stitch_params"] = {"close_radius": 5}
        del meta["close_radius"]
        assert "close_radius=5" in _qc_rerun_hint(sdata, labels_key="labels", tile_size=200)

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
