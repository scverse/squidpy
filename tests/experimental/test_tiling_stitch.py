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
from tests.conftest import PlotTester, PlotTesterMeta


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

    def test_uns_records_params_and_features(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        meta = _run_qc_and_stitch(sdata, min_confidence=0.7, max_gap=4.0).uns["tiling_stitch"]
        assert meta["min_confidence"] == 0.7
        assert meta["max_gap"] == 4.0
        assert isinstance(meta["stitch_params"], dict)
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


class TestScorePairsEarlyPrune:
    """The ``min_confidence`` early-prune is an optimization, so it cannot change
    results (covered by the public tests above). This unit test asserts the other
    half: a pair whose optimistic score can't reach the threshold skips the costly
    ``_merge_shape_features`` union reconstruction entirely.
    """

    @staticmethod
    def _edge(cell_id: int, normal_dir: int, coord: float):
        from squidpy.experimental.tl._tiling_stitch import _CutEdge

        return _CutEdge(cell_id=cell_id, axis="h", coord=coord, extent=(0.0, 10.0), normal_dir=normal_dir, length=10.0)

    def test_low_bound_pair_skips_shape_features(self, monkeypatch):
        import squidpy.experimental.tl._tiling_stitch as ts

        calls: list[tuple[int, int]] = []

        def _spy(cell_a, cell_b, *args, **kwargs):
            calls.append((cell_a, cell_b))
            return {"merge_solidity": 1.0, "merge_compactness": 1.0}

        monkeypatch.setattr(ts, "_merge_shape_features", _spy)

        # weak: optimistic bound (0 + 0 + gap_prox(0) + 2) / 5 = 0.4 < 0.5 -> pruned
        weak = (self._edge(1, 1, 0.0), self._edge(2, -1, 50.0), {"iou": 0.0, "endpoint_match": 0.0, "gap": 50.0})
        # strong: bound (1 + 1 + 1 + 2) / 5 = 1.0 -> evaluated and kept
        strong = (self._edge(3, 1, 0.0), self._edge(4, -1, 0.0), {"iou": 1.0, "endpoint_match": 1.0, "gap": 0.0})

        pairs = ts._score_pairs(
            [weak, strong], bboxes={}, outlier_crops={}, min_confidence=0.5, close_radius=2, H=100, W=100
        )

        assert calls == [(3, 4)]  # only the strong pair reached the shape step
        assert [(p.cell_a, p.cell_b) for p in pairs] == [(3, 4)]


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
        # Fixed subplot geometry (no tight_layout / titles): tight_layout sizes the
        # axes from the title text extents, which differ across platforms (fonts),
        # shifting the imshow sub-pixel so every cell edge mismatches. With a fixed
        # layout the image renders to identical pixels on every platform.
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.04)
        for ax, arr in zip(axes, [labels, regrouped], strict=True):  # left = before, right = after
            ax.imshow(colors[arr][y0:y1, x0:x1], interpolation="nearest")
            ax.axhline(self._SEAM_Y - y0, color="white", linestyle="--", linewidth=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
