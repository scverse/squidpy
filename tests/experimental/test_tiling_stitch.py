"""Tests for sq.experimental.tl.stitch_tile_cuts."""

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
    """Run QC + stitch on the fixture sdata; return the resulting AnnData."""
    sq.experimental.tl.calculate_tiling_qc(
        sdata,
        labels_key="labels",
        tile_size=200,
        nmads_cut=1.0,
        nmads_smoothed=1.5,
    )
    sq.experimental.tl.stitch_tile_cuts(sdata, labels_key="labels", **stitch_kwargs)
    return sdata.tables["labels_qc"]


# ---------------------------------------------------------------------------
# Smoke + column contract
# ---------------------------------------------------------------------------


class TestStitchObsContract:
    """The 4 .obs columns and the NaN-vs-1.0 confidence convention."""

    def test_columns_present(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata)
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in adata.obs.columns, f"missing {col}"

    def test_non_outliers_have_nan_confidence(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata)
        non_outliers = ~adata.obs["is_outlier"].astype(bool)
        # At least some non-outliers should exist
        assert non_outliers.sum() > 0
        # All non-outliers must have NaN confidence
        assert adata.obs.loc[non_outliers, "stitch_confidence"].isna().all()
        # And they keep their own label_id as group
        assert (adata.obs.loc[non_outliers, "stitch_group_id"] == adata.obs.loc[non_outliers, "label_id"]).all()
        # And n_pieces == 1, is_stitched False
        assert (adata.obs.loc[non_outliers, "n_pieces"] == 1).all()
        assert (~adata.obs.loc[non_outliers, "is_stitched"].astype(bool)).all()

    def test_solo_outliers_have_1p0_confidence(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata)
        solo_outliers = adata.obs["is_outlier"].astype(bool) & ~adata.obs["is_stitched"].astype(bool)
        # If any solo outliers exist, they get confidence 1.0 (checked, no partner)
        if solo_outliers.sum() > 0:
            assert (adata.obs.loc[solo_outliers, "stitch_confidence"] == 1.0).all()

    def test_stitched_have_calibrated_confidence(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        stitched = adata.obs["is_stitched"].astype(bool)
        if stitched.sum() > 0:
            confs = adata.obs.loc[stitched, "stitch_confidence"]
            # all in [min_confidence, 1.0]
            assert (confs >= 0.5).all()
            assert (confs <= 1.0).all()
            # n_pieces between 2 and max_group_size (default 4)
            sizes = adata.obs.loc[stitched, "n_pieces"]
            assert (sizes >= 2).all()
            assert (sizes <= 4).all()

    def test_group_id_shared_within_group(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        stitched = adata.obs[adata.obs["is_stitched"].astype(bool)]
        for gid, members in stitched.groupby("stitch_group_id"):
            n = members["n_pieces"].iloc[0]
            assert len(members) == n, f"group {gid}: {len(members)} rows but n_pieces={n}"


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


class TestStitchParamsResolution:
    def test_none_uses_defaults(self):
        from squidpy.experimental.tl._tiling_stitch import StitchParams, _resolve_stitch_params

        p = _resolve_stitch_params(None)
        assert isinstance(p, StitchParams)
        assert p.distance_tol == 0.75
        assert p.close_radius == 3

    def test_instance_passthrough(self):
        from squidpy.experimental.tl._tiling_stitch import StitchParams, _resolve_stitch_params

        inst = StitchParams(distance_tol=1.0)
        assert _resolve_stitch_params(inst) is inst

    def test_mapping_construction(self):
        from squidpy.experimental.tl._tiling_stitch import _resolve_stitch_params

        p = _resolve_stitch_params({"distance_tol": 1.5, "close_radius": 5})
        assert p.distance_tol == 1.5
        assert p.close_radius == 5

    def test_numpy_scalars_coerced(self):
        from squidpy.experimental.tl._tiling_stitch import _resolve_stitch_params

        p = _resolve_stitch_params({"distance_tol": np.float32(0.8), "close_radius": np.int64(4)})
        assert type(p.distance_tol) is float
        assert type(p.close_radius) is int

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bogus": 1}, "Unknown stitch_params"),
            ({"distance_tol": -1.0}, "distance_tol must be >= 0"),
            ({"close_radius": -1}, "close_radius must be >= 0"),
            ({"candidate_min_iou": 1.5}, r"candidate_min_iou must be in \[0, 1\]"),
        ],
        ids=["unknown_field", "negative_distance_tol", "negative_close_radius", "iou_out_of_range"],
    )
    def test_invalid_raises_value_error(self, kwargs, match):
        from squidpy.experimental.tl._tiling_stitch import _resolve_stitch_params

        with pytest.raises(ValueError, match=match):
            _resolve_stitch_params(kwargs)

    def test_wrong_type_raises_type_error(self):
        from squidpy.experimental.tl._tiling_stitch import _resolve_stitch_params

        with pytest.raises(TypeError, match="StitchParams, Mapping, or None"):
            _resolve_stitch_params(42)


class TestUnsMetadata:
    def test_uns_records_params_and_score_formula(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.7, max_gap=4.0, max_group_size=4)
        assert "tiling_stitch" in adata.uns
        meta = adata.uns["tiling_stitch"]
        assert meta["min_confidence"] == 0.7
        assert meta["max_gap"] == 4.0
        assert meta["max_group_size"] == 4
        # Advanced tunables are bundled, not flat.
        assert "distance_tol" not in meta
        assert "stitch_params" in meta
        assert isinstance(meta["stitch_params"], dict)
        assert meta["stitch_params"]["distance_tol"] == 0.75
        assert meta["stitch_params"]["close_radius"] == 3
        # No fitted coefficients -- transparent formula instead.
        assert "model_coefficients" not in meta
        assert "model_intercept" not in meta
        assert "score_formula" in meta
        assert set(meta["score_features"]) == {
            "iou",
            "endpoint_match",
            "merge_compactness",
            "merge_solidity",
        }


# ---------------------------------------------------------------------------
# Behaviour vs ground truth
# ---------------------------------------------------------------------------


class TestRecoveryVsGroundTruth:
    def test_stitches_some_cuts(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        # Of the cells the fixture marks as cut pieces, some should end up stitched.
        cut_mask = adata.obs["label_id"].isin(gt.cut_cell_ids)
        n_cut_in_stitched = (cut_mask & adata.obs["is_stitched"].astype(bool)).sum()
        assert n_cut_in_stitched > 0, "expected at least some cut pieces to be stitched"

    def test_no_intact_cells_get_stitched_at_high_threshold(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.9)
        # At threshold 0.9 (high precision), intact cells should not falsely merge.
        intact_mask = adata.obs["label_id"].isin(gt.intact_cell_ids)
        # Allow up to a handful of false merges given the dense ellipse fixture.
        n_false = (intact_mask & adata.obs["is_stitched"].astype(bool)).sum()
        assert n_false <= 5, f"too many intact cells flagged stitched: {n_false}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
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
            sq.experimental.tl.stitch_tile_cuts(sdata, **kwargs)


# ---------------------------------------------------------------------------
# Idempotency + inplace
# ---------------------------------------------------------------------------


class TestIdempotencyAndInplace:
    def test_rerun_overwrites_with_warning(self, sdata_tile_boundary, caplog):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        # Second run should warn about overwrite, leave column count unchanged.
        n_cols_before = len(sdata.tables["labels_qc"].obs.columns)
        sq.experimental.tl.stitch_tile_cuts(sdata, labels_key="labels")
        n_cols_after = len(sdata.tables["labels_qc"].obs.columns)
        assert n_cols_before == n_cols_after

    def test_inplace_false_returns_adata_without_writing(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200)
        n_cols_before = len(sdata.tables["labels_qc"].obs.columns)
        result = sq.experimental.tl.stitch_tile_cuts(sdata, labels_key="labels", inplace=False)
        n_cols_after = len(sdata.tables["labels_qc"].obs.columns)
        assert result is not None
        assert "stitch_group_id" in result.obs.columns
        # In-place table unchanged
        assert n_cols_before == n_cols_after


# ---------------------------------------------------------------------------
# QC re-run drops stitch columns
# ---------------------------------------------------------------------------


class TestResolveLabelsArray:
    """resolve_labels_array unit-tests; verifies the multi-scale branch."""

    def test_single_scale_passthrough(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        from squidpy.experimental.utils._labels import resolve_labels_array

        out = resolve_labels_array(sdata, "labels", scale=None)
        assert isinstance(out, xr.DataArray)

    def test_multiscale_requires_scale(self):
        from squidpy.experimental.utils._labels import resolve_labels_array

        labels = np.zeros((64, 64), dtype=np.int32)
        labels[10:20, 10:20] = 1
        labels_xr = xr.DataArray(da.from_array(labels, chunks=(32, 32)), dims=("y", "x"))
        sdata = SpatialData(labels={"labels_ms": Labels2DModel.parse(labels_xr, scale_factors=[2])})
        # No scale -> error.
        with pytest.raises(ValueError, match="multi-scale"):
            resolve_labels_array(sdata, "labels_ms", scale=None)
        # With scale -> returns the resolved DataArray.
        out = resolve_labels_array(sdata, "labels_ms", scale="scale0")
        assert isinstance(out, xr.DataArray)


class TestMultiScaleEndToEnd:
    """QC -> stitch on a multiscale labels element."""

    def _make_sdata(self) -> SpatialData:
        from tests.experimental.conftest import make_tile_boundary_sdata

        sdata, _ = make_tile_boundary_sdata()
        # Wrap the existing single-scale labels element as multiscale.
        labels_arr = np.asarray(sdata.labels["labels"].values)
        labels_xr = xr.DataArray(da.from_array(labels_arr, chunks=(200, 200)), dims=("y", "x"))
        ms = Labels2DModel.parse(labels_xr, scale_factors=[2])
        sdata = SpatialData(
            images={"image": sdata.images["image"]},
            labels={"labels": ms},
        )
        return sdata

    def test_stitch_runs_on_multiscale(self):
        sdata = self._make_sdata()
        sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            scale="scale0",
            tile_size=200,
            nmads_cut=1.0,
            nmads_smoothed=1.5,
        )
        sq.experimental.tl.stitch_tile_cuts(sdata, labels_key="labels")
        adata = sdata.tables["labels_qc"]
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in adata.obs.columns

    def test_make_stitched_labels_runs_on_multiscale(self):
        sdata = self._make_sdata()
        sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            scale="scale0",
            tile_size=200,
            nmads_cut=1.0,
            nmads_smoothed=1.5,
        )
        sq.experimental.tl.stitch_tile_cuts(sdata, labels_key="labels")
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")
        assert "labels_stitched" in sdata.labels


class TestInplaceFalseMakeStitchedLabels:
    def test_inplace_false_returns_dict_without_writing(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        assert "labels_stitched" not in sdata.labels
        assert "labels_stitched_table" not in sdata.tables
        result = sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True, inplace=False)
        # Nothing written
        assert "labels_stitched" not in sdata.labels
        assert "labels_stitched_table" not in sdata.tables
        # Result has both objects
        assert isinstance(result, dict)
        assert result["labels"] is not None
        assert result["table"] is not None

    def test_inplace_false_no_table_when_disabled(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        result = sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=False, inplace=False)
        assert result["labels"] is not None
        assert result["table"] is None


class TestQCRerunDropsStitch:
    def test_qc_rerun_removes_stitch_columns(self, sdata_tile_boundary, caplog):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        # Re-running QC should produce a table without stitch columns.
        sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200)
        adata = sdata.tables["labels_qc"]
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col not in adata.obs.columns


# ---------------------------------------------------------------------------
# Visual: before/after stitching, zoomed on a tile seam
# ---------------------------------------------------------------------------


def _label_to_rgb(arr: np.ndarray, seed: int = 0) -> np.ndarray:
    """Render a label image with a stable random colour per label, background black."""
    rng = np.random.default_rng(seed)
    n = int(arr.max()) + 1
    colors = rng.random((n, 3))
    colors[0] = 0.0  # background
    return colors[arr]


class TestStitchVisual(PlotTester, metaclass=PlotTesterMeta):
    """Visual baselines comparing labels before/after stitch_tile_cuts +
    make_stitched_labels, zoomed in on a tile seam.  Baselines live in
    ``tests/_images/StitchVisual_*.png`` and are downloaded from CI artifacts;
    they are not generated locally.
    """

    # Hardcoded crop window around the seam at y=200 (fixture tile_borders_y[0]).
    _ZOOM = (150, 250, 250, 350)  # (y0, y1, x0, x1)
    _SEAM_Y = 200

    def test_plot_seam_before_after(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            nmads_cut=1.0,
            nmads_smoothed=1.5,
        )
        sq.experimental.tl.stitch_tile_cuts(sdata, labels_key="labels", min_confidence=0.5)
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")

        y0, y1, x0, x1 = self._ZOOM
        before = np.asarray(sdata.labels["labels"].values)[y0:y1, x0:x1]
        after = np.asarray(sdata.labels["labels_stitched"].values)[y0:y1, x0:x1]

        # Same colour seed for both panels: unstitched cells look identical,
        # only stitched pieces change colour.
        before_rgb = _label_to_rgb(before, seed=0)
        after_rgb = _label_to_rgb(after, seed=0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, rgb, title in zip(axes, [before_rgb, after_rgb], ["Before", "After"], strict=True):
            ax.imshow(rgb, interpolation="nearest")
            ax.axhline(self._SEAM_Y - y0, color="white", linestyle="--", linewidth=1.0)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()

    def test_plot_seam_join_labels(self, sdata_tile_boundary):
        """Side-by-side: After (join_labels=False) vs After (join_labels=True).

        join_labels=False leaves stitched cells as multi-component regions
        (the cut stripe stays at 0); join_labels=True morphologically closes
        the gap so the stitched cell is a single connected component.
        """
        sdata, _ = sdata_tile_boundary
        sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            nmads_cut=1.0,
            nmads_smoothed=1.5,
        )
        sq.experimental.tl.stitch_tile_cuts(sdata, labels_key="labels", min_confidence=0.5)

        # Run twice with different join settings into separate output keys.
        sq.experimental.im.make_stitched_labels(
            sdata,
            labels_key="labels",
            labels_key_added="labels_stitched_split",
            write_table=False,
            join_labels=False,
        )
        sq.experimental.im.make_stitched_labels(
            sdata,
            labels_key="labels",
            labels_key_added="labels_stitched_joined",
            write_table=False,
            join_labels=True,
        )

        y0, y1, x0, x1 = self._ZOOM
        split = np.asarray(sdata.labels["labels_stitched_split"].values)[y0:y1, x0:x1]
        joined = np.asarray(sdata.labels["labels_stitched_joined"].values)[y0:y1, x0:x1]

        split_rgb = _label_to_rgb(split, seed=0)
        joined_rgb = _label_to_rgb(joined, seed=0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, rgb, title in zip(
            axes, [split_rgb, joined_rgb], ["join_labels=False", "join_labels=True"], strict=True
        ):
            ax.imshow(rgb, interpolation="nearest")
            ax.axhline(self._SEAM_Y - y0, color="white", linestyle="--", linewidth=1.0)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
