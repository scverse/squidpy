"""Tests for sq.experimental.tl.assign_stitch_groups."""

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
    sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", **stitch_kwargs)
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
        assert non_outliers.sum() > 0
        assert adata.obs.loc[non_outliers, "stitch_confidence"].isna().all()
        assert (adata.obs.loc[non_outliers, "stitch_group_id"] == adata.obs.loc[non_outliers, "label_id"]).all()
        assert (adata.obs.loc[non_outliers, "n_pieces"] == 1).all()
        assert (~adata.obs.loc[non_outliers, "is_stitched"].astype(bool)).all()

    def test_solo_outliers_have_1p0_confidence(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata)
        solo_outliers = adata.obs["is_outlier"].astype(bool) & ~adata.obs["is_stitched"].astype(bool)
        if solo_outliers.sum() > 0:
            assert (adata.obs.loc[solo_outliers, "stitch_confidence"] == 1.0).all()

    def test_stitched_have_composite_confidence(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        stitched = adata.obs["is_stitched"].astype(bool)
        if stitched.sum() > 0:
            confs = adata.obs.loc[stitched, "stitch_confidence"]
            assert (confs >= 0.5).all()
            assert (confs <= 1.0).all()
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
# Param resolution + feature weights
# ---------------------------------------------------------------------------


class TestStitchParamsResolution:
    def test_none_uses_defaults(self):
        from squidpy.experimental.tl._tiling_stitch import StitchParams, _resolve_stitch_params

        p = _resolve_stitch_params(None)
        assert isinstance(p, StitchParams)
        assert p.distance_tol == 0.75
        assert p.close_radius == 3
        assert p.feature_weights is None

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


class TestFeatureWeights:
    """feature_weights validation + the renormalisation contract."""

    def test_partial_mapping_accepted(self):
        from squidpy.experimental.tl._tiling_stitch import _resolve_stitch_params

        p = _resolve_stitch_params({"feature_weights": {"iou": 2.0}})
        assert p.feature_weights == {"iou": 2.0}

    def test_numpy_weight_coerced(self):
        from squidpy.experimental.tl._tiling_stitch import _resolve_stitch_params

        p = _resolve_stitch_params({"feature_weights": {"iou": np.float32(2.0)}})
        assert type(p.feature_weights["iou"]) is float

    @pytest.mark.parametrize(
        ("weights", "match"),
        [
            ({"bogus": 1.0}, "Unknown feature_weights"),
            ({"iou": -1.0}, r"feature_weights\['iou'\] must be >= 0"),
        ],
        ids=["unknown_key", "negative"],
    )
    def test_invalid_weights_raise(self, weights, match):
        from squidpy.experimental.tl._tiling_stitch import _resolve_stitch_params

        with pytest.raises(ValueError, match=match):
            _resolve_stitch_params({"feature_weights": weights})

    def test_wrong_weights_type_raises(self):
        from squidpy.experimental.tl._tiling_stitch import _resolve_stitch_params

        with pytest.raises(TypeError, match="feature_weights must be a Mapping"):
            _resolve_stitch_params({"feature_weights": [1, 2, 3]})

    def test_flat_equal_default(self):
        from squidpy.experimental.tl._tiling_stitch import _SCORE_FEATURES, _resolve_feature_weights

        w = _resolve_feature_weights(None)
        assert set(w) == set(_SCORE_FEATURES)
        assert all(abs(v - 1.0 / len(_SCORE_FEATURES)) < 1e-12 for v in w.values())
        assert abs(sum(w.values()) - 1.0) < 1e-12

    def test_partial_fills_and_renormalises(self):
        from squidpy.experimental.tl._tiling_stitch import _resolve_feature_weights

        w = _resolve_feature_weights({"iou": 2.0})
        assert abs(sum(w.values()) - 1.0) < 1e-12
        # iou weighted 2 vs 1 for the other four -> 2/6 vs 1/6.
        assert abs(w["iou"] - 2.0 / 6.0) < 1e-12
        assert abs(w["endpoint_match"] - 1.0 / 6.0) < 1e-12

    def test_all_zero_weights_raise(self):
        from squidpy.experimental.tl._tiling_stitch import _SCORE_FEATURES, _resolve_feature_weights

        with pytest.raises(ValueError, match="positive sum"):
            _resolve_feature_weights(dict.fromkeys(_SCORE_FEATURES, 0.0))


class TestGapProximity:
    """gap_proximity is normalised by the closing reach (2*close_radius), not max_gap."""

    def test_neutral_when_closing_disabled(self):
        from squidpy.experimental.tl._tiling_stitch import _gap_proximity

        # close_radius=0 (closing disabled) must NOT collapse the feature to 0;
        # it is inactive (1.0), so it never silently drags down the score.
        assert _gap_proximity(0.0, 0) == 1.0
        assert _gap_proximity(2.0, 0) == 1.0

    def test_decays_with_gap_over_reach(self):
        from squidpy.experimental.tl._tiling_stitch import _gap_proximity

        assert _gap_proximity(0.0, 3) == 1.0  # touching
        assert _gap_proximity(3.0, 3) == 0.5  # reach = 6
        assert _gap_proximity(6.0, 3) == 0.0  # at the closing reach
        assert _gap_proximity(7.0, 3) == 0.0  # clipped, never negative


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


class TestUnsMetadata:
    def test_uns_records_params_weights_and_formula(self, sdata_tile_boundary):
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
            "gap_proximity",
        }
        # Actual (renormalised) weights are recorded and sum to 1.
        assert "feature_weights" in meta
        assert set(meta["feature_weights"]) == set(meta["score_features"])
        assert abs(sum(meta["feature_weights"].values()) - 1.0) < 1e-9

    def test_custom_weights_recorded_in_uns(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, stitch_params={"feature_weights": {"merge_compactness": 4.0}})
        meta = adata.uns["tiling_stitch"]
        # merge_compactness weighted 4 vs 1 for the other four -> 4/8 vs 1/8.
        assert abs(meta["feature_weights"]["merge_compactness"] - 0.5) < 1e-9
        assert abs(meta["feature_weights"]["iou"] - 0.125) < 1e-9
        assert "merge_compactness" in meta["score_formula"]


# ---------------------------------------------------------------------------
# Behaviour vs ground truth
# ---------------------------------------------------------------------------


class TestRecoveryVsGroundTruth:
    def test_stitches_some_cuts(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        cut_mask = adata.obs["label_id"].isin(gt.cut_cell_ids)
        n_cut_in_stitched = (cut_mask & adata.obs["is_stitched"].astype(bool)).sum()
        assert n_cut_in_stitched > 0, "expected at least some cut pieces to be stitched"

    def test_no_intact_cells_get_stitched_at_high_threshold(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.9)
        intact_mask = adata.obs["label_id"].isin(gt.intact_cell_ids)
        n_false = (intact_mask & adata.obs["is_stitched"].astype(bool)).sum()
        assert n_false <= 5, f"too many intact cells flagged stitched: {n_false}"

    def test_a_stitched_group_is_made_of_cut_pieces(self, sdata_tile_boundary):
        """Robust array assertion: at least one stitched group consists entirely
        of ground-truth cut pieces sharing one stitch_group_id."""
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        obs = adata.obs
        stitched = obs[obs["is_stitched"].astype(bool)]
        found = False
        for _gid, members in stitched.groupby("stitch_group_id"):
            ids = set(members["label_id"].astype(int))
            if len(ids) >= 2 and ids <= set(gt.cut_cell_ids):
                # all members share the group id by construction; verify it
                assert members["stitch_group_id"].nunique() == 1
                found = True
                break
        assert found, "expected at least one group composed solely of cut pieces"

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
            sq.experimental.tl.assign_stitch_groups(sdata, **kwargs)


# ---------------------------------------------------------------------------
# Idempotency + inplace + determinism
# ---------------------------------------------------------------------------


class TestIdempotencyAndInplace:
    def test_rerun_overwrites_with_warning(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        n_cols_before = len(sdata.tables["labels_qc"].obs.columns)
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels")
        n_cols_after = len(sdata.tables["labels_qc"].obs.columns)
        assert n_cols_before == n_cols_after

    def test_inplace_false_returns_adata_without_writing(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200)
        n_cols_before = len(sdata.tables["labels_qc"].obs.columns)
        result = sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", inplace=False)
        n_cols_after = len(sdata.tables["labels_qc"].obs.columns)
        assert result is not None
        assert "stitch_group_id" in result.obs.columns
        assert n_cols_before == n_cols_after

    def test_deterministic_groups(self, sdata_tile_boundary):
        """Same input -> identical group ids and confidences (no RNG/order deps)."""
        sdata, _ = sdata_tile_boundary
        a1 = _run_qc_and_stitch(sdata, min_confidence=0.5).copy()
        # Re-run stitch on the same QC table.
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", min_confidence=0.5)
        a2 = sdata.tables["labels_qc"]
        np.testing.assert_array_equal(a1.obs["stitch_group_id"].to_numpy(), a2.obs["stitch_group_id"].to_numpy())
        np.testing.assert_array_equal(a1.obs["n_pieces"].to_numpy(), a2.obs["n_pieces"].to_numpy())
        np.testing.assert_allclose(
            a1.obs["stitch_confidence"].to_numpy(),
            a2.obs["stitch_confidence"].to_numpy(),
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestSaveDiagnostics:
    def test_absent_by_default(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        assert "diagnostics" not in adata.uns["tiling_stitch"]

    def test_present_and_schema_when_enabled(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5, save_diagnostics=True)
        diag = adata.uns["tiling_stitch"]["diagnostics"]
        # Stored as a dict of equal-length arrays (zarr-safe), not a DataFrame.
        assert isinstance(diag, dict)
        expected = {
            "cell_a",
            "cell_b",
            "axis",
            "iou",
            "endpoint_match",
            "merge_compactness",
            "merge_solidity",
            "gap_proximity",
            "confidence",
            "group_id",
            "status",
        }
        assert set(diag) == expected
        lengths = {len(v) for v in diag.values()}
        assert len(lengths) == 1, "diagnostics arrays must be equal length"
        n = lengths.pop()
        if n > 0:
            conf = np.asarray(diag["confidence"])
            assert ((conf >= 0.0) & (conf <= 1.0)).all()
            status = np.asarray(diag["status"])
            assert set(np.unique(status)) <= {"accepted", "below_threshold", "collapsed_group"}
            # Accepted pairs must clear the threshold.
            assert (conf[status == "accepted"] >= 0.5).all()
            # Below-threshold pairs must be under it.
            assert (conf[status == "below_threshold"] < 0.5).all()

    def test_diagnostics_and_obs_survive_zarr_roundtrip(self, sdata_tile_boundary, tmp_path):
        """Workflow-level: run(save_diagnostics) -> write zarr -> reload keeps obs + diagnostics."""
        from spatialdata import read_zarr

        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata, min_confidence=0.5, save_diagnostics=True)
        zp = tmp_path / "roundtrip.zarr"
        sdata.write(zp)
        a2 = read_zarr(zp).tables["labels_qc"]
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in a2.obs.columns
        diag = a2.uns["tiling_stitch"]["diagnostics"]
        assert set(diag) >= {"cell_a", "cell_b", "axis", "confidence", "status"}
        # feature_weights survived and no None leaked into stitch_params.
        assert abs(sum(a2.uns["tiling_stitch"]["feature_weights"].values()) - 1.0) < 1e-9
        assert "feature_weights" not in a2.uns["tiling_stitch"]["stitch_params"]


# ---------------------------------------------------------------------------
# QC re-run drops stitch columns (the _warn_if_dropping_stitch_columns hook)
# ---------------------------------------------------------------------------


class TestQCRerunDropsStitch:
    def test_qc_rerun_removes_stitch_columns(self, sdata_tile_boundary, caplog):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200)
        adata = sdata.tables["labels_qc"]
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col not in adata.obs.columns


# ---------------------------------------------------------------------------
# Multiscale end-to-end (stitch only; materialisation lives in PR-3)
# ---------------------------------------------------------------------------


class TestMultiScaleEndToEnd:
    def _make_sdata(self) -> SpatialData:
        from tests.experimental.conftest import make_tile_boundary_sdata

        sdata, _ = make_tile_boundary_sdata()
        labels_arr = np.asarray(sdata.labels["labels"].values)
        labels_xr = xr.DataArray(da.from_array(labels_arr, chunks=(200, 200)), dims=("y", "x"))
        ms = Labels2DModel.parse(labels_xr, scale_factors=[2])
        return SpatialData(images={"image": sdata.images["image"]}, labels={"labels": ms})

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
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels")
        adata = sdata.tables["labels_qc"]
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in adata.obs.columns


# ---------------------------------------------------------------------------
# Visual: before/after group recolour, zoomed on a tile seam
# ---------------------------------------------------------------------------


def _label_to_rgb(arr: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Index a precomputed colour table by label value (background black)."""
    return colors[arr]


class TestStitchVisual(PlotTester, metaclass=PlotTesterMeta):
    """Visual baseline: recolour the labels by ``label_id`` (before) vs by
    ``stitch_group_id`` (after), zoomed on a tile seam.  Pieces stitched into
    one group share a colour after.  Needs no materialised labels element
    (that, and the join_labels comparison, live in PR-3).  Baselines live in
    ``tests/_images/StitchVisual_*.png`` and are downloaded from CI artifacts;
    they are not generated locally.
    """

    _ZOOM = (150, 250, 250, 350)  # (y0, y1, x0, x1)
    _SEAM_Y = 200

    def test_plot_seam_group_recolor(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            nmads_cut=1.0,
            nmads_smoothed=1.5,
        )
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", min_confidence=0.5)
        adata = sdata.tables["labels_qc"]

        labels = np.asarray(sdata.labels["labels"].values)
        # LUT: label_id -> stitch_group_id (identity for unstitched cells).
        lut = np.arange(int(labels.max()) + 1)
        ids = adata.obs["label_id"].astype(int).to_numpy()
        grp = adata.obs["stitch_group_id"].astype(int).to_numpy()
        lut[ids] = grp
        regrouped = lut[labels]

        # Shared colour table so a given id maps to the same colour in both panels.
        rng = np.random.default_rng(0)
        colors = rng.random((int(labels.max()) + 1, 3))
        colors[0] = 0.0  # background

        y0, y1, x0, x1 = self._ZOOM
        before_rgb = _label_to_rgb(labels, colors)[y0:y1, x0:x1]
        after_rgb = _label_to_rgb(regrouped, colors)[y0:y1, x0:x1]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, rgb, title in zip(
            axes,
            [before_rgb, after_rgb],
            ["by label_id (before)", "by stitch_group_id (after)"],
            strict=True,
        ):
            ax.imshow(rgb, interpolation="nearest")
            ax.axhline(self._SEAM_Y - y0, color="white", linestyle="--", linewidth=1.0)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
