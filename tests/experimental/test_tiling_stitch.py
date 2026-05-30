"""Tests for sq.experimental.tl.assign_stitch_groups.

Scope: user-observable behaviour and the obs/uns contract that
``make_stitched_labels`` (PR-C) consumes -- not the private scoring internals.
"""

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
    sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200, nmads_cut=1.0, nmads_smoothed=1.5)
    sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", **stitch_kwargs)
    return sdata.tables["labels_qc"]


# ---------------------------------------------------------------------------
# Obs contract (the four columns PR-C consumes) + confidence convention
# ---------------------------------------------------------------------------


class TestStitchObsContract:
    def test_columns_present(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata)
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in adata.obs.columns, f"missing {col}"

    def test_confidence_convention(self, sdata_tile_boundary):
        """NaN = not evaluated (non-outlier), 1.0 = solo outlier, composite = stitched."""
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        obs = adata.obs
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
        for gid, members in stitched.groupby("stitch_group_id"):
            assert len(members) == members["n_pieces"].iloc[0], f"group {gid} size mismatch"


# ---------------------------------------------------------------------------
# Uns audit block (params, weights, formula)
# ---------------------------------------------------------------------------


class TestUnsMetadata:
    def test_uns_records_params_weights_and_formula(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.7, max_gap=4.0)
        meta = adata.uns["tiling_stitch"]
        assert meta["min_confidence"] == 0.7
        assert meta["max_gap"] == 4.0
        assert isinstance(meta["stitch_params"], dict)
        # Transparent formula, no fitted-model artefacts.
        assert "model_coefficients" not in meta and "model_intercept" not in meta
        assert set(meta["score_features"]) == {
            "iou",
            "endpoint_match",
            "merge_compactness",
            "merge_solidity",
            "gap_proximity",
        }
        assert abs(sum(meta["feature_weights"].values()) - 1.0) < 1e-9

    def test_custom_weights_recorded(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, stitch_params={"feature_weights": {"merge_compactness": 4.0}})
        meta = adata.uns["tiling_stitch"]
        # 4 vs 1 for the other four -> 4/8 vs 1/8; recorded weights are the applied ones.
        assert abs(meta["feature_weights"]["merge_compactness"] - 0.5) < 1e-9
        assert abs(meta["feature_weights"]["iou"] - 0.125) < 1e-9


# ---------------------------------------------------------------------------
# Behaviour vs ground truth
# ---------------------------------------------------------------------------


class TestRecoveryVsGroundTruth:
    def test_a_stitched_group_is_made_of_cut_pieces(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        stitched = adata.obs[adata.obs["is_stitched"].astype(bool)]
        found = any(
            len(set(m["label_id"].astype(int))) >= 2 and set(m["label_id"].astype(int)) <= set(gt.cut_cell_ids)
            for _gid, m in stitched.groupby("stitch_group_id")
        )
        assert found, "expected at least one group composed solely of cut pieces"

    def test_no_intact_cells_stitched_at_high_threshold(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.9)
        intact = adata.obs["label_id"].isin(gt.intact_cell_ids)
        n_false = int((intact & adata.obs["is_stitched"].astype(bool)).sum())
        assert n_false <= 5, f"too many intact cells flagged stitched: {n_false}"


# ---------------------------------------------------------------------------
# Errors, idempotency, the QC-rerun hook, multiscale
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


class TestIdempotencyAndInplace:
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


class TestQCRerunDropsStitch:
    def test_qc_rerun_removes_stitch_columns(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata)
        sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200)
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col not in sdata.tables["labels_qc"].obs.columns


class TestMultiScale:
    def test_stitch_runs_on_multiscale(self):
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


# ---------------------------------------------------------------------------
# Diagnostics: opt-in, and it survives a zarr round-trip (I/O contract)
# ---------------------------------------------------------------------------


class TestSaveDiagnostics:
    def test_absent_by_default(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = _run_qc_and_stitch(sdata, min_confidence=0.5)
        assert "diagnostics" not in adata.uns["tiling_stitch"]

    def test_diagnostics_and_obs_survive_zarr_roundtrip(self, sdata_tile_boundary, tmp_path):
        from spatialdata import read_zarr

        sdata, _ = sdata_tile_boundary
        _run_qc_and_stitch(sdata, min_confidence=0.5, save_diagnostics=True)
        sdata.write(tmp_path / "roundtrip.zarr")
        a2 = read_zarr(tmp_path / "roundtrip.zarr").tables["labels_qc"]
        for col in ("stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence"):
            assert col in a2.obs.columns
        diag = a2.uns["tiling_stitch"]["diagnostics"]
        assert set(diag) >= {"cell_a", "cell_b", "axis", "confidence", "status"}
        assert "feature_weights" not in a2.uns["tiling_stitch"]["stitch_params"]  # no None leaked


# ---------------------------------------------------------------------------
# Visual: before/after group recolour, zoomed on a tile seam
# ---------------------------------------------------------------------------


class TestStitchVisual(PlotTester, metaclass=PlotTesterMeta):
    """Recolour the labels by ``label_id`` (before) vs ``stitch_group_id`` (after),
    zoomed on a tile seam.  Baseline lives in ``tests/_images/StitchVisual_*.png``
    and is downloaded from CI artifacts, not generated locally.
    """

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
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, arr, title in zip(
            axes, [labels, regrouped], ["by label_id (before)", "by stitch_group_id (after)"], strict=True
        ):
            ax.imshow(colors[arr][y0:y1, x0:x1], interpolation="nearest")
            ax.axhline(self._SEAM_Y - y0, color="white", linestyle="--", linewidth=1.0)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
