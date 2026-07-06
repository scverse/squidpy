"""Tests for tiling segmentation QC metrics."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import squidpy as sq
from squidpy.experimental.im._tiling import compute_cell_info, compute_cell_info_tiled
from tests.conftest import PlotTester, PlotTesterMeta

# Core behavioural tests


class TestCalculateTilingQC:
    """Tests for sq.experimental.tl.calculate_tiling_qc using the tile-boundary fixture."""

    def test_returns_anndata_with_scores(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            inplace=False,
        )
        assert adata.n_obs == len(gt.cut_cell_ids) + len(gt.intact_cell_ids)
        assert adata.n_vars == 0
        for col in [
            "max_straight_edge_ratio",
            "cardinal_alignment_score",
            "cut_score",
            "smoothed_cut_score",
            "is_outlier",
            "nhood_outlier_fraction",
        ]:
            assert col in adata.obs.columns

    def test_cut_cells_score_higher_than_intact(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            inplace=False,
        )
        obs = adata.obs
        cut = obs[obs["label_id"].isin(gt.cut_cell_ids)]["max_straight_edge_ratio"].dropna()
        intact = obs[obs["label_id"].isin(gt.intact_cell_ids)]["max_straight_edge_ratio"].dropna()
        assert cut.mean() > intact.mean()

    def test_tiled_vs_single_tile(self, sdata_tile_boundary):
        """Tiling must not change results - scores should be identical."""
        sdata, _ = sdata_tile_boundary
        adata_tiled = sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            inplace=False,
        )
        adata_single = sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=2000,
            inplace=False,
        )
        df1 = adata_tiled.obs.set_index("label_id").sort_index()
        df2 = adata_single.obs.set_index("label_id").sort_index()

        assert set(df1.index) == set(df2.index)
        for col in [
            "max_straight_edge_ratio",
            "cardinal_alignment_score",
            "cut_score",
            "smoothed_cut_score",
            "nhood_outlier_fraction",
        ]:
            np.testing.assert_allclose(
                df1[col].values,
                df2[col].values,
                atol=1e-10,
                equal_nan=True,
            )
        np.testing.assert_array_equal(df1["is_outlier"].values, df2["is_outlier"].values)

    def test_spatial_postprocessing_columns(self, sdata_tile_boundary):
        """Spatial post-processing produces correct dtypes and value ranges."""
        sdata, _ = sdata_tile_boundary
        adata = sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200, inplace=False)
        obs = adata.obs

        # smoothed_cut_score is non-negative (product of non-negatives)
        assert (obs["smoothed_cut_score"] >= 0).all()

        # is_outlier is boolean
        assert obs["is_outlier"].dtype == bool

        # nhood_outlier_fraction is bounded [0, 1]
        assert (obs["nhood_outlier_fraction"] >= 0).all()
        assert (obs["nhood_outlier_fraction"] <= 1).all()

        # n_neighbors stored in uns
        assert adata.uns["tiling_qc"]["n_neighbors"] == 10
        # Advanced tunables are bundled, not flat.
        assert "distance_tol" not in adata.uns["tiling_qc"]
        assert "tiling_qc_params" in adata.uns["tiling_qc"]
        bundle = adata.uns["tiling_qc"]["tiling_qc_params"]
        assert isinstance(bundle, dict)
        assert bundle["distance_tol"] == 0.75
        assert bundle["min_area"] == 20
        assert bundle["max_contour_points"] == 500

    def test_outlier_fraction_consistent_with_is_outlier(self, sdata_tile_boundary):
        """nhood_outlier_fraction should be 1.0 only when all k neighbors are outliers."""
        sdata, _ = sdata_tile_boundary
        adata = sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200, inplace=False)
        obs = adata.obs
        # Cells with nhood_outlier_fraction == 0 should exist (most cells are not outliers)
        assert (obs["nhood_outlier_fraction"] == 0).any()
        # If no cell is an outlier, all fractions must be 0
        if not obs["is_outlier"].any():
            assert (obs["nhood_outlier_fraction"] == 0).all()

    def test_invalid_labels_key(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        with pytest.raises(ValueError, match="not found"):
            sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="nonexistent", inplace=False)

    def test_clean_dataset_no_outliers(self, sdata_clean):
        """No tiling artifacts → MAD-based outlier detection should flag zero cells."""
        adata = sq.experimental.tl.calculate_tiling_qc(sdata_clean, labels_key="labels", tile_size=200, inplace=False)
        obs = adata.obs
        assert not obs["is_outlier"].any(), f"Expected 0 outliers on clean data, got {obs['is_outlier'].sum()}"
        assert (obs["nhood_outlier_fraction"] == 0).all()

    def test_few_cells_below_k(self):
        """Fewer cells than k=10 should not crash."""
        import dask.array as da
        import xarray as xr
        from spatialdata import SpatialData
        from spatialdata.models import Image2DModel, Labels2DModel

        # 3 well-separated circles
        labels = np.zeros((100, 100), dtype=np.int32)
        for i, (cy, cx) in enumerate([(20, 20), (50, 50), (80, 80)], start=1):
            yy, xx = np.ogrid[-cy : 100 - cy, -cx : 100 - cx]
            labels[yy**2 + xx**2 <= 64] = i

        sdata = SpatialData(
            images={
                "image": Image2DModel.parse(xr.DataArray(np.zeros((3, 100, 100), dtype=np.uint8), dims=["c", "y", "x"]))
            },
            labels={
                "labels": Labels2DModel.parse(xr.DataArray(da.from_array(labels, chunks=(100, 100)), dims=["y", "x"]))
            },
        )
        adata = sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200, inplace=False)
        assert adata.n_obs == 3
        for col in ["smoothed_cut_score", "is_outlier", "nhood_outlier_fraction"]:
            assert col in adata.obs.columns

    def test_both_gates_disabled_raises(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        with pytest.raises(ValueError, match="At least one outlier gate"):
            sq.experimental.tl.calculate_tiling_qc(
                sdata,
                labels_key="labels",
                tile_size=200,
                inplace=False,
                outlier_use_cut=False,
                outlier_use_smoothed=False,
            )

    def test_invalid_nmads_raises(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        with pytest.raises(ValueError, match="nmads_cut must be positive"):
            sq.experimental.tl.calculate_tiling_qc(
                sdata,
                labels_key="labels",
                tile_size=200,
                inplace=False,
                nmads_cut=0,
            )
        with pytest.raises(ValueError, match="nmads_smoothed must be positive"):
            sq.experimental.tl.calculate_tiling_qc(
                sdata,
                labels_key="labels",
                tile_size=200,
                inplace=False,
                nmads_smoothed=-1,
            )

    def test_cut_only_gate(self, sdata_tile_boundary):
        """Using only cut_score gate should still produce valid output."""
        sdata, _ = sdata_tile_boundary
        adata = sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            inplace=False,
            outlier_use_cut=True,
            outlier_use_smoothed=False,
        )
        assert adata.obs["is_outlier"].dtype == bool
        assert adata.uns["tiling_qc"]["outlier_use_cut"] is True
        assert adata.uns["tiling_qc"]["outlier_use_smoothed"] is False

    def test_smoothed_only_gate(self, sdata_tile_boundary):
        """Using only smoothed gate should still produce valid output."""
        sdata, _ = sdata_tile_boundary
        adata = sq.experimental.tl.calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            inplace=False,
            outlier_use_cut=False,
            outlier_use_smoothed=True,
        )
        assert adata.obs["is_outlier"].dtype == bool


# Params resolution


class TestTilingQCParamsResolution:
    def test_none_uses_defaults(self):
        from squidpy.experimental.tl._tiling_qc import TilingQCParams, _resolve_qc_params

        p = _resolve_qc_params(None)
        assert isinstance(p, TilingQCParams)
        assert p.distance_tol == 0.75
        assert p.min_area == 20
        assert p.max_contour_points == 500

    def test_instance_passthrough(self):
        from squidpy.experimental.tl._tiling_qc import TilingQCParams, _resolve_qc_params

        inst = TilingQCParams(distance_tol=1.0)
        assert _resolve_qc_params(inst) is inst

    def test_mapping_construction(self):
        from squidpy.experimental.tl._tiling_qc import _resolve_qc_params

        p = _resolve_qc_params({"distance_tol": 1.5, "min_area": 50})
        assert p.distance_tol == 1.5
        assert p.min_area == 50

    def test_numpy_scalars_coerced(self):
        from squidpy.experimental.tl._tiling_qc import _resolve_qc_params

        p = _resolve_qc_params({"distance_tol": np.float32(0.8), "min_area": np.int64(30)})
        assert type(p.distance_tol) is float
        assert type(p.min_area) is int

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bogus": 1}, "Unknown `tiling_qc_params`"),
            ({"distance_tol": -1.0}, "`distance_tol` must be >= 0"),
            ({"min_area": 0}, "`min_area` must be >= 1"),
            ({"max_contour_points": 2}, "`max_contour_points` must be >= 3"),
        ],
        ids=["unknown_field", "negative_distance_tol", "zero_min_area", "tiny_max_contour_points"],
    )
    def test_invalid_raises_value_error(self, kwargs, match):
        from squidpy.experimental.tl._tiling_qc import _resolve_qc_params

        with pytest.raises(ValueError, match=match):
            _resolve_qc_params(kwargs)

    def test_wrong_type_raises_type_error(self):
        from squidpy.experimental.tl._tiling_qc import _resolve_qc_params

        with pytest.raises(TypeError, match="TilingQCParams, Mapping, or None"):
            _resolve_qc_params(42)


# resolve_labels_array helper


class TestResolveLabelsArray:
    def test_single_scale_passthrough(self, sdata_clean):
        from squidpy.experimental.utils._labels import resolve_labels_array

        da = resolve_labels_array(sdata_clean, "labels", scale=None)
        assert da is sdata_clean.labels["labels"]

    def test_multi_scale_without_scale_raises(self):
        from spatialdata import SpatialData
        from spatialdata.models import Labels2DModel

        from squidpy.experimental.utils._labels import resolve_labels_array

        labels = np.zeros((128, 128), dtype=np.int32)
        labels[20:40, 20:40] = 1
        ms_labels = Labels2DModel.parse(
            xr.DataArray(da.from_array(labels, chunks=(64, 64)), dims=["y", "x"]),
            scale_factors=[2],
        )
        sdata = SpatialData(labels={"labels": ms_labels})

        with pytest.raises(ValueError, match="multi-scale"):
            resolve_labels_array(sdata, "labels", scale=None)


# Tiled centroid backend


class TestComputeCellInfoTiled:
    """Direct tests for compute_cell_info_tiled against the in-memory reference.

    The tiled implementation must produce identical centroids and bounding
    boxes to :func:`compute_cell_info` for any chunking, including chunk
    sizes small enough that cells span multiple chunks.
    """

    def _reference_and_tiled(self, sdata, chunk_size: int):
        labels_da = sdata.labels["labels"]
        labels_np = np.asarray(labels_da.values)
        if labels_np.ndim > 2:
            labels_np = labels_np.squeeze()

        # Re-chunk explicitly so the tiled backend has to merge cells across
        # chunk boundaries.
        rechunked = da.from_array(labels_np, chunks=(chunk_size, chunk_size))
        tiled_da = xr.DataArray(rechunked, dims=["y", "x"])

        ref = compute_cell_info(labels_np)
        tiled = compute_cell_info_tiled(tiled_da, chunk_size=chunk_size)
        return ref, tiled

    def test_matches_reference_small_chunks(self, sdata_clean):
        """Small chunks force most cells to straddle chunk borders."""
        ref, tiled = self._reference_and_tiled(sdata_clean, chunk_size=50)

        assert set(tiled) == set(ref), "Tiled backend produced a different set of label IDs"
        for lid, ci_ref in ref.items():
            ci_tiled = tiled[lid]
            np.testing.assert_allclose(ci_tiled.centroid_y, ci_ref.centroid_y, atol=1e-9)
            np.testing.assert_allclose(ci_tiled.centroid_x, ci_ref.centroid_x, atol=1e-9)
            assert ci_tiled.bbox_h == ci_ref.bbox_h
            assert ci_tiled.bbox_w == ci_ref.bbox_w

    def test_matches_reference_single_chunk(self, sdata_clean):
        """Chunk larger than the image: tiled path must agree exactly with the reference."""
        ref, tiled = self._reference_and_tiled(sdata_clean, chunk_size=10_000)

        assert set(tiled) == set(ref)
        for lid, ci_ref in ref.items():
            ci_tiled = tiled[lid]
            np.testing.assert_allclose(ci_tiled.centroid_y, ci_ref.centroid_y, atol=1e-12)
            np.testing.assert_allclose(ci_tiled.centroid_x, ci_ref.centroid_x, atol=1e-12)
            assert ci_tiled.bbox_h == ci_ref.bbox_h
            assert ci_tiled.bbox_w == ci_ref.bbox_w


# Visual regression tests (PlotTester)


@pytest.fixture()
def sdata_with_qc(sdata_tile_boundary):
    """SpatialData with tiling QC already computed."""
    sdata, _ = sdata_tile_boundary
    sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200, inplace=True)
    return sdata


class TestTilingQCVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_tiling_qc_cut_score(self, sdata_with_qc):
        """Visual: labels coloured by cut_score."""
        sq.experimental.pl.tiling_qc(sdata_with_qc, labels_key="labels", score_col="cut_score")

    def test_plot_tiling_qc_cardinal_alignment(self, sdata_with_qc):
        """Visual: labels coloured by cardinal_alignment_score."""
        sq.experimental.pl.tiling_qc(
            sdata_with_qc,
            labels_key="labels",
            score_col="cardinal_alignment_score",
        )

    def test_plot_tiling_qc_straight_edge_ratio(self, sdata_with_qc):
        """Visual: labels coloured by max_straight_edge_ratio."""
        sq.experimental.pl.tiling_qc(
            sdata_with_qc,
            labels_key="labels",
            score_col="max_straight_edge_ratio",
        )

    def test_plot_tiling_qc_nhood_outlier_fraction(self, sdata_with_qc):
        """Visual: default plot (nhood_outlier_fraction, RdYlGn_r, colorbar)."""
        sq.experimental.pl.tiling_qc(sdata_with_qc, labels_key="labels")

    def test_plot_tiling_qc_is_outlier(self, sdata_with_qc):
        """Visual: labels coloured by is_outlier (boolean)."""
        sq.experimental.pl.tiling_qc(sdata_with_qc, labels_key="labels", score_col="is_outlier")

    def test_plot_tiling_qc_smoothed_cut_score(self, sdata_with_qc):
        """Visual: labels coloured by smoothed_cut_score."""
        sq.experimental.pl.tiling_qc(
            sdata_with_qc,
            labels_key="labels",
            score_col="smoothed_cut_score",
        )
