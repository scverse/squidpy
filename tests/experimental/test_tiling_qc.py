"""Tests for tiling segmentation QC metrics."""

from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.pl._tiling_qc import tiling_qc
from squidpy.experimental.tl._tiling_qc import (
    _cardinal_alignment,
    _longest_collinear_segment,
    _score_tile,
    _straight_edge_metrics,
    calculate_tiling_qc,
)
from tests.conftest import PlotTester, PlotTesterMeta

# ---------------------------------------------------------------------------
# Unit tests for geometry helpers
# ---------------------------------------------------------------------------


class TestLongestCollinearSegment:
    """Tests for _longest_collinear_segment."""

    def test_perfectly_straight_horizontal(self):
        """A horizontal line should be detected as fully straight."""
        contour = np.array([[0.0, i] for i in range(20)])
        length, angle = _longest_collinear_segment(contour)
        assert length == pytest.approx(19.0, abs=0.1)
        # angle of horizontal line (row=0, col increasing) -> arctan2(0, 1) = 0
        assert abs(angle) < 0.1

    def test_perfectly_straight_vertical(self):
        """A vertical line should be detected as fully straight."""
        contour = np.array([[i, 0.0] for i in range(15)])
        length, angle = _longest_collinear_segment(contour)
        assert length == pytest.approx(14.0, abs=0.1)
        assert abs(angle - np.pi / 2) < 0.1

    def test_staircase_is_straight(self):
        """A pixel staircase (alternating cardinal/diagonal steps) is collinear."""
        contour = np.array([
            [0.0, 0.5],
            [0.5, 1.0],
            [0.5, 2.0],
            [1.0, 2.5],
            [1.0, 3.5],
            [1.5, 4.0],
        ])
        length, _ = _longest_collinear_segment(contour, distance_tol=0.75)
        assert length > 3.0

    def test_circle_has_short_segments(self):
        """A circle should have no long straight segments."""
        t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        contour = np.column_stack([10 * np.sin(t), 10 * np.cos(t)])
        length, _ = _longest_collinear_segment(contour)
        assert length < 10.0

    def test_too_few_points(self):
        """Contours with fewer than 3 points return zero."""
        length, angle = _longest_collinear_segment(np.array([[0, 0], [1, 1]]))
        assert length == 0.0

    def test_empty_contour(self):
        length, angle = _longest_collinear_segment(np.array([]).reshape(0, 2))
        assert length == 0.0


class TestCardinalAlignment:
    """Tests for _cardinal_alignment."""

    def test_horizontal(self):
        assert _cardinal_alignment(0.0) == pytest.approx(1.0)

    def test_vertical(self):
        assert _cardinal_alignment(np.pi / 2) == pytest.approx(1.0)

    def test_diagonal(self):
        assert _cardinal_alignment(np.pi / 4) == pytest.approx(0.0)

    def test_negative_angle(self):
        assert _cardinal_alignment(-np.pi / 2) == pytest.approx(1.0)

    def test_near_pi(self):
        assert _cardinal_alignment(np.pi - 0.01) == pytest.approx(1.0, abs=0.05)


class TestStraightEdgeMetrics:
    """Tests for _straight_edge_metrics."""

    def test_output_range(self):
        contour = np.array([[0.0, i] for i in range(10)])
        ser, cas, cs = _straight_edge_metrics(contour, cell_area=50.0)
        assert ser >= 0
        assert 0 <= cas <= 1.0
        assert cs >= 0

    def test_zero_area(self):
        contour = np.array([[0.0, i] for i in range(10)])
        ser, cas, cs = _straight_edge_metrics(contour, cell_area=0.0)
        assert ser == 0.0
        assert cas == 0.0
        assert cs == 0.0


# ---------------------------------------------------------------------------
# Per-tile scoring
# ---------------------------------------------------------------------------


class TestScoreTile:
    """Tests for _score_tile."""

    def test_empty_labels(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        df = _score_tile(labels)
        assert df.empty
        assert list(df.columns) == [
            "max_straight_edge_ratio",
            "cardinal_alignment_score",
            "cut_score",
        ]

    def test_single_cell(self):
        from skimage.draw import disk
        labels = np.zeros((50, 50), dtype=np.int32)
        rr, cc = disk((25, 25), 15, shape=(50, 50))
        labels[rr, cc] = 1
        df = _score_tile(labels)
        assert len(df) == 1
        assert df.index[0] == 1
        assert not df.isna().any().any()

    def test_cell_below_min_area_gets_nan(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10, 10] = 1
        df = _score_tile(labels, min_area=5)
        assert len(df) == 1
        assert df.iloc[0].isna().all()

    def test_rectangle_has_high_straight_ratio(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[5:45, 5:15] = 1
        df = _score_tile(labels)
        assert df.loc[1, "max_straight_edge_ratio"] > 1.0

    def test_downsample(self):
        from skimage.draw import disk
        labels = np.zeros((100, 100), dtype=np.int32)
        rr, cc = disk((50, 50), 30, shape=(100, 100))
        labels[rr, cc] = 1
        df1 = _score_tile(labels, downsample=1)
        df2 = _score_tile(labels, downsample=2)
        assert len(df2) == 1
        assert abs(df1.loc[1, "max_straight_edge_ratio"] - df2.loc[1, "max_straight_edge_ratio"]) < 0.3


# ---------------------------------------------------------------------------
# End-to-end tests with fixture
# ---------------------------------------------------------------------------


class TestCalculateTilingQC:
    """Tests for calculate_tiling_qc using the tile-boundary fixture."""

    def test_returns_anndata_with_scores_in_obs(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=False,
        )
        assert adata.n_obs == len(gt.cut_cell_ids) + len(gt.intact_cell_ids)
        assert adata.n_vars == 0
        for col in ["max_straight_edge_ratio", "cardinal_alignment_score", "cut_score"]:
            assert col in adata.obs.columns

    def test_inplace_stores_in_default_table_key(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        result = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=True,
        )
        assert result is None
        assert "labels_qc" in sdata.tables

    def test_spatialdata_attrs(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=False,
        )
        assert adata.uns["spatialdata_attrs"]["region"] == "labels"
        assert "label_id" in adata.obs.columns

    def test_centroids_stored_in_obs(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=False,
        )
        assert "centroid_y" in adata.obs.columns
        assert "centroid_x" in adata.obs.columns
        # Centroids should be within image bounds (400x400)
        assert (adata.obs["centroid_y"] >= 0).all()
        assert (adata.obs["centroid_x"] >= 0).all()
        assert (adata.obs["centroid_y"] < 400).all()
        assert (adata.obs["centroid_x"] < 400).all()

    def test_params_stored_in_uns(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200,
            distance_tol=1.0, min_area=10, downsample=2, inplace=False,
        )
        params = adata.uns["tiling_qc"]
        assert params["tile_size"] == 200
        assert params["distance_tol"] == 1.0
        assert params["min_area"] == 10
        assert params["downsample"] == 2
        assert params["scale"] is None

    def test_cut_cells_score_higher_than_intact(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=False,
        )
        obs = adata.obs
        cut = obs[obs["label_id"].isin(gt.cut_cell_ids)]["max_straight_edge_ratio"].dropna()
        intact = obs[obs["label_id"].isin(gt.intact_cell_ids)]["max_straight_edge_ratio"].dropna()
        assert cut.mean() > intact.mean()

    def test_score_ranges(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=False,
        )
        for col in ["max_straight_edge_ratio", "cardinal_alignment_score", "cut_score"]:
            valid = adata.obs[col].dropna()
            assert (valid >= 0).all(), f"{col} has negative values"

    def test_cardinal_score_bounded(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=False,
        )
        cardinal = adata.obs["cardinal_alignment_score"].dropna()
        assert (cardinal >= 0).all()
        assert (cardinal <= 1.0).all()

    def test_tiled_vs_single_tile(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        adata_tiled = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=False,
        )
        adata_single = calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=2000, inplace=False,
        )
        df1 = adata_tiled.obs.set_index("label_id").sort_index()
        df2 = adata_single.obs.set_index("label_id").sort_index()

        assert set(df1.index) == set(df2.index)
        for col in ["max_straight_edge_ratio", "cardinal_alignment_score", "cut_score"]:
            np.testing.assert_allclose(
                df1[col].values, df2[col].values, atol=1e-10, equal_nan=True,
            )

    def test_invalid_labels_key(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        with pytest.raises(ValueError, match="not found"):
            calculate_tiling_qc(sdata, labels_key="nonexistent", inplace=False)

    def test_custom_adata_key(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200,
            adata_key_added="my_qc", inplace=True,
        )
        assert "my_qc" in sdata.tables


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------


class TestPlotTilingQC:
    """Tests for tiling_qc plot (delegates to spatialdata-plot)."""

    def test_plot_renders(self, sdata_tile_boundary):
        import matplotlib
        matplotlib.use("Agg")

        sdata, _ = sdata_tile_boundary
        calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=True,
        )
        tiling_qc(sdata, labels_key="labels")
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_custom_score_col(self, sdata_tile_boundary):
        import matplotlib
        matplotlib.use("Agg")

        sdata, _ = sdata_tile_boundary
        calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=True,
        )
        tiling_qc(
            sdata, labels_key="labels", score_col="max_straight_edge_ratio",
        )
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_missing_qc_table(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        with pytest.raises(ValueError, match="not found"):
            tiling_qc(sdata, labels_key="labels", qc_key="nonexistent")

    def test_plot_invalid_score_col(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        calculate_tiling_qc(
            sdata, labels_key="labels", tile_size=200, inplace=True,
        )
        with pytest.raises(ValueError, match="not found"):
            tiling_qc(sdata, labels_key="labels", score_col="invalid")


# ---------------------------------------------------------------------------
# Visual regression tests (PlotTester)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sdata_with_qc(sdata_tile_boundary):
    """SpatialData with tiling QC already computed."""
    sdata, _ = sdata_tile_boundary
    calculate_tiling_qc(sdata, labels_key="labels", tile_size=200, inplace=True)
    return sdata


class TestTilingQCVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_tiling_qc_cut_score(self, sdata_with_qc):
        """Visual: labels coloured by cut_score."""
        tiling_qc(sdata_with_qc, labels_key="labels", score_col="cut_score")

    def test_plot_tiling_qc_straight_edge_ratio(self, sdata_with_qc):
        """Visual: labels coloured by max_straight_edge_ratio."""
        tiling_qc(
            sdata_with_qc, labels_key="labels",
            score_col="max_straight_edge_ratio",
        )
