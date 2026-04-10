"""Tests for tiling segmentation QC metrics."""

from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.pl._tiling_qc import tiling_qc
from squidpy.experimental.tl._tiling_qc import calculate_tiling_qc
from tests.conftest import PlotTester, PlotTesterMeta

# ---------------------------------------------------------------------------
# Core behavioural tests
# ---------------------------------------------------------------------------


class TestCalculateTilingQC:
    """Tests for calculate_tiling_qc using the tile-boundary fixture."""

    def test_returns_anndata_with_scores(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            inplace=False,
        )
        assert adata.n_obs == len(gt.cut_cell_ids) + len(gt.intact_cell_ids)
        assert adata.n_vars == 0
        for col in ["max_straight_edge_ratio", "cardinal_alignment_score", "cut_score"]:
            assert col in adata.obs.columns

    def test_cut_cells_score_higher_than_intact(self, sdata_tile_boundary):
        sdata, gt = sdata_tile_boundary
        adata = calculate_tiling_qc(
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
        """Tiling must not change results — scores should be identical."""
        sdata, _ = sdata_tile_boundary
        adata_tiled = calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=200,
            inplace=False,
        )
        adata_single = calculate_tiling_qc(
            sdata,
            labels_key="labels",
            tile_size=2000,
            inplace=False,
        )
        df1 = adata_tiled.obs.set_index("label_id").sort_index()
        df2 = adata_single.obs.set_index("label_id").sort_index()

        assert set(df1.index) == set(df2.index)
        for col in ["max_straight_edge_ratio", "cardinal_alignment_score", "cut_score"]:
            np.testing.assert_allclose(
                df1[col].values,
                df2[col].values,
                atol=1e-10,
                equal_nan=True,
            )

    def test_invalid_labels_key(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        with pytest.raises(ValueError, match="not found"):
            calculate_tiling_qc(sdata, labels_key="nonexistent", inplace=False)


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

    def test_plot_tiling_qc_cardinal_alignment(self, sdata_with_qc):
        """Visual: labels coloured by cardinal_alignment_score."""
        tiling_qc(
            sdata_with_qc,
            labels_key="labels",
            score_col="cardinal_alignment_score",
        )

    def test_plot_tiling_qc_straight_edge_ratio(self, sdata_with_qc):
        """Visual: labels coloured by max_straight_edge_ratio."""
        tiling_qc(
            sdata_with_qc,
            labels_key="labels",
            score_col="max_straight_edge_ratio",
        )
