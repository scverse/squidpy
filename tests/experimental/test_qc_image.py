from __future__ import annotations

import copy

import pytest
import spatialdata_plot as sdp

import squidpy as sq
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp

# Large tile size to keep tests fast (fewer tiles = less computation).
_FAST_TILE = (1000, 1000)


@pytest.fixture(scope="session")
def _sdata_hne_with_tissue():
    """Load the Visium H&E dataset once and pre-compute the tissue mask.

    The tissue mask is the most expensive setup step (~16 s).  Pre-computing
    it here and deepcopying per test avoids paying that cost repeatedly.
    """
    sdata = sq.datasets.visium_hne_sdata()
    sq.experimental.im.detect_tissue(sdata, image_key="hne", scale="scale0", inplace=True, new_labels_key="hne_tissue")
    return sdata


@pytest.fixture()
def sdata_hne(_sdata_hne_with_tissue):
    """Per-test deep copy so each test gets a fresh SpatialData with tissue mask."""
    return copy.deepcopy(_sdata_hne_with_tissue)


class TestQCImage(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_calc_qc_image_hne(self, sdata_hne):
        """Test QC image on Visium H&E dataset with default H&E metrics."""
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            metrics=[sq.experimental.im.QCMetric.TENENGRAD],
            progress=False,
        )

        (
            sdata_hne.pl.render_images()
            .pl.render_shapes("qc_img_hne_grid", color="qc_outlier", groups="True", palette="red", fill_alpha=1.0)
            .pl.show()
        )

    def test_plot_calc_qc_image_not_hne(self, sdata_hne):
        """Test QC image with is_hne=False (generic metrics only)."""
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            is_hne=False,
            metrics=[sq.experimental.im.QCMetric.TENENGRAD],
            progress=False,
        )

        (
            sdata_hne.pl.render_images()
            .pl.render_shapes("qc_img_hne_grid", color="qc_outlier", groups="True", palette="red", fill_alpha=1.0)
            .pl.show()
        )

    def test_plot_plot_qc_image(self, sdata_hne):
        """Test QC image plotting function."""
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            metrics=[sq.experimental.im.QCMetric.TENENGRAD],
            progress=False,
        )

        sq.experimental.pl.qc_image(
            sdata_hne,
            image_key="hne",
        )


def test_qc_image_hne_metric_without_hne_flag(sdata_hne):
    """Test that H&E metrics raise ValueError when is_hne=False."""
    with pytest.raises(ValueError, match="H&E-specific metrics"):
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            is_hne=False,
            metrics=[sq.experimental.im.QCMetric.HEMATOXYLIN_MEAN],
        )


def test_qc_image_default_metrics_hne(sdata_hne):
    """Test that default H&E metrics produce expected var_names."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        metrics=None,
        is_hne=True,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    expected = {
        "qc_tenengrad",
        "qc_var_of_laplacian",
        "qc_entropy",
        "qc_brightness_mean",
        "qc_hematoxylin_mean",
        "qc_eosin_mean",
    }
    assert set(adata.var_names) == expected


def test_qc_image_default_metrics_generic(sdata_hne):
    """Test that default generic metrics produce expected var_names."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        metrics=None,
        is_hne=False,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    expected = {
        "qc_tenengrad",
        "qc_var_of_laplacian",
        "qc_entropy",
        "qc_brightness_mean",
    }
    assert set(adata.var_names) == expected


def test_qc_image_rgb_metric(sdata_hne):
    """Test running a single RGB metric."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        metrics=[sq.experimental.im.QCMetric.HEMATOXYLIN_MEAN],
        detect_tissue=False,
        detect_outliers=False,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    assert "qc_hematoxylin_mean" in adata.var_names
