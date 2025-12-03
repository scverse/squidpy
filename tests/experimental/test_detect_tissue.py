from __future__ import annotations

import spatialdata_plot as sdp

import squidpy as sq
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp


class TestDetectTissue(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_detect_tissue_otsu(self):
        """Test OTSU tissue detection on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.detect_tissue(
            sdata,
            image_key="hne",
            method="otsu",
        )

        sdata.pl.render_labels("hne_tissue").pl.show()

    def test_plot_detect_tissue_felzenszwalb(self):
        """Test OTSU tissue detection on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.detect_tissue(
            sdata,
            image_key="hne",
            method="felzenszwalb",
        )

        sdata.pl.render_labels("hne_tissue").pl.show()

    def test_plot_detect_tissue_weka(self):
        """Test OTSU tissue detection on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.detect_tissue(
            sdata,
            image_key="hne",
            method="weka",
            # We'll have to manually correct for the Visium frame here - nothing's perfect.
            border_margin_px=1500,
        )

        sdata.pl.render_labels("hne_tissue").pl.show()
