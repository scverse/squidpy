"""Test for experimental tissue detection."""

from __future__ import annotations

import squidpy as sq
from tests.conftest import PlotTester, PlotTesterMeta


class TestDetectTissue(PlotTester, metaclass=PlotTesterMeta):
    def test_detect_tissue_otsu(self):
        """Test OTSU tissue detection on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.detect_tissue(
            sdata,
            image_key="hne",
            method="otsu",
        )

        sdata.pl.render_labels("hne_tissue").pl.show()
