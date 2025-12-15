from __future__ import annotations

import spatialdata_plot as sdp

import squidpy as sq
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp


class TestQCSharpness(PlotTester, metaclass=PlotTesterMeta):
    def test_calc_qc_sharpness(self):
        """Test QC sharpness on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.qc_sharpness(
            sdata,
            image_key="hne",
            metrics=[sq.experimental.im.SharpnessMetric.TENENGRAD],
        )

        adata = sdata.tables["qc_img_hne_sharpness"]
        assert "sharpness_tenengrad" in adata.var_names
        assert adata.X.size > 0

    def test_plot_qc_sharpness(self):
        """Test QC sharpness plotting on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.qc_sharpness(
            sdata,
            image_key="hne",
            metrics=[sq.experimental.im.SharpnessMetric.TENENGRAD],
        )

        sq.experimental.pl.qc_sharpness(
            sdata,
            image_key="hne",
        )
