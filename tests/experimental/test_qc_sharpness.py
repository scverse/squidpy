from __future__ import annotations

import spatialdata_plot as sdp

import squidpy as sq
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp


class TestQCSharpness(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_calc_qc_sharpness(self):
        """Test QC sharpness on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.qc_sharpness(
            sdata,
            image_key="hne",
            # Only one metric for speed
            metrics=[sq.experimental.im.SharpnessMetric.TENENGRAD],
        )

        (
            sdata.pl.render_images()
            .pl.render_shapes(
                "qc_img_hne_sharpness_grid", color="sharpness_outlier", groups="True", palette="red", fill_alpha=1.0
            )
            .pl.show()
        )

    def test_plot_plot_qc_sharpness(self):
        """Test QC sharpness on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.qc_sharpness(
            sdata,
            image_key="hne",
            # Only one metric for speed
            metrics=[sq.experimental.im.SharpnessMetric.TENENGRAD],
        )

        sq.experimental.pl.qc_sharpness(
            sdata,
            image_key="hne",
        )
