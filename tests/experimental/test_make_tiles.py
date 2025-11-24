from __future__ import annotations

import spatialdata_plot as sdp

import squidpy as sq
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp


class TestMakeTiles(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_make_tiles(self):
        """Test make tiles on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.make_tiles(sdata, image_key="hne")

        sdata.pl.render_shapes("hne_tiles", color="tile_classification", fill_alpha=0.5).pl.show()


class TestMakeTilesFromSpots(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_make_tiles_from_spots(self):
        """Test make tiles from spots on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.make_tiles_from_spots(
            sdata,
            spots_key="spots",
        )

        # background cannot be classified, no img or mask, grey tiles in plot
        sdata.pl.render_shapes("spots_tiles", color="tile_classification", fill_alpha=0.5).pl.show()

    def test_plot_make_tiles_from_spots_with_image_key(self):
        """Test make tiles from spots with image key on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.make_tiles_from_spots(
            sdata,
            spots_key="spots",
            image_key="hne",
        )

        # background is automatically classified
        sdata.pl.render_shapes("spots_tiles", color="tile_classification", fill_alpha=0.5).pl.show()

    def test_plot_make_tiles_from_spots_with_tissue_mask_key(self):
        """Test make tiles from spots with tissue mask key on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.detect_tissue(
            sdata,
            image_key="hne",
        )

        sq.experimental.im.make_tiles_from_spots(
            sdata,
            spots_key="spots",
            tissue_mask_key="hne_tissue",
        )

        # background is classified but preview plot doesn't have img
        sdata.pl.render_shapes("spots_tiles", color="tile_classification", fill_alpha=0.5).pl.show()
