from __future__ import annotations

import spatialdata_plot as sdp
from spatialdata.transformations import Identity, get_transformation, set_transformation

import squidpy as sq
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp


class TestMakeTiles(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_make_tiles(self):
        """Test make tiles on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.make_tiles(sdata, image_key="hne")

    def test_plot_make_tiles_with_different_size(self):
        """Test make tiles on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.make_tiles(sdata, image_key="hne", tile_size=(1000, 1000))

    def test_plot_make_tiles_can_center_grid_on_tissue(self):
        """Test make tiles on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.make_tiles(
            sdata,
            image_key="hne",
            tile_size=(1900, 1900),  # Weird size so that we get a gap at the edges
            center_grid_on_tissue=True,
        )

    def test_plot_make_tiles_uses_min_tissue_fraction(self):
        """Test make tiles on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.make_tiles(
            sdata,
            image_key="hne",
            min_tissue_fraction=0.00001,  # Basically any non-bg tile is now tissue
        )


class TestMakeTilesFromSpots(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_make_tiles_from_spots(self):
        """Test make tiles from spots on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        # background cannot be classified, no img or mask, grey tiles in plot
        sq.experimental.im.make_tiles_from_spots(
            sdata,
            spots_key="spots",
        )

    def test_plot_make_tiles_from_spots_with_image_key(self):
        """Test make tiles from spots with image key on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        # background is automatically classified
        sq.experimental.im.make_tiles_from_spots(
            sdata,
            spots_key="spots",
            image_key="hne",
        )

    def test_plot_make_tiles_from_spots_with_tissue_mask_key(self):
        """Test make tiles from spots with tissue mask key on Visium H&E dataset."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.detect_tissue(
            sdata,
            image_key="hne",
        )

        # background is classified but preview plot doesn't have img
        sq.experimental.im.make_tiles_from_spots(
            sdata,
            spots_key="spots",
            tissue_mask_key="hne_tissue",
        )

    def test_plot_make_tiles_from_spots_auto_mask(self):
        """Ensure make_tiles_from_spots auto-creates mask when only image_key is provided."""
        sdata = sq.datasets.visium_hne_sdata()

        sq.experimental.im.make_tiles_from_spots(
            sdata,
            spots_key="spots",
            image_key="hne",
            preview=True,
        )

    def test_plot_make_tiles_center_grid_on_tissue(self):
        """Ensure centering on tissue shifts the grid when a mask is provided."""
        sdata = sq.datasets.visium_hne_sdata()
        sq.experimental.im.detect_tissue(sdata, image_key="hne")

        sq.experimental.im.make_tiles(
            sdata,
            image_key="hne",
            image_mask_key="hne_tissue",
            center_grid_on_tissue=True,
            preview=True,
        )


def test_make_tiles_copies_image_transformations(sdata_hne):
    """Tiles saved from images inherit the image transformations."""
    image_key = "hne"
    custom_cs = Identity()
    set_transformation(sdata_hne.images[image_key], {"custom_cs": custom_cs}, set_all=True)

    sq.experimental.im.make_tiles(sdata_hne, image_key=image_key, preview=False)

    img_tfs = get_transformation(sdata_hne.images[image_key], get_all=True)
    tile_tfs = get_transformation(sdata_hne.shapes[f"{image_key}_tiles"], get_all=True)

    assert "custom_cs" in img_tfs and "custom_cs" in tile_tfs
    assert isinstance(img_tfs["custom_cs"], Identity)
    assert isinstance(tile_tfs["custom_cs"], Identity)
