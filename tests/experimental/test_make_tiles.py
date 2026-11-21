from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
import spatialdata_plot as sdp
from anndata import AnnData
from shapely import box, points
from shapely.geometry import MultiPolygon, Polygon
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale, get_transformation, set_transformation

import squidpy as sq
from squidpy.experimental.utils import derive_mpp_from_shapes
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


# ---------------------------------------------------------------------------
# make_tiles: units, centred crops, tissue polygons, cohorts and observation links
# ---------------------------------------------------------------------------


def _image(h: int = 512, w: int = 512, cs: str = "global") -> Image2DModel:
    return Image2DModel.parse(
        np.zeros((3, h, w), dtype=np.uint8),
        dims=("c", "y", "x"),
        scale_factors=[2, 2],
        transformations={cs: Identity()},
    )


def _tissue(*geoms, cs: str = "global", names=None) -> ShapesModel:
    names = names or [f"specimen_{i}" for i in range(len(geoms))]
    return ShapesModel.parse(
        gpd.GeoDataFrame(geometry=list(geoms), index=pd.Index(names)), transformations={cs: Identity()}
    )


def _cells(xy: np.ndarray, cs: str = "global") -> ShapesModel:
    gdf = gpd.GeoDataFrame({"radius": np.full(len(xy), 2.0)}, geometry=points(xy))
    return ShapesModel.parse(gdf, transformations={cs: Identity()})


def _omics_table(region: str | list[str], n: list[int]) -> AnnData:
    regions = [region] if isinstance(region, str) else region
    obs = pd.DataFrame(
        {
            "region": pd.Categorical(np.repeat(regions, n), categories=regions),
            "cell_id": np.concatenate([np.arange(k) for k in n]),
        }
    )
    obs.index = obs.index.astype(str)
    return TableModel.parse(AnnData(obs=obs), region=regions, region_key="region", instance_key="cell_id")


def test_make_tiles_grid_over_tissue_bbox_writes_shapes_and_table():
    sdata = sd.SpatialData(images={"img": _image()}, shapes={"img_tissue": _tissue(box(100, 100, 400, 400))})

    sq.experimental.im.make_tiles(sdata, image_key="img", tile_size=64, preview=False)

    tiles = sdata.shapes["img_tiles"]
    table = sdata.tables["img_tiles_table"]
    assert len(tiles) == 25  # ceil(300 / 64) = 5 per side, starting at the tissue bbox
    assert tiles["pixel_y0"].min() == 100 and tiles["pixel_x0"].min() == 100
    assert tiles.index.is_unique
    assert table.n_obs == 25
    assert list(table.obs["tile_id"]) == list(tiles.index)
    assert table.uns["spatialdata_attrs"]["region"] == ["img_tiles"]
    info = table.uns["make_tiles"]["per_image"]["img"]
    assert info["units"] == "px"
    assert info["tissue_source"] == "shapes:img_tissue"
    assert isinstance(get_transformation(tiles, get_all=True)["global"], Identity)


def test_make_tiles_min_tissue_fraction_keeps_and_flags():
    # one tile of 100 px holds exactly 25% tissue; the other specimen fixes the bbox
    tissue = _tissue(box(0, 0, 50, 50), box(150, 150, 200, 200))
    sdata = sd.SpatialData(images={"img": _image(256, 256)}, shapes={"img_tissue": tissue})

    sq.experimental.im.make_tiles(sdata, image_key="img", tile_size=100, preview=False)

    tiles = sdata.shapes["img_tiles"]
    assert len(tiles) == 4  # tiles below the threshold stay
    first = tiles.loc["img_x0_y0"]
    assert first["tissue_fraction"] == pytest.approx(0.25)
    assert bool(first["in_tissue"])
    assert not bool(tiles.loc["img_x1_y0"]["in_tissue"])
    assert tiles.loc["img_x1_y0"]["tissue_fraction"] == 0.0


def test_make_tiles_polygon_fraction_with_hole_and_multipolygon():
    outer = Polygon([(0, 0), (300, 0), (300, 300), (0, 300)], holes=[[(100, 100), (200, 100), (200, 200), (100, 200)]])
    second = MultiPolygon([box(300, 0, 350, 100), box(300, 200, 400, 300)])
    sdata = sd.SpatialData(
        images={"img": _image(512, 512)}, shapes={"img_tissue": _tissue(outer, second, names=["a", "b"])}
    )

    sq.experimental.im.make_tiles(sdata, image_key="img", tile_size=100, preview=False)

    tiles = sdata.shapes["img_tiles"]
    assert tiles.loc["img_x0_y0", "tissue_fraction"] == 1.0
    assert tiles.loc["img_x1_y1", "tissue_fraction"] == 0.0  # the hole
    assert tiles.loc["img_x3_y0", "tissue_fraction"] == pytest.approx(0.5)
    assert tiles.loc["img_x0_y0", "specimen_id"] == "a"
    assert tiles.loc["img_x3_y2", "specimen_id"] == "b"
    assert pd.isna(tiles.loc["img_x1_y1", "specimen_id"])


@pytest.mark.parametrize("labels_key", ["img_tissue", "img_tissue_labels"])
def test_make_tiles_raster_fallback_matches_polygons(labels_key):
    polys = [box(0, 0, 160, 96), box(200, 40, 320, 300)]
    mask = np.zeros((512, 512), dtype=np.int32)
    for i, p in enumerate(polys, start=1):
        x0, y0, x1, y1 = (int(v) for v in p.bounds)
        mask[y0:y1, x0:x1] = i
    from_polys = sd.SpatialData(images={"img": _image()}, shapes={"img_tissue": _tissue(*polys)})
    from_mask = sd.SpatialData(
        images={"img": _image()}, labels={labels_key: Labels2DModel.parse(mask, dims=("y", "x"))}
    )

    for sdata in (from_polys, from_mask):
        sq.experimental.im.make_tiles(sdata, image_key="img", tile_size=64, preview=False)

    a = from_polys.shapes["img_tiles"]["tissue_fraction"]
    b = from_mask.shapes["img_tiles"]["tissue_fraction"]
    np.testing.assert_allclose(a.to_numpy(), b.loc[a.index].to_numpy(), atol=1e-6)
    assert from_mask.tables["img_tiles_table"].uns["make_tiles"]["per_image"]["img"]["tissue_source"] == (
        f"labels:{labels_key}@scale0"
    )


def test_make_tiles_specimen_id_comes_from_detect_tissue_column():
    tissue = _tissue(box(0, 0, 100, 100), box(100, 0, 200, 100), names=["row_a", "row_b"])
    tissue["specimen_id"] = [1, 2]
    sdata = sd.SpatialData(images={"img": _image(256, 256)}, shapes={"img_tissue": tissue})

    sq.experimental.im.make_tiles(sdata, image_key="img", tile_size=100, preview=False)

    tiles = sdata.shapes["img_tiles"]
    assert list(tiles["specimen_id"]) == ["1", "2"]


def test_make_tiles_physical_units_from_visium_spot_pitch():
    pitch_px = 200.0  # 100 um pitch at 0.5 um/px
    rows, cols = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    xy = np.column_stack([(cols * pitch_px + (rows % 2) * pitch_px / 2).ravel(), (rows * pitch_px * 0.866).ravel()])
    sdata = sd.SpatialData(
        images={"img": _image(2048, 2048)},
        shapes={"spots": _cells(xy + 100), "img_tissue": _tissue(box(0, 0, 2048, 2048))},
    )
    mpp = derive_mpp_from_shapes(sdata, shapes_key="spots", coordinate_system="global", um_between_centers=100)

    sq.experimental.im.make_tiles(sdata, image_key="img", image_mpp=mpp, tile_size_um=100, preview=False)

    tiles = sdata.shapes["img_tiles"]
    assert (tiles.loc["img_x0_y0", "pixel_x1"] - tiles.loc["img_x0_y0", "pixel_x0"]) == 200
    info = sdata.tables["img_tiles_table"].uns["make_tiles"]["per_image"]["img"]
    assert info["units"] == "um"
    assert info["mpp_source"] == "user"


def test_make_tiles_mpp_from_micron_element():
    transcripts = PointsModel.parse(
        pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]}),
        transformations={"global": Scale([1 / 0.2125, 1 / 0.2125], axes=("x", "y"))},
    )
    sdata = sd.SpatialData(
        images={"img": _image()},
        points={"transcripts": transcripts},
        shapes={"img_tissue": _tissue(box(0, 0, 512, 512))},
    )

    sq.experimental.im.make_tiles(sdata, image_key="img", mpp_from="transcripts", tile_size_um=20, preview=False)

    info = sdata.tables["img_tiles_table"].uns["make_tiles"]["per_image"]["img"]
    assert info["mpp0"] == pytest.approx(0.2125)
    assert info["mpp_source"] == "element:transcripts"

    set_transformation(sdata.points["transcripts"], Scale([1.0, 2.0], axes=("x", "y")), to_coordinate_system="global")
    with pytest.raises(ValueError, match="anisotropic"):
        sq.experimental.im.make_tiles(sdata, image_key="img", mpp_from="transcripts", tile_size_um=20, preview=False)


@pytest.mark.parametrize(("target_mpp", "level", "actual"), [(1.0, "scale2", 1.0), (0.7, "scale1", 0.5)])
def test_make_tiles_picks_pyramid_level_closest_to_target_mpp(target_mpp, level, actual):
    sdata = sd.SpatialData(images={"img": _image(1024, 1024)}, shapes={"img_tissue": _tissue(box(0, 0, 1024, 1024))})

    sq.experimental.im.make_tiles(
        sdata, image_key="img", image_mpp=0.25, tile_size_um=64, target_mpp=target_mpp, preview=False
    )

    info = sdata.tables["img_tiles_table"].uns["make_tiles"]["per_image"]["img"]
    assert info["level"] == level
    assert info["actual_mpp"] == pytest.approx(actual)
    assert info["tile_size_px_scale0"] == [256, 256]
    tiles = sdata.shapes["img_tiles"]
    first = tiles.loc["img_x0_y0"]
    assert first["level_x1"] - first["level_x0"] == 64 / actual


def test_make_tiles_unknown_pixel_size_stays_in_pixels():
    sdata = sd.SpatialData(images={"img": _image()}, shapes={"img_tissue": _tissue(box(0, 0, 512, 512))})

    sq.experimental.im.make_tiles(sdata, image_key="img", tile_size_um=50, preview=False)

    info = sdata.tables["img_tiles_table"].uns["make_tiles"]["per_image"]["img"]
    assert info["units"] == "px"
    assert "mpp_source" not in info
    first = sdata.shapes["img_tiles"].loc["img_x0_y0"]
    assert first["pixel_x1"] - first["pixel_x0"] == 224


def test_make_tiles_centered_on_spots_and_links_one_to_one():
    xy = np.array([[50.0, 50.0], [200.0, 120.0], [400.0, 400.0]])
    sdata = sd.SpatialData(
        images={"img": _image()},
        shapes={"spots": _cells(xy), "img_tissue": _tissue(box(0, 0, 512, 512))},
        tables={"table": _omics_table("spots", [3])},
    )

    sq.experimental.im.make_tiles(
        sdata, image_key="img", tile_size=64, centered_on="spots", table_key="table", preview=False
    )

    tiles = sdata.shapes["img_tiles"]
    assert list(tiles.index) == ["img_0", "img_1", "img_2"]
    centres = np.column_stack([(tiles.pixel_x0 + tiles.pixel_x1) / 2, (tiles.pixel_y0 + tiles.pixel_y1) / 2])
    np.testing.assert_allclose(centres[1:], xy[1:], atol=0.5)
    assert (tiles.pixel_x1 - tiles.pixel_x0).iloc[1] == 64
    assert list(sdata.tables["table"].obs["tile_id"]) == ["img_0", "img_1", "img_2"]


def test_make_tiles_links_grid_tiles_to_observations():
    xy = np.array([[110.0, 110.0], [250.0, 170.0], [5.0, 5.0]])  # the last cell is outside the tissue bbox
    sdata = sd.SpatialData(
        images={"img": _image()},
        shapes={"cells": _cells(xy), "img_tissue": _tissue(box(100, 100, 400, 400))},
        tables={"table": _omics_table("cells", [3])},
    )

    sq.experimental.im.make_tiles(sdata, image_key="img", tile_size=64, table_key="table", preview=False)

    tile_id = sdata.tables["table"].obs["tile_id"]
    assert tile_id.iloc[0] == "img_x0_y0"
    assert tile_id.iloc[1] == "img_x2_y1"
    assert pd.isna(tile_id.iloc[2])


def test_make_tiles_cohort_one_table_per_image_shapes():
    sdata = sd.SpatialData(
        images={"he_a": _image(512, 512, cs="cs_a"), "he_b": _image(256, 256, cs="cs_b")},
        shapes={
            "he_a_tissue": _tissue(box(0, 0, 256, 256), cs="cs_a"),
            "he_b_tissue": _tissue(box(0, 0, 128, 128), cs="cs_b"),
            "cells_a": _cells(np.array([[10.0, 10.0]]), cs="cs_a"),
            "cells_b": _cells(np.array([[70.0, 70.0]]), cs="cs_b"),
        },
        tables={"table": _omics_table(["cells_a", "cells_b"], [1, 1])},
    )

    sq.experimental.im.make_tiles(
        sdata,
        image_key=["he_a", "he_b"],
        tile_size=64,
        image_mpp={"he_a": 0.5, "he_b": 0.25},
        table_key="table",
        preview=False,
    )

    table = sdata.tables["tiles_table"]
    assert list(table.obs["slide"].cat.categories) == ["he_a", "he_b"]
    assert table.uns["spatialdata_attrs"]["region"] == ["he_a_tiles", "he_b_tiles"]
    assert table.obs_names.is_unique
    assert "cs_a" in get_transformation(sdata.shapes["he_a_tiles"], get_all=True)
    assert "cs_b" in get_transformation(sdata.shapes["he_b_tiles"], get_all=True)
    assert list(sdata.tables["table"].obs["tile_id"]) == ["he_a_x0_y0", "he_b_x1_y1"]


def test_make_tiles_rejects_conflicting_arguments():
    sdata = sd.SpatialData(images={"img": _image()}, shapes={"img_tissue": _tissue(box(0, 0, 512, 512))})
    with pytest.raises(ValueError, match="not both"):
        sq.experimental.im.make_tiles(sdata, image_key="img", tile_size=64, tile_size_um=10, preview=False)
    with pytest.raises(ValueError, match="requires `tile_size_um`"):
        sq.experimental.im.make_tiles(sdata, image_key="img", target_mpp=0.5, preview=False)
    sdata.images["img2"] = _image()
    with pytest.raises(ValueError, match="single image"):
        sq.experimental.im.make_tiles(sdata, image_key=["img", "img2"], new_shapes_key="x", preview=False)


def test_make_tiles_table_roundtrips_through_zarr(tmp_path):
    sdata = sd.SpatialData(images={"img": _image()}, shapes={"img_tissue": _tissue(box(100, 100, 400, 400))})
    sq.experimental.im.make_tiles(sdata, image_key="img", image_mpp=0.5, tile_size_um=32, preview=False)

    sdata.write(tmp_path / "tiles.zarr")
    read = sd.read_zarr(tmp_path / "tiles.zarr")

    table = read.tables["img_tiles_table"]
    assert table.uns["make_tiles"]["per_image"]["img"]["units"] == "um"
    assert table.obs["in_tissue"].dtype == bool
    assert read.shapes["img_tiles"]["tissue_fraction"].between(0, 1).all()


def test_make_tiles_centered_on_per_image_in_a_cohort():
    sdata = sd.SpatialData(
        images={"he_a": _image(512, 512, cs="cs_a"), "he_b": _image(512, 512, cs="cs_b")},
        shapes={
            "he_a_tissue": _tissue(box(0, 0, 512, 512), cs="cs_a"),
            "he_b_tissue": _tissue(box(0, 0, 512, 512), cs="cs_b"),
            "spots_a": _cells(np.array([[100.0, 100.0], [300.0, 300.0]]), cs="cs_a"),
            "spots_b": _cells(np.array([[200.0, 50.0]]), cs="cs_b"),
        },
        tables={"table": _omics_table(["spots_a", "spots_b"], [2, 1])},
    )

    sq.experimental.im.make_tiles(
        sdata,
        image_key=["he_a", "he_b"],
        tile_size=64,
        centered_on={"he_a": "spots_a", "he_b": "spots_b"},
        table_key="table",
        preview=False,
    )

    assert list(sdata.shapes["he_a_tiles"].index) == ["he_a_0", "he_a_1"]
    assert list(sdata.shapes["he_b_tiles"].index) == ["he_b_0"]
    assert list(sdata.tables["table"].obs["tile_id"]) == ["he_a_0", "he_a_1", "he_b_0"]
    with pytest.raises(KeyError, match="no entry"):
        sq.experimental.im.make_tiles(
            sdata, image_key=["he_a", "he_b"], tile_size=64, centered_on={"he_a": "spots_a"}, preview=False
        )


def test_make_tiles_from_spots_reads_tissue_polygons():
    xy = np.array([[50.0, 50.0], [250.0, 250.0], [450.0, 450.0]])
    sdata = sd.SpatialData(
        images={"img": _image()},
        shapes={"spots": _cells(xy), "img_tissue": _tissue(box(0, 0, 300, 300))},
    )

    sq.experimental.im.make_tiles_from_spots(
        sdata, spots_key="spots", image_key="img", min_tissue_fraction=0.25, preview=False
    )

    classes = sdata.shapes["spots"]["tile_classification"].astype(str).tolist()
    assert "img_tissue" not in sdata.labels
    assert classes[0] == "tissue" and classes[2] == "background"
