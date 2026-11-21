from __future__ import annotations

import numpy as np
import pytest
import spatialdata_plot as sdp
from shapely import Polygon

import squidpy as sq
from squidpy.types import FelzenszwalbParams, WekaParams
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp


class TestDetectTissue(PlotTester, metaclass=PlotTesterMeta):
    # test segmentation methods

    def test_plot_detect_tissue_otsu(self, sdata_hne):
        """Test OTSU tissue detection."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="otsu",
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()

    def test_plot_detect_tissue_felzenszwalb(self, sdata_hne):
        """Test OTSU tissue detection."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="felzenszwalb",
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()

    def test_plot_detect_tissue_weka(self, sdata_hne):
        """Test OTSU tissue detection."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="weka",
            # We'll have to manually correct for the Visium frame here - nothing's perfect.
            border_margin_px=1500,
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()

    # testing method parameters
    def test_plot_detect_tissue_using_felzenszwalb_params(self, sdata_hne):
        """Test tissue detection using Felzenszwalb parameters."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="felzenszwalb",
            # yields smaller mask
            method_params=FelzenszwalbParams(
                grid_rows=4,
                grid_cols=4,
            ),
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()

    def test_plot_detect_tissue_using_weka_params(self, sdata_hne):
        """Test tissue detection using Weka parameters."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="weka",
            method_params=WekaParams(
                # Cripple RF estimators to see effect
                rf_estimators=1,
            ),
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()

    # testing parameters

    def test_plot_detect_tissue_using_border_margins(self, sdata_hne):
        """Test tissue detection using border margins."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="otsu",
            border_margin_px=(
                3000,  # top
                4500,  # bottom
                3500,  # left
                4000,  # right
            ),
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()

    def test_plot_detect_tissue_using_mask_smoothing(self, sdata_hne):
        """Test tissue detection using mask smoothing."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="felzenszwalb",
            mask_smoothing_cycles=5,  # closes holes
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()

    def test_plot_detect_tissue_using_close_holes_smaller_than_frac(self, sdata_hne):
        """Test tissue detection using close holes smaller than a fraction of the image area."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="felzenszwalb",
            close_holes_smaller_than_frac=0.1,  # closes all holes
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()

    def test_detect_tissue_using_manual_scale(self, sdata_hne):
        """Test tissue detection using a manual scale."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            as_labels=True,
            method="otsu",
            scale="scale3",
        )

        sdata_hne.pl.render_labels("hne_tissue_labels").pl.show()


@pytest.mark.parametrize(
    ("corners", "expected"),
    [
        (False, (False,) * 4),
        (np.bool_(False), (False,) * 4),
        (0, (False,) * 4),
        (1, (True,) * 4),
        (np.array(False), (False,) * 4),
        ([True, True, False, False], (True, True, False, False)),
        (np.array([False, True, False, False]), (False, True, False, False)),
    ],
)
def test_normalize_corners(corners, expected) -> None:
    from squidpy.experimental.im._detect_tissue import _normalize_corners

    assert _normalize_corners(corners) == expected


def test_normalize_corners_rejects_generator() -> None:
    from squidpy.experimental.im._detect_tissue import _normalize_corners

    with pytest.raises(ValueError, match="sequence of 4 bools"):
        _normalize_corners(c for c in [False] * 4)


def test_weka_border_margin_is_not_a_params_key() -> None:
    from squidpy._params import resolve_params

    # the margin is detect_tissue's own `border_margin_px` argument only
    with pytest.raises(ValueError, match="Unknown `method_params` field"):
        resolve_params({"border_margin_px": 20}, WekaParams)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"corners_are_background": "False"}, TypeError, "not a string"),  # `bool("False")` is True
        ({"corners_are_background": (True, False)}, ValueError, "sequence of 4 bools"),
        ({"corners_are_background": np.ones((2, 2), dtype=bool)}, ValueError, "sequence of 4 bools"),
        ({"corner_size_pct": 0.0}, ValueError, "`corner_size_pct` must be in"),
        ({"method": "bogus"}, ValueError, "Expected `method` to be one of"),
        ({"method": None}, ValueError, "Expected `method` to be one of"),
    ],
    ids=["string", "wrong_length", "not_flat", "zero_corner_size", "unknown_method", "none_method"],
)
def test_invalid_corners_raise(sdata_hne, kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        sq.experimental.im.detect_tissue(sdata_hne, image_key="hne", inplace=False, **kwargs)


@pytest.mark.parametrize(
    ("i", "rows", "cols"),
    [
        (0, slice(None, 2), slice(None, 2)),
        (1, slice(None, 2), slice(-2, None)),
        (2, slice(-2, None), slice(None, 2)),
        (3, slice(-2, None), slice(-2, None)),
    ],
    ids=["top_left", "top_right", "bottom_left", "bottom_right"],
)
def test_corner_mask_lights_only_its_corner(i, rows, cols) -> None:
    from squidpy.experimental.im._detect_tissue import _corner_mask

    mask = _corner_mask((10, 10), tuple(j == i for j in range(4)), 0.2)
    assert mask[rows, cols].all()
    assert mask.sum() == 4


def _ring_sdata(size: int = 400):
    """White slide with one tissue ring around a background hole."""
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel

    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    img[(r > 60) & (r < 150)] = (150, 60, 140)
    return SpatialData(images={"img": Image2DModel.parse(img, dims=("y", "x", "c"))})


def test_detect_tissue_writes_tissue_polygons(sdata_hne) -> None:
    from spatialdata import get_extent

    sq.experimental.im.detect_tissue(sdata_hne, image_key="hne")

    assert "hne_tissue" in sdata_hne.shapes
    assert "hne_tissue" not in sdata_hne.labels
    tissue = sdata_hne.shapes["hne_tissue"]
    assert tissue.is_valid.all()
    assert set(tissue.geom_type) <= {"Polygon", "MultiPolygon"}
    assert tissue["specimen_id"].tolist() == list(range(1, len(tissue) + 1))
    t, img = (
        get_extent(tissue, coordinate_system="global"),
        get_extent(sdata_hne.images["hne"], coordinate_system="global"),
    )
    for ax in ("x", "y"):
        assert img[ax][0] <= t[ax][0] < t[ax][1] <= img[ax][1]


def test_tissue_polygons_keep_holes() -> None:
    tissue = sq.experimental.im.detect_tissue(_ring_sdata(), image_key="img", inplace=False)

    assert len(tissue) == 1
    ring = tissue.geometry.iloc[0]
    assert ring.geom_type == "Polygon"
    assert len(ring.interiors) == 1
    # the hole has radius 60 px; simplification moves edges by at most one pixel
    hole_area = np.pi * 60**2
    assert abs(Polygon(ring.interiors[0]).area - hole_area) / hole_area < 0.05


def test_tissue_polygon_area_matches_mask() -> None:
    sdata = _ring_sdata()
    tissue = sq.experimental.im.detect_tissue(sdata, image_key="img", inplace=False)
    labels = sq.experimental.im.detect_tissue(sdata, image_key="img", inplace=False, as_labels=True)

    assert abs(tissue.area.sum() - (labels > 0).sum()) / (labels > 0).sum() < 0.01


def test_detect_tissue_name_clash_raises(sdata_hne) -> None:
    from spatialdata.models import Labels2DModel

    sdata_hne.labels["hne_tissue"] = Labels2DModel.parse(np.ones((8, 8), dtype=np.int32), dims=("y", "x"))
    with pytest.raises(ValueError, match="older squidpy version"):
        sq.experimental.im.detect_tissue(sdata_hne, image_key="hne")


def test_as_labels_writes_working_resolution_labels(sdata_hne) -> None:
    from spatialdata import get_extent

    sq.experimental.im.detect_tissue(sdata_hne, image_key="hne", as_labels=True)

    labels = sdata_hne.labels["hne_tissue_labels"]
    assert labels.shape[0] * labels.shape[1] <= 5_000_000
    lab, img = (
        get_extent(labels, coordinate_system="global"),
        get_extent(sdata_hne.images["hne"], coordinate_system="global"),
    )
    for ax in ("x", "y"):
        np.testing.assert_allclose(lab[ax], img[ax], atol=1e-6)


def test_tissue_consumers_match_on_polygons_and_raster(sdata_hne) -> None:
    """qc_image and the stain fit give the same answer from the polygons and from the same mask as pixels."""
    from spatialdata.models import Labels2DModel
    from spatialdata.transformations import get_transformation

    from squidpy.experimental.im._utils import rasterize_polygons

    sq.experimental.im.detect_tissue(sdata_hne, image_key="hne")
    h, w = sdata_hne.images["hne"]["scale0"]["image"].shape[1:]
    raster = rasterize_polygons(sdata_hne.shapes["hne_tissue"], (h, w), scale_xy=(1.0, 1.0)).compute()
    sdata_hne.labels["hne_raster"] = Labels2DModel.parse(
        raster.astype(np.int32),
        dims=("y", "x"),
        transformations=get_transformation(sdata_hne.images["hne"], get_all=True),
    )

    def tissue_tiles(mask_key):
        sq.experimental.im.qc_image(
            sdata_hne, image_key="hne", tile_size=(1000, 1000), metrics=["brightness_mean"],
            tissue_mask_key=mask_key, progress=False, preview=False,
        )  # fmt: skip
        return (sdata_hne.tables["qc_img_hne"].obs["is_tissue"] == "True").to_numpy()

    from_polygons, from_raster = tissue_tiles("hne_tissue"), tissue_tiles("hne_raster")
    assert (from_polygons != from_raster).sum() <= max(1, int(0.005 * from_polygons.size))

    fit_p = sq.experimental.im.fit_stain_reference(sdata_hne, image_key="hne", tissue_mask_key="hne_tissue")
    fit_r = sq.experimental.im.fit_stain_reference(sdata_hne, image_key="hne", tissue_mask_key="hne_raster")
    # the polygons are drawn at the fit scale, the raster is downsampled to it, so edge pixels differ slightly
    np.testing.assert_allclose(fit_p.stain_matrix, fit_r.stain_matrix, atol=5e-3)
