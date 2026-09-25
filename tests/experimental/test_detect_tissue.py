from __future__ import annotations

import numpy as np
import pytest
import spatialdata_plot as sdp

import squidpy as sq
from squidpy.experimental.im import FelzenszwalbParams, WekaParams
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp


class TestDetectTissue(PlotTester, metaclass=PlotTesterMeta):
    # test segmentation methods

    def test_plot_detect_tissue_otsu(self, sdata_hne):
        """Test OTSU tissue detection."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="otsu",
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()

    def test_plot_detect_tissue_felzenszwalb(self, sdata_hne):
        """Test OTSU tissue detection."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="felzenszwalb",
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()

    def test_plot_detect_tissue_weka(self, sdata_hne):
        """Test OTSU tissue detection."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="weka",
            # We'll have to manually correct for the Visium frame here - nothing's perfect.
            border_margin_px=1500,
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()

    # testing method parameters
    def test_plot_detect_tissue_using_felzenszwalb_params(self, sdata_hne):
        """Test tissue detection using Felzenszwalb parameters."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="felzenszwalb",
            # yields smaller mask
            method_params=FelzenszwalbParams(
                grid_rows=4,
                grid_cols=4,
            ),
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()

    def test_plot_detect_tissue_using_weka_params(self, sdata_hne):
        """Test tissue detection using Weka parameters."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="weka",
            method_params=WekaParams(
                # Cripple RF estimators to see effect
                rf_estimators=1,
            ),
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()

    # testing parameters

    def test_plot_detect_tissue_using_border_margins(self, sdata_hne):
        """Test tissue detection using border margins."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="otsu",
            border_margin_px=(
                3000,  # top
                4500,  # bottom
                3500,  # left
                4000,  # right
            ),
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()

    def test_plot_detect_tissue_using_mask_smoothing(self, sdata_hne):
        """Test tissue detection using mask smoothing."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="felzenszwalb",
            mask_smoothing_cycles=5,  # closes holes
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()

    def test_plot_detect_tissue_using_close_holes_smaller_than_frac(self, sdata_hne):
        """Test tissue detection using close holes smaller than a fraction of the image area."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="felzenszwalb",
            close_holes_smaller_than_frac=0.1,  # closes all holes
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()

    def test_detect_tissue_using_manual_scale(self, sdata_hne):
        """Test tissue detection using a manual scale."""
        sq.experimental.im.detect_tissue(
            sdata_hne,
            image_key="hne",
            method="otsu",
            scale="scale3",
        )

        sdata_hne.pl.render_labels("hne_tissue").pl.show()


@pytest.mark.parametrize(
    ("corners", "expected"),
    [
        (False, (False,) * 4),
        (np.bool_(False), (False,) * 4),
        ([True, False, False, True], (True, False, False, True)),
        (np.array([True, False, False, True]), (True, False, False, True)),
    ],
)
def test_normalize_corners(corners, expected) -> None:
    from squidpy.experimental.im._detect_tissue import _normalize_corners

    assert _normalize_corners(corners) == expected


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"corners_are_background": "False"}, TypeError, "not a string"),  # `bool("False")` is True
        ({"corners_are_background": (True, False)}, ValueError, "sequence of 4 bools"),
        ({"corners_are_background": np.ones((2, 2), dtype=bool)}, ValueError, "sequence of 4 bools"),
        ({"corner_size_pct": 0.0}, ValueError, "`corner_size_pct` must be in"),
    ],
    ids=["string", "wrong_length", "not_flat", "zero_corner_size"],
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


class TestWekaSeeding:
    @staticmethod
    def _synthetic_rgb() -> np.ndarray:
        img = np.full((48, 48, 3), 240, dtype=np.uint8)  # bright background
        img[18:30, 18:30] = 60  # a small dark blob of "tissue"
        return img

    def test_seed_floor_and_no_refinement(self) -> None:
        # forces the top-z seed fallback and skips refinement, which the default tests never reach
        from squidpy._params import resolve_params
        from squidpy.experimental.im._detect_tissue import _segment_weka
        from squidpy.types import WekaParams

        weka = resolve_params(
            WekaParams(rf_estimators=1, pseudo_min_pixels=5000, refine_with_classifier=False, rng=0),
            WekaParams,
        )
        mask = _segment_weka(self._synthetic_rgb(), (True,) * 4, 0.01, weka)
        assert mask.dtype == bool
        assert mask.shape == (48, 48)
        assert mask.any()  # the dark blob is found
