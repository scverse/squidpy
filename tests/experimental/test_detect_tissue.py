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


class TestBackgroundPriorKeywords:
    """The `BackgroundDetectionParams` keys are keyword arguments of `detect_tissue`."""

    def test_explicit_corner_overrides_broadcast(self) -> None:
        # `corners_are_background` broadcasts to all four corners; a corner passed
        # explicitly wins over it (the old `or` dropped the broadcast entirely)
        from squidpy.experimental.im._detect_tissue import _BACKGROUND_DEFAULTS, BackgroundDetectionParams
        from squidpy.experimental.utils._params import resolve_params

        corner_priors = BackgroundDetectionParams(
            ymin_xmin_is_bg=False, ymax_xmin_is_bg=False, ymin_xmax_is_bg=False, ymax_xmax_is_bg=False
        )
        bgp = resolve_params({**corner_priors, "ymin_xmin_is_bg": True}, defaults=_BACKGROUND_DEFAULTS)
        assert bgp["ymin_xmin_is_bg"] is True
        assert bgp["ymax_xmax_is_bg"] is False  # the broadcast still applies to the rest

    def test_unknown_keyword_raises(self, sdata_hne) -> None:
        with pytest.raises(ValueError, match="Unknown `background_detection_params` field"):
            sq.experimental.im.detect_tissue(sdata_hne, image_key="hne", inplace=False, corner_size_pctt=0.02)


class TestCornerMask:
    """Each corner prior is honoured on its own, not just all-on / all-off."""

    def test_single_corner_only(self) -> None:
        from squidpy.experimental.im._detect_tissue import _BACKGROUND_DEFAULTS, _corner_mask
        from squidpy.experimental.utils._params import resolve_params

        for corner, (rows, cols) in {
            "ymin_xmin_is_bg": (slice(None, 2), slice(None, 2)),
            "ymin_xmax_is_bg": (slice(None, 2), slice(-2, None)),
            "ymax_xmin_is_bg": (slice(-2, None), slice(None, 2)),
            "ymax_xmax_is_bg": (slice(-2, None), slice(-2, None)),
        }.items():
            off = dict.fromkeys(("ymin_xmin_is_bg", "ymin_xmax_is_bg", "ymax_xmin_is_bg", "ymax_xmax_is_bg"), False)
            params = resolve_params({**off, corner: True, "corner_size_pct": 0.2}, defaults=_BACKGROUND_DEFAULTS)
            mask = _corner_mask((10, 10), params)
            assert mask[rows, cols].all(), corner
            assert mask.sum() == 4, f"{corner} lit up more than its own corner"


class TestWekaSeeding:
    """The WEKA seeding fallback and the optional refinement stage."""

    @staticmethod
    def _synthetic_rgb() -> np.ndarray:
        img = np.full((48, 48, 3), 240, dtype=np.uint8)  # bright background
        img[18:30, 18:30] = 60  # a small dark blob of "tissue"
        return img

    def test_seed_floor_and_no_refinement(self) -> None:
        # `pseudo_min_pixels` above the seeded count forces the top-z fallback, and
        # `refine_with_classifier=False` skips the second stage -- both branches that
        # the default-parameter tests never take.
        from squidpy.experimental.im._detect_tissue import _BACKGROUND_DEFAULTS, _segment_weka
        from squidpy.experimental.types import _WEKA_DEFAULTS, WekaParams
        from squidpy.experimental.utils._params import resolve_params

        weka = resolve_params(
            WekaParams(rf_estimators=1, pseudo_min_pixels=5000, refine_with_classifier=False, rng=0),
            defaults=_WEKA_DEFAULTS,
        )
        bgp = resolve_params({}, defaults=_BACKGROUND_DEFAULTS)
        mask = _segment_weka(self._synthetic_rgb(), bgp, weka)
        assert mask.dtype == bool
        assert mask.shape == (48, 48)
        assert mask.any()  # the dark blob is found
