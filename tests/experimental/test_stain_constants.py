from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.im._stain._constants import (
    DEFAULT_BACKGROUND_INTENSITY,
    RUDERMAN_LAB_TO_LMS,
    RUDERMAN_LMS_TO_LAB,
    RUDERMAN_LMS_TO_RGB,
    RUDERMAN_RGB_TO_LMS,
    RUIFROK_HE,
    SDA_SCALE,
    STAIN_REFERENCE_SCHEMA_VERSION,
)


class TestRuifrok:
    @pytest.mark.parametrize("stain", ["hematoxylin", "eosin", "dab"])
    def test_unit_norm(self, stain: str) -> None:
        v = RUIFROK_HE[stain]
        assert v.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-12)

    def test_arrays_are_immutable(self) -> None:
        with pytest.raises(ValueError):
            RUIFROK_HE["hematoxylin"][0] = 0.0


class TestRuderman:
    def test_rgb_lms_round_trip(self) -> None:
        np.testing.assert_allclose(RUDERMAN_RGB_TO_LMS @ RUDERMAN_LMS_TO_RGB, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(RUDERMAN_LMS_TO_RGB @ RUDERMAN_RGB_TO_LMS, np.eye(3), atol=1e-10)

    def test_lms_lab_round_trip(self) -> None:
        np.testing.assert_allclose(RUDERMAN_LMS_TO_LAB @ RUDERMAN_LAB_TO_LMS, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(RUDERMAN_LAB_TO_LMS @ RUDERMAN_LMS_TO_LAB, np.eye(3), atol=1e-10)


class TestModuleDefaults:
    def test_background_intensity(self) -> None:
        np.testing.assert_array_equal(DEFAULT_BACKGROUND_INTENSITY, [255.0, 255.0, 255.0])
        with pytest.raises(ValueError):
            DEFAULT_BACKGROUND_INTENSITY[0] = 0.0

    def test_sda_scale_positive(self) -> None:
        assert SDA_SCALE > 0.0

    def test_schema_version(self) -> None:
        assert STAIN_REFERENCE_SCHEMA_VERSION == 1
