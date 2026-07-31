from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.im._stain._constants import (
    RUDERMAN_LAB_TO_LMS,
    RUDERMAN_LMS_TO_LAB,
    RUDERMAN_LMS_TO_RGB,
    RUDERMAN_RGB_TO_LMS,
    RUIFROK_HE,
    SDA_SCALE,
)


@pytest.mark.parametrize("stain", ["hematoxylin", "eosin", "dab"])
def test_ruifrok_unit_norm(stain: str) -> None:
    v = RUIFROK_HE[stain]
    assert v.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-12)


def test_rgb_lms_round_trip() -> None:
    np.testing.assert_allclose(RUDERMAN_RGB_TO_LMS @ RUDERMAN_LMS_TO_RGB, np.eye(3), atol=1e-10)


def test_lms_lab_round_trip() -> None:
    np.testing.assert_allclose(RUDERMAN_LMS_TO_LAB @ RUDERMAN_LAB_TO_LMS, np.eye(3), atol=1e-10)


def test_sda_scale_positive() -> None:
    assert SDA_SCALE > 0.0
