"""Canonical stain vectors, color-space matrices, and module-wide defaults.

Constants live at module scope and must not be mutated by callers.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from skimage.color import rgb_from_hed

STAIN_REFERENCE_SCHEMA_VERSION: int = 1

DEFAULT_BACKGROUND_INTENSITY: np.ndarray = np.array([255.0, 255.0, 255.0], dtype=np.float64)
DEFAULT_BACKGROUND_INTENSITY.flags.writeable = False

DEFAULT_LUMINOSITY_THRESHOLD: float = 0.15

NEAR_ZERO_NORM: float = 1e-8


class StainMethod(StrEnum):
    """Fitting methods supported by :class:`StainReference`."""

    MACENKO = "macenko"
    VAHADANE = "vahadane"
    REINHARD = "reinhard"


def _unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    a /= np.linalg.norm(a)
    a.flags.writeable = False
    return a


# Ruifrok and Johnston (2001) canonical stain vectors, unit-normalised.
# Sourced from `skimage.color.rgb_from_hed` so the values stay consistent with
# downstream scikit-image consumers; skimage stores them un-normalised, hence
# the `_unit` step.
RUIFROK_HE: dict[str, np.ndarray] = {
    "hematoxylin": _unit(rgb_from_hed[0]),
    "eosin": _unit(rgb_from_hed[1]),
    "dab": _unit(rgb_from_hed[2]),
}

# HistomicsTK-compatible SDA scale: encodes optical density so that luminosity
# thresholds taken from the literature transfer directly.
SDA_SCALE: float = 255.0 / np.log(256.0)

# Ruderman, Cronin, Chiao (1998) decorrelated colour space, as used by
# Reinhard et al. (2001). NOT equivalent to CIE Lab and NOT interchangeable
# with `skimage.color.rgb2lab`. Matrices are taken verbatim from the Reinhard
# 2001 paper and reproduced by HistomicsTK.
RUDERMAN_RGB_TO_LMS: np.ndarray = np.array(
    [
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444],
    ],
    dtype=np.float64,
)
RUDERMAN_RGB_TO_LMS.flags.writeable = False

RUDERMAN_LMS_TO_RGB: np.ndarray = np.linalg.inv(RUDERMAN_RGB_TO_LMS)
RUDERMAN_LMS_TO_RGB.flags.writeable = False

# Reinhard 2001 eq. 4: a diagonal (1/sqrt(3), 1/sqrt(6), 1/sqrt(2)) scaling
# followed by an orthogonal mixing matrix.
_DIAG = np.diag([1.0 / np.sqrt(3.0), 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(2.0)])
_MIX = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -2.0],
        [1.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
RUDERMAN_LMS_TO_LAB: np.ndarray = _DIAG @ _MIX
RUDERMAN_LMS_TO_LAB.flags.writeable = False

RUDERMAN_LAB_TO_LMS: np.ndarray = np.linalg.inv(RUDERMAN_LMS_TO_LAB)
RUDERMAN_LAB_TO_LMS.flags.writeable = False
