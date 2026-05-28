"""Canonical stain vectors, color-space matrices, and module-wide defaults."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb_from_hed


def _unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


# Ruifrok and Johnston (2001) canonical stain vectors, unit-normalised.
# Sourced from `skimage.color.rgb_from_hed`; skimage stores them
# un-normalised so the `_unit` step is load-bearing.
RUIFROK_HE: dict[str, np.ndarray] = {
    "hematoxylin": _unit(rgb_from_hed[0]),
    "eosin": _unit(rgb_from_hed[1]),
    "dab": _unit(rgb_from_hed[2]),
}

# HistomicsTK-compatible SDA scale so luminosity thresholds taken from
# the H&E literature transfer directly.
SDA_SCALE: float = 255.0 / np.log(256.0)

# Ruderman, Cronin, Chiao (1998) decorrelated colour space, as used by
# Reinhard et al. (2001). NOT equivalent to CIE Lab and NOT interchangeable
# with `skimage.color.rgb2lab`.
RUDERMAN_RGB_TO_LMS: np.ndarray = np.array(
    [
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444],
    ],
    dtype=np.float64,
)
RUDERMAN_LMS_TO_RGB: np.ndarray = np.linalg.inv(RUDERMAN_RGB_TO_LMS)

# Reinhard 2001 eq. 4.
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
RUDERMAN_LAB_TO_LMS: np.ndarray = np.linalg.inv(RUDERMAN_LMS_TO_LAB)
