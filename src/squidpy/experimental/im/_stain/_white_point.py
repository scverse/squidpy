"""Background (white-point) intensity estimation for absorbance methods.

The decomposition methods convert RGB to absorbance against a per-channel
white point ``I_0``. Rather than assume pure white (255), estimate it from the
brightest pixels of the slide, which are the unstained background.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from squidpy.experimental.im._stain._conversion import _check_channel_dim
from squidpy.experimental.im._stain._validation import StainFittingError

#: Default per-channel white point ``I_0`` for the absorbance methods. A fixed
#: full-white reference (8-bit), matching HistomicsTK (255/256) and the Macenko
#: literature (240). The absorbance origin must be at least as bright as the
#: slide background, otherwise unstained pixels get a non-zero absorbance and
#: cannot round-trip back to white. Estimate from the image (see
#: ``estimate_white_point``) only when the slide has a genuinely
#: non-white background you want to anchor to.
DEFAULT_WHITE_POINT: np.ndarray = np.array([255.0, 255.0, 255.0])


def estimate_white_point(rgb: xr.DataArray, *, percentile: float = 99.0) -> np.ndarray:
    """Estimate the per-channel white point from the brightest pixels.

    Parameters
    ----------
    rgb
        Image with a ``"c"`` dimension of length 3. Numpy- or dask-backed.
    percentile
        Per-channel intensity percentile to take as the white point. The
        default (99) picks near-saturated background while ignoring the few
        truly-saturated outlier pixels.

    Returns
    -------
    Shape-``(3,)`` float64 white point, suitable as ``white_point``
    for :func:`~squidpy.experimental.im._stain._conversion.rgb_to_sda`.

    Notes
    -----
    The exact percentile is computed eagerly (the input is materialised), so
    the result is identical for numpy- and dask-backed inputs and independent
    of chunking - important for reproducible references across a cohort. Pass
    a coarse pyramid level for whole-slide images.

    Raises
    ------
    StainFittingError
        If the estimate is not strictly positive in every channel (e.g. a
        blank/black image with no bright background).
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"`percentile` must be in (0, 100], got {percentile}.")
    _check_channel_dim(rgb)
    flat = np.asarray(rgb.transpose("c", "y", "x").data, dtype=np.float64).reshape(3, -1)
    bg = np.percentile(flat, percentile, axis=1)

    if np.any(bg <= 0):
        raise StainFittingError(
            "estimated background intensity is non-positive; the image may be blank or all-tissue. "
            "Pass an explicit `white_point` if this is expected."
        )
    return bg
