"""White-point (``I_0``) handling for the absorbance methods.

The decomposition methods measure absorbance against a per-channel white point
``I_0`` (the intensity that counts as fully unstained). The default is a fixed
full-white reference at the image's bit depth (255 for uint8, 65535 for uint16,
1.0 for float), matching HistomicsTK (255/256) and the Macenko literature (240):
``I_0`` must be at least as bright as the slide background, otherwise unstained
pixels get a non-zero absorbance and cannot round-trip back to white. Use
:func:`estimate_white_point` only for a slide with a genuinely non-white
background you want to anchor to.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from squidpy.experimental.im._stain._conversion import _check_channel_dim, dtype_max
from squidpy.experimental.im._stain._validation import StainFittingError


def default_white_point(rgb: xr.DataArray) -> np.ndarray:
    """Dtype-aware default white point ``I_0`` (full white), with a range check.

    Returns ``(3,)`` filled with the dtype's full-white value. Raises with
    guidance when the data clearly does not match its dtype's range (e.g. 8-bit
    values stored in a uint16 container, or 0-255 values stored as float), since
    that would silently mis-scale the absorbance.
    """
    m = dtype_max(rgb.dtype)
    data_max = float(np.asarray(rgb.max()))
    if np.issubdtype(rgb.dtype, np.integer):
        if m >= 256 and data_max <= 255:
            raise ValueError(
                f"{rgb.dtype} image but the maximum value is {data_max:.0f} (<= 255) - this looks like "
                f"8-bit data stored in a {rgb.dtype} container. Convert to uint8, or pass `white_point`."
            )
    elif data_max > 1.5:
        raise ValueError(
            f"float image but the maximum value is {data_max:.1f} (> 1) - this looks like 0-255 data "
            "stored as float. Rescale to [0, 1], or pass `white_point`."
        )
    return np.full(3, m, dtype=np.float64)


def white_point_from_background(rgb: xr.DataArray, background_mask: np.ndarray) -> np.ndarray:
    """Per-channel median intensity over background pixels -> ``(3,)`` white point.

    ``background_mask`` is a ``(y, x)`` boolean, ``True`` over non-tissue
    (background) pixels. Sampling the *median* of true background (rather than a
    whole-image percentile) anchors ``I_0`` to the actual unstained intensity,
    matching HistomicsTK's ``background_intensity`` semantics.

    Raises
    ------
    StainFittingError
        If the mask selects no background pixels, or the median is non-positive
        (e.g. a black background).
    """
    _check_channel_dim(rgb)
    flat = np.asarray(rgb.transpose("c", "y", "x").data, dtype=np.float64)  # (3, y, x)
    bg_pixels = flat[:, np.asarray(background_mask, dtype=bool)]  # (3, N_background)
    if bg_pixels.shape[1] == 0:
        raise StainFittingError(
            "no background pixels to estimate the white point; the tissue mask covers the whole image. "
            "Pass an explicit `white_point`."
        )
    wp = np.median(bg_pixels, axis=1)
    if np.any(wp <= 0):
        raise StainFittingError(
            "estimated white point is non-positive; the background may be black. Pass an explicit `white_point`."
        )
    return wp
