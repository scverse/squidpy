"""RGB <-> optical density (SDA) and RGB <-> Ruderman Lab conversions.

All functions operate on :class:`xarray.DataArray` with a channel dimension
named ``"c"`` of length 3. Numpy-backed and dask-backed inputs are both
supported transparently; nothing here forces materialisation of large arrays.
Each public function compiles down to a single :func:`xarray.apply_ufunc`
call so that dask schedules one task per chunk regardless of how many
elementwise and matrix steps the transform contains.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from squidpy.experimental.im._stain._constants import (
    DEFAULT_BACKGROUND_INTENSITY,
    RUDERMAN_LAB_TO_LMS,
    RUDERMAN_LMS_TO_LAB,
    RUDERMAN_LMS_TO_RGB,
    RUDERMAN_RGB_TO_LMS,
    SDA_SCALE,
)

_CHANNEL_DIM = "c"


def _check_channel_dim(arr: xr.DataArray) -> None:
    if _CHANNEL_DIM not in arr.dims:
        raise ValueError(f"Input must have a dimension named {_CHANNEL_DIM!r}; got dims {arr.dims}.")
    if arr.sizes[_CHANNEL_DIM] != 3:
        raise ValueError(f"Channel dimension {_CHANNEL_DIM!r} must have length 3; got {arr.sizes[_CHANNEL_DIM]}.")


def _background_as_numpy(
    background_intensity: xr.DataArray | np.ndarray | float | int,
) -> np.ndarray:
    if isinstance(background_intensity, xr.DataArray):
        arr = np.asarray(background_intensity.values, dtype=np.float64)
    else:
        arr = np.asarray(background_intensity, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.repeat(arr, 3)
    if arr.shape != (3,):
        raise ValueError(f"background_intensity must be scalar or shape (3,); got shape {arr.shape}.")
    return arr


def _working_dtype(arr: xr.DataArray) -> np.dtype:
    # Integer/uint inputs are promoted to float32 to keep dask graphs cheap
    # on whole-slide images; already-float inputs preserve caller dtype.
    return arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.dtype(np.float32)


def _apply_along_channel(
    arr: xr.DataArray,
    kernel,
    *,
    out_dtype: np.dtype,
    **kwargs,
) -> xr.DataArray:
    """Run a per-chunk kernel that consumes and returns the channel axis.

    ``apply_ufunc`` moves the ``c`` core dim to the end of the output; we
    transpose back to the caller's original dim order so downstream
    consumers see a stable layout.
    """
    original_dims = arr.dims
    out = xr.apply_ufunc(
        kernel,
        arr,
        input_core_dims=[[_CHANNEL_DIM]],
        output_core_dims=[[_CHANNEL_DIM]],
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[out_dtype],
    )
    return out.transpose(*original_dims)


def _rgb_to_sda_kernel(x: np.ndarray, *, bg: np.ndarray, dtype: np.dtype) -> np.ndarray:
    x = x.astype(dtype, copy=False)
    return (-np.log((x + 1.0) / (bg + 1.0)) * SDA_SCALE).astype(dtype, copy=False)


def _sda_to_rgb_kernel(x: np.ndarray, *, bg: np.ndarray, dtype: np.dtype) -> np.ndarray:
    rgb = (bg + 1.0) * np.exp(-x.astype(dtype, copy=False) / SDA_SCALE) - 1.0
    np.clip(rgb, 0.0, 255.0, out=rgb)
    return rgb.astype(dtype, copy=False)


def _rgb_to_lab_kernel(x: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    x = x.astype(dtype, copy=False)
    lms = x @ RUDERMAN_RGB_TO_LMS.T.astype(dtype, copy=False)
    np.log(lms + 1.0, out=lms)
    return (lms @ RUDERMAN_LMS_TO_LAB.T.astype(dtype, copy=False)).astype(dtype, copy=False)


def _lab_to_rgb_kernel(x: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    x = x.astype(dtype, copy=False)
    log_lms = x @ RUDERMAN_LAB_TO_LMS.T.astype(dtype, copy=False)
    # The +1.0 / -1.0 pair is paired with the matching offset in
    # `_rgb_to_lab_kernel` so the round trip remains exact for valid RGB.
    lms = np.exp(log_lms) - 1.0
    rgb = lms @ RUDERMAN_LMS_TO_RGB.T.astype(dtype, copy=False)
    np.clip(rgb, 0.0, 255.0, out=rgb)
    return rgb.astype(dtype, copy=False)


def rgb_to_sda(
    rgb: xr.DataArray,
    background_intensity: xr.DataArray | np.ndarray | float = DEFAULT_BACKGROUND_INTENSITY,
) -> xr.DataArray:
    """Convert RGB intensities to standard deviation per absorbance (SDA).

    Equivalent to optical density with a per-channel background ``I_0``::

        sda = -log((rgb + 1) / (I_0 + 1)) * SDA_SCALE

    The matched ``+1`` terms avoid ``log(0)`` at fully saturated black
    pixels and guarantee that pixels at the white point map exactly to
    zero. Scaling matches the HistomicsTK convention so that luminosity
    thresholds from the published H&E literature transfer directly.

    Parameters
    ----------
    rgb
        Image with a ``"c"`` dimension of length 3. May be numpy- or
        dask-backed; the operation is purely elementwise and stays lazy.
    background_intensity
        Per-channel white-point ``I_0``. Scalar, shape-``(3,)`` array, or a
        DataArray with a ``"c"`` dim of length 3.

    Returns
    -------
    SDA-space DataArray, float dtype. Lazy if and only if ``rgb`` was lazy.
    """
    _check_channel_dim(rgb)
    dtype = _working_dtype(rgb)
    bg = _background_as_numpy(background_intensity).astype(dtype, copy=False)
    return _apply_along_channel(rgb, _rgb_to_sda_kernel, out_dtype=dtype, bg=bg, dtype=dtype)


def sda_to_rgb(
    sda: xr.DataArray,
    background_intensity: xr.DataArray | np.ndarray | float = DEFAULT_BACKGROUND_INTENSITY,
) -> xr.DataArray:
    """Convert SDA back to RGB intensities in ``[0, 255]``.

    Inverse of :func:`rgb_to_sda`. The result is clipped to ``[0, 255]`` but
    kept in float dtype; uint8 conversion is the caller's choice.
    """
    _check_channel_dim(sda)
    dtype = _working_dtype(sda)
    bg = _background_as_numpy(background_intensity).astype(dtype, copy=False)
    return _apply_along_channel(sda, _sda_to_rgb_kernel, out_dtype=dtype, bg=bg, dtype=dtype)


def rgb_to_lab_ruderman(rgb: xr.DataArray) -> xr.DataArray:
    """Convert RGB to Ruderman et al. (1998) decorrelated Lab space.

    This is the Lab variant used by Reinhard et al. (2001) for colour
    transfer, not CIE Lab. Results differ from
    :func:`skimage.color.rgb2lab`.

    The pipeline is RGB -> LMS via :data:`RUDERMAN_RGB_TO_LMS`, then
    ``log(LMS + 1)``, then LMS -> Lab via :data:`RUDERMAN_LMS_TO_LAB`. All
    steps fuse into a single per-chunk numpy kernel.
    """
    _check_channel_dim(rgb)
    dtype = _working_dtype(rgb)
    return _apply_along_channel(rgb, _rgb_to_lab_kernel, out_dtype=dtype, dtype=dtype)


def lab_ruderman_to_rgb(lab: xr.DataArray) -> xr.DataArray:
    """Inverse of :func:`rgb_to_lab_ruderman`.

    Returns RGB clipped to ``[0, 255]`` in float dtype; uint8 conversion is
    the caller's choice.
    """
    _check_channel_dim(lab)
    dtype = _working_dtype(lab)
    return _apply_along_channel(lab, _lab_to_rgb_kernel, out_dtype=dtype, dtype=dtype)
