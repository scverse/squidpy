"""Foreground (tissue) masking for stain fitting.

Method-agnostic on purpose: Reinhard fits its channel statistics over tissue
pixels only, and the Macenko/Vahadane fits need the same kind of mask. Two
variants live here - luminosity (Reinhard, intensity space) and absorbance
(decomposition, optical-density space) - both returning the same ``(y, x)``
boolean contract so downstream statistics code stays mask-source-agnostic.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from squidpy.experimental.im._stain._constants import (
    RUDERMAN_LMS_TO_LAB,
    RUDERMAN_RGB_TO_LMS,
)


def _white_luminosity() -> float:
    """Ruderman Lab-L value of a pure-white pixel.

    L is a positive sum of ``log(LMS + 1)`` and every LMS coefficient is
    positive, so L is maximised at ``(255, 255, 255)``. Dividing the L
    channel by this constant maps luminosity into ``[0, 1]`` (black -> 0,
    white -> 1).
    """
    white = np.array([255.0, 255.0, 255.0], dtype=np.float64)
    log_lms = np.log(white @ RUDERMAN_RGB_TO_LMS.T + 1.0)
    return float((RUDERMAN_LMS_TO_LAB @ log_lms)[0])


_L_WHITE: float = _white_luminosity()


def foreground_mask_from_lab(lab: xr.DataArray, threshold: float) -> xr.DataArray:
    """Tissue mask from an already-computed Ruderman Lab image.

    Lets callers that have converted to Lab already (the fit/apply paths)
    derive the mask without a second RGB->Lab conversion: both the mask and
    the channel statistics then read from the same lazy ``lab`` graph, so
    dask materialises the conversion once.
    """
    luminosity = lab.isel(c=0, drop=True) / _L_WHITE
    return luminosity <= threshold


def as_spatial_mask(mask: np.ndarray, like: xr.DataArray) -> xr.DataArray:
    """Wrap a ``(y, x)`` boolean array as a DataArray aligned to ``like``'s y/x.

    Copies ``like``'s ``y``/``x`` coords (when present) so ``like.where(...)``
    aligns by coordinate rather than silently broadcasting. ``mask`` must match
    ``like`` in the spatial dims.
    """
    coords = {d: like.coords[d] for d in ("y", "x") if d in like.coords}
    return xr.DataArray(np.asarray(mask, dtype=bool), dims=("y", "x"), coords=coords)


def foreground_mask_from_sda(sda: xr.DataArray, beta: float = 0.15) -> xr.DataArray:
    """Tissue mask from an already-computed optical-density (SDA) image.

    The absorbance-space sibling of :func:`foreground_mask_from_lab`: lets the
    decomposition fit derive the mask from the same lazy ``sda`` graph it
    already needs for the optical densities, so dask materialises the
    RGB->SDA conversion once. ``True`` = tissue (mean absorbance ``> beta``).
    """
    return sda.mean(dim="c") > beta
