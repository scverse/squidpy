"""Reinhard (2001) colour transfer in Ruderman Lab space.

Pure DataArray layer: every function takes and returns ``xr.DataArray`` (or
numpy), stays lazy, touches no ``sdata``, and exposes no public surface. The
thin ``sdata`` wrapper lives in :mod:`._normalize`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from squidpy.experimental.im._stain._conversion import (
    _apply_along_channel,
    _check_channel_dim,
    _working_dtype,
    lab_ruderman_to_rgb,
    rgb_to_lab_ruderman,
)
from squidpy.experimental.im._stain._mask import as_spatial_mask, foreground_mask_from_lab
from squidpy.experimental.im._stain._reference import StainFit
from squidpy.experimental.utils._params import resolve_params
from squidpy.types import _REINHARD_DEFAULTS, ReinhardParams

# Numerical safeguard against divide-by-zero on flat (constant-colour)
# channels. Not a tuning knob, so kept off the public ReinhardParams surface.
_SIGMA_FLOOR: float = 1e-6


def validate_reinhard_params(params: dict[str, Any]) -> None:
    """Coerce ``params`` in place and range-check it. Raises on invalid values."""
    params["luminosity_threshold"] = float(params["luminosity_threshold"])
    params["mask_background"] = bool(params["mask_background"])
    if not 0.0 < params["luminosity_threshold"] <= 1.0:
        raise ValueError(f"`luminosity_threshold` must be in (0, 1], got {params['luminosity_threshold']}.")


def _resolve_reinhard_params(method_params: ReinhardParams | Mapping[str, Any] | None) -> ReinhardParams:
    """Normalise the ``method_params`` argument to a validated :class:`~squidpy.types.ReinhardParams`."""
    return resolve_params(method_params, defaults=_REINHARD_DEFAULTS, validate=validate_reinhard_params)


def _masked_channel_stats(lab: xr.DataArray, mask: xr.DataArray | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over the spatial dims, tissue pixels only.

    Lazy: the masked mean and std are bundled into one dataset and computed
    in a single pass, never materialising the full image. Returns two
    shape-``(3,)`` float64 arrays in channel order.

    Raises ``ValueError`` if the mask leaves no tissue pixels in any channel
    (the mean would be NaN), with an actionable message.
    """
    masked = lab.where(mask) if mask is not None else lab
    stats = xr.Dataset(
        {
            "mu": masked.mean(dim=("y", "x"), skipna=True),
            "sigma": masked.std(dim=("y", "x"), skipna=True),
        }
    ).compute()
    mu = np.asarray(stats["mu"].values, dtype=np.float64)
    sigma = np.asarray(stats["sigma"].values, dtype=np.float64)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        raise ValueError(
            "Foreground mask leaves zero tissue pixels in at least one channel; "
            "the luminosity_threshold may be too low or the image may be blank."
        )
    return mu, sigma


def _transfer_kernel(
    x: np.ndarray,
    *,
    mu_src: np.ndarray,
    sigma_src: np.ndarray,
    mu_ref: np.ndarray,
    sigma_ref: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    x = x.astype(dtype, copy=False)
    return ((x - mu_src) / sigma_src * sigma_ref + mu_ref).astype(dtype, copy=False)


def _reinhard_mask(lab: xr.DataArray, params: ReinhardParams, tissue_mask: np.ndarray | None) -> xr.DataArray | None:
    """Resolve the tissue mask for the Reinhard stats: external mask wins, else
    the param-driven luminosity mask (or ``None`` for vanilla Reinhard).

    ``params`` is resolved here rather than assumed complete: :class:`~squidpy.types.ReinhardParams`
    is ``total=False``, so a caller may legitimately pass a partial mapping."""
    params = _resolve_reinhard_params(params)
    if tissue_mask is not None:
        return as_spatial_mask(tissue_mask, lab)
    if params["mask_background"]:
        return foreground_mask_from_lab(lab, params["luminosity_threshold"])
    return None


def fit_reinhard(image_rgb: xr.DataArray, params: ReinhardParams, *, tissue_mask: np.ndarray | None = None) -> StainFit:
    """Fit Reinhard channel statistics on a reference image.

    Converts to Ruderman Lab, computes per-channel ``mu``/``sigma`` over
    tissue pixels, and packs them into a ``StainFit(method="reinhard")``.
    ``tissue_mask`` (a ``(y, x)`` boolean aligned to ``image_rgb``) selects the
    tissue pixels when given; otherwise the ``mask_background`` /
    ``luminosity_threshold`` params drive the mask.
    """
    _check_channel_dim(image_rgb)
    lab = rgb_to_lab_ruderman(image_rgb)
    mu, sigma = _masked_channel_stats(lab, _reinhard_mask(lab, params, tissue_mask))
    return StainFit(method="reinhard", mu=mu, sigma=sigma)


def apply_reinhard(
    image_rgb: xr.DataArray,
    reference: StainFit,
    params: ReinhardParams,
    *,
    fit_rgb: xr.DataArray | None = None,
    tissue_mask: np.ndarray | None = None,
    out_dtype: np.dtype | type = np.uint8,
) -> xr.DataArray:
    """Apply a Reinhard reference to a source image.

    Standardises by the source's own tissue statistics, rescales to the
    reference statistics, and converts back to RGB. The transform is applied
    to every pixel of ``image_rgb`` (the map is global); the defining
    statistics are reduced on ``fit_rgb`` (a coarse level) when given, so the
    full-resolution image is never materialised to compute them.
    ``tissue_mask`` (aligned to ``fit_rgb``) selects the source tissue pixels.
    Lazy if and only if ``image_rgb`` is lazy.
    """
    _check_channel_dim(image_rgb)
    fit_lab = rgb_to_lab_ruderman(fit_rgb if fit_rgb is not None else image_rgb)
    mu_src, sigma_src = _masked_channel_stats(fit_lab, _reinhard_mask(fit_lab, params, tissue_mask))
    sigma_src = np.maximum(sigma_src, _SIGMA_FLOOR)

    lab = rgb_to_lab_ruderman(image_rgb)

    dtype = _working_dtype(lab)
    lab_out = _apply_along_channel(
        lab,
        _transfer_kernel,
        out_dtype=dtype,
        mu_src=mu_src.astype(dtype, copy=False),
        sigma_src=sigma_src.astype(dtype, copy=False),
        mu_ref=np.asarray(reference.mu, dtype=dtype),
        sigma_ref=np.asarray(reference.sigma, dtype=dtype),
        dtype=dtype,
    )
    return lab_ruderman_to_rgb(lab_out, out_dtype=out_dtype)
