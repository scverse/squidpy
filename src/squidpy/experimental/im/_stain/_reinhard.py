"""Reinhard (2001) colour transfer in Ruderman Lab space.

Pure DataArray layer: every function takes and returns ``xr.DataArray`` (or
numpy), stays lazy, touches no ``sdata``, and exposes no public surface. The
thin ``sdata`` wrapper lives in :mod:`._normalize`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import xarray as xr

from squidpy.experimental.im._stain._constants import DEFAULT_LUMINOSITY_THRESHOLD
from squidpy.experimental.im._stain._conversion import (
    _apply_along_channel,
    _check_channel_dim,
    _working_dtype,
    lab_ruderman_to_rgb,
    rgb_to_lab_ruderman,
)
from squidpy.experimental.im._stain._mask import luminosity_foreground_mask
from squidpy.experimental.im._stain._reference import StainReference

# Numerical safeguard against divide-by-zero on flat (constant-colour)
# channels. Not a tuning knob, so kept off the public ReinhardParams surface.
_SIGMA_FLOOR: float = 1e-6


@dataclass(slots=True, frozen=True)
class ReinhardParams:
    """Tuning knobs for Reinhard stain normalization.

    Pass an instance (or a ``Mapping`` of field names to values) as
    ``method_params``. Frozen so validation in ``__post_init__`` cannot be
    silently bypassed by later mutation.
    """

    luminosity_threshold: float = DEFAULT_LUMINOSITY_THRESHOLD
    """Normalised Ruderman Lab-L cutoff in ``(0, 1]``; pixels brighter than this are excluded from the fit."""

    mask_background: bool = True
    """If ``True``, fit channel statistics over tissue pixels only; if ``False``, use every pixel (vanilla Reinhard)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "luminosity_threshold", float(self.luminosity_threshold))
        object.__setattr__(self, "mask_background", bool(self.mask_background))
        if not 0.0 < self.luminosity_threshold <= 1.0:
            raise ValueError(f"`luminosity_threshold` must be in (0, 1], got {self.luminosity_threshold}.")


_REINHARD_DEFAULTS = ReinhardParams()
_REINHARD_FIELDS = frozenset(f.name for f in fields(ReinhardParams))


def _resolve_reinhard_params(method_params: ReinhardParams | Mapping[str, Any] | None) -> ReinhardParams:
    """Normalise the ``method_params`` argument to a :class:`ReinhardParams` instance."""
    if method_params is None:
        return _REINHARD_DEFAULTS
    if isinstance(method_params, ReinhardParams):
        return method_params
    if isinstance(method_params, Mapping):
        unknown = set(method_params) - _REINHARD_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown `method_params` field(s): {sorted(unknown)}; expected from {sorted(_REINHARD_FIELDS)}."
            )
        return ReinhardParams(**method_params)
    raise TypeError(f"`method_params` must be ReinhardParams, Mapping, or None; got {type(method_params).__name__}.")


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


def fit_reinhard(image_rgb: xr.DataArray, params: ReinhardParams) -> StainReference:
    """Fit Reinhard channel statistics on a reference image.

    Converts to Ruderman Lab, computes per-channel ``mu``/``sigma`` over
    tissue pixels (or all pixels when ``mask_background=False``), and packs
    them into a ``StainReference(method="reinhard")``.
    """
    _check_channel_dim(image_rgb)
    lab = rgb_to_lab_ruderman(image_rgb)
    mask = luminosity_foreground_mask(image_rgb, params.luminosity_threshold) if params.mask_background else None
    mu, sigma = _masked_channel_stats(lab, mask)
    return StainReference(method="reinhard", mu=mu, sigma=sigma)


def apply_reinhard(image_rgb: xr.DataArray, reference: StainReference, params: ReinhardParams) -> xr.DataArray:
    """Apply a Reinhard reference to a source image.

    Standardises by the source's own tissue statistics, rescales to the
    reference statistics, and converts back to RGB. The transform is applied
    to every pixel (the map is global); only the statistics that define it
    are tissue-only. Lazy if and only if the input is lazy.
    """
    _check_channel_dim(image_rgb)
    lab = rgb_to_lab_ruderman(image_rgb)
    mask = luminosity_foreground_mask(image_rgb, params.luminosity_threshold) if params.mask_background else None
    mu_src, sigma_src = _masked_channel_stats(lab, mask)
    sigma_src = np.maximum(sigma_src, _SIGMA_FLOOR)

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
    return lab_ruderman_to_rgb(lab_out)
