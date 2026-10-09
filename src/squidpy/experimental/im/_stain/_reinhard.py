"""Reinhard (2001) colour transfer in Ruderman Lab space.

Pure DataArray layer: every function takes and returns ``xr.DataArray`` (or
numpy), stays lazy, touches no ``sdata``, and exposes no public surface. The
thin ``sdata`` wrapper lives in :mod:`._normalize`.

``params`` arguments must already be resolved by :func:`squidpy._params.resolve_params`
(the public dispatchers do this once); they are not re-validated here.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from squidpy.experimental.im._stain._validation import StainFittingError
from squidpy.types import ReinhardParams

# Numerical safeguard against divide-by-zero on flat (constant-colour)
# channels. Not a tuning knob, so kept off the public ReinhardParams surface.
_SIGMA_FLOOR: float = 1e-6


def validate_reinhard_params(params: dict[str, Any]) -> None:
    """Coerce ``params`` in place and range-check it. Raises on invalid values."""
    params["luminosity_threshold"] = float(params["luminosity_threshold"])
    params["mask_background"] = bool(params["mask_background"])
    if not 0.0 < params["luminosity_threshold"] <= 1.0:
        raise ValueError(f"`luminosity_threshold` must be in (0, 1], got {params['luminosity_threshold']}.")


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
    the param-driven luminosity mask (or ``None`` for vanilla Reinhard)."""
    if tissue_mask is not None:
        return as_spatial_mask(tissue_mask, lab)
    if params["mask_background"]:
        return foreground_mask_from_lab(lab, params["luminosity_threshold"])
    return None


def _tissue_lab_moments(
    image_rgb: xr.DataArray, params: ReinhardParams, tissue_mask: np.ndarray | None, *, image_key: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _check_channel_dim(image_rgb)
    lab = rgb_to_lab_ruderman(image_rgb)
    mask = _reinhard_mask(lab, params, tissue_mask)
    masked = lab.where(mask) if mask is not None else lab
    # Accumulate in float64: this is a reduction, not a map, so the float32
    # working dtype is not enough. sigma falls out of E[x^2] - E[x]^2, whose two
    # terms are ~5000x the variance they bracket; in float32 that cancellation
    # costs ~1% of sigma and makes the answer depend on chunking. The cast is
    # elementwise, so it fuses into the per-chunk graph and stays lazy.
    wide = masked.astype(np.float64)
    stats = xr.Dataset(
        {
            "n": wide.count(dim=("y", "x")),
            "s": wide.sum(dim=("y", "x"), skipna=True),
            "s2": (wide**2).sum(dim=("y", "x"), skipna=True),
        }
    ).compute()
    n = np.asarray(stats["n"].values, dtype=np.float64)
    if not np.all(n > 0):
        raise StainFittingError(
            "foreground mask leaves zero tissue pixels in at least one channel; "
            "the luminosity_threshold may be too low or the image may be blank.",
            image_key=image_key,
        )
    return n, np.asarray(stats["s"].values, dtype=np.float64), np.asarray(stats["s2"].values, dtype=np.float64)


def _stats_from_moments(n: np.ndarray, s: np.ndarray, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = s / n
    # ponytail: one-pass. Two-pass would be robust even in float32, but the
    # pooled mu is not known until every slide has been read, so it would cost a
    # second traversal of the cohort. float64 one-pass lands at ~1e-13; the clamp
    # absorbs the residual cancellation on a flat channel, where _SIGMA_FLOOR
    # takes over downstream. Revisit only if a real slide shows drift.
    return mu, np.sqrt(np.maximum(s2 / n - mu**2, 0.0))


def fit_reinhard(
    image_rgb: xr.DataArray | Sequence[xr.DataArray],
    params: ReinhardParams,
    *,
    tissue_mask: np.ndarray | Sequence[np.ndarray | None] | None = None,
    image_key: str | Sequence[str | None] | None = None,
) -> StainFit:
    """Fit Reinhard channel statistics on one or more reference images.

    Converts to Ruderman Lab and computes per-channel ``mu``/``sigma``
    (population std) over the tissue pixels. Several images pool into one
    reference by summing their per-channel moments, so a slide contributes in
    proportion to its **tissue pixel count**, not equally. A single image is a
    pool of one. ``tissue_mask`` (``(y, x)`` booleans aligned to each image)
    selects the tissue pixels when given; otherwise the ``mask_background`` /
    ``luminosity_threshold`` params drive the mask. ``image_key`` only names the
    image in error messages.
    """
    das = [image_rgb] if isinstance(image_rgb, xr.DataArray) else list(image_rgb)
    masks = (
        [tissue_mask] * len(das) if tissue_mask is None or isinstance(tissue_mask, np.ndarray) else list(tissue_mask)
    )
    keys = [image_key] * len(das) if image_key is None or isinstance(image_key, str) else list(image_key)
    moments = [_tissue_lab_moments(da, params, m, image_key=k) for da, m, k in zip(das, masks, keys, strict=True)]
    pooled = (np.sum(np.stack(x), axis=0) for x in zip(*moments, strict=True))
    mu, sigma = _stats_from_moments(*pooled)
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
    fit_src = fit_rgb if fit_rgb is not None else image_rgb
    mu_src, sigma_src = _stats_from_moments(*_tissue_lab_moments(fit_src, params, tissue_mask, image_key=None))
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
