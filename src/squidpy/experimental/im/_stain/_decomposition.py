"""Macenko and Vahadane stain decomposition (fit + apply).

Pure DataArray/numpy layer: no ``sdata``, no public export. The stain-matrix
fits run on tissue pixels (a bounded reduction at the chosen scale); the apply
transform is a single per-pixel matmul and stays lazy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from squidpy._utils import legacy_random
from squidpy._validators import assert_non_negative, assert_positive
from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._conversion import (
    _apply_along_channel,
    _check_channel_dim,
    _working_dtype,
    rgb_to_sda,
    sda_to_rgb,
)
from squidpy.experimental.im._stain._mask import as_spatial_mask, foreground_mask_from_sda
from squidpy.experimental.im._stain._reference import StainFit, StainMethod
from squidpy.experimental.im._stain._validation import (
    StainFittingError,
    _unit_columns,
    complement_third_column,
    reorder_to_canonical,
    validate_stain_matrix,
)
from squidpy.experimental.utils._params import resolve_params
from squidpy.types import (
    _MACENKO_DEFAULTS,
    _VAHADANE_DEFAULTS,
    MacenkoParams,
    VahadaneParams,
)

_MAXC_PERCENTILE = 99.0
_MAXC_FLOOR = 1e-6


def validate_macenko_params(params: dict[str, Any]) -> None:
    """Coerce ``params`` in place and range-check it. Raises on invalid values."""
    params["alpha"] = float(params["alpha"])
    params["beta"] = float(params["beta"])
    if not 0.0 < params["alpha"] < 50.0:  # open interval, so `assert_in_range` does not fit
        raise ValueError(f"`alpha` must be in (0, 50), got {params['alpha']}.")
    assert_non_negative(params["beta"], name="beta")


def validate_vahadane_params(params: dict[str, Any]) -> None:
    """Coerce ``params`` in place and range-check it. Raises on invalid values."""
    params["beta"] = float(params["beta"])
    params["lambda1"] = float(params["lambda1"])
    params["n_iter"] = int(params["n_iter"])
    assert_non_negative(params["beta"], name="beta")
    assert_non_negative(params["lambda1"], name="lambda1")
    assert_positive(params["n_iter"], name="n_iter")


def _resolve_macenko_params(params: MacenkoParams | Mapping[str, Any] | None) -> MacenkoParams:
    return resolve_params(params, defaults=_MACENKO_DEFAULTS, validate=validate_macenko_params)


def _resolve_vahadane_params(params: VahadaneParams | Mapping[str, Any] | None) -> VahadaneParams:
    return resolve_params(params, defaults=_VAHADANE_DEFAULTS, validate=validate_vahadane_params)


def _tissue_od(
    image_rgb: xr.DataArray,
    white_point: np.ndarray,
    beta: float,
    *,
    tissue_mask: np.ndarray | None = None,
    image_key: str | None,
) -> np.ndarray:
    """Flatten tissue pixels to an ``(N, 3)`` optical-density matrix.

    Reduces over the chosen scale only (bounded); the stain fits need the
    tissue pixels resident for SVD/NMF, so this is the one materialising step.
    When ``tissue_mask`` (a ``(y, x)`` boolean aligned to ``image_rgb``) is
    given it selects the tissue pixels; otherwise the absorbance threshold
    ``beta`` is used.
    """
    sda = rgb_to_sda(image_rgb, white_point)
    mask = as_spatial_mask(tissue_mask, sda) if tissue_mask is not None else foreground_mask_from_sda(sda, beta)
    od = np.asarray(sda.where(mask).transpose("c", "y", "x").data.reshape(3, -1)).T
    od = od[np.all(np.isfinite(od), axis=1)]
    if od.shape[0] == 0:
        raise StainFittingError("no tissue pixels for stain fitting; the mask is empty.", image_key=image_key)
    # Keep signed OD: pixels brighter than the estimated background carry
    # negative absorbance that Macenko's SVD legitimately uses. Only Vahadane's
    # NMF requires non-negativity, and clips locally.
    return od


def _macenko_stain_matrix(od: np.ndarray, alpha: float) -> np.ndarray:
    """Recover a ``(3, 2)`` H/E matrix via Macenko's angular-extreme method."""
    # right singular vectors of OD = principal absorbance directions through 0
    _, _, vh = np.linalg.svd(od, full_matrices=False)
    plane = vh[:2].T  # (3, 2)
    # SVD sign is arbitrary; orient the basis into the data so the projected
    # angles cluster around 0 instead of straddling the atan2 branch cut at
    # +-180 deg (which would collapse the angular percentiles).
    signs = np.sign(od.mean(axis=0) @ plane)
    signs[signs == 0] = 1.0
    plane = plane * signs
    proj = od @ plane  # (N, 2)
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100.0 - alpha])
    extremes = np.stack(
        [plane @ np.array([np.cos(lo), np.sin(lo)]), plane @ np.array([np.cos(hi), np.sin(hi)])],
        axis=1,
    )
    return _unit_columns(extremes)


def _vahadane_stain_matrix(od: np.ndarray, params: VahadaneParams) -> np.ndarray:
    """Recover a ``(3, 2)`` H/E matrix via sparse NMF (Vahadane)."""
    from sklearn.decomposition import NMF

    nmf = NMF(
        n_components=2,
        init="nndsvda",
        random_state=legacy_random(np.random.default_rng(params["rng"])),
        alpha_W=params["lambda1"],
        l1_ratio=1.0,
        max_iter=params["n_iter"],
    )
    nmf.fit(np.clip(od, 0.0, None))  # NMF requires non-negative absorbance
    stains = nmf.components_.T  # (3, 2)
    if np.any(np.linalg.norm(stains, axis=0) < 1e-8):
        raise StainFittingError("Vahadane NMF produced a zero-norm stain vector.")
    return _unit_columns(stains)


def _stain_matrix(
    od: np.ndarray,
    method: StainMethod,
    params: Any,
    *,
    image_key: str | None,
    reference: dict[str, np.ndarray] = RUIFROK_HE,
    max_angle_deg: float = 45.0,
) -> np.ndarray:
    """Fit, canonicalise, complete and validate a ``(3, 3)`` stain matrix.

    ``reference`` (the canonical H/E vectors) drives both the column ordering and
    the deviation gate; ``max_angle_deg`` is the gate tolerance.
    """
    raw = _macenko_stain_matrix(od, params["alpha"]) if method == "macenko" else _vahadane_stain_matrix(od, params)
    matrix = complement_third_column(reorder_to_canonical(raw, reference))
    validate_stain_matrix(matrix, reference=reference, max_angle_deg=max_angle_deg, image_key=image_key)
    return matrix


def _concentrations(od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Per-pixel stain concentrations ``(N, 3)`` from optical density."""
    return od @ np.linalg.pinv(stain_matrix).T


def _max_concentrations(concentrations: np.ndarray) -> np.ndarray:
    """Robust per-stain (H, E) maximum concentrations ``(2,)`` from an ``(N, 3)`` array."""
    return np.maximum(np.percentile(concentrations[:, :2], _MAXC_PERCENTILE, axis=0), _MAXC_FLOOR)


def fit_decomposition(
    image_rgb: xr.DataArray,
    method: StainMethod,
    params: Any,
    white_point: np.ndarray,
    *,
    tissue_mask: np.ndarray | None = None,
    image_key: str | None = None,
    reference: dict[str, np.ndarray] = RUIFROK_HE,
    max_angle_deg: float = 45.0,
) -> StainFit:
    """Fit a decomposition :class:`~squidpy.experimental.im.StainFit` (stain matrix + max concentrations)."""
    # `params` is `total=False`, so resolve rather than assume every key is present.
    params = _resolve_macenko_params(params) if method == "macenko" else _resolve_vahadane_params(params)
    od = _tissue_od(image_rgb, white_point, params["beta"], tissue_mask=tissue_mask, image_key=image_key)
    matrix = _stain_matrix(od, method, params, image_key=image_key, reference=reference, max_angle_deg=max_angle_deg)
    return StainFit(
        method=method,
        stain_matrix=matrix,
        white_point=np.asarray(white_point, dtype=np.float64),
        max_concentrations=_max_concentrations(_concentrations(od, matrix)),
    )


def _matmul_kernel(x: np.ndarray, *, matrix: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return (x.astype(dtype, copy=False) @ matrix.T).astype(dtype, copy=False)


def apply_decomposition(
    image_rgb: xr.DataArray,
    reference: StainFit,
    params: Any,
    *,
    fit_rgb: xr.DataArray | None = None,
    tissue_mask: np.ndarray | None = None,
    out_dtype: np.dtype | type = np.uint8,
) -> xr.DataArray:
    """Normalize a source image to a decomposition reference (colour-basis transfer).

    Deconvolves with the *source's* own stain matrix and reconvolves with the
    reference matrix: only the stain *colour basis* changes, the *amount*
    (concentration magnitude) is untouched (HistomicsTK's
    ``deconvolution_based_normalization``). Use ``method="reinhard"`` for
    intensity/appearance matching.

    The source matrix is a colour property, so it is fit on ``fit_rgb`` (a
    coarse level) when given, while ``image_rgb`` (which may be full
    resolution) is only ever touched by the lazy operator - never
    materialised to fit a matrix.
    """
    _check_channel_dim(image_rgb)
    bg = reference.white_point
    # `params` is `total=False`, so resolve rather than assume every key is present.
    params = _resolve_macenko_params(params) if reference.method == "macenko" else _resolve_vahadane_params(params)

    od_src = _tissue_od(
        fit_rgb if fit_rgb is not None else image_rgb, bg, params["beta"], tissue_mask=tissue_mask, image_key=None
    )
    w_src = _stain_matrix(od_src, reference.method, params, image_key=None)
    operator = reference.stain_matrix @ np.linalg.pinv(w_src)

    sda = rgb_to_sda(image_rgb, bg)
    dtype = _working_dtype(sda)
    sda_out = _apply_along_channel(sda, _matmul_kernel, out_dtype=dtype, matrix=operator.astype(dtype), dtype=dtype)
    return sda_to_rgb(sda_out, bg, out_dtype=out_dtype)


def decompose_to_concentrations(
    image_rgb: xr.DataArray, stain_matrix: np.ndarray, white_point: np.ndarray
) -> xr.DataArray:
    """Project an image onto a stain matrix, returning a 3-channel concentration image.

    Channels are ``(hematoxylin, eosin, residual)``; the residual is the
    concentration along the complement vector and is a diagnostic, not a stain.
    """
    _check_channel_dim(image_rgb)
    sda = rgb_to_sda(image_rgb, white_point)
    dtype = _working_dtype(sda)
    pinv = np.linalg.pinv(stain_matrix)
    return _apply_along_channel(sda, _matmul_kernel, out_dtype=dtype, matrix=pinv.astype(dtype), dtype=dtype)
