"""Public, ``sdata``-aware entry points for stain normalization.

The single integration boundary for the stain module: the only file that
reads ``sdata.images[...]``, writes back via :class:`Image2DModel`, and is
re-exported publicly. Everything it calls is a pure DataArray-layer
primitive (:mod:`._reinhard`, :mod:`._mask`, :mod:`._conversion`).

Both entry points dispatch on the fitting ``method`` (``"reinhard"`` colour
transfer, or ``"macenko"``/``"vahadane"`` absorbance decomposition); a third
entry, :meth:`~squidpy.experimental.im.StainFit.decompose`, projects an image onto its stain matrix.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel
from spatialdata.transformations import get_transformation

from squidpy._params import resolve_params
from squidpy._utils import _get_scale_factors
from squidpy._validators import normalize_choice
from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._conversion import _check_channel_dim
from squidpy.experimental.im._stain._decomposition import (
    MacenkoParams,
    VahadaneParams,
    fit_decomposition,
)
from squidpy.experimental.im._stain._reference import StainFit, StainMethod
from squidpy.experimental.im._stain._reinhard import (
    ReinhardParams,
    fit_reinhard,
)
from squidpy.experimental.im._stain._white_point import (
    default_white_point,
    validate_rgb_range,
    white_point_from_background,
)
from squidpy.experimental.im._utils import (
    _choose_label_scale_for_image,
    get_element_data,
    get_mask_materialized,
    resolve_tissue_mask,
)

#: The params type each method takes.
_METHOD_PARAMS: dict[str, type[ReinhardParams | MacenkoParams | VahadaneParams]] = {
    "reinhard": ReinhardParams,
    "macenko": MacenkoParams,
    "vahadane": VahadaneParams,
}
_VALID_METHODS = tuple(_METHOD_PARAMS)

# Public union accepted by the method_params argument of the dispatchers.
MethodParams = ReinhardParams | MacenkoParams | VahadaneParams | Mapping[str, Any] | None


def _resolve_image(
    sdata: sd.SpatialData,
    *,
    image_key: str,
    scale: str,
    prefer: Literal["coarsest", "finest"],
) -> xr.DataArray:
    if image_key not in sdata.images:
        raise ValueError(f"image_key {image_key!r} not found, valid keys: {list(sdata.images.keys())}")
    node = sdata.images[image_key]
    da = get_element_data(node, scale, "image", image_key, prefer=prefer)
    _check_channel_dim(da)
    return da


def _resolve_mask_key_and_scale(
    sdata: sd.SpatialData, *, image_key: str, target_da: xr.DataArray, tissue_mask_key: str | None
) -> tuple[str, str, tuple[int, int]]:
    """Resolve the (mandatory) tissue-mask key and the label scale closest to ``target_da``.

    Shared by the two mask consumers below. Consumes a
    :func:`!detect_tissue` labels element - raises if
    none exists.
    """
    mask_key = resolve_tissue_mask(sdata, image_key, "auto", tissue_mask_key, auto_create=False)
    target_hw = (int(target_da.sizes["y"]), int(target_da.sizes["x"]))
    label_scale = _choose_label_scale_for_image(sdata.labels[mask_key], target_hw)
    return mask_key, label_scale, target_hw


def _resolve_tissue_bool_mask(
    sdata: sd.SpatialData, *, image_key: str, fit_da: xr.DataArray, tissue_mask_key: str | None
) -> np.ndarray:
    """Return a materialised ``(y, x)`` boolean tissue mask aligned to ``fit_da``.

    For the (coarse) fit: nearest-resizes to ``fit_da``'s ``(y, x)`` when the
    closest label scale differs. The fits run on a coarse level, so the mask
    stays small.
    """
    mask_key, label_scale, target_hw = _resolve_mask_key_and_scale(
        sdata, image_key=image_key, target_da=fit_da, tissue_mask_key=tissue_mask_key
    )
    mask = get_mask_materialized(sdata, mask_key, label_scale) > 0
    if mask.shape != target_hw:
        from skimage.transform import resize

        mask = resize(mask, target_hw, order=0, preserve_range=True) > 0.5
    return mask


def _resolve_output_tissue_mask(
    sdata: sd.SpatialData, *, image_key: str, target_da: xr.DataArray, tissue_mask_key: str | None
) -> xr.DataArray:
    """Return a lazy ``(y, x)`` boolean tissue mask aligned to ``target_da``.

    Like :func:`_resolve_tissue_bool_mask` but kept lazy and at the (full-res)
    output resolution, for compositing the original background back into the
    normalized image without materialising the full frame. The label pyramid
    shares the image's scale factors, so the matching level usually lines up
    exactly; only a residual size mismatch forces a (small) eager resize.
    """
    mask_key, label_scale, target_hw = _resolve_mask_key_and_scale(
        sdata, image_key=image_key, target_da=target_da, tissue_mask_key=tissue_mask_key
    )
    coords = {d: target_da.coords[d] for d in ("y", "x") if d in target_da.coords}
    mask = get_element_data(sdata.labels[mask_key], label_scale, "label", mask_key).squeeze() > 0
    if (int(mask.sizes["y"]), int(mask.sizes["x"])) == target_hw:
        return mask.assign_coords(coords)
    from skimage.transform import resize

    resized = resize(np.asarray(mask.data) > 0, target_hw, order=0, preserve_range=True) > 0.5
    return xr.DataArray(resized, dims=("y", "x"), coords=coords)


def _write_image(
    sdata: sd.SpatialData,
    *,
    source_node: Any,
    image_key_added: str,
    data_array: xr.DataArray,
    c_coords: list[Any] | None = None,
) -> None:
    """Write a derived image element, preserving the source's transforms/pyramid.

    Reconstructs the element from the bare array (a derived DataArray would
    carry the source's ``transform`` attr and collide with the transformations
    we pass) plus the dims/channel-coords/transforms to preserve. The same
    idiom as detect_tissue. ``_get_scale_factors`` returns ``[]`` for a
    single-scale source; parse needs ``None`` there (an empty list builds a
    degenerate single-level pyramid).
    """
    if image_key_added in sdata.images:
        raise ValueError(f"image_key_added={image_key_added!r} already exists in sdata.images.")
    if c_coords is None:
        c_coords = data_array.coords["c"].values.tolist() if "c" in data_array.coords else None
    sdata.images[image_key_added] = Image2DModel.parse(
        data_array.data,
        dims=data_array.dims,
        c_coords=c_coords,
        transformations=get_transformation(source_node, get_all=True),
        scale_factors=_get_scale_factors(source_node) or None,
    )


def estimate_white_point(
    sdata: sd.SpatialData,
    *,
    image_key: str,
    tissue_mask_key: str | None = None,
    scale: str | Literal["auto"] = "auto",
) -> np.ndarray:
    """Estimate the white point ``I_0`` from a slide's background (non-tissue median).

    Opt-in alternative to the fixed dtype-aware default white point, for a slide
    whose unstained background is genuinely not full white. Samples the
    per-channel median over **non-tissue** pixels (background = the complement of
    the :func:`!detect_tissue` mask).

    Parameters
    ----------
    sdata, image_key
        The SpatialData object and the RGB image key.
    tissue_mask_key
        Tissue-label element key (defaults to ``f"{image_key}_tissue"``); a
        tissue mask is required, as for :func:`fit_stain_reference`.
    scale
        Scale level to sample on. ``"auto"`` uses the coarsest level.
        The sampled level is materialised to take the median, so keep this
        coarse - do not pass a fine level on a whole-slide image.

    Returns
    -------
    Shape-``(3,)`` white point; pass it as ``white_point`` to
    :func:`fit_stain_reference` / :meth:`~squidpy.experimental.im.StainFit.decompose`.
    """
    da = _resolve_image(sdata, image_key=image_key, scale=scale, prefer="coarsest")
    validate_rgb_range(da)
    tissue_mask = _resolve_tissue_bool_mask(sdata, image_key=image_key, fit_da=da, tissue_mask_key=tissue_mask_key)
    return white_point_from_background(da, ~tissue_mask)


def fit_stain_reference(
    sdata: sd.SpatialData,
    *,
    image_key: str,
    method: StainMethod = "macenko",
    scale: str | Literal["auto"] = "auto",
    method_params: MethodParams = None,
    white_point: np.ndarray | None = None,
    tissue_mask_key: str | None = None,
    max_angle_deg: float = 45.0,
    canonical_reference: Mapping[str, np.ndarray] | None = None,
) -> StainFit:
    """Fit a stain reference from an image in a :class:`~spatialdata.SpatialData` object.

    Parameters
    ----------
    sdata
        SpatialData object containing the image.
    image_key
        Key of the RGB image in ``sdata.images`` to fit on.
    method
        Fitting method: ``"macenko"`` or ``"vahadane"`` (physical
        stain-matrix decomposition, usable by both :meth:`~squidpy.experimental.im.StainFit.transform` and
        :meth:`~squidpy.experimental.im.StainFit.decompose`), or ``"reinhard"`` (faster statistical colour
        transfer, no stain separation). Macenko is the default because its one
        documented weakness - artifact pixels contaminating the fit - is removed
        by the mandatory tissue mask.
    scale
        Scale level to fit on. ``"auto"`` uses the coarsest level,
        which is cheap and sufficient for colour statistics.
    method_params
        A mapping of ``ReinhardParams``/``MacenkoParams``/``VahadaneParams`` keys,
        or ``None`` for defaults. Must match ``method``.
    white_point
        Per-channel white point ``I_0`` ``(3,)`` for the decomposition methods.
        If ``None``, a fixed full-white ``[255, 255, 255]`` is used (the
        HistomicsTK/Macenko convention), so unstained pixels round-trip to
        white. Pass :func:`estimate_white_point` only for slides with a
        known non-white background. Ignored by Reinhard.
    tissue_mask_key
        Key of a tissue-label element in ``sdata.labels`` (as produced by
        :func:`!detect_tissue`) restricting the fit to
        tissue pixels. If ``None``, ``f"{image_key}_tissue"`` is used. A tissue
        mask is **required**: if neither exists, a :class:`KeyError` asks you to
        run :func:`!detect_tissue` first.
    max_angle_deg
        Tolerance of the H/E sanity gate for the decomposition methods: the fit
        raises :class:`!StainFittingError` if either recovered stain vector
        deviates more than this many degrees from its canonical reference.
        Default ``45``. Ignored by Reinhard.
    canonical_reference
        Canonical H/E reference for the decomposition methods, a mapping with
        ``"hematoxylin"`` and ``"eosin"`` keys to ``(3,)`` RGB optical-density
        unit vectors. Drives both the H/E column ordering and the deviation
        gate. If ``None``, the Ruifrok H&E vectors are used. Ignored by Reinhard.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.im.StainFit`. Nothing is written to ``sdata``.
    """
    method = normalize_choice(method, _VALID_METHODS, name="method")
    da = _resolve_image(sdata, image_key=image_key, scale=scale, prefer="coarsest")
    validate_rgb_range(da)
    params = resolve_params(method_params, _METHOD_PARAMS[method])
    tissue_mask = _resolve_tissue_bool_mask(sdata, image_key=image_key, fit_da=da, tissue_mask_key=tissue_mask_key)
    if method == "reinhard":
        return fit_reinhard(da, params, tissue_mask=tissue_mask)
    bg = default_white_point(da) if white_point is None else np.asarray(white_point, np.float64)
    reference = RUIFROK_HE if canonical_reference is None else dict(canonical_reference)
    return fit_decomposition(
        da,
        method,
        params,
        bg,
        tissue_mask=tissue_mask,
        image_key=image_key,
        reference=reference,
        max_angle_deg=max_angle_deg,
    )
