"""Public, ``sdata``-aware entry points for stain normalization.

The single integration boundary for the stain module: the only file that
reads ``sdata.images[...]``, writes back via :class:`Image2DModel`, and is
re-exported publicly. Everything it calls is a pure DataArray-layer
primitive (:mod:`._reinhard`, :mod:`._mask`, :mod:`._conversion`).

Both entry points dispatch on the fitting ``method`` (``"reinhard"`` colour
transfer, or ``"macenko"``/``"vahadane"`` absorbance decomposition); a third
entry, :func:`decompose_stains`, projects an image onto its stain matrix.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import spatialdata as sd
import xarray as xr
from numpy.typing import DTypeLike
from spatialdata.models import Image2DModel
from spatialdata.transformations import get_transformation

from squidpy._utils import _get_scale_factors
from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._conversion import _check_channel_dim, cast_to_image_dtype
from squidpy.experimental.im._stain._decomposition import (
    MacenkoParams,
    VahadaneParams,
    _resolve_macenko_params,
    _resolve_vahadane_params,
    apply_decomposition,
    decompose_to_concentrations,
    fit_decomposition,
)
from squidpy.experimental.im._stain._reference import StainMethod, StainReference
from squidpy.experimental.im._stain._reinhard import (
    ReinhardParams,
    _resolve_reinhard_params,
    apply_reinhard,
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

_VALID_METHODS = ("reinhard", "macenko", "vahadane")
_DECOMPOSITION_METHODS = ("macenko", "vahadane")
_CONCENTRATION_CHANNELS = ["hematoxylin", "eosin", "residual"]

# Public union accepted by the method_params argument of the dispatchers.
MethodParams = ReinhardParams | MacenkoParams | VahadaneParams | Mapping[str, Any] | None


def _resolve_image(
    sdata: sd.SpatialData,
    image_key: str,
    scale: str,
    *,
    prefer: Literal["coarsest", "finest"],
) -> xr.DataArray:
    if image_key not in sdata.images:
        raise ValueError(f"image_key {image_key!r} not found, valid keys: {list(sdata.images.keys())}")
    node = sdata.images[image_key]
    da = get_element_data(node, scale, "image", image_key, prefer=prefer)
    _check_channel_dim(da)
    return da


def _resolve_mask_key_and_scale(
    sdata: sd.SpatialData, image_key: str, target_da: xr.DataArray, tissue_mask_key: str | None
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
    sdata: sd.SpatialData, image_key: str, fit_da: xr.DataArray, tissue_mask_key: str | None
) -> np.ndarray:
    """Return a materialised ``(y, x)`` boolean tissue mask aligned to ``fit_da``.

    For the (coarse) fit: nearest-resizes to ``fit_da``'s ``(y, x)`` when the
    closest label scale differs. The fits run on a coarse level, so the mask
    stays small.
    """
    mask_key, label_scale, target_hw = _resolve_mask_key_and_scale(sdata, image_key, fit_da, tissue_mask_key)
    mask = get_mask_materialized(sdata, mask_key, label_scale) > 0
    if mask.shape != target_hw:
        from skimage.transform import resize

        mask = resize(mask, target_hw, order=0, preserve_range=True) > 0.5
    return mask


def _resolve_output_tissue_mask(
    sdata: sd.SpatialData, image_key: str, target_da: xr.DataArray, tissue_mask_key: str | None
) -> xr.DataArray:
    """Return a lazy ``(y, x)`` boolean tissue mask aligned to ``target_da``.

    Like :func:`_resolve_tissue_bool_mask` but kept lazy and at the (full-res)
    output resolution, for compositing the original background back into the
    normalized image without materialising the full frame. The label pyramid
    shares the image's scale factors, so the matching level usually lines up
    exactly; only a residual size mismatch forces a (small) eager resize.
    """
    mask_key, label_scale, target_hw = _resolve_mask_key_and_scale(sdata, image_key, target_da, tissue_mask_key)
    coords = {d: target_da.coords[d] for d in ("y", "x") if d in target_da.coords}
    mask = get_element_data(sdata.labels[mask_key], label_scale, "label", mask_key).squeeze() > 0
    if (int(mask.sizes["y"]), int(mask.sizes["x"])) == target_hw:
        return mask.assign_coords(coords)
    from skimage.transform import resize

    resized = resize(np.asarray(mask.data) > 0, target_hw, order=0, preserve_range=True) > 0.5
    return xr.DataArray(resized, dims=("y", "x"), coords=coords)


def _resolve_method_params(method: str, method_params: MethodParams) -> Any:
    """Pick the right Params dataclass for ``method`` and resolve a mapping/instance/None."""
    if method == "reinhard":
        return _resolve_reinhard_params(method_params)
    if method == "macenko":
        return _resolve_macenko_params(method_params)
    if method == "vahadane":
        return _resolve_vahadane_params(method_params)
    raise ValueError(f"Unknown method {method!r}; expected one of {list(_VALID_METHODS)}.")


def _write_image(
    sdata: sd.SpatialData,
    source_node: Any,
    image_key_added: str,
    data_array: xr.DataArray,
    *,
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
    image_key: str,
    *,
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
        Scale level to sample on. ``"auto"`` (default) uses the coarsest level.
        The sampled level is materialised to take the median, so keep this
        coarse - do not pass a fine level on a whole-slide image.

    Returns
    -------
    Shape-``(3,)`` white point; pass it as ``white_point`` to
    :func:`fit_stain_reference` / :func:`decompose_stains`.
    """
    da = _resolve_image(sdata, image_key, scale, prefer="coarsest")
    validate_rgb_range(da)
    tissue_mask = _resolve_tissue_bool_mask(sdata, image_key, da, tissue_mask_key)
    return white_point_from_background(da, ~tissue_mask)


def fit_stain_reference(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    method: StainMethod = "macenko",
    scale: str | Literal["auto"] = "auto",
    method_params: MethodParams = None,
    white_point: np.ndarray | None = None,
    tissue_mask_key: str | None = None,
    max_angle_deg: float = 45.0,
    canonical_reference: Mapping[str, np.ndarray] | None = None,
) -> StainReference:
    """Fit a stain reference from an image in a :class:`~spatialdata.SpatialData` object.

    Parameters
    ----------
    sdata
        SpatialData object containing the image.
    image_key
        Key of the RGB image in ``sdata.images`` to fit on.
    method
        Fitting method: ``"macenko"`` (default) or ``"vahadane"`` (physical
        stain-matrix decomposition, usable by both :func:`normalize_stains` and
        :func:`decompose_stains`), or ``"reinhard"`` (faster statistical colour
        transfer, no stain separation). Macenko is the default because its one
        documented weakness - artifact pixels contaminating the fit - is removed
        by the mandatory tissue mask.
    scale
        Scale level to fit on. ``"auto"`` (default) uses the coarsest level,
        which is cheap and sufficient for colour statistics.
    method_params
        A :class:`ReinhardParams`/:class:`MacenkoParams`/:class:`VahadaneParams`
        instance, a mapping of its fields, or ``None`` for defaults. Must match
        ``method``.
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
    The fitted :class:`StainReference`. Nothing is written to ``sdata``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {list(_VALID_METHODS)}.")
    da = _resolve_image(sdata, image_key, scale, prefer="coarsest")
    validate_rgb_range(da)
    params = _resolve_method_params(method, method_params)
    tissue_mask = _resolve_tissue_bool_mask(sdata, image_key, da, tissue_mask_key)
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


def normalize_stains(
    sdata: sd.SpatialData,
    image_key: str,
    reference: StainReference,
    *,
    scale: str | Literal["auto"] = "auto",
    method_params: MethodParams = None,
    image_key_added: str | None = None,
    inplace: bool = True,
    output_dtype: DTypeLike | None = None,
    tissue_mask_key: str | None = None,
    preserve_background: bool = True,
) -> xr.DataArray | None:
    """Normalize an image to a fitted stain reference.

    Parameters
    ----------
    sdata
        SpatialData object containing the source image.
    image_key
        Key of the RGB image in ``sdata.images`` to normalize.
    reference
        A :class:`StainReference` fitted with :func:`fit_stain_reference`.
        Dispatch is on ``reference.method``.
    scale
        Scale level to normalize. ``"auto"`` (default) uses the finest level
        so the result is not downsampled; source statistics are reduced
        lazily so memory stays bounded.
    method_params
        Params matching ``reference.method`` (instance, mapping, or ``None``).
    image_key_added
        Key for the written image when ``inplace=True``. If ``None`` (default),
        ``f"{image_key}_normalized"`` is used. Ignored when ``inplace=False``.
    inplace
        If ``True`` (default), write the normalized image to
        ``sdata.images[image_key_added]`` (rebuilding the pyramid for multiscale
        sources, preserving transforms) and return ``None``; raises if the key
        already exists. If ``False``, leave ``sdata`` untouched and return the
        lazy normalized :class:`~xarray.DataArray`.
    output_dtype
        Dtype of the result. If ``None`` (default), the source image's dtype is
        used. The reconstruction is clipped to that dtype's valid range and
        rounded (for integer dtypes) at the write boundary.
    tissue_mask_key
        Key of a tissue-label element in ``sdata.labels`` restricting the
        *source* statistics to tissue pixels. As for
        :func:`fit_stain_reference`, a tissue mask is required (defaults to
        ``f"{image_key}_tissue"``; raises if missing).
    preserve_background
        If ``True`` (default), non-tissue (background) pixels are passed through
        unchanged from the source image, so the normalization recolours only
        tissue. The colour map is a global linear transform that would otherwise
        tint background/white pixels. Set ``False`` for full-frame normalization.

    Returns
    -------
    ``None`` if ``inplace=True`` (the image is written), otherwise the lazy
    normalized :class:`xarray.DataArray`.
    """
    da = _resolve_image(sdata, image_key, scale, prefer="finest")
    target_key = image_key_added if image_key_added is not None else f"{image_key}_normalized"
    if inplace and target_key in sdata.images:
        raise ValueError(f"image_key_added={target_key!r} already exists in sdata.images.")
    params = _resolve_method_params(reference.method, method_params)
    # Source statistics (Reinhard mu/sigma or the decomposition source matrix)
    # are reduced on a coarse level with a tissue mask; the lazy transform is
    # then applied to the full-resolution `da`.
    fit_rgb = _resolve_image(sdata, image_key, scale, prefer="coarsest")
    validate_rgb_range(fit_rgb)  # reject mis-typed source (e.g. 0-255 float) before the dtype-clipped reconstruction
    tissue_mask = _resolve_tissue_bool_mask(sdata, image_key, fit_rgb, tissue_mask_key)
    out_dtype = da.dtype if output_dtype is None else np.dtype(output_dtype)  # clip range + final cast
    if reference.method == "reinhard":
        normalized = apply_reinhard(
            da, reference, params, fit_rgb=fit_rgb, tissue_mask=tissue_mask, out_dtype=out_dtype
        )
    else:
        normalized = apply_decomposition(
            da, reference, params, fit_rgb=fit_rgb, tissue_mask=tissue_mask, out_dtype=out_dtype
        )

    if preserve_background:
        # Keep non-tissue pixels byte-identical to the source: the global colour
        # map would otherwise recolour background/white pixels (HistomicsTK's
        # `mask_out`). Stays lazy - the mask aligns to `da` without materialising.
        keep = _resolve_output_tissue_mask(sdata, image_key, da, tissue_mask_key)
        normalized = normalized.where(keep, da)

    # Deferred cast at the write boundary: the reconstruction was kept in float
    # (clipped to `out_dtype`'s range); round + cast here so the stored image is
    # the requested dtype and integer background stays byte-identical.
    normalized = cast_to_image_dtype(normalized, out_dtype)

    if not inplace:
        return normalized
    _write_image(sdata, sdata.images[image_key], target_key, normalized)
    return None


def decompose_stains(
    sdata: sd.SpatialData,
    image_key: str,
    reference_or_method: StainReference | Literal["macenko", "vahadane"],
    *,
    scale: str | Literal["auto"] = "auto",
    method_params: MethodParams = None,
    white_point: np.ndarray | None = None,
    image_key_added: str | None = None,
    inplace: bool = True,
    output_dtype: DTypeLike = np.float16,
    tissue_mask_key: str | None = None,
    include_residual: bool = True,
) -> dict[str, xr.DataArray] | None:
    """Decompose an image into separate per-stain concentration maps.

    Parameters
    ----------
    sdata, image_key
        The SpatialData object and the RGB image key to decompose.
    reference_or_method
        Either a decomposition :class:`StainReference` (its stain matrix and
        white point are used) or a method name (``"macenko"``/``"vahadane"``)
        to fit on this image first. The reference is the provenance record of
        how the maps were produced (method, stain matrix, white point).
    scale, method_params, white_point, tissue_mask_key
        As for :func:`fit_stain_reference` (only used when a method name is
        given; a reference is projected as-is and needs no tissue mask).
    image_key_added
        Key *prefix* for the written images when ``inplace=True``. If ``None``
        (default), ``image_key`` is used, so each stain is written as its own
        single-channel image ``sdata.images[f"{image_key}_{stain}"]`` (e.g.
        ``f"{image_key}_hematoxylin"``). Ignored when ``inplace=False``.
    inplace
        If ``True`` (default), write each stain as a separate single-channel
        image under the ``image_key_added`` prefix and return ``None``; the
        write is atomic (all target keys are validated free before any is
        written). If ``False``, leave ``sdata`` untouched and return the maps
        as a dict.
    output_dtype
        Dtype of the concentration maps. Defaults to ``float16`` (half the
        storage; ~3 significant figures, adequate for concentrations); pass
        ``float32`` for strict quantification.
    include_residual
        If ``True`` (default), also produce the ``"residual"`` map. The residual
        is the absorbance along the complement direction - a diagnostic of
        decomposition quality (extra chromogen, artifacts, or a poor fit), not a
        biological stain. Set ``False`` to keep only ``hematoxylin``/``eosin``.

    Returns
    -------
    ``None`` if ``inplace=True`` (the maps are written as separate images),
    otherwise a ``dict`` mapping each stain name to its ``(y, x)`` concentration
    :class:`~xarray.DataArray` (``"hematoxylin"``, ``"eosin"``, and
    ``"residual"`` unless dropped).
    """
    da = _resolve_image(sdata, image_key, scale, prefer="finest")
    if isinstance(reference_or_method, StainReference):
        reference = reference_or_method
        if reference.method not in _DECOMPOSITION_METHODS or reference.stain_matrix is None:
            raise ValueError("decompose_stains requires a macenko/vahadane reference with a stain matrix.")
        stain_matrix, bg = reference.stain_matrix, reference.white_point
    else:
        if reference_or_method not in _DECOMPOSITION_METHODS:
            raise ValueError(f"method must be one of {list(_DECOMPOSITION_METHODS)}; got {reference_or_method!r}.")
        reference = fit_stain_reference(
            sdata,
            image_key,
            method=reference_or_method,
            scale=scale,
            method_params=method_params,
            white_point=white_point,
            tissue_mask_key=tissue_mask_key,
        )
        stain_matrix, bg = reference.stain_matrix, reference.white_point

    names = ["hematoxylin", "eosin"] + (["residual"] if include_residual else [])
    prefix = image_key_added if image_key_added is not None else image_key
    target_keys = [f"{prefix}_{name}" for name in names]
    if inplace:  # validate all keys free up front, so a partial write can't leave a half-decomposed sdata
        clashes = [k for k in target_keys if k in sdata.images]
        if clashes:
            raise ValueError(f"decompose_stains would overwrite existing image(s): {clashes}.")

    concentrations = decompose_to_concentrations(da, stain_matrix, bg).assign_coords(c=_CONCENTRATION_CHANNELS)
    concentrations = concentrations.astype(np.dtype(output_dtype))

    if not inplace:
        return {name: concentrations.sel(c=name) for name in names}

    source = sdata.images[image_key]
    for name, key in zip(names, target_keys, strict=True):
        # keep the c dim (length 1) so Image2DModel.parse accepts it
        _write_image(sdata, source, key, concentrations.sel(c=[name]), c_coords=[name])
    return None
