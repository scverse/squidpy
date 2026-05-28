"""Public, ``sdata``-aware entry points for stain normalization.

The single integration boundary for the stain module: the only file that
reads ``sdata.images[...]``, writes back via :class:`Image2DModel`, and is
re-exported publicly. Everything it calls is a pure DataArray-layer
primitive (:mod:`._reinhard`, :mod:`._mask`, :mod:`._conversion`).

Both entry points dispatch on the fitting ``method``. Only ``"reinhard"`` is
implemented here; ``"macenko"``/``"vahadane"`` raise ``NotImplementedError``
and are filled in without changing these signatures.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel
from spatialdata.transformations import get_transformation

from squidpy._utils import _get_scale_factors
from squidpy.experimental.im._stain._conversion import _check_channel_dim
from squidpy.experimental.im._stain._reference import StainMethod, StainReference
from squidpy.experimental.im._stain._reinhard import (
    ReinhardParams,
    _resolve_reinhard_params,
    apply_reinhard,
    fit_reinhard,
)
from squidpy.experimental.im._utils import get_element_data

_DECOMPOSITION_NOT_IMPLEMENTED = "macenko/vahadane decomposition is not yet implemented"


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


def fit_stain_reference(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    method: StainMethod = "reinhard",
    scale: str | Literal["auto"] = "auto",
    method_params: ReinhardParams | Mapping[str, Any] | None = None,
) -> StainReference:
    """Fit a stain reference from an image in a :class:`~spatialdata.SpatialData` object.

    Parameters
    ----------
    sdata
        SpatialData object containing the image.
    image_key
        Key of the RGB image in ``sdata.images`` to fit on.
    method
        Fitting method. Only ``"reinhard"`` is implemented; ``"macenko"`` and
        ``"vahadane"`` raise :class:`NotImplementedError`.
    scale
        Scale level to fit on. ``"auto"`` (default) uses the coarsest level,
        which is cheap and sufficient for colour statistics.
    method_params
        :class:`ReinhardParams` instance, a mapping of its fields, or ``None``
        for defaults.

    Returns
    -------
    The fitted :class:`StainReference`. Nothing is written to ``sdata``.
    """
    da = _resolve_image(sdata, image_key, scale, prefer="coarsest")
    if method == "reinhard":
        return fit_reinhard(da, _resolve_reinhard_params(method_params))
    if method in {"macenko", "vahadane"}:
        raise NotImplementedError(_DECOMPOSITION_NOT_IMPLEMENTED)
    raise ValueError(f"Unknown method {method!r}; expected one of ['macenko', 'reinhard', 'vahadane'].")


def apply_stain_normalization(
    sdata: sd.SpatialData,
    image_key: str,
    reference: StainReference,
    *,
    scale: str | Literal["auto"] = "auto",
    method_params: ReinhardParams | Mapping[str, Any] | None = None,
    image_key_added: str | None = None,
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
        :class:`ReinhardParams` instance, a mapping of its fields, or ``None``
        for defaults.
    image_key_added
        If ``None`` (default), return the lazy normalized DataArray and leave
        ``sdata`` untouched. If given, write the result to
        ``sdata.images[image_key_added]`` (rebuilding the pyramid for
        multiscale sources, preserving transforms) and return ``None``.
        Raises if the key already exists.

    Returns
    -------
    The lazy normalized :class:`xarray.DataArray` if ``image_key_added`` is
    ``None``, otherwise ``None``.
    """
    da = _resolve_image(sdata, image_key, scale, prefer="finest")
    if reference.method == "reinhard":
        normalized = apply_reinhard(da, reference, _resolve_reinhard_params(method_params))
    elif reference.method in {"macenko", "vahadane"}:
        raise NotImplementedError(_DECOMPOSITION_NOT_IMPLEMENTED)
    else:  # pragma: no cover - StainReference validates method on construction
        raise ValueError(f"Unknown reference method {reference.method!r}.")

    if image_key_added is None:
        return normalized
    if image_key_added in sdata.images:
        raise ValueError(f"image_key_added={image_key_added!r} already exists in sdata.images.")

    node = sdata.images[image_key]
    # Reconstruct the element explicitly from the underlying array: parse a
    # DataArray would carry over the source's transform attr and collide with
    # the transformations we pass, so we hand it the bare array plus the
    # dims/channel coords/transforms we want to preserve (the same idiom as
    # detect_tissue). `_get_scale_factors` returns [] for a single-scale
    # source; parse needs None there (an empty list builds a degenerate
    # single-level pyramid).
    c_coords = normalized.coords["c"].values.tolist() if "c" in normalized.coords else None
    sdata.images[image_key_added] = Image2DModel.parse(
        normalized.data,
        dims=normalized.dims,
        c_coords=c_coords,
        transformations=get_transformation(node, get_all=True),
        scale_factors=_get_scale_factors(node) or None,
    )
    return None
