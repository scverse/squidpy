"""Numeric helpers for STalign point-cloud registration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from squidpy.experimental.im._rasterize_points import rasterize

from .._common import validate_xy
from ._core import jax_dtype, reverse_axes

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any

__all__ = [
    "affine_from_points",
    "affine_xy_to_rc",
    "as_chw",
    "centred_axes",
    "explicit_axes",
    "rasterize",
    "rasterize_cloud",
    "validate_points",
]


def rasterize_cloud(
    points_rc: JaxArray, *, dx: float, blur: float | Sequence[float], expand: float
) -> tuple[tuple[JaxArray, JaxArray], JaxArray]:
    """Rasterize a row-col cloud into a ``((grid_y, grid_x), image)`` density."""
    grid_x, grid_y, image = rasterize(points_rc[:, 1], points_rc[:, 0], dx=dx, blur=blur, expand=expand)
    return (grid_y, grid_x), image


def validate_points(points: Any, *, name: str) -> JaxArray:
    """Coerce ``points`` to a finite ``(n, 2)`` JAX array."""
    return jnp.asarray(validate_xy(points, name=name), dtype=jax_dtype())


def as_chw(image: Any, *, name: str, ndim: int = 2) -> JaxArray:
    """Coerce an image to channels-first, promoting an unchannelled array.

    ``ndim`` is the number of *spatial* axes: 2 for a ``(y, x)`` section, 3 for a
    ``(z, y, x)`` volume. Not :func:`jnp.atleast_3d`, which appends the new axis --
    turning a ``(y, x)`` image into ``(y, x, 1)``, y channels of x by 1, instead of a
    single ``(1, y, x)`` channel.
    """
    arr = jnp.asarray(image, dtype=jax_dtype())
    if arr.ndim == ndim:
        return arr[None]
    if arr.ndim != ndim + 1:
        spatial = ", ".join("zyx"[-ndim:])
        raise ValueError(f"Expected `{name}` to be a `({spatial})` or `(c, {spatial})` image, found shape {arr.shape}.")
    return arr


def centred_axes(shape: tuple[int, ...], scale: Sequence[float]) -> tuple[JaxArray, ...]:
    """Physical axes for ``shape``, centred on the origin and spaced by ``scale``.

    Centring is what lets the affine initialise near identity: with axes running from 0
    the translation would have to carry the whole half-extent before the fit even starts.
    """
    dtype = jax_dtype()
    if len(scale) != len(shape):
        raise ValueError(f"Expected {len(shape)} scale entries for a {len(shape)}-axis image, found {len(scale)}.")
    return tuple(
        (jnp.arange(size, dtype=dtype) - (size - 1) / 2.0) * step for size, step in zip(shape, scale, strict=True)
    )


def explicit_axes(value: Sequence[Any], shape: tuple[int, ...], name: str) -> tuple[JaxArray, ...]:
    """Validate caller-supplied physical axes against the image they describe."""
    dtype = jax_dtype()
    resolved = tuple(jnp.asarray(axis, dtype=dtype) for axis in value)
    if len(resolved) != len(shape):
        raise ValueError(f"Expected `{name}` to hold {len(shape)} coordinate vectors, found {len(resolved)}.")
    lengths = tuple(axis.shape[0] for axis in resolved)
    if any(axis.ndim != 1 for axis in resolved) or lengths != shape:
        raise ValueError(f"Expected `{name}` lengths {shape}, found {lengths}.")
    if any(length < 2 for length in lengths):
        raise ValueError(f"Expected each `{name}` axis to contain at least two coordinates.")
    return resolved


def affine_xy_to_rc(matrix: Any, *, name: str = "initial_affine", ndim: int = 2) -> tuple[JaxArray, JaxArray]:
    """Split a homogeneous ``(x, y[, z])`` affine into array-order ``(linear, translation)``.

    The solver works in array order -- ``(y, x)`` at rank 2, ``(z, y, x)`` at rank 3 --
    so conjugating by the axis reversal converts the caller's convention without them
    having to think in the solver's.
    """
    dtype = jax_dtype()
    affine_xy = jnp.asarray(matrix, dtype=dtype)
    if affine_xy.shape != (ndim + 1, ndim + 1):
        raise ValueError(f"Expected `{name}` to have shape ({ndim + 1}, {ndim + 1}), found {affine_xy.shape}.")
    swap = reverse_axes(ndim)
    affine_rc = swap @ affine_xy @ swap
    return affine_rc[:ndim, :ndim], affine_rc[:ndim, ndim]


def affine_from_points(
    points_source: JaxArray,
    points_target: JaxArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an affine initialization from corresponding landmarks."""
    source = np.asarray(points_source, dtype=float)
    target = np.asarray(points_target, dtype=float)
    if source.shape != target.shape:
        raise ValueError(
            f"Expected `points_source` and `points_target` to have the same shape, found "
            f"`{source.shape}` and `{target.shape}`."
        )

    if source.shape[0] < 3:
        linear = np.eye(2, dtype=float)
        translation = np.mean(target, axis=0) - np.mean(source, axis=0)
        return linear, translation

    from skimage.transform import estimate_transform

    model_obj = estimate_transform("affine", src=source, dst=target)
    affine = np.asarray(model_obj.params)
    return affine[:2, :2], affine[:2, -1]
