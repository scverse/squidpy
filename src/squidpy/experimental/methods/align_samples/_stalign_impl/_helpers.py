"""Numeric helpers for STalign point-cloud registration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from squidpy.experimental.methods._common import validate_xy

from ._core import _axis, jax_dtype

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any

__all__ = [
    "affine_from_points",
    "affine_xy_to_rc",
    "as_chw",
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


def as_chw(image: Any, *, name: str) -> JaxArray:
    """Coerce an image to channels-first ``(c, y, x)``, promoting a bare ``(y, x)``.

    Not :func:`jnp.atleast_3d`: it appends the new axis, turning a ``(y, x)`` image into
    ``(y, x, 1)`` -- y channels of x by 1 -- instead of a single ``(1, y, x)`` channel.
    """
    arr = jnp.asarray(image, dtype=jax_dtype())
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Expected `{name}` to be a `(y, x)` or `(c, y, x)` image, found shape {arr.shape}.")
    return arr


def affine_xy_to_rc(matrix: Any, *, name: str = "initial_affine") -> tuple[JaxArray, JaxArray]:
    """Split a homogeneous ``(3, 3)`` ``(x, y)`` affine into row-col ``(linear, translation)``.

    The solver works in row-col; conjugating by the axis swap converts the caller's
    ``(x, y)`` convention without them having to think in the solver's.
    """
    dtype = jax_dtype()
    affine_xy = jnp.asarray(matrix, dtype=dtype)
    if affine_xy.shape != (3, 3):
        raise ValueError(f"Expected `{name}` to have shape (3, 3), found {affine_xy.shape}.")
    swap = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    affine_rc = swap @ affine_xy @ swap
    return affine_rc[:2, :2], affine_rc[:2, 2]


def rasterize(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dx: float,
    blur: float | Sequence[float],
    expand: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a point cloud into a multi-scale Gaussian density image.

    Each point deposits unit mass bilinearly across its four neighbouring cells of a
    regular ``dx``-spaced grid; every ``blur`` scale is then an isotropic Gaussian blur
    of that histogram, so each point becomes a unit-integral Gaussian. ``blur`` sets the
    kernel width: ``sigma = 2 * blur`` pixels, i.e. ``2 * blur * dx`` in physical units.
    Total mass is preserved exactly, including for points near the border.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Expected `x` and `y` to be 1D arrays with the same length.")
    if x.size == 0:
        raise ValueError("Expected at least one point to rasterize.")
    if dx <= 0:
        raise ValueError("Expected `dx` to be positive.")
    if expand <= 0:
        raise ValueError("Expected `expand` to be positive.")

    blur_values = np.atleast_1d(np.asarray(blur, dtype=float))
    if blur_values.ndim != 1 or np.any(blur_values <= 0):
        raise ValueError("Expected `blur` to be a positive scalar or a 1D sequence of positive values.")

    min_x = float(np.min(x))
    max_x = float(np.max(x))
    min_y = float(np.min(y))
    max_y = float(np.max(y))

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half_x = (max_x - min_x) * expand / 2.0
    half_y = (max_y - min_y) * expand / 2.0

    grid_x = _axis(center_x - half_x, center_x + half_x, dx)
    grid_y = _axis(center_y - half_y, center_y + half_y, dx)
    if grid_x.size < 2 or grid_y.size < 2:
        raise ValueError("Rasterized grid is too small. Increase the point spread or lower `dx`.")

    histogram = _deposit(x, y, grid_x, grid_y, dx)
    out = np.stack([_blur_conserving(histogram, sigma=2.0 * float(b)) for b in blur_values])
    return grid_x, grid_y, out


def _deposit(x: np.ndarray, y: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, dx: float) -> np.ndarray:
    """Spread each point's unit mass bilinearly over its four neighbouring cells.

    Snapping to the nearest cell instead would quantise every position by up to half a
    cell before any blurring happens, which at a typical ``dx`` is comparable to the
    features being registered. ``np.bincount`` keeps this a handful of vectorised passes
    rather than a Python loop over points.
    """
    n_rows, n_cols = grid_y.size, grid_x.size
    col_f = (x - grid_x[0]) / dx
    row_f = (y - grid_y[0]) / dx
    col_0 = np.floor(col_f).astype(np.intp)
    row_0 = np.floor(row_f).astype(np.intp)
    col_w = col_f - col_0
    row_w = row_f - row_0

    flat = np.zeros(n_rows * n_cols, dtype=float)
    for row_offset, row_weight in ((0, 1.0 - row_w), (1, row_w)):
        for col_offset, col_weight in ((0, 1.0 - col_w), (1, col_w)):
            rows, cols = row_0 + row_offset, col_0 + col_offset
            inside = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
            flat += np.bincount(
                rows[inside] * n_cols + cols[inside],
                weights=(row_weight * col_weight)[inside],
                minlength=flat.size,
            )
    return flat.reshape(n_rows, n_cols)


def _blur_conserving(histogram: np.ndarray, *, sigma: float) -> np.ndarray:
    """Gaussian blur that keeps every point's mass on the grid.

    ``mode="constant"`` lets a kernel centred near the border spill off the edge, so
    points there contribute less than one unit and the density is biased low around the
    rim. The mass a point at cell ``c`` retains is ``sum_p K(p - c)``, which by symmetry
    of ``K`` equals ``gaussian_filter(ones)[c]`` -- dividing by that before blurring
    makes the total exactly the number of points, wherever they lie.
    """
    from scipy.ndimage import gaussian_filter

    retained = gaussian_filter(np.ones_like(histogram), sigma=sigma, mode="constant")
    return gaussian_filter(histogram / retained, sigma=sigma, mode="constant")


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
