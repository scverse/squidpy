"""Numeric helpers for STalign point-cloud registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from ._core import jax_dtype

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any

__all__ = [
    "affine_from_points",
    "rasterize",
    "rasterize_cloud",
    "validate_points",
]


def rasterize_cloud(
    points_rc: JaxArray, *, dx: float, blur: float | list[float], expand: float
) -> tuple[tuple[JaxArray, JaxArray], JaxArray]:
    """Rasterize a row-col cloud into a ``((grid_y, grid_x), image)`` density."""
    grid_x, grid_y, image = rasterize(points_rc[:, 1], points_rc[:, 0], dx=dx, blur=blur, expand=expand)
    return (grid_y, grid_x), image


def validate_points(points: Any, *, name: str) -> JaxArray:
    """Coerce ``points`` to a finite ``(n, 2)`` JAX array."""
    arr = jnp.asarray(points, dtype=jax_dtype())
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected `{name}` to have shape `(n, 2)`, found `{arr.shape}`.")
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"Expected `{name}` to contain only finite values.")
    return arr


def rasterize(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dx: float = 30.0,
    blur: float | list[float] = 1.0,
    expand: float = 1.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a point cloud into a multi-scale Gaussian density image.

    Each point deposits unit mass into its nearest cell of a regular
    ``dx``-spaced grid; every ``blur`` scale is then an isotropic Gaussian blur
    of that histogram, so each point becomes a unit-integral Gaussian. ``blur``
    is the kernel width in units of ``2 * dx`` (``sigma = 2 * blur`` cells).
    Mass within ~``blur`` cells of the border leaks off-grid (``mode="constant"``).
    """
    from scipy.ndimage import gaussian_filter

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

    grid_x = np.arange(center_x - half_x, center_x + half_x + dx, dx, dtype=float)
    grid_y = np.arange(center_y - half_y, center_y + half_y + dx, dx, dtype=float)
    if grid_x.size < 2 or grid_y.size < 2:
        raise ValueError("Rasterized grid is too small. Increase the point spread or lower `dx`.")

    # Bin each point onto its nearest grid cell (unit mass each), dropping any
    # whose nearest cell falls outside the padded grid.
    col = np.rint((x - grid_x[0]) / dx).astype(np.intp)
    row = np.rint((y - grid_y[0]) / dx).astype(np.intp)
    inside = (col >= 0) & (col < grid_x.size) & (row >= 0) & (row < grid_y.size)
    histogram = np.zeros((grid_y.size, grid_x.size), dtype=float)
    np.add.at(histogram, (row[inside], col[inside]), 1.0)

    out = np.stack([gaussian_filter(histogram, sigma=2.0 * float(b), mode="constant") for b in blur_values])
    return grid_x, grid_y, out


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
