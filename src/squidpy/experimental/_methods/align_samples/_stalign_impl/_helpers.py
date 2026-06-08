"""Numeric helpers for STalign point-cloud registration.

Lifted from scverse/squidpy#1150 (Selman Özleyen). Container readers
(``extract_points`` / ``extract_landmarks``) were dropped on the move into
:mod:`squidpy.experimental._methods`: reading coordinates out of an ``AnnData`` /
``SpatialData`` is the caller's concern, so everything here operates purely on
in-memory arrays. Rasterization runs on JAX so the point cloud never leaves the
device between preprocessing and the LDDMM solve.
"""

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
    "validate_points",
]


def validate_points(points: Any, *, name: str) -> JaxArray:
    """Coerce ``points`` to a finite ``(n, 2)`` JAX array."""
    arr = jnp.asarray(points, dtype=jax_dtype())
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected `{name}` to have shape `(n, 2)`, found `{arr.shape}`.")
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"Expected `{name}` to contain only finite values.")
    return arr


def rasterize(
    x: JaxArray,
    y: JaxArray,
    *,
    g: JaxArray | None = None,
    dx: float = 30.0,
    blur: float | list[float] = 1.0,
    expand: float = 1.1,
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Rasterize a point cloud into a multi-scale Gaussian density image.

    Each point splats a normalized Gaussian over a fixed ``(2r + 1)`` patch and
    the patches are scatter-added onto the grid -- the JAX analogue of the
    original per-point NumPy loop. Out-of-grid patch pixels are masked out.
    """
    dtype = jax_dtype()
    x = jnp.asarray(x, dtype=dtype).reshape(-1)
    y = jnp.asarray(y, dtype=dtype).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Expected `x` and `y` to be 1D arrays with the same length.")
    if x.size == 0:
        raise ValueError("Expected at least one point to rasterize.")
    if dx <= 0:
        raise ValueError("Expected `dx` to be positive.")
    if expand <= 0:
        raise ValueError("Expected `expand` to be positive.")

    blur_values = jnp.atleast_1d(jnp.asarray(blur, dtype=dtype))
    if blur_values.ndim != 1 or bool(jnp.any(blur_values <= 0)):
        raise ValueError("Expected `blur` to be a positive scalar or a 1D sequence of positive values.")

    if g is None:
        weights = jnp.ones_like(x)
    else:
        weights = jnp.asarray(g, dtype=dtype).reshape(-1)
        if weights.shape != x.shape:
            raise ValueError("Expected `g` to have the same shape as `x` and `y`.")
        if not bool(jnp.allclose(weights, 1.0)):
            weights = _normalize(weights)

    min_x = float(jnp.min(x))
    max_x = float(jnp.max(x))
    min_y = float(jnp.min(y))
    max_y = float(jnp.max(y))

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half_x = (max_x - min_x) * expand / 2.0
    half_y = (max_y - min_y) * expand / 2.0

    grid_x = jnp.arange(center_x - half_x, center_x + half_x + dx, dx, dtype=dtype)
    grid_y = jnp.arange(center_y - half_y, center_y + half_y + dx, dx, dtype=dtype)
    n_cols = int(grid_x.shape[0])
    n_rows = int(grid_y.shape[0])
    if n_cols < 2 or n_rows < 2:
        raise ValueError("Rasterized grid is too small. Increase the point spread or lower `dx`.")

    radius = int(np.ceil(float(jnp.max(blur_values)) * 4.0))
    offsets = jnp.arange(-radius, radius + 1)

    # Nearest grid cell for every point, then its fixed (2r+1) patch indices.
    col0 = jnp.round((x - grid_x[0]) / dx).astype(jnp.int32)  # (N,)
    row0 = jnp.round((y - grid_y[0]) / dx).astype(jnp.int32)  # (N,)
    patch_cols = col0[:, None] + offsets[None, :]  # (N, K)
    patch_rows = row0[:, None] + offsets[None, :]  # (N, K)

    patch_x = grid_x[0] + patch_cols.astype(dtype) * dx  # (N, K)
    patch_y = grid_y[0] + patch_rows.astype(dtype) * dx  # (N, K)

    # (N, K, K) squared distance from each patch pixel to its source point.
    dist2 = (patch_x[:, None, :] - x[:, None, None]) ** 2 + (patch_y[:, :, None] - y[:, None, None]) ** 2

    denom = 2.0 * (dx * blur_values * 2.0) ** 2  # (B,)
    kernels = jnp.exp(-dist2[None] / denom[:, None, None, None])  # (B, N, K, K)
    kernels_sum = kernels.sum(axis=(2, 3), keepdims=True)
    kernels = kernels / jnp.where(kernels_sum == 0.0, 1.0, kernels_sum)
    kernels = kernels * weights[None, :, None, None]

    valid = (
        (patch_rows >= 0)[:, :, None]
        & (patch_rows < n_rows)[:, :, None]
        & (patch_cols >= 0)[:, None, :]
        & (patch_cols < n_cols)[:, None, :]
    )  # (N, K, K)
    kernels = kernels * valid[None]

    rows = jnp.broadcast_to(jnp.clip(patch_rows, 0, n_rows - 1)[:, :, None], dist2.shape)  # (N, K, K)
    cols = jnp.broadcast_to(jnp.clip(patch_cols, 0, n_cols - 1)[:, None, :], dist2.shape)  # (N, K, K)

    out = jnp.zeros((blur_values.shape[0], n_rows, n_cols), dtype=dtype)
    out = out.at[:, rows, cols].add(kernels)
    return grid_x, grid_y, out


def _normalize(values: JaxArray) -> JaxArray:
    vmin = jnp.min(values)
    vmax = jnp.max(values)
    return jnp.where(jnp.isclose(vmin, vmax), jnp.ones_like(values), (values - vmin) / (vmax - vmin))


def affine_from_points(
    points_source: JaxArray,
    points_target: JaxArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an affine initialization from corresponding landmarks.

    Delegates to skimage's least-squares estimator, so this is the one place
    that drops to NumPy on the CPU; the small ``(2, 2)`` / ``(2,)`` result is
    lifted back to JAX by the caller.
    """
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
