"""Helpers for experimental STalign point-cloud registration."""

from __future__ import annotations

from typing import Literal

import numpy as np
from anndata import AnnData

PointOrder = Literal["row_col", "xy"]

__all__ = [
    "PointOrder",
    "affine_from_points",
    "extract_landmarks",
    "extract_points",
    "rasterize",
]


def _validate_points(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected `{name}` to have shape `(n, 2)`, found `{arr.shape}`.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Expected `{name}` to contain only finite values.")
    return arr


def extract_points(adata: AnnData, key: str = "spatial") -> np.ndarray:
    """Return a validated coordinate array from ``adata.obsm``."""
    if key not in adata.obsm:
        raise KeyError(f"Key `{key}` not found in `adata.obsm`.")

    return _validate_points(np.asarray(adata.obsm[key]), name=f"adata.obsm[{key!r}]")


def extract_landmarks(adata: AnnData, key: str) -> np.ndarray:
    """Return a validated landmark array from ``adata.obsm`` or ``adata.uns``."""
    if key in adata.obsm:
        arr = np.asarray(adata.obsm[key], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected `adata.obsm[{key!r}]` to have shape `(n, 2)`, found `{arr.shape}`.")
        mask = np.all(np.isfinite(arr), axis=1)
        landmarks = arr[mask]
        if landmarks.size == 0:
            raise ValueError(f"No finite landmark rows were found in `adata.obsm[{key!r}]`.")
        return landmarks

    if key in adata.uns:
        return _validate_points(np.asarray(adata.uns[key]), name=f"adata.uns[{key!r}]")

    raise KeyError(f"Key `{key}` not found in `adata.obsm` or `adata.uns`.")


# TODO: are these duplicated? I would imagine its
# better to keep image transform functions under some place

def to_row_col(points: np.ndarray, *, point_order: PointOrder) -> np.ndarray:
    """Convert coordinates to row-column order."""
    arr = _validate_points(points, name="points")
    if point_order == "row_col":
        return arr
    if point_order == "xy":
        return arr[:, [1, 0]]
    raise ValueError(f"Unknown `point_order`: `{point_order}`.")


def from_row_col(points: np.ndarray, *, point_order: PointOrder) -> np.ndarray:
    """Convert row-column coordinates to the requested order."""
    arr = _validate_points(points, name="points")
    if point_order == "row_col":
        return arr
    if point_order == "xy":
        return arr[:, [1, 0]]
    raise ValueError(f"Unknown `point_order`: `{point_order}`.")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = np.min(values)
    vmax = np.max(values)
    if np.isclose(vmin, vmax):
        return np.ones_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def rasterize(
    x: np.ndarray,
    y: np.ndarray,
    *,
    g: np.ndarray | None = None,
    dx: float = 30.0,
    blur: float | list[float] = 1.0,
    expand: float = 1.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a point cloud into a multi-scale density image."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
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

    if g is None:
        weights = np.ones_like(x, dtype=float)
    else:
        weights = np.asarray(g, dtype=float)
        if weights.shape != x.shape:
            raise ValueError("Expected `g` to have the same shape as `x` and `y`.")
        if not np.allclose(weights, 1.0):
            weights = _normalize(weights)

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

    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    out = np.zeros((len(blur_values), grid_y.size, grid_x.size), dtype=float)
    radius = int(np.ceil(float(np.max(blur_values)) * 4.0))

    for x_i, y_i, w_i in zip(x, y, weights, strict=False):
        col = int(np.rint((x_i - grid_x[0]) / dx))
        row = int(np.rint((y_i - grid_y[0]) / dx))

        row0 = max(row - radius, 0)
        row1 = min(row + radius, out.shape[1] - 1)
        col0 = max(col - radius, 0)
        col1 = min(col + radius, out.shape[2] - 1)

        patch_x = mesh_x[row0 : row1 + 1, col0 : col1 + 1]
        patch_y = mesh_y[row0 : row1 + 1, col0 : col1 + 1]
        denom = 2.0 * (dx * blur_values * 2.0) ** 2

        kernels = np.exp(-((patch_x[..., None] - x_i) ** 2 + (patch_y[..., None] - y_i) ** 2) / denom)
        kernels_sum = kernels.sum(axis=(0, 1), keepdims=True)
        kernels /= np.where(kernels_sum == 0.0, 1.0, kernels_sum)
        out[:, row0 : row1 + 1, col0 : col1 + 1] += np.moveaxis(kernels * w_i, -1, 0)

    return grid_x, grid_y, out


def affine_from_points(
    points_source: np.ndarray,
    points_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an affine initialization from corresponding landmarks."""
    source = _validate_points(points_source, name="points_source")
    target = _validate_points(points_target, name="points_target")
    if source.shape != target.shape:
        raise ValueError(
            f"Expected `points_source` and `points_target` to have the same shape, found "
            f"`{source.shape}` and `{target.shape}`."
        )

    if source.shape[0] < 3:
        linear = np.eye(2, dtype=float)
        translation = np.mean(target, axis=0) - np.mean(source, axis=0)
        return linear, translation

    source_h = np.concatenate((source, np.ones((source.shape[0], 1), dtype=float)), axis=1)
    target_h = np.concatenate((target, np.ones((target.shape[0], 1), dtype=float)), axis=1)
    affine = np.linalg.lstsq(source_h, target_h, rcond=None)[0].T
    return affine[:2, :2], affine[:2, -1]
