"""Turn a point cloud into a density image.

The numerics here are deliberately free of JAX: the STalign solver in
:mod:`squidpy.experimental.tl` imports :func:`rasterize` and :func:`axis` from this
module, so the primitive is usable -- and installable -- without the optional JAX extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spatialdata import SpatialData
    from xarray import DataArray

#: ``axis`` and ``rasterize`` are the JAX-free primitives the STalign solver imports, so
#: they are part of this module's surface even though only ``rasterize_points`` is public API.
__all__ = ["axis", "rasterize", "rasterize_points"]


def axis(start: float, stop: float, step: float) -> np.ndarray:
    """``step``-spaced samples covering ``[start, stop)``, with a stable length.

    ``arange(start, stop, step)`` on floats derives its length from the arguments by
    floating-point division, so a ``stop`` that is itself a sum of floats can yield one
    more or one fewer sample than intended. Taking the count first makes the length a
    function of the interval alone.
    """
    count = max(int(np.ceil((stop - start) / step)), 1)
    return start + step * np.arange(count, dtype=float)


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

    Returns
    -------
    ``(grid_x, grid_y, image)``, the image being ``(len(blur), len(grid_y), len(grid_x))``.
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

    grid_x = axis(center_x - half_x, center_x + half_x, dx)
    grid_y = axis(center_y - half_y, center_y + half_y, dx)
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


def rasterize_points(
    sdata: SpatialData,
    points_key: str,
    *,
    dx: float = 30.0,
    blur: float | Sequence[float] = 1.0,
    expand: float = 1.1,
    target_coordinate_system: str = "global",
    key_added: str | None = None,
) -> DataArray:
    """Rasterize a points element into a Gaussian density image.

    Each point deposits unit mass bilinearly over its four neighbouring pixels, and the
    result is blurred once per ``blur`` scale, so every point becomes a unit-integral
    Gaussian and the total intensity is exactly the number of points. Passing several
    ``blur`` values returns them as channels, which is what a coarse-to-fine registration
    wants.

    Parameters
    ----------
    sdata
        The :class:`~spatialdata.SpatialData` holding the points.
    points_key
        Name of the points element to rasterize.
    dx
        Pixel size of the output, in the units of ``target_coordinate_system``.
    blur
        Gaussian width(s), in pixels: ``sigma = 2 * blur``. A sequence produces one
        channel per value.
    expand
        Field-of-view padding factor around the points' bounding box. The default leaves
        a 5% margin on each side, so a deformation has somewhere to move into.
    target_coordinate_system
        Coordinate system to read the point positions in, and the one the returned image
        is registered to.
    key_added
        Image element name to store the result under. ``None`` (default) returns the
        image without touching ``sdata``.

    Returns
    -------
    A channels-first ``(c, y, x)`` :class:`~xarray.DataArray`, registered to
    ``target_coordinate_system`` by the scale-and-translation that maps its pixel grid
    onto the point coordinates.
    """
    import spatialdata
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import Scale, Translation
    from spatialdata.transformations import Sequence as TransformSequence

    if points_key not in sdata.points:
        raise KeyError(f"`points_key={points_key!r}`: no such points element. Available: {sorted(sdata.points)}.")

    element = spatialdata.transform(sdata.points[points_key], to_coordinate_system=target_coordinate_system)
    missing = [axis_name for axis_name in ("x", "y") if axis_name not in element.columns]
    if missing:
        raise ValueError(f"`points_key={points_key!r}` has no {missing} column(s); found {list(element.columns)}.")
    frame = element[["x", "y"]].compute() if hasattr(element, "compute") else element[["x", "y"]]

    grid_x, grid_y, image = rasterize(np.asarray(frame["x"]), np.asarray(frame["y"]), dx=dx, blur=blur, expand=expand)

    # The raster's own pixel grid is index space; this puts it back where the points are.
    # `Scale` first, then `Translation`, since the grid origin is a physical offset.
    transformation = TransformSequence(
        [Scale([dx, dx], axes=("y", "x")), Translation([grid_y[0], grid_x[0]], axes=("y", "x"))]
    )
    parsed = Image2DModel.parse(
        image,
        dims=("c", "y", "x"),
        transformations={target_coordinate_system: transformation},
    )
    if key_added is not None:
        sdata.images[key_added] = parsed
    return parsed
