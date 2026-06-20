"""Result type and rasterization helper for experimental STalign.

The fitted map is returned as a single :class:`StalignResult`; container
write-back lives in the caller, not here. The solver orchestration (rasterize ->
LDDMM) lives in the estimator adapter :func:`...align_samples._stalign.fit_stalign`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any

type BlurScales = float | tuple[float, ...] | list[float]

__all__ = ["StalignResult"]


@dataclass(slots=True)
class StalignResult:
    """A fitted STalign diffeomorphism, ready to transform arbitrary points.

    :meth:`transform` works in ``(x, y)``; ``aligned_points`` is the fitted query
    cloud already mapped into the reference frame.
    """

    affine: JaxArray
    velocity: JaxArray
    velocity_grid: tuple[JaxArray, JaxArray]
    aligned_points: JaxArray

    def transform(
        self,
        points: JaxArray,
        *,
        direction: Literal["forward", "backward"] = "forward",
    ) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` points with the fitted diffeomorphism."""
        import jax.numpy as jnp

        from ._core import jax_dtype, transform_points_row_col

        pts = jnp.asarray(points, dtype=jax_dtype())
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) `(x, y)` array, found shape {pts.shape}.")
        transformed_rc = transform_points_row_col(
            self.velocity_grid,
            self.velocity,
            self.affine,
            pts[:, ::-1],
            direction=direction,
        )
        return transformed_rc[:, ::-1]


def _rasterize_cloud(
    points_rc: JaxArray, *, dx: float, blur: BlurScales, expand: float
) -> tuple[tuple[JaxArray, JaxArray], JaxArray]:
    """Rasterize a row-col cloud into a ``((grid_y, grid_x), image)`` density."""
    from ._helpers import rasterize

    grid_x, grid_y, image = rasterize(points_rc[:, 1], points_rc[:, 0], dx=dx, blur=blur, expand=expand)
    return (grid_y, grid_x), image
