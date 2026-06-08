"""Point-cloud tools for experimental STalign.

The fitted map is returned as a single :class:`StalignResult`; container
write-back lives in the caller, not here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any

BlurScales: TypeAlias = float | tuple[float, ...] | list[float]

__all__ = [
    "STalignConfig",
    "STalignPreprocessConfig",
    "STalignRegistrationConfig",
    "StalignResult",
    "stalign_points",
]


@dataclass(slots=True)
class STalignPreprocessConfig:
    dx: float = 30.0
    blur: BlurScales = (2.0, 1.0, 0.5)
    expand: float = 1.1


@dataclass(slots=True)
class STalignRegistrationConfig:
    """LDDMM registration hyperparameters.

    Field names (``sigmaM``, ``epL``, etc.) preserve the conventions from
    the STalign paper and reference implementation to keep them
    recognisable when cross-referencing the literature.
    """

    a: float = 500.0
    p: float = 2.0
    expand: float = 2.0
    nt: int = 3
    niter: int = 5000
    diffeo_start: int = 0
    epL: float = 2e-8
    epT: float = 2e-1
    epV: float = 2e3
    sigmaM: float = 1.0
    sigmaB: float = 2.0
    sigmaA: float = 5.0
    sigmaR: float = 5e5
    sigmaP: float = 2e1


@dataclass(slots=True)
class STalignConfig:
    preprocess: STalignPreprocessConfig = field(default_factory=STalignPreprocessConfig)
    registration: STalignRegistrationConfig = field(default_factory=STalignRegistrationConfig)


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


def _rasterize_cloud(points_rc: JaxArray, config: STalignPreprocessConfig) -> tuple[tuple[JaxArray, JaxArray], JaxArray]:
    """Rasterize a row-col cloud into a ``((grid_y, grid_x), image)`` density."""
    from ._helpers import rasterize

    grid_x, grid_y, image = rasterize(
        points_rc[:, 1],
        points_rc[:, 0],
        dx=config.dx,
        blur=config.blur,
        expand=config.expand,
    )
    return (grid_y, grid_x), image


def stalign_points(
    source_points: JaxArray,
    target_points: JaxArray,
    *,
    config: STalignConfig | None = None,
    landmarks_source: JaxArray | None = None,
    landmarks_target: JaxArray | None = None,
) -> StalignResult:
    """Align a source point cloud onto a target with a JAX LDDMM solver.

    All point arrays are in the solver's row-column frame; the returned
    :class:`StalignResult` speaks ``(x, y)``.
    """
    import jax.numpy as jnp

    from ._core import jax_dtype, lddmm, transform_points_row_col
    from ._helpers import affine_from_points, validate_points

    config = STalignConfig() if config is None else config
    registration = config.registration
    source_points = validate_points(source_points, name="source_points")
    target_points = validate_points(target_points, name="target_points")
    source_grid, source_image = _rasterize_cloud(source_points, config.preprocess)
    target_grid, target_image = _rasterize_cloud(target_points, config.preprocess)

    if (landmarks_source is None) != (landmarks_target is None):
        raise ValueError("Expected both landmark arrays to be provided together.")

    dtype = jax_dtype()
    if landmarks_source is None:
        linear = jnp.eye(2, dtype=dtype)
        translation = jnp.zeros(2, dtype=dtype)
        source_landmarks = None
        target_landmarks = None
    else:
        source_landmarks = validate_points(landmarks_source, name="landmarks_source")
        target_landmarks = validate_points(landmarks_target, name="landmarks_target")
        linear_np, translation_np = affine_from_points(source_landmarks, target_landmarks)
        linear = jnp.asarray(linear_np, dtype=dtype)
        translation = jnp.asarray(translation_np, dtype=dtype)

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        points_source=source_landmarks,
        points_target=target_landmarks,
        **asdict(registration),
    )
    transformed_rc = transform_points_row_col(
        result["xv"],
        result["v"],
        result["A"],
        source_points,
        direction="forward",
    )
    return StalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=transformed_rc[:, ::-1],
    )
