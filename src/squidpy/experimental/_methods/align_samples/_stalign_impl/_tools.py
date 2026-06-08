"""Low-level point-cloud tools for experimental STalign.

Lifted from scverse/squidpy#1150 (Selman Özleyen) with import paths adjusted
and minor cleanups (config unpacking, lazy dtype resolution). The container
write-back path (``STalignResult.transform_adata``) was dropped on the move
into :mod:`squidpy.experimental._methods`: results stay in memory and the caller
owns write-back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import jax.numpy as jnp
import numpy as np

from ._core import jax_dtype, lddmm, transform_points_row_col
from ._helpers import (
    PointOrder,
    affine_from_points,
    from_row_col,
    rasterize,
    to_row_col,
)

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any

BlurScales: TypeAlias = float | tuple[float, ...] | list[float]

__all__ = [
    "STalignConfig",
    "STalignPreprocessConfig",
    "STalignPreprocessResult",
    "STalignRegistrationConfig",
    "STalignResult",
    "stalign_points",
    "stalign_preprocess",
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
class STalignPreprocessResult:
    source_grid: tuple[np.ndarray, np.ndarray]
    source_image: np.ndarray
    target_grid: tuple[np.ndarray, np.ndarray]
    target_image: np.ndarray


@dataclass(slots=True)
class STalignResult:
    affine: JaxArray
    velocity: JaxArray
    velocity_grid: tuple[JaxArray, JaxArray]
    aligned_points: JaxArray
    point_order: PointOrder = "row_col"

    def transform_points(
        self,
        points: np.ndarray,
        *,
        direction: Literal["forward", "backward"] = "forward",
        point_order: PointOrder | None = None,
    ) -> JaxArray:
        """Transform arbitrary point arrays with the fitted map."""
        order = self.point_order if point_order is None else point_order
        points_rc = to_row_col(points, point_order=order)
        transformed = transform_points_row_col(
            self.velocity_grid,
            jnp.asarray(self.velocity),
            jnp.asarray(self.affine),
            jnp.asarray(points_rc, dtype=jax_dtype()),
            direction=direction,
        )
        return jnp.asarray(from_row_col(np.asarray(transformed), point_order=order))


def stalign_preprocess(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    config: STalignPreprocessConfig | None = None,
) -> STalignPreprocessResult:
    """Rasterize source and target point clouds for LDDMM registration."""
    config = STalignPreprocessConfig() if config is None else config
    source_points = to_row_col(source_points, point_order="row_col")
    target_points = to_row_col(target_points, point_order="row_col")

    source_x, source_y, source_image = rasterize(
        source_points[:, 1],
        source_points[:, 0],
        dx=config.dx,
        blur=config.blur,
        expand=config.expand,
    )
    target_x, target_y, target_image = rasterize(
        target_points[:, 1],
        target_points[:, 0],
        dx=config.dx,
        blur=config.blur,
        expand=config.expand,
    )

    return STalignPreprocessResult(
        source_grid=(source_y, source_x),
        source_image=source_image,
        target_grid=(target_y, target_x),
        target_image=target_image,
    )


def stalign_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    preprocessed: STalignPreprocessResult | None = None,
    config: STalignConfig | None = None,
    landmarks_source: np.ndarray | None = None,
    landmarks_target: np.ndarray | None = None,
) -> STalignResult:
    """Align source point cloud to target with a JAX LDDMM solver."""
    config = STalignConfig() if config is None else config
    registration = config.registration
    source_points = to_row_col(source_points, point_order="row_col")
    target_points = to_row_col(target_points, point_order="row_col")
    if preprocessed is None:
        preprocessed = stalign_preprocess(source_points, target_points, config=config.preprocess)

    if (landmarks_source is None) != (landmarks_target is None):
        raise ValueError("Expected both landmark arrays to be provided together.")

    if landmarks_source is None:
        linear = np.eye(2, dtype=float)
        translation = np.zeros(2, dtype=float)
        source_landmarks = None
        target_landmarks = None
    else:
        source_landmarks = to_row_col(landmarks_source, point_order="row_col")
        target_landmarks = to_row_col(landmarks_target, point_order="row_col")
        linear, translation = affine_from_points(source_landmarks, target_landmarks)

    dtype = jax_dtype()
    result = lddmm(
        preprocessed.source_grid,
        preprocessed.source_image,
        preprocessed.target_grid,
        preprocessed.target_image,
        L=jnp.asarray(linear, dtype=dtype),
        T=jnp.asarray(translation, dtype=dtype),
        points_source=None if source_landmarks is None else jnp.asarray(source_landmarks, dtype=dtype),
        points_target=None if target_landmarks is None else jnp.asarray(target_landmarks, dtype=dtype),
        **asdict(registration),
    )
    points_rc = to_row_col(source_points, point_order="row_col")
    transformed = transform_points_row_col(
        result["xv"],
        jnp.asarray(result["v"]),
        jnp.asarray(result["A"]),
        jnp.asarray(points_rc, dtype=dtype),
        direction="forward",
    )
    aligned_points = jnp.asarray(from_row_col(np.asarray(transformed), point_order="row_col"))
    return STalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=aligned_points,
        point_order="row_col",
    )
