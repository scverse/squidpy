"""Low-level point-cloud tools for experimental STalign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import numpy as np
from anndata import AnnData

from squidpy.experimental.tl._stalign_core import JAX_DTYPE, lddmm, transform_points_row_col
from squidpy.experimental.tl._stalign_helpers import (
    PointOrder,
    affine_from_points,
    extract_points,
    from_row_col,
    rasterize,
    to_row_col,
)

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any

__all__ = ["STalignPreprocessResult", "STalignResult", "stalign_points", "stalign_preprocess", "transform_points"]


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
        return transform_points(
            self.velocity_grid,
            self.velocity,
            self.affine,
            points,
            direction=direction,
            point_order=self.point_order if point_order is None else point_order,
        )

    def transform_adata(
        self,
        adata: AnnData,
        *,
        spatial_key: str = "spatial",
        key_added: str | None = None,
        direction: Literal["forward", "backward"] = "forward",
        copy: bool = False,
    ) -> AnnData | np.ndarray | None:
        """Apply the fitted transform to coordinates stored on an AnnData object."""
        points = extract_points(adata, key=spatial_key)
        transformed = np.asarray(self.transform_points(points, direction=direction, point_order="xy"))

        if copy:
            adata = adata.copy()

        if key_added is None:
            if copy:
                adata.obsm[spatial_key] = transformed
                return adata
            return transformed

        adata.obsm[key_added] = transformed
        if copy:
            return adata
        return None


def stalign_preprocess(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    dx: float = 30.0,
    blur: float | list[float] = (2.0, 1.0, 0.5),
    expand: float = 1.1,
) -> STalignPreprocessResult:
    """Rasterize source and target point clouds for LDDMM registration."""
    source_points = to_row_col(source_points, point_order="row_col")
    target_points = to_row_col(target_points, point_order="row_col")

    source_x, source_y, source_image = rasterize(
        source_points[:, 1],
        source_points[:, 0],
        dx=dx,
        blur=blur,
        expand=expand,
    )
    target_x, target_y, target_image = rasterize(
        target_points[:, 1],
        target_points[:, 0],
        dx=dx,
        blur=blur,
        expand=expand,
    )

    return STalignPreprocessResult(
        source_grid=(source_y, source_x),
        source_image=source_image,
        target_grid=(target_y, target_x),
        target_image=target_image,
    )


def transform_points(
    xv: tuple[JaxArray, JaxArray],
    v: JaxArray,
    A: JaxArray,
    points: np.ndarray,
    *,
    direction: Literal["forward", "backward"] = "forward",
    point_order: PointOrder = "row_col",
) -> JaxArray:
    """Transform point arrays with a fitted STalign map."""
    points_rc = to_row_col(points, point_order=point_order)
    transformed = transform_points_row_col(
        xv,
        jnp.asarray(v),
        jnp.asarray(A),
        jnp.asarray(points_rc, dtype=JAX_DTYPE),
        direction=direction,
    )
    return jnp.asarray(from_row_col(np.asarray(transformed), point_order=point_order))


def stalign_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    preprocessed: STalignPreprocessResult | None = None,
    dx: float = 30.0,
    blur: float | list[float] = (2.0, 1.0, 0.5),
    landmarks_source: np.ndarray | None = None,
    landmarks_target: np.ndarray | None = None,
    a: float = 500.0,
    p: float = 2.0,
    expand: float = 2.0,
    nt: int = 3,
    niter: int = 5000,
    diffeo_start: int = 0,
    epL: float = 2e-8,
    epT: float = 2e-1,
    epV: float = 2e3,
    sigmaM: float = 1.0,
    sigmaB: float = 2.0,
    sigmaA: float = 5.0,
    sigmaR: float = 5e5,
    sigmaP: float = 2e1,
) -> STalignResult:
    """Align source point cloud to target with a JAX LDDMM solver."""
    source_points = to_row_col(source_points, point_order="row_col")
    target_points = to_row_col(target_points, point_order="row_col")
    if preprocessed is None:
        preprocessed = stalign_preprocess(source_points, target_points, dx=dx, blur=blur)

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

    result = lddmm(
        preprocessed.source_grid,
        preprocessed.source_image,
        preprocessed.target_grid,
        preprocessed.target_image,
        L=jnp.asarray(linear, dtype=JAX_DTYPE),
        T=jnp.asarray(translation, dtype=JAX_DTYPE),
        points_source=None if source_landmarks is None else jnp.asarray(source_landmarks, dtype=JAX_DTYPE),
        points_target=None if target_landmarks is None else jnp.asarray(target_landmarks, dtype=JAX_DTYPE),
        a=a,
        p=p,
        expand=expand,
        nt=nt,
        niter=niter,
        diffeo_start=diffeo_start,
        epL=epL,
        epT=epT,
        epV=epV,
        sigmaM=sigmaM,
        sigmaB=sigmaB,
        sigmaA=sigmaA,
        sigmaR=sigmaR,
        sigmaP=sigmaP,
    )
    aligned_points = transform_points(
        result["xv"],
        result["v"],
        result["A"],
        source_points,
        direction="forward",
        point_order="row_col",
    )
    return STalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=aligned_points,
        point_order="row_col",
    )
