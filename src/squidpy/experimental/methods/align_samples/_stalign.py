"""STalign estimator: JAX LDDMM point-cloud registration.

Holds both the estimator adapter :func:`fit_stalign` and its result type
:class:`StalignResult`; the pure numerics live under :mod:`._stalign_impl`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy.typing as npt

from squidpy.experimental.methods.registry import ALIGN_SAMPLES

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any


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

        from ._stalign_impl._core import jax_dtype, transform_points_row_col

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


@ALIGN_SAMPLES.register("stalign", requires=("jax",))
def fit_stalign(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    landmarks_source: npt.ArrayLike | None = None,
    landmarks_target: npt.ArrayLike | None = None,
    # rasterization
    dx: float = 30.0,
    blur: float | Sequence[float] = (2.0, 1.0, 0.5),
    raster_expand: float = 1.1,
    # LDDMM registration
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
) -> StalignResult:
    """Fit a deformation mapping ``query`` onto ``ref``.

    Parameters
    ----------
    ref, query
        ``(N, 2)`` / ``(M, 2)`` reference and query point clouds in ``(x, y)``
        order; the query is aligned onto the reference. Both are plain in-memory
        arrays -- extracting them from an ``AnnData`` / ``SpatialData`` is the
        caller's responsibility.
    landmarks_source, landmarks_target
        Optional corresponding ``(x, y)`` landmark arrays used to initialise the
        affine. Must be provided together.
    dx, blur, raster_expand
        Rasterization of the point clouds into density images: grid spacing,
        Gaussian blur scale(s), and field-of-view padding factor.
    a, p, expand, nt, niter, diffeo_start
        LDDMM controls: kernel width ``a``, regularisation power ``p``,
        velocity-grid padding ``expand``, number of integration time steps
        ``nt``, iterations ``niter``, and the iteration at which the
        diffeomorphic (non-affine) part starts updating ``diffeo_start``.
    epL, epT, epV
        Gradient-descent step sizes for the linear part, translation, and
        velocity field.
    sigmaM, sigmaB, sigmaA, sigmaR, sigmaP
        Noise scales for the matching, background, artifact, regularisation, and
        landmark-point terms of the objective.

    Returns
    -------
    A :class:`StalignResult` whose :meth:`~StalignResult.transform` maps
    ``(x, y)`` points into the reference frame; ``aligned_points`` is the fitted
    ``query`` already mapped.
    """
    # Import the JAX-backed solver only after the registry's requirements check
    # passes, so callers without JAX get a clean ImportError rather than a
    # confusing failure from a module-level `import jax`.
    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype, lddmm, transform_points_row_col
    from ._stalign_impl._helpers import affine_from_points, rasterize_cloud, validate_points

    if (landmarks_source is None) != (landmarks_target is None):
        raise ValueError("Expected both landmark arrays to be provided together.")

    # The solver runs internally in row-col (y, x); inputs are (x, y) -- swap at the boundary.
    source_rc = validate_points(query, name="query")[:, ::-1]
    target_rc = validate_points(ref, name="ref")[:, ::-1]
    source_grid, source_image = rasterize_cloud(source_rc, dx=dx, blur=blur, expand=raster_expand)
    target_grid, target_image = rasterize_cloud(target_rc, dx=dx, blur=blur, expand=raster_expand)

    dtype = jax_dtype()
    if landmarks_source is None:
        linear, translation = jnp.eye(2, dtype=dtype), jnp.zeros(2, dtype=dtype)
        src_lm = tgt_lm = None
    else:
        src_lm = validate_points(landmarks_source, name="landmarks_source")[:, ::-1]
        tgt_lm = validate_points(landmarks_target, name="landmarks_target")[:, ::-1]
        linear_np, translation_np = affine_from_points(src_lm, tgt_lm)
        linear, translation = jnp.asarray(linear_np, dtype=dtype), jnp.asarray(translation_np, dtype=dtype)

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        points_source=src_lm,
        points_target=tgt_lm,
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
    aligned_rc = transform_points_row_col(result["xv"], result["v"], result["A"], source_rc, direction="forward")
    return StalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=aligned_rc[:, ::-1],
    )
