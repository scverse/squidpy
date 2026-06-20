"""STalign estimator: JAX LDDMM point-cloud registration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy.typing as npt

from squidpy.experimental._methods._families import ALIGN_SAMPLES

if TYPE_CHECKING:
    from ._stalign_impl._tools import StalignResult


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

    from ._stalign_impl._helpers import validate_points
    from ._stalign_impl._tools import (
        STalignConfig,
        STalignPreprocessConfig,
        STalignRegistrationConfig,
        stalign_points,
    )

    config = STalignConfig(
        preprocess=STalignPreprocessConfig(dx=dx, blur=blur, expand=raster_expand),
        registration=STalignRegistrationConfig(
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
        ),
    )

    ref_xy = validate_points(ref, name="ref")
    query_xy = validate_points(query, name="query")

    # The solver runs internally in row-col (y, x); inputs are (x, y) -- swap at the boundary.
    lm_src = None if landmarks_source is None else jnp.asarray(landmarks_source)[:, ::-1]
    lm_tgt = None if landmarks_target is None else jnp.asarray(landmarks_target)[:, ::-1]

    return stalign_points(
        source_points=query_xy[:, ::-1],
        target_points=ref_xy[:, ::-1],
        config=config,
        landmarks_source=lm_src,
        landmarks_target=lm_tgt,
    )
