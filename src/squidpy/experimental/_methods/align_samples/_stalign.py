"""STalign estimator: JAX LDDMM point-cloud registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy.typing as npt

from squidpy.experimental._methods._families import ALIGN_SAMPLES

if TYPE_CHECKING:
    from ._stalign_impl._tools import STalignConfig, StalignResult


@ALIGN_SAMPLES.register("stalign", requires=("jax",))
def fit_stalign(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    config: STalignConfig | None = None,
    landmarks_source: npt.ArrayLike | None = None,
    landmarks_target: npt.ArrayLike | None = None,
) -> StalignResult:
    """Fit a deformation mapping ``query`` onto ``ref``.

    Parameters
    ----------
    ref
        ``(N, 2)`` reference point cloud in ``(x, y)`` order.
    query
        ``(M, 2)`` query point cloud in ``(x, y)`` order, to be aligned to
        ``ref``. Both are plain in-memory arrays; extracting them from an
        ``AnnData`` / ``SpatialData`` is the caller's responsibility.
    config
        Optional :class:`STalignConfig` of solver hyperparameters.
    landmarks_source, landmarks_target
        Optional corresponding ``(x, y)`` landmark arrays used to
        initialise the affine. Must be provided together.

    Returns
    -------
    A :class:`StalignResult` whose :meth:`~StalignResult.transform` maps
    ``(x, y)`` points into the reference frame; ``aligned_points`` is the fitted
    ``query`` already mapped.
    """
    # Import the JAX-backed solver only after requirements pass, so callers
    # without JAX get the clean ImportError from check_requirements rather
    # than a confusing failure from a module-level `import jax`.
    import jax.numpy as jnp

    from ._stalign_impl._helpers import validate_points
    from ._stalign_impl._tools import stalign_points

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
