"""STalign estimator: JAX LDDMM point-cloud registration.

The JAX solver is lifted from scverse/squidpy#1150 (Selman Özleyen); see
:mod:`._stalign_impl`. This module only adds the thin :class:`Estimator` /
:class:`FitResult` adapter onto the :mod:`squidpy.experimental._fit` core. JAX
is imported lazily inside :meth:`StalignEstimator.fit`, so importing this module
is cheap and does not require JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from squidpy.experimental._fit._families import ALIGN_SAMPLES
from squidpy.experimental._fit._result import FitResult

if TYPE_CHECKING:
    from ._stalign_impl._tools import STalignConfig


@dataclass
class StalignFitResult(FitResult):
    """Fitted STalign deformation, in ``(x, y)`` convention.

    ``deltas`` is the per-point displacement that maps the *query* points into
    the *reference* frame; :meth:`transform` bakes it in. STalign produces a
    diffeomorphic (non-affine) map, so ``affine`` is ``None`` here -- the full
    map (velocity field, velocity grid, affine initialisation) is kept under
    ``metadata['stalign_result']`` for power users who need to transform
    arbitrary points via ``stalign_result.transform_points``.
    """

    deltas: np.ndarray
    affine: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted displacement to the ``(N, 2)`` ``(x, y)`` points it was fit on.

        ``deltas`` is per-point, so ``x`` must have the same shape as the query
        cloud passed to :meth:`fit_stalign`. To map *arbitrary* points,
        use ``metadata['stalign_result'].transform_points``.
        """
        coords = np.asarray(x, dtype=float)
        if coords.shape != self.deltas.shape:
            raise ValueError(
                f"`transform` expects coordinates of shape {self.deltas.shape} (the fitted query cloud), "
                f"found {coords.shape}. Use `metadata['stalign_result'].transform_points` for arbitrary points."
            )
        return coords + self.deltas


@ALIGN_SAMPLES.register("stalign", requires=("jax",))
def fit_stalign(
    ref: np.ndarray,
    query: np.ndarray,
    *,
    config: STalignConfig | None = None,
    landmarks_source: np.ndarray | None = None,
    landmarks_target: np.ndarray | None = None,
) -> StalignFitResult:
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
    A :class:`StalignFitResult` whose ``deltas`` move ``query`` into the
    reference frame.
    """
    # Import the JAX-backed solver only after requirements pass, so callers
    # without JAX get the clean ImportError from check_requirements rather
    # than a confusing failure from a module-level `import jax`.
    from ._stalign_impl._tools import stalign_points

    ref_xy = np.asarray(ref, dtype=float)
    query_xy = np.asarray(query, dtype=float)
    for arr_name, arr in (("ref", ref_xy), ("query", query_xy)):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected `{arr_name}` to be an (N, 2) array, found shape {arr.shape}.")

    # The solver runs internally in row-col (y, x); inputs are (x, y) -- swap at the boundary.
    src_rc = query_xy[:, [1, 0]]
    tgt_rc = ref_xy[:, [1, 0]]
    lm_src_rc = None if landmarks_source is None else np.asarray(landmarks_source, dtype=float)[:, [1, 0]]
    lm_tgt_rc = None if landmarks_target is None else np.asarray(landmarks_target, dtype=float)[:, [1, 0]]

    result = stalign_points(
        source_points=src_rc,
        target_points=tgt_rc,
        config=config,
        landmarks_source=lm_src_rc,
        landmarks_target=lm_tgt_rc,
    )

    aligned_xy = np.asarray(result.aligned_points)[:, [1, 0]]
    deltas_xy = aligned_xy - query_xy
    return StalignFitResult(
        deltas=deltas_xy,
        affine=None,
        metadata={"method": "stalign", "stalign_result": result},
    )
