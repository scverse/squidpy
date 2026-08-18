from __future__ import annotations

# `AlignResult` is the estimator contract: a `transform` mapping points into the
# reference frame, satisfied by every fit whatever method produced it. The concrete
# results and the solver-tuning TypedDicts come along because callers need them --
# `StalignResult` is what a fit returns, and the TypedDicts document the knobs.
from ._align import (
    AffineFitResult,
    AlignResult,
    StalignObsSolverKwargs,
    StalignResult,
    StalignSliceResult,
    StalignSliceSolverKwargs,
    StalignSolverKwargs,
    align_landmarks,
    align_stalign_image,
    align_stalign_obs,
    align_stalign_slice,
)
from ._stitched_labels import make_stitched_labels
from ._tiling_qc import TilingQCParams, calculate_tiling_qc
from ._tiling_stitch import StitchParams, assign_stitch_groups

__all__ = [
    "AffineFitResult",
    "AlignResult",
    "StalignObsSolverKwargs",
    "StalignResult",
    "StalignSliceResult",
    "StalignSliceSolverKwargs",
    "StalignSolverKwargs",
    "StitchParams",
    "TilingQCParams",
    "align_landmarks",
    "align_stalign_image",
    "align_stalign_obs",
    "align_stalign_slice",
    "assign_stitch_groups",
    "calculate_tiling_qc",
    "make_stitched_labels",
]
