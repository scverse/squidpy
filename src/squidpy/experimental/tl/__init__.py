from __future__ import annotations

# The fit classes and the solver-tuning TypedDicts are exported because callers need them:
# a fit is what an alignment returns, and the TypedDicts document the solver knobs.
from ._align import (
    StalignFit,
    StalignImageFit,
    StalignImageSolverKwargs,
    StalignObsFit,
    StalignObsSolverKwargs,
    StalignVolumeFit,
    StalignVolumeSolverKwargs,
    align_landmarks,
    stalign_align_image,
    stalign_align_obs,
    stalign_align_volume,
)
from ._stitched_labels import make_stitched_labels
from ._tiling_qc import TilingQCParams, calculate_tiling_qc
from ._tiling_stitch import StitchParams, assign_stitch_groups

__all__ = [
    "StalignFit",
    "StalignImageFit",
    "StalignImageSolverKwargs",
    "StalignObsFit",
    "StalignObsSolverKwargs",
    "StalignVolumeFit",
    "StalignVolumeSolverKwargs",
    "StitchParams",
    "TilingQCParams",
    "align_landmarks",
    "assign_stitch_groups",
    "calculate_tiling_qc",
    "make_stitched_labels",
    "stalign_align_image",
    "stalign_align_obs",
    "stalign_align_volume",
]
