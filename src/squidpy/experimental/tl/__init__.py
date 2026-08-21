from __future__ import annotations

# The concrete result classes and the solver-tuning TypedDicts are exported because
# callers need them: a result is what a fit returns, and the TypedDicts document the knobs.
from ._align import (
    Stalign2DResult,
    Stalign3DResult,
    StalignImageSolverKwargs,
    StalignObsSolverKwargs,
    StalignResult,
    StalignVolumeSolverKwargs,
    align_landmarks,
    align_stalign_image,
    align_stalign_obs,
    align_stalign_volume,
    stalign_affine_xyz,
    stalign_apply_transform,
    stalign_apply_warp,
    stalign_deformation_grid,
    stalign_from_uns,
    stalign_to_uns,
    stalign_transform_points,
    stalign_warp_image,
)
from ._stitched_labels import make_stitched_labels
from ._tiling_qc import TilingQCParams, calculate_tiling_qc
from ._tiling_stitch import StitchParams, assign_stitch_groups

__all__ = [
    "StalignObsSolverKwargs",
    "Stalign2DResult",
    "Stalign3DResult",
    "StalignResult",
    "StalignVolumeSolverKwargs",
    "StalignImageSolverKwargs",
    "StitchParams",
    "TilingQCParams",
    "align_landmarks",
    "align_stalign_image",
    "align_stalign_obs",
    "align_stalign_volume",
    "assign_stitch_groups",
    "calculate_tiling_qc",
    "make_stitched_labels",
    "stalign_affine_xyz",
    "stalign_apply_transform",
    "stalign_apply_warp",
    "stalign_from_uns",
    "stalign_to_uns",
    "stalign_deformation_grid",
    "stalign_transform_points",
    "stalign_warp_image",
]
