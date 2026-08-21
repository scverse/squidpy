"""Alignment for :mod:`squidpy.experimental.tl`.

Layered so the numerics never touch a container: :mod:`._api` holds the public
``align_*`` functions, which resolve the ``*_key`` arguments to plain arrays and hand
them to the array-in / array-out estimators in :mod:`._landmark` and :mod:`._stalign`.
Importing stays cheap -- JAX is pulled in only when a STalign fit actually runs.
"""

from __future__ import annotations

from ._api import (
    align_landmarks,
    align_stalign_image,
    align_stalign_obs,
    align_stalign_volume,
    stalign_apply_transform,
    stalign_apply_warp,
    stalign_from_uns,
    stalign_to_uns,
)
from ._stalign import (
    Stalign2DResult,
    Stalign3DResult,
    StalignImageSolverKwargs,
    StalignObsSolverKwargs,
    StalignResult,
    StalignVolumeSolverKwargs,
    stalign_affine_xyz,
    stalign_deformation_grid,
    stalign_transform_points,
    stalign_warp_image,
)

__all__ = [
    "StalignObsSolverKwargs",
    "Stalign2DResult",
    "Stalign3DResult",
    "StalignResult",
    "StalignVolumeSolverKwargs",
    "StalignImageSolverKwargs",
    "align_landmarks",
    "align_stalign_image",
    "align_stalign_obs",
    "align_stalign_volume",
    "stalign_affine_xyz",
    "stalign_apply_transform",
    "stalign_apply_warp",
    "stalign_from_uns",
    "stalign_to_uns",
    "stalign_deformation_grid",
    "stalign_transform_points",
    "stalign_warp_image",
]
