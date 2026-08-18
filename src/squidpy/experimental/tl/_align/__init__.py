"""Alignment for :mod:`squidpy.experimental.tl`.

Layered so the numerics never touch a container: :mod:`._api` holds the public
``align_*`` functions, which resolve the ``*_key`` arguments to plain arrays and hand
them to the array-in / array-out estimators in :mod:`._landmark` and :mod:`._stalign`.
Importing stays cheap -- JAX is pulled in only when a STalign fit actually runs.
"""

from __future__ import annotations

from ._api import align_landmarks, align_stalign_image, align_stalign_obs, align_stalign_slice
from ._common import AlignResult
from ._landmark import AffineFitResult
from ._stalign import (
    StalignObsSolverKwargs,
    StalignResult,
    StalignSliceResult,
    StalignSliceSolverKwargs,
    StalignSolverKwargs,
)

__all__ = [
    "AffineFitResult",
    "AlignResult",
    "StalignObsSolverKwargs",
    "StalignResult",
    "StalignSliceResult",
    "StalignSliceSolverKwargs",
    "StalignSolverKwargs",
    "align_landmarks",
    "align_stalign_image",
    "align_stalign_obs",
    "align_stalign_slice",
]
