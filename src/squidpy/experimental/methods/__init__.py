"""In-memory model-fitting core for experimental methods.

Each family subpackage (e.g. :mod:`.align_samples`, :mod:`.align_landmarks`) holds
array-in / array-out estimator functions; the public wrappers in
:mod:`squidpy.experimental.tl` handle container access and write-back. Every subpackage
stays cheap to import -- heavy or optional dependencies (e.g. JAX) are pulled in lazily,
only when an estimator actually runs.
"""

from __future__ import annotations

from squidpy.experimental.methods._common import AlignResult
from squidpy.experimental.methods.align_landmarks import AffineFitResult, fit_affine, fit_similarity
from squidpy.experimental.methods.align_samples import (
    StalignObsSolverKwargs,
    StalignResult,
    StalignSolverKwargs,
    fit_stalign_image,
    fit_stalign_obs,
)

__all__ = [
    "AffineFitResult",
    "AlignResult",
    "StalignObsSolverKwargs",
    "StalignResult",
    "StalignSolverKwargs",
    "fit_affine",
    "fit_similarity",
    "fit_stalign_image",
    "fit_stalign_obs",
]
