"""``align_samples`` family: align two samples' point clouds or images (STalign).

Array-in / array-out estimators; container access lives in the public wrappers
:func:`squidpy.experimental.tl.align_stalign_obs` /
:func:`squidpy.experimental.tl.align_stalign_image`. Importing stays cheap -- JAX is
pulled in lazily, only when a fit actually runs.
"""

from __future__ import annotations

from squidpy.experimental.methods.align_samples._stalign import (
    StalignObsSolverKwargs,
    StalignResult,
    StalignSolverKwargs,
    fit_stalign_image,
    fit_stalign_obs,
)

__all__ = [
    "StalignObsSolverKwargs",
    "StalignResult",
    "StalignSolverKwargs",
    "fit_stalign_image",
    "fit_stalign_obs",
]
