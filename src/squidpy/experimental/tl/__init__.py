from __future__ import annotations

# `AlignResult` is the only result type on the public surface: it is the estimator
# contract (a `transform` mapping points into the reference frame) and the declared
# return of `align` / `align_by_landmarks`. The concrete results (`StalignResult`,
# `AffineFitResult`) stay in their home modules under `squidpy.experimental._methods`
# for callers that need raw fields -- the public API stays method-agnostic.
from squidpy.experimental._methods import AlignResult

from ._align import align, align_by_landmarks
from ._tiling_qc import TilingQCParams, calculate_tiling_qc
from ._tiling_stitch import StitchParams, assign_stitch_groups

__all__ = [
    "align",
    "align_by_landmarks",
    "calculate_tiling_qc",
    "TilingQCParams",
    "AlignResult",
    "StitchParams",
    "assign_stitch_groups",
]
