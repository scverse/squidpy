from __future__ import annotations

# `AlignResult` is the only result type on the public surface: it is the estimator
# contract (a `transform` mapping points into the reference frame) and the declared
# return of `align`. The concrete results (`StalignResult`,
# `AffineFitResult`) stay in their home modules under `squidpy.experimental.methods`
# for callers that need raw fields -- the public API stays method-agnostic.
from squidpy.experimental.methods import AlignResult

from ._align import align
from ._stitched_labels import make_stitched_labels
from ._tiling_qc import TilingQCParams, calculate_tiling_qc
from ._tiling_stitch import StitchParams, assign_stitch_groups

__all__ = [
    "align",
    "calculate_tiling_qc",
    "make_stitched_labels",
    "TilingQCParams",
    "AlignResult",
    "StitchParams",
    "assign_stitch_groups",
]
