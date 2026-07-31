"""In-memory model-fitting core for experimental methods.

The :mod:`.registry` subpackage holds the registry machinery and the :data:`ALIGN`
registry; each family subpackage (e.g. :mod:`.align_samples`, :mod:`.align_landmarks`)
holds the estimator implementations. Importing this package imports those subpackages so
the estimators register themselves. Each subpackage stays cheap to import -- heavy or
optional dependencies (e.g. JAX) are pulled in lazily, only when an estimator actually
runs.
"""

from __future__ import annotations

# Import for side effects: populates ALIGN.
from squidpy.experimental.methods import align_landmarks, align_samples  # noqa: F401
from squidpy.experimental.methods.registry import (
    ALIGN,
    AlignImagesFn,
    AlignLandmarksFn,
    AlignMethod,
    AlignObsFn,
    AlignResult,
    Modality,
    Registry,
)

__all__ = [
    "ALIGN",
    "AlignImagesFn",
    "AlignLandmarksFn",
    "AlignMethod",
    "AlignObsFn",
    "AlignResult",
    "Modality",
    "Registry",
]
