"""In-memory model-fitting core for experimental methods.

The :mod:`.registry` subpackage holds the registry machinery and the family
registries; :mod:`.methods` holds the estimator implementations. Importing this
package imports :mod:`.methods` so the estimators register themselves into the
(public) family registries.
"""

from __future__ import annotations

from squidpy.experimental.method_registry.registry import (
    ALIGN_LANDMARKS,
    ALIGN_SAMPLES,
    AlignLandmarksFn,
    AlignResult,
    AlignSamplesFn,
    Registry,
)

# Import for side effects: populates ALIGN_SAMPLES / ALIGN_LANDMARKS.
from squidpy.experimental.method_registry import methods  # noqa: F401

__all__ = [
    "Registry",
    "AlignResult",
    "AlignSamplesFn",
    "AlignLandmarksFn",
    "ALIGN_SAMPLES",
    "ALIGN_LANDMARKS",
]
