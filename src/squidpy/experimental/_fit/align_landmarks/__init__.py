"""``align_landmarks`` family: closed-form alignment from paired landmarks.

Importing this package registers the family's estimators into
:data:`~squidpy.experimental._fit._families.ALIGN_LANDMARKS`. Pure NumPy /
spatialdata / skimage -- no JAX.
"""

from __future__ import annotations

from squidpy.experimental._fit._families import ALIGN_LANDMARKS
from squidpy.experimental._fit.align_landmarks._landmark import (
    AffineFitResult,
    LandmarkAffineEstimator,
    LandmarkSimilarityEstimator,
)

__all__ = [
    "ALIGN_LANDMARKS",
    "AffineFitResult",
    "LandmarkAffineEstimator",
    "LandmarkSimilarityEstimator",
]
