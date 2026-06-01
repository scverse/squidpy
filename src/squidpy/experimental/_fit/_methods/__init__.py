"""Concrete estimators for :mod:`squidpy.experimental._fit`.

Importing this package registers the bundled estimators into their family
registries (``ALIGN``: STalign; ``LANDMARK``: similarity / affine) but stays
cheap: optional dependencies such as JAX are pulled in lazily, only when an
estimator's ``fit`` is actually called.
"""

from __future__ import annotations

from squidpy.experimental._fit._methods._families import ALIGN, LANDMARK
from squidpy.experimental._fit._methods._landmark import (
    AffineFitResult,
    LandmarkAffineEstimator,
    LandmarkSimilarityEstimator,
)
from squidpy.experimental._fit._methods._stalign import StalignEstimator, StalignFitResult

__all__ = [
    "ALIGN",
    "LANDMARK",
    "AffineFitResult",
    "LandmarkAffineEstimator",
    "LandmarkSimilarityEstimator",
    "StalignEstimator",
    "StalignFitResult",
]
