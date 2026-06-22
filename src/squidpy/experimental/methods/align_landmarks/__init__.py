"""``align_landmarks`` family: closed-form alignment from paired landmarks.

Importing this package registers the family's estimators into
:data:`~squidpy.experimental.methods.registry.ALIGN_LANDMARKS`. Only the
implementations are re-exported here; the registry itself lives in (and is public
from) :mod:`squidpy.experimental.methods`.
"""

from __future__ import annotations

from squidpy.experimental.methods.align_landmarks._landmark import (
    AffineFitResult,
    fit_affine,
    fit_similarity,
)

__all__ = [
    "AffineFitResult",
    "fit_affine",
    "fit_similarity",
]
