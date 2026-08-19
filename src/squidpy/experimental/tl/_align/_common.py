"""Shared input contract for the experimental estimators.

:func:`validate_xy` is the ``(n, 2)`` ``(x, y)`` contract every estimator's inputs have
to satisfy, whichever array library the estimator itself works in: the landmark fits work
in NumPy and the STalign solver in JAX, but what they validate is the same.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from squidpy._utils import NDArrayA

__all__ = ["validate_xy"]


def validate_xy(points: npt.ArrayLike, *, name: str) -> NDArrayA:
    """Coerce ``points`` to a finite ``(n, 2)`` float array of ``(x, y)`` pairs."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected `{name}` to be a sequence of (x, y) pairs, found shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Expected `{name}` to contain only finite values.")
    return arr
