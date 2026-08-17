"""Shared contracts for the experimental estimators.

Two small pieces every estimator family uses:

* :class:`AlignResult` -- the structural :class:`~typing.Protocol` the public align
  functions are typed against, so they stay agnostic to which estimator produced the
  fit (``StalignResult``, ``AffineFitResult``, ...).
* :func:`validate_xy` -- the ``(n, 2)`` ``(x, y)`` contract every estimator's inputs
  have to satisfy, whichever array library the estimator itself works in.

Optional heavy dependencies (e.g. JAX) are imported inside the function that needs
them, so importing :mod:`squidpy.experimental` stays cheap.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from squidpy._utils import NDArrayA

__all__ = ["AlignResult", "validate_xy"]


def validate_xy(points: npt.ArrayLike, *, name: str) -> NDArrayA:
    """Coerce ``points`` to a finite ``(n, 2)`` float array of ``(x, y)`` pairs.

    Shared by every estimator family: the landmark fits work in NumPy and the STalign
    solver in JAX, but the contract they validate is the same one.
    """
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected `{name}` to be a sequence of (x, y) pairs, found shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Expected `{name}` to contain only finite values.")
    return arr


@runtime_checkable
class AlignResult(Protocol):
    """A fitted alignment that maps ``(N, 2)`` ``(x, y)`` points into the reference frame.

    This is the only thing the public align functions require of an estimator's
    result, so returning the fit is agnostic to the method that produced it.
    """

    def transform(self, points: npt.ArrayLike, /) -> NDArrayA:
        """Map an ``(N, 2)`` ``(x, y)`` array into the reference frame."""
        ...
