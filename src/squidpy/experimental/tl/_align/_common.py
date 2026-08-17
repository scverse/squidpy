"""Shared contracts for the experimental estimators.

Two small pieces every estimator family uses:

* :class:`AlignResult` -- the structural :class:`~typing.Protocol` the public align
  functions are typed against, so they stay agnostic to which estimator produced the
  fit (``StalignResult``, ``AffineFitResult``, ...).
* :func:`requires` -- a decorator deferring optional heavy dependencies (e.g. JAX) to
  call time, so importing :mod:`squidpy.experimental` stays cheap and a missing
  dependency fails with an install hint rather than a bare :class:`ImportError`.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

from squidpy._utils import NDArrayA

__all__ = ["AlignResult", "requires", "validate_xy"]

F = TypeVar("F", bound=Callable[..., Any])


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


def requires(*packages: str) -> Callable[[F], F]:
    """Defer ``packages`` to call time: raise an installing-hint ImportError if absent.

    The wrapped function's module can then be imported without the dependency, keeping
    :mod:`squidpy.experimental` cheap to import.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            missing = [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]
            if missing:
                verb = "is" if len(missing) == 1 else "are"
                names = ", ".join(repr(p) for p in missing)
                extras = ",".join(missing)
                raise ImportError(
                    f"`{func.__name__}` requires {names}, which {verb} not installed. "
                    f'Install with `pip install "squidpy[{extras}]"`.'
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
