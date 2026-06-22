"""The registry machinery and the family registries it powers.

This module holds three things that belong together:

* :class:`Registry` -- a flat ``name -> function`` map, one per method *family*.
* The structural :class:`~typing.Protocol` contracts each family advertises, so
  the public API and the registries are typed against a contract rather than a
  concrete estimator result (e.g. ``StalignResult``). A new estimator only has to
  satisfy :class:`AlignResult` -- a ``transform`` that maps points into the
  reference frame -- to plug into :func:`squidpy.experimental.tl.align`.
* The family registry instances (:data:`ALIGN_SAMPLES`, :data:`ALIGN_LANDMARKS`)
  the estimator implementations register into.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy.typing as npt

from squidpy._utils import NDArrayA

if TYPE_CHECKING:
    from squidpy.experimental.method_registry.methods.align_landmarks._landmark import AffineFitResult

__all__ = [
    "Registry",
    "AlignResult",
    "AlignSamplesFn",
    "AlignLandmarksFn",
    "ALIGN_SAMPLES",
    "ALIGN_LANDMARKS",
]

#: The calling convention a family's registry advertises (returned by :meth:`Registry.get`).
F = TypeVar("F", bound=Callable[..., Any])
#: The concrete function being registered. Kept separate from ``F`` so an estimator may
#: declare specific keyword parameters (e.g. ``config=``) without having to structurally
#: match the family's open-ended ``**kwargs`` calling convention.
RegisteredT = TypeVar("RegisteredT", bound=Callable[..., Any])


class Registry(Generic[F]):
    """A flat ``name -> function`` registry for one *family* of methods.

    One :class:`Registry` is created per family (e.g. ``align``, ``impute``),
    so keys are plain method names -- there is no ``(method, mode)`` compound
    key, because the family already pins the rest.

    The type parameter ``F`` is the family's calling convention (a callable
    :class:`~typing.Protocol`); :meth:`get` returns it, so dispatch sites are
    typed against the family contract rather than ``Callable[..., Any]``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, F] = {}

    def register(self, key: str, *, requires: tuple[str, ...] = ()) -> Callable[[RegisteredT], RegisteredT]:
        """Return a decorator registering a method/function under ``key``."""

        def decorator(func: RegisteredT) -> RegisteredT:
            if key in self._registry:
                raise ValueError(f"Method {key!r} is already registered in the {self.name!r} registry.")

            if requires:

                @functools.wraps(func)
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    missing = [pkg for pkg in requires if importlib.util.find_spec(pkg) is None]
                    if missing:
                        verb = "is" if len(missing) == 1 else "are"
                        names = ", ".join(repr(p) for p in missing)
                        extras = ",".join(missing)
                        raise ImportError(
                            f"Method {key!r} requires {names}, which {verb} not installed. "
                            f'Install with `pip install "squidpy[{extras}]"`.'
                        )
                    return func(*args, **kwargs)

                self._registry[key] = wrapped  # type: ignore[assignment]
                return wrapped  # type: ignore[return-value]
            else:
                self._registry[key] = func  # type: ignore[assignment]
                return func

        return decorator

    def get(self, key: str) -> F:
        """Return the function registered under ``key``."""
        try:
            return self._registry[key]
        except KeyError:
            raise ValueError(f"Unknown {self.name} method {key!r}. Available: {sorted(self._registry)}.") from None

    def keys(self) -> tuple[str, ...]:
        """Return the registered method names."""
        return tuple(self._registry)


@runtime_checkable
class AlignResult(Protocol):
    """A fitted alignment that maps ``(N, 2)`` ``(x, y)`` points into the reference frame.

    This is the only thing the public ``align*`` functions require of an
    estimator's result, so ``output_mode="object"`` is agnostic to the method
    that produced it.
    """

    def transform(self, points: npt.ArrayLike, /) -> NDArrayA:
        """Map an ``(N, 2)`` ``(x, y)`` array into the reference frame."""
        ...


class AlignSamplesFn(Protocol):
    """Calling convention for ``align_samples`` estimators.

    Two point clouds in (passed by keyword as ``ref`` / ``query`` so the
    direction can never be silently swapped), one :class:`AlignResult` out.
    Solver-specific options arrive through ``**kwargs``.
    """

    def __call__(self, ref: npt.ArrayLike, query: npt.ArrayLike, **kwargs: Any) -> AlignResult: ...


class AlignLandmarksFn(Protocol):
    """Calling convention for ``align_landmarks`` estimators: paired landmarks in, affine out."""

    def __call__(
        self,
        ref: npt.ArrayLike,
        query: npt.ArrayLike,
        *,
        source_cs: str | None = ...,
        target_cs: str | None = ...,
    ) -> AffineFitResult: ...


#: Sample-to-sample alignment estimators -- ref/query point clouds in, transform out.
#: Consumed by ``squidpy.experimental.tl.align``.
ALIGN_SAMPLES: Registry[AlignSamplesFn] = Registry("align_samples")

#: Closed-form landmark alignment estimators -- paired landmarks in, affine out.
#: Consumed by ``squidpy.experimental.tl.align_by_landmarks``.
ALIGN_LANDMARKS: Registry[AlignLandmarksFn] = Registry("align_landmarks")
