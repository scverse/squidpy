"""The registry machinery and the align family it powers.

This module holds three things that belong together:

* :class:`Registry` -- a ``name -> AlignMethod`` map for one method *family*.
* :class:`AlignMethod` -- what one named method can align. A method fills the slots it
  implements (``obs``, ``images``, ``landmarks``) and leaves the rest ``None``, so
  "``stalign`` cannot align landmarks" is data rather than a special case buried in a
  dispatch site.
* The structural :class:`~typing.Protocol` contracts each slot advertises, so the public
  API is typed against a contract rather than a concrete result (e.g. ``StalignResult``).
  A new estimator only has to satisfy :class:`AlignResult` -- a ``transform`` that maps
  points into the reference frame -- to plug into :func:`squidpy.experimental.tl.align`.

The three slots do **not** all take the same kind of input. ``obs`` and ``images`` take
the data being aligned; ``landmarks`` takes pre-paired correspondences that annotate it.
Each public entry point only ever reads its own slot, so no caller sees a mixed
convention, but the distinction is why these are named slots rather than one callable
with a mode flag.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args, runtime_checkable

import numpy.typing as npt

from squidpy._utils import NDArrayA

if TYPE_CHECKING:
    from squidpy.experimental.methods.align_landmarks._landmark import AffineFitResult

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

#: What an alignment can be driven by. Also the slot names on :class:`AlignMethod`, and
#: the modalities :mod:`squidpy.experimental.tl._align._paths` resolves a path to.
Modality = Literal["obs", "images", "landmarks"]
MODALITIES: tuple[Modality, ...] = get_args(Modality)


@runtime_checkable
class AlignResult(Protocol):
    """A fitted alignment that maps ``(N, 2)`` ``(x, y)`` points into the reference frame.

    This is the only thing the public ``align`` function requires of an estimator's
    result, so ``out=None`` is agnostic to the method that produced it.
    """

    def transform(self, points: npt.ArrayLike, /) -> NDArrayA:
        """Map an ``(N, 2)`` ``(x, y)`` array into the reference frame."""
        ...


class AlignObsFn(Protocol):
    """Calling convention for the ``obs`` slot.

    Two point clouds in (by keyword as ``ref`` / ``query``, so the direction can never be
    silently swapped), one :class:`AlignResult` out.
    """

    def __call__(self, ref: npt.ArrayLike, query: npt.ArrayLike, **kwargs: Any) -> AlignResult: ...


class AlignImagesFn(Protocol):
    """Calling convention for the ``images`` slot: two ``(c, y, x)`` rasters in."""

    def __call__(self, ref: npt.ArrayLike, query: npt.ArrayLike, **kwargs: Any) -> AlignResult: ...


class AlignLandmarksFn(Protocol):
    """Calling convention for the ``landmarks`` slot: paired correspondences in, affine out.

    Unlike the other two slots, ``ref`` / ``query`` here are *not* the data being aligned
    -- they are ``(N, 2)`` landmark arrays annotating it, matched by row order.
    """

    def __call__(
        self,
        ref: npt.ArrayLike,
        query: npt.ArrayLike,
        *,
        source_cs: str | None = ...,
        target_cs: str | None = ...,
    ) -> AffineFitResult: ...


@dataclass(frozen=True, slots=True)
class AlignMethod:
    """One named alignment method and the modalities it implements."""

    name: str
    obs: AlignObsFn | None = None
    images: AlignImagesFn | None = None
    landmarks: AlignLandmarksFn | None = None

    def supports(self) -> tuple[Modality, ...]:
        """Modalities this method implements."""
        return tuple(m for m in MODALITIES if getattr(self, m) is not None)

    def implementation(self, modality: Modality) -> Callable[..., Any]:
        """Return the estimator for ``modality``, or explain what this method does instead."""
        fn = getattr(self, modality)
        if fn is None:
            supported = ", ".join(self.supports())
            raise ValueError(f"Method {self.name!r} does not support {modality} alignment. It supports: {supported}.")
        return fn


class Registry:
    """A ``name -> AlignMethod`` registry for one *family* of methods.

    Registration is per slot, so a method that gains a modality later does not have to be
    declared in one place::

        @ALIGN.register("stalign", "obs", requires=("jax",))
        def fit_stalign_obs(...): ...

        @ALIGN.register("stalign", "images", requires=("jax",))
        def fit_stalign_image(...): ...
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, AlignMethod] = {}

    def register(
        self,
        key: str,
        modality: Modality,
        *,
        requires: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator registering a function as ``key``'s ``modality`` estimator."""
        if modality not in MODALITIES:
            raise ValueError(f"Unknown modality {modality!r}. Expected one of {', '.join(MODALITIES)}.")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            existing = self._registry.get(key, AlignMethod(name=key))
            if getattr(existing, modality) is not None:
                raise ValueError(f"Method {key!r} already has a {modality} estimator in the {self.name!r} registry.")

            registered = func
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

                registered = wrapped

            self._registry[key] = replace(existing, **{modality: registered})
            return registered

        return decorator

    def get(self, key: str) -> AlignMethod:
        """Return the method registered under ``key``."""
        try:
            return self._registry[key]
        except KeyError:
            raise ValueError(f"Unknown {self.name} method {key!r}. Available: {sorted(self._registry)}.") from None

    def keys(self) -> tuple[str, ...]:
        """Return every registered method name."""
        return tuple(self._registry)

    def supporting(self, modality: Modality) -> tuple[str, ...]:
        """Return the method names implementing ``modality``, for docs and error messages."""
        return tuple(sorted(key for key, m in self._registry.items() if getattr(m, modality) is not None))


#: Alignment estimators, keyed by method name. Each declares which of ``obs`` / ``images``
#: / ``landmarks`` it can align. Consumed by ``squidpy.experimental.tl.align``.
ALIGN: Registry = Registry("align")
