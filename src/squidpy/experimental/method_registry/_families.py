"""Estimator family registries and the structural contracts they advertise.

The :class:`~typing.Protocol` types are what the public API and the registries
are typed against, so the orchestration layer never names a concrete estimator
result (e.g. ``StalignResult``). A new estimator only has to satisfy
:class:`AlignResult` -- a ``transform`` that maps points into the reference
frame -- to plug into :func:`squidpy.experimental.tl.align`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy.typing as npt

from squidpy._utils import NDArrayA
from squidpy.experimental.method_registry._registry import Registry

if TYPE_CHECKING:
    from squidpy.experimental.method_registry.align_landmarks._landmark import AffineFitResult

__all__ = ["AlignResult", "AlignSamplesFn", "AlignLandmarksFn", "ALIGN_SAMPLES", "ALIGN_LANDMARKS"]


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
