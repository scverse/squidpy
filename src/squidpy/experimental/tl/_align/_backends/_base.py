"""Backend Protocol shared by every alignment flavour."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from squidpy.experimental.tl._align._types import AlignPair, AlignResult


@runtime_checkable
class AlignBackend(Protocol):
    """Minimal contract every alignment backend must satisfy.

    Backends are constructed cheaply (no heavy imports in ``__init__``) and
    only pull in their optional dependencies on the first call into ``align_obs``
    or ``align_images``.  ``requires_jax`` advertises whether the backend
    needs JAX so callers / dispatch can short-circuit.
    """

    name: str
    requires_jax: bool

    def align_obs(
        self,
        pair: AlignPair,
        **kwargs: Any,
    ) -> AlignResult: ...

    def align_images(
        self,
        pair: AlignPair,
        **kwargs: Any,
    ) -> AlignResult: ...
