"""Moscot backend stub.

Moscot only exposes ``align_obs``; image alignment is not a moscot use case.
The dispatch layer rejects ``flavour='moscot'`` for :func:`align_images`
before ever reaching this file, so the ``align_images`` method below is here
purely to satisfy the :class:`AlignBackend` Protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from squidpy.experimental.tl._align._types import AlignPair, AlignResult


class MoscotBackend:
    name = "moscot"
    requires_jax = True

    def align_obs(
        self,
        pair: AlignPair,
        *,
        device: Literal["cpu", "gpu"] | None = None,
        **kwargs: Any,
    ) -> AlignResult:
        from squidpy.experimental.tl._align._jax import require_jax

        require_jax(device)
        raise NotImplementedError(
            "moscot backend `align_obs`: TODO. Skeleton landed; the moscot "
            "solver will replace this body in a follow-up PR."
        )

    def align_images(
        self,
        pair: AlignPair,
        *,
        device: Literal["cpu", "gpu"] | None = None,
        **kwargs: Any,
    ) -> AlignResult:
        raise NotImplementedError("moscot does not implement image alignment; use `flavour='stalign'`.")
