"""Backend dispatch for the alignment skeleton.

Imports of individual backends happen *inside* the dispatch branches so that
``import squidpy.experimental.tl`` never pulls in ``stalign``, ``moscot``, or
``jax`` transitively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from squidpy.experimental.tl._align._backends._base import AlignBackend


def get_backend(flavour: str) -> AlignBackend:
    """Return a backend instance for the requested ``flavour``."""
    if flavour == "stalign":
        from squidpy.experimental.tl._align._backends._stalign import StAlignBackend

        return StAlignBackend()
    if flavour == "moscot":
        from squidpy.experimental.tl._align._backends._moscot import MoscotBackend

        return MoscotBackend()
    raise ValueError(f"Unknown alignment flavour {flavour!r}; expected 'stalign' or 'moscot'.")
