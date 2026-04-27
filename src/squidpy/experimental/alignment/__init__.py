"""Experimental alignment tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from squidpy.experimental.alignment._api import align_obs
from squidpy.experimental.alignment._deps import import_stalign_tools

if TYPE_CHECKING:
    from squidpy.experimental.tl.stalign_tools import (
        STalignConfig,
        STalignPreprocessConfig,
        STalignPreprocessResult,
        STalignRegistrationConfig,
        STalignResult,
    )

__all__ = [
    "STalignConfig",
    "STalignPreprocessConfig",
    "STalignPreprocessResult",
    "STalignRegistrationConfig",
    "STalignResult",
    "align_obs",
]

_STALIGN_REEXPORTS = frozenset(
    {
        "STalignConfig",
        "STalignPreprocessConfig",
        "STalignPreprocessResult",
        "STalignRegistrationConfig",
        "STalignResult",
    }
)


def __getattr__(name: str) -> Any:
    """Lazy access to JAX-only STalign config dataclasses."""
    if name in _STALIGN_REEXPORTS:
        return getattr(import_stalign_tools(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
