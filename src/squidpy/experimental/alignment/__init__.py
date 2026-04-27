"""Experimental alignment tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from squidpy.experimental.alignment._api import align_obs
from squidpy.experimental.alignment._deps import import_stalign_method

if TYPE_CHECKING:
    from squidpy.experimental.alignment._methods._stalign import (
        STalignConfig,
        STalignPreprocessConfig,
        STalignPreprocessResult,
        STalignRegistrationConfig,
        STalignResult,
        stalign_points,
        stalign_preprocess,
        transform_points,
    )

__all__ = [
    "STalignConfig",
    "STalignPreprocessConfig",
    "STalignPreprocessResult",
    "STalignRegistrationConfig",
    "STalignResult",
    "align_obs",
    "stalign_points",
    "stalign_preprocess",
    "transform_points",
]

_STALIGN_REEXPORTS = frozenset(
    {
        "STalignConfig",
        "STalignPreprocessConfig",
        "STalignPreprocessResult",
        "STalignRegistrationConfig",
        "STalignResult",
        "stalign_points",
        "stalign_preprocess",
        "transform_points",
    }
)


def __getattr__(name: str) -> Any:
    """Lazy access to JAX-only STalign config dataclasses."""
    if name in _STALIGN_REEXPORTS:
        # TODO maybe this needs to be removed
        return getattr(import_stalign_method(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# TODO maybe this needs to be removed
def __dir__() -> list[str]:
    return sorted(__all__)
