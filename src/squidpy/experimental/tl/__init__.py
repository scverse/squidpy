from __future__ import annotations

from typing import TYPE_CHECKING, Any

from squidpy.experimental.tl._align import align_by_landmarks, align_obs

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
    "align_by_landmarks",
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
        try:
            from squidpy.experimental.tl import stalign_tools as _tools
        except ModuleNotFoundError as e:
            if e.name == "jax":
                raise ImportError(
                    'STalign requires the optional dependency `jax`. Install it with `pip install "squidpy[jax]"`.'
                ) from e
            raise
        return getattr(_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
