from __future__ import annotations

from typing import TYPE_CHECKING, Any

from squidpy.experimental.tl._align import align_by_landmarks, align_obs

if TYPE_CHECKING:
    from squidpy.experimental.tl._align._backends._stalign_tools import (
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
    """Lazy access to the JAX-only STalign config dataclasses.

    Importing :mod:`squidpy.experimental.tl._align._backends._stalign_tools` pulls in
    :mod:`jax` at module-load time, so we defer the import until the first
    attribute access.  This preserves the lazy-import contract pinned by
    ``test_optional_deps_not_imported_at_import_time``.
    """
    if name in _STALIGN_REEXPORTS:
        try:
            from squidpy.experimental.tl._align._backends import _stalign_tools as _tools
        except ModuleNotFoundError as e:
            if e.name == "jax":
                raise ImportError(
                    'STalign requires the optional dependency `jax`. Install it with `pip install "squidpy[jax]"`.'
                ) from e
            raise
        return getattr(_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
