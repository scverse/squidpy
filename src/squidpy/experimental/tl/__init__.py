from __future__ import annotations

from importlib import import_module

__all__ = ["stalign", "stalign_tools"]


def __getattr__(name: str):
    # Module-level lazy imports are a common scientific Python pattern for
    # optional or heavy dependencies.
    if name == "stalign":
        return import_module("squidpy.experimental.tl._stalign").stalign
    if name == "stalign_tools":
        return import_module("squidpy.experimental.tl.stalign_tools")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
