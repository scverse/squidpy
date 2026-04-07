from __future__ import annotations

from importlib import import_module

__all__ = ["stalign", "stalign_tools"]


def _import_stalign_module(module_name: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == "jax":
            raise ImportError(
                'STalign requires the optional dependency `jax`. Install it with `pip install "squidpy[jax]"`.'
            ) from e
        raise


def __getattr__(name: str):
    # Module-level lazy imports are a common scientific Python pattern for
    # optional or heavy dependencies.
    if name == "stalign":
        return _import_stalign_module("squidpy.experimental.tl._stalign").stalign
    if name == "stalign_tools":
        return _import_stalign_module("squidpy.experimental.tl.stalign_tools")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
