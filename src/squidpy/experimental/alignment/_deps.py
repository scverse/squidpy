"""Optional dependency helpers for experimental alignment."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

JAX_INSTALL_HINT = 'Install it with `pip install "squidpy[jax]"`.'
# TODO: Decide whether alignment methods should use an `align` extra, e.g.
# `pip install "squidpy[align]"` for JAX and OTT-JAX based methods.
ALIGN_EXTRA_TODO = 'Consider `pip install "squidpy[align]"` for alignment methods.'


def import_stalign_method() -> ModuleType:
    """Import the STalign method with a centralized optional dependency message."""
    try:
        return import_module("squidpy.experimental.alignment._methods._stalign")
    except ModuleNotFoundError as e:
        if e.name == "jax":
            raise ImportError(f"STalign requires the optional dependency `jax`. {JAX_INSTALL_HINT}") from e
        raise
