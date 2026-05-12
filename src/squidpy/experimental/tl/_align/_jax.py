"""Lazy JAX import guard for JAX-backed alignment backends.

JAX is an optional dependency.  Importing this module is cheap; calling
:func:`require_jax` is what actually pulls JAX in, and only the
JAX-backed backends do so on first call.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = 'JAX is required for the requested align_* flavour. Install with `pip install "squidpy[jax]"`.'


def require_jax() -> Any:
    """Import JAX lazily and return the module.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    try:
        import jax
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e

    return jax
