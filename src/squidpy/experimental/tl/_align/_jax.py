"""Lazy JAX import + device selection for JAX-backed alignment backends.

JAX is an optional dependency.  Importing this module is cheap; calling
:func:`require_jax` is what actually pulls JAX in, and only the
JAX-backed backends do so on first call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    Device = Any  # jax.Device, but importing it eagerly defeats the purpose


_INSTALL_HINT = (
    "JAX is required for the requested align_* flavour. "
    "Install with `pip install jax` (CPU) or follow the JAX install guide for GPU."
)


def require_jax(device: Literal["cpu", "gpu"] | None = None) -> tuple[Any, Any]:
    """Import JAX lazily and return ``(jax, device)``.

    Parameters
    ----------
    device
        ``"cpu"``/``"gpu"`` to force a platform, or ``None`` to use whatever
        JAX picks as the default.

    Returns
    -------
    jax_module
        The imported :mod:`jax` module.
    device
        A :class:`jax.Device` of the requested platform.

    Raises
    ------
    ImportError
        If JAX is not installed.
    RuntimeError
        If the requested device platform is not available on this host.
    """
    try:
        import jax
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e

    if device is None:
        return jax, jax.devices()[0]

    matching = [d for d in jax.devices() if d.platform == device]
    if not matching:
        raise RuntimeError(f"No JAX device of kind {device!r} available; have {[d.platform for d in jax.devices()]}.")
    return jax, matching[0]
