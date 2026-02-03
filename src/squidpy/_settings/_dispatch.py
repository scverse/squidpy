"""GPU dispatch decorator for squidpy."""

from __future__ import annotations

import functools
import importlib
import re
from collections.abc import Callable
from typing import Any, TypeVar

from squidpy._settings._settings import GPU_UNAVAILABLE_MSG, settings

__all__ = ["gpu_dispatch"]

F = TypeVar("F", bound=Callable[..., Any])


def _get_effective_device() -> str:
    """Get effective device from settings, resolving 'auto'."""
    device = settings.device
    if device == "cpu":
        return "cpu"
    if device == "gpu":
        if not settings.gpu_available:
            raise RuntimeError(GPU_UNAVAILABLE_MSG)
        return "gpu"
    # auto
    return "gpu" if settings.gpu_available else "cpu"


def _make_gpu_note(func_name: str, gpu_module: str, indent: str = "") -> str:
    lines = [
        ".. note::",
        "    This function supports GPU acceleration via :doc:`rapids_singlecell <rapids_singlecell:index>`.",
        f"    See :func:`{gpu_module}.{func_name}` for the GPU implementation.",
    ]
    return "\n".join(indent + line for line in lines)


def _inject_gpu_note(doc: str | None, func_name: str, gpu_module: str) -> str | None:
    """Inject GPU note into docstring before the Parameters section."""
    if doc is None:
        return None

    # Find "Parameters\n    ----------" and capture the indentation (spaces only, not newline)
    match = re.search(r"\n([ \t]*)Parameters\s*\n\s*-+", doc)
    if match:
        indent = match.group(1)  # Capture only the spaces/tabs before Parameters
        gpu_note = _make_gpu_note(func_name, gpu_module, indent)
        insert_pos = match.start()
        return doc[:insert_pos] + "\n\n" + gpu_note + "\n" + doc[insert_pos:]

    # Fallback: append at the end
    return doc + "\n\n" + _make_gpu_note(func_name, gpu_module)


# Cache for GPU functions
_GPU_FUNC_CACHE: dict[tuple[str, str], Callable[..., Any]] = {}


def gpu_dispatch(
    gpu_module: str = "rapids_singlecell.gr",
    gpu_func_name: str | None = None,
) -> Callable[[F], F]:
    """Decorator to dispatch to GPU implementation based on settings.device.

    When device is 'gpu', calls the GPU implementation from the specified module.
    The `device_kwargs` parameter (if present) is passed to the GPU function.

    Parameters
    ----------
    gpu_module
        Module path containing the GPU implementation.
    gpu_func_name
        Name of GPU function. Defaults to same name as decorated function.
    """

    def decorator(func: F) -> F:
        func_name = func.__name__
        _gpu_func_name = gpu_func_name or func_name

        # Inject GPU note into docstring
        func.__doc__ = _inject_gpu_note(func.__doc__, _gpu_func_name, gpu_module)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_device = _get_effective_device()

            if resolved_device == "cpu":
                # CPU path - just remove device_kwargs if present and call CPU func
                kwargs.pop("device_kwargs", None)
                return func(*args, **kwargs)

            # GPU path
            key = (gpu_module, _gpu_func_name)
            if key not in _GPU_FUNC_CACHE:
                try:
                    module = importlib.import_module(gpu_module)
                    _GPU_FUNC_CACHE[key] = getattr(module, _gpu_func_name)
                except (ImportError, AttributeError):
                    # GPU module not available, fall back to CPU
                    kwargs.pop("device_kwargs", None)
                    return func(*args, **kwargs)

            gpu_func = _GPU_FUNC_CACHE[key]

            # Extract device_kwargs and merge into kwargs for GPU call
            device_kwargs = kwargs.pop("device_kwargs", None) or {}
            kwargs.update(device_kwargs)

            return gpu_func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
