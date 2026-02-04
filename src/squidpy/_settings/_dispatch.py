"""GPU dispatch decorator for squidpy."""

from __future__ import annotations

import functools
import importlib
import re
from collections.abc import Callable, Mapping
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
        indent = match.group(1)
        gpu_note = _make_gpu_note(func_name, gpu_module, indent)
        insert_pos = match.start()
        return doc[:insert_pos] + "\n\n" + gpu_note + "\n" + doc[insert_pos:]

    # Fallback: append at the end
    return doc + "\n\n" + _make_gpu_note(func_name, gpu_module)


@functools.cache
def _get_gpu_func(gpu_module: str, func_name: str) -> Callable[..., Any]:
    """Get GPU function from module, with caching.


    Raises
    ------
    ImportError
        If the GPU module cannot be imported.
    AttributeError
        If the function does not exist in the GPU module.
    """
    module = importlib.import_module(gpu_module)
    return getattr(module, func_name)


def gpu_dispatch(
    gpu_module: str = "rapids_singlecell.gr",
    validate_args: Mapping[str, Callable[[Any], None]] | None = None,
) -> Callable[[F], F]:
    """Decorator to dispatch to GPU implementation based on settings.device.

    When device is 'gpu', calls the GPU implementation from the specified module.
    The ``device_kwargs`` parameter from the decorated function is merged into the
    call for GPU-specific options. Arguments with ``None`` values are filtered out
    to let the GPU function use its defaults.

    Parameters
    ----------
    gpu_module
        Module path containing the GPU implementation.
    validate_args
        Mapping of parameter names to validation functions. Each validator is called
        with the parameter value before GPU dispatch and should raise ValueError
        if the value is not supported on GPU. Validated arguments are removed from
        kwargs before calling the GPU function. Only called when dispatching to GPU.
    """
    _validate_args = validate_args or {}

    def decorator(func: F) -> F:
        func_name = func.__name__

        func.__doc__ = _inject_gpu_note(func.__doc__, func_name, gpu_module)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_device = _get_effective_device()

            if resolved_device == "cpu":
                kwargs.pop("device_kwargs", None)
                return func(*args, **kwargs)

            # GPU path
            # run validators and remove validated args
            for param_name, validator in _validate_args.items():
                if param_name in kwargs:
                    validator(kwargs[param_name])
                    kwargs.pop(param_name)

            # get GPU function
            gpu_func = _get_gpu_func(gpu_module, func_name)

            # merge device_kwargs and call GPU function
            device_kwargs = kwargs.pop("device_kwargs", None) or {}
            kwargs.update(device_kwargs)

            # filter out None values to let GPU function use its defaults
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            return gpu_func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
