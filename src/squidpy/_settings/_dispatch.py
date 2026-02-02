"""GPU dispatch decorator for squidpy."""

from __future__ import annotations

import functools
import importlib
import inspect
import re
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from squidpy._settings._settings import GPU_UNAVAILABLE_MSG, settings
from squidpy.gr._gpu import GPU_PARAM_REGISTRY, check_exclusive_params, get_or_create_registry_entry

__all__ = ["gpu_dispatch"]

F = TypeVar("F", bound=Callable[..., Any])


def _resolve_device(device: Literal["auto", "cpu", "gpu"] | None) -> Literal["cpu", "gpu"]:
    """Resolve device arg to 'cpu' or 'gpu'."""
    if device is None:
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


def gpu_dispatch(
    gpu_module: str = "rapids_singlecell.gr",
    gpu_func_name: str | None = None,
) -> Callable[[F], F]:
    """Decorator to dispatch to GPU implementation when device='gpu'.

    Automatically determines CPU-only and GPU-only parameters by comparing
    function signatures. Errors if user explicitly provides a value for
    an exclusive parameter on the wrong device.

    Also injects a GPU note into the function's docstring.

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
            sig = inspect.signature(func)
            try:
                bound = sig.bind(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)

            # Track what user actually provided (before defaults)
            user_provided = set(bound.arguments.keys())

            bound.apply_defaults()
            all_args = dict(bound.arguments)

            device = all_args.pop("device", None)
            resolved_device = _resolve_device(device)

            # Get or create registry entry (populated once per function)
            key = (gpu_module, _gpu_func_name)
            if key not in GPU_PARAM_REGISTRY:
                try:
                    module = importlib.import_module(gpu_module)
                    gpu_func = getattr(module, _gpu_func_name)
                    get_or_create_registry_entry(gpu_module, _gpu_func_name, func, gpu_func)
                except (ImportError, AttributeError):
                    # GPU module not available, just run CPU function
                    return func(**all_args)

            entry = GPU_PARAM_REGISTRY[key]

            if resolved_device == "gpu":
                # Check if user explicitly provided any CPU-only params
                cpu_only_names = set(entry.cpu_only_params.keys())
                user_provided_cpu_only = user_provided & cpu_only_names
                check_exclusive_params(func_name, user_provided_cpu_only, all_args, "gpu", entry)

                # Remove CPU-only params before calling GPU func
                for k in cpu_only_names:
                    all_args.pop(k, None)

                return entry.gpu_func(**all_args)

            # CPU path
            # Check if user explicitly provided any GPU-only params
            gpu_only_names = set(entry.gpu_only_params.keys())
            user_provided_gpu_only = user_provided & gpu_only_names
            check_exclusive_params(func_name, user_provided_gpu_only, all_args, "cpu", entry)

            # Remove GPU-only params before calling CPU func
            for k in gpu_only_names:
                all_args.pop(k, None)

            return func(**all_args)

        return wrapper  # type: ignore[return-value]

    return decorator
