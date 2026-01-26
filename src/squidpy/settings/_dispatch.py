"""GPU dispatch decorator for squidpy."""

from __future__ import annotations

import functools
import importlib
import inspect
import re
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from squidpy.gr._gpu import GPU_PARAM_REGISTRY, check_cpu_params, check_gpu_params
from squidpy.settings._settings import settings

__all__ = ["gpu_dispatch"]

F = TypeVar("F", bound=Callable[..., Any])


def _resolve_device(device: Literal["auto", "cpu", "gpu"] | None) -> Literal["cpu", "gpu"]:
    """Resolve device arg to 'cpu' or 'gpu'."""
    if device is None:
        device = settings.device
    if device == "cpu":
        return "cpu"
    if device == "gpu":
        if not settings.gpu_available():
            raise RuntimeError("GPU unavailable. Install with: pip install squidpy[gpu-cuda12]")
        return "gpu"
    # auto
    return "gpu" if settings.gpu_available() else "cpu"


def _make_gpu_note(func_name: str, indent: str = "") -> str:
    lines = [
        ".. note::",
        "    This function supports GPU acceleration via :doc:`rapids_singlecell <rapids_singlecell:index>`.",
        f"    See :func:`rapids_singlecell.gr.{func_name}` for the GPU implementation.",
    ]
    return "\n".join(indent + line for line in lines)


def _inject_gpu_note(doc: str | None, func_name: str) -> str | None:
    """Inject GPU note into docstring before the Parameters section."""
    if doc is None:
        return None

    # Find "Parameters\n    ----------" and capture the indentation (spaces only, not newline)
    match = re.search(r"\n([ \t]*)Parameters\s*\n\s*-+", doc)
    if match:
        indent = match.group(1)  # Capture only the spaces/tabs before Parameters
        gpu_note = _make_gpu_note(func_name, indent)
        insert_pos = match.start()
        return doc[:insert_pos] + "\n\n" + gpu_note + "\n" + doc[insert_pos:]

    # Fallback: append at the end
    return doc + "\n\n" + _make_gpu_note(func_name)


def gpu_dispatch(
    gpu_module: str = "rapids_singlecell.gr",
    gpu_func_name: str | None = None,
) -> Callable[[F], F]:
    """Decorator to dispatch to GPU implementation when device='gpu'.

    Uses the GPU_PARAM_REGISTRY from squidpy.gr._gpu to:
    - Warn about CPU-only parameters that differ from defaults, then filter them out
    - Filter out GPU-only parameters on CPU (they only affect GPU)

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

        # Inject GPU note into docstring
        func.__doc__ = _inject_gpu_note(func.__doc__, func_name)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            try:
                bound = sig.bind(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)

            bound.apply_defaults()
            all_args = dict(bound.arguments)

            device = all_args.pop("device", None)

            # Handle **kwargs: unpack instead of passing as kwargs=dict
            extra_kwargs = all_args.pop("kwargs", {})

            # Get registry for this function
            registry = GPU_PARAM_REGISTRY.get(func_name, {"cpu_only": {}, "gpu_only": {}})

            if _resolve_device(device) == "gpu":
                # Collect CPU-only param values and check them (warn if non-default)
                cpu_only_values = {k: all_args.pop(k) for k in list(all_args) if k in registry["cpu_only"]}
                cpu_only_values.update(
                    {k: extra_kwargs.pop(k) for k in list(extra_kwargs) if k in registry["cpu_only"]}
                )

                check_gpu_params(func_name, **cpu_only_values)

                # Import and call GPU function
                module = importlib.import_module(gpu_module)
                gpu_func = getattr(module, gpu_func_name or func_name)

                return gpu_func(**all_args, **extra_kwargs)

            # CPU path: check gpu_only params (error if non-default), then filter them out
            gpu_only_values = {k: all_args.pop(k) for k in list(all_args) if k in registry["gpu_only"]}
            gpu_only_values.update({k: extra_kwargs.pop(k) for k in list(extra_kwargs) if k in registry["gpu_only"]})

            check_cpu_params(func_name, **gpu_only_values)

            return func(**all_args, **extra_kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
