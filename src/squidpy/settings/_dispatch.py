"""GPU dispatch decorator for squidpy."""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Callable
from typing import Any, Literal, TypeVar

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


def gpu_dispatch(gpu_func_name: str | None = None) -> Callable[[F], F]:
    """Decorator to dispatch to GPU adapter when device='gpu'.

    Also injects a GPU note into the function's docstring.
    """

    def decorator(func: F) -> F:
        # Inject GPU note into docstring
        func.__doc__ = _inject_gpu_note(func.__doc__, func.__name__)

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

            if _resolve_device(device) == "gpu":
                from squidpy.gr import _gpu

                adapter_name = gpu_func_name if gpu_func_name is not None else f"{func.__name__}_gpu"
                gpu_adapter = getattr(_gpu, adapter_name)
                return gpu_adapter(**all_args, **extra_kwargs)

            return func(**all_args, **extra_kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
