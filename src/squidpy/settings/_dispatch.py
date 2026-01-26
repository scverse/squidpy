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

_GPU_NOTE_TEMPLATE = """
.. note::
    This function supports GPU acceleration via :doc:`rapids_singlecell <rapids_singlecell:index>`.
    See :func:`rapids_singlecell.squidpy_gpu.{func_name}` for the GPU implementation.
"""


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


def _inject_gpu_note(doc: str | None, func_name: str) -> str | None:
    """Inject GPU note into docstring after the first paragraph."""
    if doc is None:
        return None

    gpu_note = _GPU_NOTE_TEMPLATE.format(func_name=func_name)

    # Find "Parameters\n----------" and insert note before it
    match = re.search(r"(\n\s*Parameters\s*\n\s*-+)", doc)
    if match:
        insert_pos = match.start()
        return doc[:insert_pos] + "\n" + gpu_note + doc[insert_pos:]

    # Fallback: append at the end
    return doc + "\n" + gpu_note


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
