"""GPU dispatch decorator for squidpy."""

from __future__ import annotations

import functools
import inspect
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


def gpu_dispatch(gpu_func_name: str | None = None) -> Callable[[F], F]:
    """Decorator to dispatch to GPU adapter when device='gpu'."""

    def decorator(func: F) -> F:
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

                func_name = gpu_func_name if gpu_func_name is not None else f"{func.__name__}_gpu"
                gpu_adapter = getattr(_gpu, func_name)
                return gpu_adapter(**all_args, **extra_kwargs)

            return func(**all_args, **extra_kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
