"""GPU parameter handling for squidpy functions with GPU acceleration.

Automatically determines CPU-only and GPU-only parameters by introspecting function signatures.
Only special cases (custom validators) need explicit registry entries.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["SPECIAL_PARAM_REGISTRY", "check_exclusive_params", "get_exclusive_params"]


@dataclass
class GpuParamSpec:
    """Specification for a parameter with custom validation."""

    validate_fn: Callable[[Any], str | None]


def _attr_validator(value: Any) -> str | None:
    """Validator for attr param - error if not 'X' on GPU."""
    if value == "X":
        return None
    return f"attr={value!r} is not supported on GPU. Set device='cpu' to use other attributes."


# Minimal registry: only for params that need custom validators
# Format: {func_name: {"cpu_only": {param: GpuParamSpec}, "gpu_only": {param: GpuParamSpec}}}
SPECIAL_PARAM_REGISTRY: dict[str, dict[str, dict[str, GpuParamSpec]]] = {
    "spatial_autocorr": {
        "cpu_only": {
            "attr": GpuParamSpec(validate_fn=_attr_validator),
        },
        "gpu_only": {},
    },
}


def get_exclusive_params(cpu_func: Callable[..., Any], gpu_func: Callable[..., Any]) -> tuple[set[str], set[str]]:
    """Get CPU-only and GPU-only params by comparing function signatures.

    Parameters
    ----------
    cpu_func
        The CPU implementation function.
    gpu_func
        The GPU implementation function.

    Returns
    -------
    Tuple of (cpu_only_params, gpu_only_params) as sets of param names.
    """
    cpu_sig = inspect.signature(cpu_func)
    gpu_sig = inspect.signature(gpu_func)

    cpu_params = set(cpu_sig.parameters.keys())
    gpu_params = set(gpu_sig.parameters.keys())

    # CPU-only: in CPU sig but not in GPU sig (excluding 'device' which is handled separately)
    cpu_only = cpu_params - gpu_params - {"device"}

    # GPU-only: in GPU sig but not in CPU sig
    gpu_only = gpu_params - cpu_params

    return cpu_only, gpu_only


def check_exclusive_params(
    func_name: str,
    user_provided_exclusive: set[str],
    param_values: dict[str, Any],
    target_device: str,
) -> None:
    """Check exclusive params, raise error if user explicitly provided any.

    Parameters
    ----------
    func_name
        Name of the function (for registry lookup).
    user_provided_exclusive
        Set of param names that user explicitly provided AND are exclusive to other device.
    param_values
        All argument values (for error messages and custom validators).
    target_device
        The device being used ('cpu' or 'gpu').

    Raises
    ------
    ValueError
        If user explicitly provided an exclusive parameter.
    """
    other_device = "gpu" if target_device == "cpu" else "cpu"
    registry_key = "gpu_only" if target_device == "cpu" else "cpu_only"
    registry = SPECIAL_PARAM_REGISTRY.get(func_name, {"cpu_only": {}, "gpu_only": {}})

    for name in user_provided_exclusive:
        value = param_values.get(name)

        # Check special validate_fn first (they may allow certain values)
        if name in registry[registry_key]:
            spec = registry[registry_key][name]
            msg = spec.validate_fn(value)
            if msg:
                raise ValueError(msg)
            continue

        # User explicitly provided an exclusive param - error
        msg = f"{name}={value!r} is only supported on {other_device.upper()}. Use device={other_device!r} or remove this argument."
        raise ValueError(msg)
