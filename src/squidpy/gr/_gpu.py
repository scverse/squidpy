"""GPU parameter registry for squidpy.gr functions.

Defines which parameters are CPU-only (ignored on GPU) and GPU-only (ignored on CPU).
The gpu_dispatch decorator uses this registry to automatically handle parameter filtering.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["GPU_PARAM_REGISTRY", "GpuParamSpec", "check_gpu_params", "check_cpu_params"]


@dataclass
class GpuParamSpec:
    """Specification for a parameter's GPU compatibility."""

    default: Any
    message: str | None = None
    validator: Callable[[Any], str | None] | None = None


def _attr_validator(value: Any) -> str | None:
    """Special validator for attr param - only warn if not 'X'."""
    if value == "X":
        return None
    return f"attr={value!r} is not supported on GPU, using attr='X'. Set device='cpu' to use other attributes."


# Common CPU-only param specs (reusable)
_PARALLELIZE: dict[str, GpuParamSpec] = {
    "n_jobs": GpuParamSpec(None),
    "backend": GpuParamSpec("loky"),
    "show_progress_bar": GpuParamSpec(True),
}
_SEED: dict[str, GpuParamSpec] = {"seed": GpuParamSpec(None)}

# Registry: {func_name: {"cpu_only": {...}, "gpu_only": {...}}}
# - cpu_only: parameters ignored on GPU (warn if non-default, then filter out)
# - gpu_only: parameters ignored on CPU (error if non-default, pass through to GPU)
GPU_PARAM_REGISTRY: dict[str, dict[str, dict[str, GpuParamSpec]]] = {
    "spatial_autocorr": {
        "cpu_only": {
            "attr": GpuParamSpec("X", validator=_attr_validator),
            **_SEED,
            **_PARALLELIZE,
        },
        "gpu_only": {
            "use_sparse": GpuParamSpec(True),
        },
    },
    "co_occurrence": {
        "cpu_only": {
            "n_splits": GpuParamSpec(None),
            **_PARALLELIZE,
        },
        "gpu_only": {},
    },
    "ligrec": {
        "cpu_only": {
            "clusters": GpuParamSpec(None),
            "numba_parallel": GpuParamSpec(None),
            "transmitter_params": GpuParamSpec(None),
            "receiver_params": GpuParamSpec(None),
            "interactions_params": GpuParamSpec(None),
            "alpha": GpuParamSpec(0.05),
            **_SEED,
            **_PARALLELIZE,
        },
        "gpu_only": {},
    },
}


def check_gpu_params(func_name: str, **cpu_only_values: Any) -> None:
    """Check CPU-only params on GPU, raise error if non-default.

    Parameters
    ----------
    func_name
        Name of the function in GPU_PARAM_REGISTRY.
    **cpu_only_values
        CPU-only parameter values to check.

    Raises
    ------
    ValueError
        If a CPU-only parameter has a non-default value on GPU.
    """
    registry = GPU_PARAM_REGISTRY.get(func_name, {"cpu_only": {}, "gpu_only": {}})

    for name, spec in registry["cpu_only"].items():
        if name not in cpu_only_values:
            continue
        value = cpu_only_values[name]

        # Use custom validator if provided, else default behavior
        if spec.validator:
            msg = spec.validator(value)
        elif value != spec.default:
            msg = spec.message or f"{name}={value!r} is only supported on CPU. Use device='cpu' or remove this argument."
        else:
            msg = None

        if msg:
            raise ValueError(msg.format(name=name, value=value))


def check_cpu_params(func_name: str, **gpu_only_values: Any) -> None:
    """Check GPU-only params on CPU, raise error if non-default.

    Parameters
    ----------
    func_name
        Name of the function in GPU_PARAM_REGISTRY.
    **gpu_only_values
        GPU-only parameter values to check.

    Raises
    ------
    ValueError
        If a GPU-only parameter has a non-default value on CPU.
    """
    registry = GPU_PARAM_REGISTRY.get(func_name, {"cpu_only": {}, "gpu_only": {}})

    for name, spec in registry["gpu_only"].items():
        if name not in gpu_only_values:
            continue
        value = gpu_only_values[name]

        if value != spec.default:
            msg = f"{name}={value!r} is only supported on GPU. Use device='gpu' or remove this argument."
            raise ValueError(msg)
