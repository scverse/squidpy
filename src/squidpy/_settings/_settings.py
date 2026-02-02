"""Squidpy global settings."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Literal, get_args

__all__ = ["settings", "DeviceType"]

DeviceType = Literal["auto", "cpu", "gpu"]
_device_var: ContextVar[DeviceType] = ContextVar("device", default="auto")


def _check_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    try:
        import rapids_singlecell  # noqa: F401

        return True
    except ImportError:
        return False


class SqSettings:
    """Global configuration for squidpy."""

    def __init__(self) -> None:
        self.gpu_available: bool = _check_gpu_available()

    @property
    def device(self) -> DeviceType:
        """Compute device: ``'auto'``, ``'cpu'``, or ``'gpu'``."""
        return _device_var.get()

    @device.setter
    def device(self, value: DeviceType) -> None:
        if value not in get_args(DeviceType):
            raise ValueError(f"device must be one of {get_args(DeviceType)}, got {value!r}")
        if value == "gpu" and not self.gpu_available:
            raise RuntimeError("GPU unavailable. Install: pip install squidpy[gpu-cuda12] or with [gpu-cuda11] for CUDA 11 support.")
        _device_var.set(value)


settings = SqSettings()
