from __future__ import annotations

from contextvars import ContextVar
from typing import Literal, get_args

__all__ = ["settings", "DeviceType"]

DeviceType = Literal["auto", "cpu", "gpu"]

_device_var: ContextVar[DeviceType] = ContextVar("device", default="auto")


class _SqSettings:
    """Global settings for squidpy."""

    @property
    def device(self) -> DeviceType:
        """Current compute device setting."""
        return _device_var.get()

    @device.setter
    def device(self, value: DeviceType) -> None:
        valid = get_args(DeviceType)
        if value not in valid:
            raise ValueError(f"Invalid device {value!r}. Must be one of: {valid}")
        if value == "gpu" and not self.gpu_available():
            raise RuntimeError(
                "Cannot set device='gpu': rapids-singlecell not installed. "
                "Install with: pip install squidpy[gpu-cuda12] or squidpy[gpu-cuda11]"
            )
        _device_var.set(value)

    @staticmethod
    def gpu_available() -> bool:
        """
        Check if GPU acceleration is available.

        Returns
        -------
        bool
            True if rapids-singlecell is installed and importable.
        """
        try:
            import rapids_singlecell  # noqa: F401

            return True
        except ImportError:
            return False


settings = _SqSettings()
