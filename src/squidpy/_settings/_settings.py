"""Squidpy global settings."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["settings", "DeviceType"]

DeviceType = Literal["cpu", "gpu"]
GPU_UNAVAILABLE_MSG = (
    "GPU unavailable. Install: pip install squidpy[gpu-cuda12] or with [gpu-cuda11] for CUDA 11 support."
)
_device_var: ContextVar[DeviceType | None] = ContextVar("device", default=None)


def _check_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    try:
        import rapids_singlecell  # noqa: F401

        return True
    except ImportError:
        return False


class SqSettings:
    """Global configuration for squidpy.

    Attributes
    ----------
    gpu_available
        Whether GPU acceleration via rapids-singlecell is available.
    device
        Compute device.
        Defaults to ``'gpu'`` if available, otherwise ``'cpu'``.
    """

    def __init__(self) -> None:
        self.gpu_available: bool = _check_gpu_available()

    @property
    def device(self) -> DeviceType:
        """Compute device: ``'cpu'`` or ``'gpu'``.

        Defaults to ``'gpu'`` if rapids-singlecell is installed, otherwise ``'cpu'``.
        Setting to ``'gpu'`` when GPU is unavailable raises a RuntimeError.
        """
        value = _device_var.get()
        if value is None:
            return "gpu" if self.gpu_available else "cpu"
        return value

    @device.setter
    def device(self, value: DeviceType) -> None:
        if value not in get_args(DeviceType):
            raise ValueError(f"device must be one of {get_args(DeviceType)}, got {value!r}")
        if value == "gpu" and not self.gpu_available:
            raise RuntimeError(GPU_UNAVAILABLE_MSG)
        _device_var.set(value)

    @contextmanager
    def use_device(self, device: DeviceType) -> Generator[None, None, None]:
        """Temporarily set the compute device within a context.

        Parameters
        ----------
        device
            The device to use.

        Examples
        --------
        >>> with sq.settings.use_device("cpu"):
        ...     sq.gr.spatial_neighbors(adata)
        """
        token: Token[DeviceType | None] = _device_var.set(_device_var.get())
        try:
            self.device = device
            yield
        finally:
            _device_var.reset(token)


settings = SqSettings()
