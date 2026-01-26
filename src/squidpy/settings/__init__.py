"""Squidpy settings."""

from squidpy.settings._dispatch import gpu_dispatch
from squidpy.settings._settings import DeviceType, settings

__all__ = ["settings", "DeviceType", "gpu_dispatch"]
