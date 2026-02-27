"""Squidpy settings."""

from __future__ import annotations

from squidpy._settings._dispatch import gpu_dispatch
from squidpy._settings._settings import DeviceType, settings

__all__ = ["settings", "DeviceType", "gpu_dispatch"]
