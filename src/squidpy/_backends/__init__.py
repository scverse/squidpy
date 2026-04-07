"""Pluggable backend dispatch system for squidpy."""

from squidpy._backends._dispatch import dispatch
from squidpy._backends._registry import available_backend_names, get_backend
from squidpy._backends._settings import settings

__all__ = ["dispatch", "get_backend", "available_backend_names", "settings"]
