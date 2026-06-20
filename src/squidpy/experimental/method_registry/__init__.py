"""In-memory model-fitting core for experimental methods."""

from __future__ import annotations

from squidpy.experimental.method_registry._protocols import AlignLandmarksFn, AlignResult, AlignSamplesFn
from squidpy.experimental.method_registry._registry import Registry

__all__ = ["Registry", "AlignResult", "AlignSamplesFn", "AlignLandmarksFn"]
