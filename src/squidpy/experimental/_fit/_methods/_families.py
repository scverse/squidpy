"""Estimator family registries.

Defined in their own module so individual estimator modules can register into
them without importing :mod:`squidpy.experimental._fit._methods` (which imports
the estimator modules in turn -- that would be circular).
"""

from __future__ import annotations

from squidpy.experimental._fit._registry import Registry

#: Data-driven alignment estimators (consumed by ``experimental.tl.align``).
ALIGN = Registry("align")

#: Closed-form landmark estimators (consumed by ``experimental.tl.align_by_landmarks``).
LANDMARK = Registry("landmark")
