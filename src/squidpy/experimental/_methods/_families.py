"""Estimator family registries.

A *family* is the set of interchangeable algorithms behind one public function,
selected by ``method=`` and sharing one ``fit`` contract. One :class:`Registry`
per family. Registries live in this small module so the estimator modules can
register into them without importing their package ``__init__`` (which imports
the estimator modules in turn -- that would be circular).
"""

from __future__ import annotations

from squidpy.experimental._fit._registry import Registry

#: Sample-to-sample alignment estimators -- ref/query point clouds in, transform out.
#: Consumed by ``squidpy.experimental.tl.align``.
ALIGN_SAMPLES = Registry("align_samples")

#: Closed-form landmark alignment estimators -- paired landmarks in, affine out.
#: Consumed by ``squidpy.experimental.tl.align_by_landmarks``.
ALIGN_LANDMARKS = Registry("align_landmarks")
