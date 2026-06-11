"""Estimator family registries."""

from __future__ import annotations

from squidpy.experimental._methods._protocols import AlignLandmarksFn, AlignSamplesFn
from squidpy.experimental._methods._registry import Registry

#: Sample-to-sample alignment estimators -- ref/query point clouds in, transform out.
#: Consumed by ``squidpy.experimental.tl.align``.
ALIGN_SAMPLES: Registry[AlignSamplesFn] = Registry("align_samples")

#: Closed-form landmark alignment estimators -- paired landmarks in, affine out.
#: Consumed by ``squidpy.experimental.tl.align_by_landmarks``.
ALIGN_LANDMARKS: Registry[AlignLandmarksFn] = Registry("align_landmarks")
