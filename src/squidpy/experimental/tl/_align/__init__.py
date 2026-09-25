"""Alignment for :mod:`squidpy.experimental.tl`.

Layered so the numerics never touch a container: :mod:`._api` holds the public
``align_*`` functions, which resolve the ``*_key`` arguments to plain arrays and hand
them to the array-in / array-out estimators in :mod:`._landmark` and :mod:`._stalign`.
Importing stays cheap -- JAX is pulled in only when a STalign fit actually runs.
"""

from __future__ import annotations

from ._api import (
    align_landmarks,
    stalign_align_image,
    stalign_align_obs,
    stalign_align_volume,
)
from ._landmark import apply_affine
from ._stalign import (
    StalignFit,
    StalignImageFit,
    StalignImageParams,
    StalignObsFit,
    StalignObsParams,
    StalignVolumeFit,
    StalignVolumeParams,
)

__all__ = [
    "StalignFit",
    "StalignImageFit",
    "StalignObsFit",
    "StalignVolumeFit",
    "align_landmarks",
    "apply_affine",
    "stalign_align_image",
    "stalign_align_obs",
    "stalign_align_volume",
]
