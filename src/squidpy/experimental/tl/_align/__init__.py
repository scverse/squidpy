"""Alignment for :mod:`squidpy.experimental.tl`.

JAX is imported only when a STalign fit runs, so importing this module stays cheap.
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
    StalignObsFit,
    StalignVolumeFit,
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
