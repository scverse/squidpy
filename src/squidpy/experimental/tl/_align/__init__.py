"""Alignment skeleton under :mod:`squidpy.experimental.tl`.

Public surface:

- :func:`align_obs` -- align two ``obs``-level point clouds (cells / spots).
- :func:`align_by_landmarks` -- closed-form fit from user-provided landmarks.

Optional backends (``stalign``, ``moscot``) and JAX are imported lazily -- only
the function call that needs them pulls them in.
"""

from __future__ import annotations

from squidpy.experimental.tl._align._api import (
    align_by_landmarks,
    align_obs,
)

__all__ = ["align_by_landmarks", "align_obs"]
