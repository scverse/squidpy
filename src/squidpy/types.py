"""Squidpy's public type surface: the parameter bags and the result tuples.

Collected by *kind* rather than by domain. A ``*Params`` is inert input the caller fills in;
a ``*Result`` is what a function hands back. Neither carries behaviour -- anything that does
is a fit, and lives with the functions that produce it. Gathering both here makes that split
an import path instead of a convention, and keeps the domain modules to the functions they
are about.

Each class is declared next to the code that consumes it -- moving the declarations here
would make this module import the packages whose ``__init__`` imports it -- so this is the
single *public* route to them rather than a second one.
"""

from __future__ import annotations

from squidpy.experimental.im._detect_tissue import (
    BackgroundDetectionParams,
    FelzenszwalbParams,
    WekaParams,
)
from squidpy.experimental.im._stain._decomposition import MacenkoParams, VahadaneParams
from squidpy.experimental.im._stain._reinhard import ReinhardParams
from squidpy.experimental.tl._align._stalign import (
    StalignImageParams,
    StalignObsParams,
    StalignVolumeParams,
)
from squidpy.experimental.tl._tiling_qc import TilingQCParams
from squidpy.experimental.tl._tiling_stitch import StitchParams
from squidpy.gr._build import SpatialNeighborsResult
from squidpy.gr._nhood import NhoodEnrichmentResult

__all__ = [
    # Parameters
    "BackgroundDetectionParams",
    "FelzenszwalbParams",
    "MacenkoParams",
    "ReinhardParams",
    "StalignImageParams",
    "StalignObsParams",
    "StalignVolumeParams",
    "StitchParams",
    "TilingQCParams",
    "VahadaneParams",
    "WekaParams",
    # Results
    "NhoodEnrichmentResult",
    "SpatialNeighborsResult",
]
