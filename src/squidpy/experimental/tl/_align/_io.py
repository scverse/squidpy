"""SpatialData write-back for the public align functions.

Holds the one write that is *not* an array write: registering a fitted affine as a
SpatialData transformation, which changes how elements are placed rather than
materialising anything.

The fit estimators in :mod:`squidpy.experimental.methods` operate on plain arrays and
never see a container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

if TYPE_CHECKING:
    from squidpy.experimental.methods.align_landmarks import AffineFitResult

__all__ = ["shallow_copy_sdata", "writeback_affine_sdata"]


def writeback_affine_sdata(
    result: AffineFitResult,
    sdata: SpatialData,
    *,
    inplace: bool,
    moving_cs: str | None,
    target_cs: str | None,
) -> SpatialData | None:
    """Register the fitted affine on every element living in ``moving_cs``.

    Non-destructive: it adds a transformation into ``target_cs`` so the whole
    coordinate system inherits the alignment. Nothing is materialised.
    """
    from spatialdata import deepcopy as sd_deepcopy
    from spatialdata.transformations import Affine, Sequence, get_transformation, set_transformation

    if moving_cs is None or target_cs is None:
        raise ValueError("`moving_cs` and `target_cs` are required to register a transform on a SpatialData.")

    out = sdata if inplace else shallow_copy_sdata(sdata)
    sd_affine = Affine(np.asarray(result.matrix), input_axes=("x", "y"), output_axes=("x", "y"))
    touched = False
    for etype, name, element in list(out.gen_elements()):
        if isinstance(element, AnnData):
            continue
        if moving_cs not in get_transformation(element, get_all=True):
            continue
        if not inplace:
            # `shallow_copy_sdata` shares element objects with the original; deep-copy each
            # element we register a transform on so `copy=True` leaves the input untouched.
            element = sd_deepcopy(element)
            getattr(out, etype)[name] = element
        # The fitted affine maps `moving_cs` coords into `target_cs`, not the element's
        # intrinsic frame. Compose it after the element's existing intrinsic -> `moving_cs`
        # transform so a non-identity placement into `moving_cs` is preserved.
        existing = get_transformation(element, to_coordinate_system=moving_cs)
        set_transformation(element, Sequence([existing, sd_affine]), to_coordinate_system=target_cs)
        touched = True
    if not touched:
        raise KeyError(f"No elements in the SpatialData are registered to coordinate system {moving_cs!r}.")
    return None if inplace else out


def shallow_copy_sdata(sdata: SpatialData) -> SpatialData:
    """Shallow copy of a SpatialData for ``copy=True`` (via ``subset``)."""
    names = [name for _, name, _ in sdata.gen_elements()]
    return sdata.subset(names, filter_tables=False, include_orphan_tables=True)
