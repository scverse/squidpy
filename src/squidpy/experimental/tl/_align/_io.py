"""SpatialData write-back for the public align functions.

Holds the one write that is *not* an array write: registering a fitted affine as a
SpatialData transformation, which changes how elements are placed rather than
materialising anything.

The fit estimators in :mod:`._stalign` / :mod:`._landmark` operate on plain arrays and
never see a container.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

__all__ = ["writeback_affine_sdata"]


def writeback_affine_sdata(
    matrix: np.ndarray,
    sdata: SpatialData,
    *,
    moving_cs: str | None,
    target_cs: str | None,
) -> None:
    """Register the fitted affine on every element living in ``moving_cs``.

    Non-destructive: it adds a transformation into ``target_cs`` so the whole
    coordinate system inherits the alignment. Nothing is materialised.
    """
    from spatialdata.transformations import Affine, Sequence, get_transformation, set_transformation

    sd_affine = Affine(np.asarray(matrix), input_axes=("x", "y"), output_axes=("x", "y"))
    touched = False
    for _, _, element in sdata.gen_elements():
        if isinstance(element, AnnData):
            continue
        if moving_cs not in get_transformation(element, get_all=True):
            continue
        # The fitted affine maps `moving_cs` coords into `target_cs`, not the element's
        # intrinsic frame. Compose it after the element's existing intrinsic -> `moving_cs`
        # transform so a non-identity placement into `moving_cs` is preserved.
        existing = get_transformation(element, to_coordinate_system=moving_cs)
        set_transformation(element, Sequence([existing, sd_affine]), to_coordinate_system=target_cs)
        touched = True
    if not touched:
        raise KeyError(f"No elements in the SpatialData are registered to coordinate system {moving_cs!r}.")
