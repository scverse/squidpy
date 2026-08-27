"""Container write-back for the public align functions.

Holds the writes that are *not* plain array writes: registering a fitted affine as a
SpatialData transformation, and encoding a fit for ``uns``.

The fit estimators in :mod:`._stalign` / :mod:`._landmark` operate on plain arrays and
never see a container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

if TYPE_CHECKING:
    from ._stalign import StalignResult

__all__ = ["fit_from_uns", "fit_to_uns", "writeback_affine_sdata"]

#: Fit keys holding one array per axis. Stored as a mapping keyed by axis index: anndata has
#: no tuple writer, a ragged list raises, and an object array round-trips to `StringDType`.
_AXIS_KEYS = frozenset({"velocity_grid", "ref_axes", "query_axes"})
_SCALAR_KEYS = frozenset({"rank", "n_iter"})


def fit_to_uns(fit_result: StalignResult, adata: AnnData, key: str) -> None:
    """Encode a fit into ``adata.uns[key]`` in a form that survives a write."""
    stored: dict[str, object] = {}
    for name, value in fit_result.items():
        if name in _AXIS_KEYS:
            stored[name] = {str(axis): np.asarray(a) for axis, a in enumerate(value)}
        elif name in _SCALAR_KEYS:
            stored[name] = int(value)
        else:
            stored[name] = np.asarray(value)
    adata.uns[key] = stored


def fit_from_uns(adata: AnnData, key: str) -> StalignResult:
    """Decode a fit written by :func:`fit_to_uns`, as numpy arrays."""
    if key not in adata.uns:
        raise KeyError(f"`key={key!r}`: no `uns[{key!r}]`. Available: {sorted(adata.uns)}.")
    stored = adata.uns[key]
    if not isinstance(stored, dict) or "rank" not in stored:
        found = sorted(stored) if isinstance(stored, dict) else type(stored).__name__
        raise ValueError(f"`uns[{key!r}]` carries no `rank`, so it is not a stored STalign fit. Found: {found}.")
    decoded: dict[str, object] = {}
    for name, value in stored.items():
        if name in _AXIS_KEYS:
            decoded[name] = tuple(np.asarray(value[str(axis)]) for axis in range(len(value)))
        elif name in _SCALAR_KEYS:
            decoded[name] = int(value)
        else:
            decoded[name] = np.asarray(value)
    return decoded  # type: ignore[return-value]


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
