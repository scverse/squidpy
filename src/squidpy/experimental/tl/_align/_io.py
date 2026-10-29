"""Container write-back for the public align functions.

Registering a fitted affine as a SpatialData transformation, and encoding a fit for ``uns``.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from ._stalign import StalignFit, StalignImageFit, StalignObsFit, StalignVolumeFit

__all__ = ["fit_from_uns", "fit_to_uns", "writeback_affine_sdata"]

#: Fit fields holding one array per axis, stored keyed by axis index: anndata has no tuple writer.
_AXIS_KEYS = frozenset({"velocity_grid", "ref_axes", "query_axes"})
_SCALAR_KEYS = frozenset({"n_iter"})
#: Fit fields that are plain strings; `np.asarray` would round-trip them as 0-d arrays.
_STR_KEYS = frozenset({"coordinate_system"})

_KINDS: dict[str, type[StalignFit]] = {
    "obs": StalignObsFit,
    "image": StalignImageFit,
    "volume": StalignVolumeFit,
}


def fit_to_uns(fit_result: StalignFit, adata: AnnData, key: str) -> None:
    """Encode a fit into ``adata.uns[key]`` in a form that survives a write."""
    stored: dict[str, object] = {"kind": fit_result.kind}
    for spec in fields(fit_result):
        value = getattr(fit_result, spec.name)
        if value is None:
            # anndata cannot encode None; the class default restores it
            continue
        if spec.name in _AXIS_KEYS:
            stored[spec.name] = {str(axis): np.asarray(a) for axis, a in enumerate(value)}
        elif spec.name in _SCALAR_KEYS:
            stored[spec.name] = int(value)
        elif spec.name in _STR_KEYS:
            stored[spec.name] = str(value)
        else:
            stored[spec.name] = np.asarray(value)
    adata.uns[key] = stored


def fit_from_uns(adata: AnnData, key: str) -> StalignFit:
    """Decode a fit written by :func:`fit_to_uns`, as numpy arrays."""
    if key not in adata.uns:
        raise KeyError(f"`key={key!r}`: no `uns[{key!r}]`. Available: {sorted(adata.uns)}.")
    stored = adata.uns[key]
    if not isinstance(stored, dict) or "kind" not in stored:
        found = sorted(stored) if isinstance(stored, dict) else type(stored).__name__
        raise ValueError(f"`uns[{key!r}]` carries no `kind`, so it is not a stored STalign fit. Found: {found}.")
    kind = str(stored["kind"])
    if kind not in _KINDS:
        raise ValueError(f"`uns[{key!r}]` names an unknown fit kind {kind!r}. Expected one of {sorted(_KINDS)}.")
    decoded: dict[str, object] = {}
    for name, value in stored.items():
        if name == "kind":
            continue
        if name in _AXIS_KEYS:
            decoded[name] = tuple(np.asarray(value[str(axis)]) for axis in range(len(value)))
        elif name in _SCALAR_KEYS:
            decoded[name] = int(value)
        elif name in _STR_KEYS:
            decoded[name] = str(value)
        else:
            decoded[name] = np.asarray(value)
    return _KINDS[kind](**decoded)  # type: ignore[arg-type]


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
        if target_cs in get_transformation(element, get_all=True):
            raise ValueError(
                f"An element is already registered to {target_cs!r}; writing the alignment there "
                f"would replace its existing placement. Pass a `target_coordinate_system` that is "
                f"not already in use."
            )
        # compose after the element's existing transform into `moving_cs`
        existing = get_transformation(element, to_coordinate_system=moving_cs)
        set_transformation(element, Sequence([existing, sd_affine]), to_coordinate_system=target_cs)
        touched = True
    if not touched:
        raise KeyError(f"No elements in the SpatialData are registered to coordinate system {moving_cs!r}.")
