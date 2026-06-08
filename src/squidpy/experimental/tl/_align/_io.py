"""Input resolvers and output write-back for the public align functions.

This is the *only* layer that knows about the ``AnnData | SpatialData`` argument
shapes and the ``output_mode`` write-back strategies. The fit estimators in
:mod:`squidpy.experimental._methods` operate on plain arrays and never see a
container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from squidpy._validators import assert_key_in_sdata

if TYPE_CHECKING:
    from squidpy.experimental._methods.align_landmarks import AffineFitResult
    from squidpy.experimental._methods.align_samples._stalign_impl._tools import StalignResult


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


def resolve_obs_pair(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None,
    ref_key: str | None,
    query_key: str | None,
) -> tuple[AnnData, AnnData, SpatialData | None, str | None]:
    """Normalise ``align(on="obs")`` inputs.

    Returns ``(ref_adata, query_adata, query_container, query_key)`` where
    ``query_container`` is the SpatialData to write back into (``None`` for plain
    AnnData inputs).
    """
    if isinstance(data_ref, AnnData):
        if data_query is None:
            raise ValueError("`data_query` is required when `data_ref` is an AnnData.")
        if not isinstance(data_query, AnnData):
            raise TypeError(
                f"Mixed AnnData/SpatialData inputs are not supported; `data_query` is {type(data_query).__name__}."
            )
        if ref_key is not None or query_key is not None:
            raise ValueError("`ref_key`/`query_key` are only valid for SpatialData inputs.")
        return data_ref, data_query, None, None

    if not isinstance(data_ref, SpatialData):
        raise TypeError(f"`data_ref` must be AnnData or SpatialData, got {type(data_ref).__name__}.")

    sdata_query = data_ref if data_query is None else data_query
    if not isinstance(sdata_query, SpatialData):
        raise TypeError(
            f"Mixed AnnData/SpatialData inputs are not supported; `data_query` is {type(data_query).__name__}."
        )
    if ref_key is None or query_key is None:
        raise ValueError("`ref_key` and `query_key` are required for SpatialData inputs.")
    assert_key_in_sdata(data_ref, ref_key, attr="tables")
    assert_key_in_sdata(sdata_query, query_key, attr="tables")
    return data_ref.tables[ref_key], sdata_query.tables[query_key], sdata_query, query_key


def get_coords(adata: AnnData, spatial_key: str) -> np.ndarray:
    """Return a validated ``(N, 2)`` ``(x, y)`` coordinate array from ``obsm``."""
    if spatial_key not in adata.obsm:
        raise KeyError(f"`obsm[{spatial_key!r}]` not found; pass `spatial_key=` to select the coordinate key.")
    arr = np.asarray(adata.obsm[spatial_key], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"`obsm[{spatial_key!r}]` must be an (N, 2) array, found shape {arr.shape}.")
    return arr


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


def writeback_obs(
    result: StalignResult | AffineFitResult,
    *,
    output_mode: str,
    query_adata: AnnData,
    container: SpatialData | None,
    element_key: str | None,
    spatial_key: str,
    key_added: str | None,
) -> StalignResult | AffineFitResult | AnnData | SpatialData | None:
    """Bake ``result.transform(coords)`` into the query ``obsm`` per ``output_mode``."""
    if output_mode == "object":
        return result

    dest = _resolve_dest(query_adata, spatial_key=spatial_key, key_added=key_added)
    new_coords = np.asarray(result.transform(get_coords(query_adata, spatial_key)))

    if container is None:
        target = query_adata if output_mode == "inplace" else query_adata.copy()
        target.obsm[dest] = new_coords
        return None if output_mode == "inplace" else target

    sdata = container if output_mode == "inplace" else shallow_copy_sdata(container)
    sdata.tables[element_key].obsm[dest] = new_coords
    return None if output_mode == "inplace" else sdata


def writeback_affine_sdata(
    result: AffineFitResult,
    sdata: SpatialData,
    *,
    output_mode: str,
    moving_cs: str | None,
    target_cs: str | None,
) -> SpatialData | None:
    """Register the fitted affine on every element living in ``moving_cs``.

    Non-destructive: it adds a transformation into ``target_cs`` so the whole
    coordinate system inherits the alignment. Nothing is materialised.
    """
    from spatialdata.transformations import Affine, get_transformation, set_transformation

    if moving_cs is None or target_cs is None:
        raise ValueError("`cs_name_query` and `cs_name_ref` are required to register a transform on a SpatialData.")

    out = sdata if output_mode == "inplace" else shallow_copy_sdata(sdata)
    sd_affine = Affine(np.asarray(result.matrix), input_axes=("x", "y"), output_axes=("x", "y"))
    touched = False
    for _etype, _name, element in out.gen_elements():
        if isinstance(element, AnnData):
            continue
        if moving_cs not in get_transformation(element, get_all=True):
            continue
        set_transformation(element, sd_affine, to_coordinate_system=target_cs)
        touched = True
    if not touched:
        raise KeyError(f"No elements in the SpatialData are registered to coordinate system {moving_cs!r}.")
    return None if output_mode == "inplace" else out


def _resolve_dest(adata: AnnData, *, spatial_key: str, key_added: str | None) -> str:
    """Resolve the destination obsm key, guarding against silent overwrite."""
    if key_added is not None:
        return key_added
    dest = f"aligned_{spatial_key}"
    if dest in adata.obsm:
        raise ValueError(
            f"`obsm[{dest!r}]` already exists. Pass `key_added` explicitly to choose the destination "
            f"(or to overwrite it intentionally)."
        )
    return dest


def shallow_copy_sdata(sdata: SpatialData) -> SpatialData:
    """Shallow copy of a SpatialData for ``output_mode='copy'`` (via ``subset``)."""
    names = [name for _, name, _ in sdata.gen_elements()]
    return sdata.subset(names, filter_tables=False, include_orphan_tables=True)
