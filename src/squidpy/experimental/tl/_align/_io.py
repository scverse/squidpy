"""Container reads and write-backs shared by the align functions and the fit classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from squidpy._validators import validate_xy

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_array_utils.types import HasArrayNamespace as Array

__all__ = ["assert_table_coords_share_frame", "read_coords", "resolve_table", "write_coords", "writeback_affine_sdata"]


def writeback_affine_sdata(
    matrix: np.ndarray,
    sdata: SpatialData,
    *,
    moving_cs: str | None,
    target_cs: str | None,
) -> None:
    """Register the fitted affine on every element living in ``moving_cs``."""
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


def resolve_table(container: AnnData | SpatialData, table_key: str | None, *, side: str) -> AnnData:
    """Resolve a container plus ``table_key`` to the AnnData holding the data."""
    if isinstance(container, AnnData):
        if table_key is not None:
            raise ValueError(
                f"`table_key` was given for the {side}, but the {side} is an AnnData, which has no tables. "
                f"Pass `table_key=None` for it (use a `(ref, query)` pair to address mixed containers)."
            )
        return container
    if not isinstance(container, SpatialData):
        raise TypeError(f"Expected the {side} to be an AnnData or SpatialData, got {type(container).__name__}.")
    if table_key is None:
        raise ValueError(
            f"The {side} is a SpatialData, which may hold several tables; pass `table_key` to say "
            f"which one holds the data. Available: {sorted(container.tables)}."
        )
    if table_key not in container.tables:
        raise KeyError(
            f"`table_key={table_key!r}`: no such table in the {side}. Available: {sorted(container.tables)}."
        )
    return container.tables[table_key]


def read_coords(adata: AnnData, *, key: str, side: str, name: str) -> np.ndarray:
    """Read a validated ``(N, 2)`` coordinate array from ``obsm``."""
    if key not in adata.obsm:
        raise KeyError(f"`{name}={key!r}`: no `obsm[{key!r}]` on the {side}. Available: {sorted(adata.obsm)}.")
    return validate_xy(adata.obsm[key], name=f"{name}={key!r}")


def write_coords(
    container: AnnData | SpatialData,
    table_key: str | None,
    spatial_key: str,
    key_added: str,
    *,
    transform: Callable[[np.ndarray], Array],
    spatial_key_name: str,
    inplace: bool = True,
) -> np.ndarray | None:
    """Transform ``obsm[spatial_key]`` and write it to ``obsm[key_added]`` on the query."""
    adata = resolve_table(container, table_key, side="query")
    coords = read_coords(adata, key=spatial_key, side="query", name=spatial_key_name)
    transformed = np.asarray(transform(coords))
    if not inplace:
        return transformed
    adata.obsm[key_added] = transformed
    return None


def assert_table_coords_share_frame(
    sdata: SpatialData, *, adata: AnnData, spatial_key: str, coordinate_system: str
) -> None:
    """Refuse to transform ``obsm`` coordinates that are not in ``coordinate_system``."""
    from spatialdata.transformations import get_transformation

    region = adata.uns.get("spatialdata_attrs", {}).get("region")
    regions = [region] if isinstance(region, str) else list(region or [])
    for name in regions:
        if name not in sdata:
            continue
        matrix = np.asarray(
            get_transformation(sdata[name], to_coordinate_system=coordinate_system).to_affine_matrix(
                input_axes=("x", "y"), output_axes=("x", "y")
            ),
            dtype=float,
        )
        if not np.allclose(matrix, np.eye(3)):
            raise ValueError(
                f"`spatial_key={spatial_key!r}` lives in the intrinsic frame of element {name!r}, which "
                f"carries a non-identity transformation into {coordinate_system!r}: the coordinate system "
                f"the fit's units come from. Transforming it would silently produce wrong reference "
                f"coordinates. Store the coordinates in {coordinate_system!r} units, or apply "
                f"`StalignFit.transform_points` to coordinates you have placed there yourself."
            )
