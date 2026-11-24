"""Resolve the reference and query containers of the experimental tools that take both."""

from __future__ import annotations

from collections.abc import Sequence

from anndata import AnnData
from spatialdata import SpatialData

from squidpy.gr._utils import extract_adata_if_sdata

__all__ = ["query_of", "resolve_pair", "resolve_table"]


def resolve_pair(value: str | Sequence[str | None] | None, *, name: str) -> tuple[str | None, str | None]:
    """Normalize a single key, applying to both sides, or a ``(ref, query)`` pair to a pair."""
    if value is None or isinstance(value, str):
        return value, value
    if isinstance(value, tuple | list) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(f"`{name}` must be a single key or a `(ref, query)` pair, got {value!r}.")


def query_of(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None,
    *,
    ref_address: tuple[str | None, ...],
    query_address: tuple[str | None, ...],
    key_name: str,
) -> AnnData | SpatialData:
    """The container holding the query sample, validating the single-SpatialData form."""
    if data_query is not None:
        return data_query
    if not isinstance(data_ref, SpatialData):
        raise ValueError("`data_query` is required unless `data_ref` is a SpatialData holding both samples.")
    if ref_address == query_address:
        raise ValueError(
            f"With a single SpatialData, pass `{key_name}=(ref, query)` to say which of its elements holds each sample."
        )
    return data_ref


def resolve_table(container: AnnData | SpatialData, table_key: str | None, *, side: str) -> AnnData:
    """Resolve a container plus ``table_key`` to the AnnData holding the data."""
    if isinstance(container, AnnData) and table_key is not None:
        raise ValueError(
            f"`table_key` was given for the {side}, but the {side} is an AnnData, which has no tables. "
            f"Pass `table_key=None` for it (use a `(ref, query)` pair to address mixed containers)."
        )
    if isinstance(container, SpatialData) and table_key is None:
        raise ValueError(
            f"The {side} is a SpatialData, which may hold several tables; pass `table_key` to say "
            f"which one holds the data. Available: {sorted(container.tables)}."
        )
    return extract_adata_if_sdata(container, table_key=table_key)
