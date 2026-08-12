"""Input-source abstraction for the delegation backend.

Capture is almost input-agnostic: its only coupling to the concrete input is
(a) resolving the list of libraries and (b) naming the SpatialData elements each
panel renders. A source encapsulates exactly those two concerns so one capture path
serves both AnnData (via the transient-sdata shim) and native SpatialData input.
"""

from __future__ import annotations

from typing import Protocol

from anndata import AnnData
from spatialdata import SpatialData

from squidpy._constants._pkg_constants import Key

from ._adapter import _image_name, _labels_name, _points_name, _shapes_name, _table_name
from ._intent import ElementKind

_ELEMENT_CONTAINER: dict[ElementKind, str] = {"shapes": "shapes", "labels": "labels", "points": "points"}


class _Source(Protocol):
    def library_ids(self, library_key: str | None, library_id: object) -> tuple[str, ...]: ...
    def element_name(self, library_id: str, kind: ElementKind) -> str: ...
    def image_name(self, library_id: str) -> str | None: ...
    def table_name(self, library_id: str) -> str | None: ...


class _AnnDataSource:
    """Names follow the transient-sdata shim convention (see _adapter)."""

    def __init__(self, adata: AnnData) -> None:
        self.adata = adata

    def library_ids(self, library_key: str | None, library_id: object) -> tuple[str, ...]:
        if library_id is not None:
            return (library_id,) if isinstance(library_id, str) else tuple(library_id)
        if library_key is not None:
            return tuple(map(str, self.adata.obs[library_key].cat.categories))
        if Key.uns.spatial in self.adata.uns:
            return tuple(self.adata.uns[Key.uns.spatial].keys())
        raise ValueError("No library_id or library_key provided and no 'spatial' key in adata.uns.")

    def element_name(self, library_id: str, kind: ElementKind) -> str:
        return {"shapes": _shapes_name, "points": _points_name, "labels": _labels_name}[kind](library_id)

    def image_name(self, library_id: str) -> str | None:
        return _image_name(library_id)

    def table_name(self, library_id: str) -> str | None:
        return _table_name(library_id)


class _SpatialDataSource:
    """Resolve element/table names from a user's SpatialData.

    Libraries are coordinate systems (subset by ``library_id``). Within a coordinate
    system an element type is auto-resolved when unique; otherwise the caller must
    disambiguate with the matching ``*_layer`` kwarg, else a ValueError lists the
    candidates (mirrors scanpy's ``layer=`` ergonomics).
    """

    def __init__(
        self,
        sdata: SpatialData,
        *,
        shapes_layer: str | None = None,
        labels_layer: str | None = None,
        points_layer: str | None = None,
        image_layer: str | None = None,
        table: str | None = None,
    ) -> None:
        self.sdata = sdata
        self._explicit: dict[str, str | None] = {
            "shapes": shapes_layer,
            "labels": labels_layer,
            "points": points_layer,
            "images": image_layer,
        }
        self._table = table

    def library_ids(self, library_key: str | None, library_id: object) -> tuple[str, ...]:
        if library_key is not None:
            raise ValueError(
                "`library_key` is AnnData-only. On SpatialData input, libraries are coordinate "
                "systems; select them with `library_id`."
            )
        systems = tuple(self.sdata.coordinate_systems)
        if library_id is None:
            return systems
        wanted = (library_id,) if isinstance(library_id, str) else tuple(map(str, library_id))
        missing = [w for w in wanted if w not in systems]
        if missing:
            raise ValueError(f"Coordinate system(s) {missing} not in SpatialData; available: {list(systems)}.")
        return wanted

    def _resolve(self, library_id: str, container: str, *, required: bool) -> str | None:
        sub = self.sdata.filter_by_coordinate_system(library_id)
        keys = list(getattr(sub, container))
        explicit = self._explicit.get(container)
        if explicit is not None:
            if explicit not in keys:
                raise ValueError(
                    f"{container} layer {explicit!r} not found in coordinate system {library_id!r}; available: {keys}."
                )
            return explicit
        if len(keys) == 1:
            return keys[0]
        if not keys:
            if required:
                raise ValueError(f"No {container} element in coordinate system {library_id!r}.")
            return None
        raise ValueError(
            f"Multiple {container} elements in coordinate system {library_id!r}: {keys}. "
            f"Disambiguate with the matching *_layer kwarg."
        )

    def element_name(self, library_id: str, kind: ElementKind) -> str:
        name = self._resolve(library_id, _ELEMENT_CONTAINER[kind], required=True)
        assert name is not None  # required=True guarantees non-None
        return name

    def image_name(self, library_id: str) -> str | None:
        return self._resolve(library_id, "images", required=False)

    def table_name(self, library_id: str) -> str | None:
        if self._table is not None:
            if self._table not in self.sdata.tables:
                raise ValueError(f"table {self._table!r} not found; available: {list(self.sdata.tables)}.")
            return self._table
        # find a table annotating any element in this coordinate system
        sub = self.sdata.filter_by_coordinate_system(library_id)
        element_names = set(sub.shapes) | set(sub.labels) | set(sub.points)
        for tname, tbl in self.sdata.tables.items():
            region = tbl.uns.get("spatialdata_attrs", {}).get("region")
            regions = {region} if isinstance(region, str) else set(region or ())
            if regions & element_names:
                return tname
        return None
