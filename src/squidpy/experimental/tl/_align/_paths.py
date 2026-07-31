"""Location paths for the ``in_`` / ``out`` arguments of the align functions.

A path names one location inside an :class:`~anndata.AnnData` or
:class:`~spatialdata.SpatialData`:

- ``obsm/spatial`` -- an AnnData ``obsm`` key
- ``tables/slice1/obsm/spatial`` -- an ``obsm`` key of a SpatialData table
- ``images/he`` -- a SpatialData image
- ``shapes/landmarks`` -- a SpatialData shapes element holding landmark correspondences
- ``cs/aligned`` -- a coordinate system to register a transformation into (``out`` only)

Naming a location this way rather than through a stack of ``*_key`` arguments means one
argument covers *which element* and *which array inside it*, and the modality follows
from the path instead of a separate ``on=`` switch. This is the shape proposed for
scanpy in https://github.com/scverse/scanpy/issues/4007.

The path also decides what drives the alignment: ``obsm``/``tables`` paths align point
clouds, ``images`` paths align raster intensities, and ``shapes`` paths align by paired
landmarks. ``cs`` is the one write-only form -- it registers a transformation rather than
materialising an array, which only an affine-representable fit can do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

__all__ = ["DataPath", "parse_path", "read_path", "write_path"]

from squidpy.experimental.methods.registry import Modality

#: Element collections a path may address on a SpatialData.
_SDATA_ATTRS = ("tables", "images", "labels", "points", "shapes")
#: Accepted spellings for the coordinate-system namespace, which is not an element
#: collection: writing there registers a transformation instead of storing an array.
_COORDINATE_SYSTEM_ATTRS = ("cs", "coordinate_systems")
_EXAMPLES = '"obsm/spatial", "tables/<table>/obsm/spatial", "images/<name>", "shapes/<name>", or "cs/<name>"'


@dataclass(frozen=True, slots=True)
class DataPath:
    """A parsed location inside a container."""

    modality: Modality
    #: SpatialData element name, or ``None`` for a bare AnnData path.
    element: str | None
    #: ``obsm`` key for ``obs`` paths; ``None`` for every other modality.
    key: str | None
    #: The original string, for error messages.
    raw: str
    #: True for ``cs/<name>``: a coordinate system to register a transformation into,
    #: rather than a location holding an array.
    coordinate_system: bool = False

    def __str__(self) -> str:
        return self.raw


def parse_path(path: str, *, name: str) -> DataPath:
    """Parse a location string, or raise :class:`ValueError` explaining the grammar."""
    if not isinstance(path, str):
        raise TypeError(f"`{name}` must be a string path, got {type(path).__name__}. Expected one of {_EXAMPLES}.")

    parts = tuple(part for part in path.strip("/").split("/") if part)
    if not parts:
        raise ValueError(f"`{name}` is empty. Expected one of {_EXAMPLES}.")

    head = parts[0]
    if head == "obsm":
        if len(parts) != 2:
            raise ValueError(f"`{name}={path!r}` is not a valid obsm path. Expected `obsm/<key>`.")
        return DataPath(modality="obs", element=None, key=parts[1], raw=path)

    if head == "tables":
        if len(parts) != 4 or parts[2] != "obsm":
            raise ValueError(f"`{name}={path!r}` is not a valid table path. Expected `tables/<table>/obsm/<key>`.")
        return DataPath(modality="obs", element=parts[1], key=parts[3], raw=path)

    if head == "images":
        if len(parts) != 2:
            raise ValueError(f"`{name}={path!r}` is not a valid image path. Expected `images/<name>`.")
        return DataPath(modality="images", element=parts[1], key=None, raw=path)

    if head == "shapes":
        if len(parts) != 2:
            raise ValueError(f"`{name}={path!r}` is not a valid landmark path. Expected `shapes/<name>`.")
        return DataPath(modality="landmarks", element=parts[1], key=None, raw=path)

    if head in _COORDINATE_SYSTEM_ATTRS:
        if len(parts) != 2:
            raise ValueError(f"`{name}={path!r}` is not a valid coordinate system. Expected `cs/<name>`.")
        return DataPath(modality="landmarks", element=parts[1], key=None, raw=path, coordinate_system=True)

    if head in _SDATA_ATTRS:
        raise ValueError(
            f"`{name}={path!r}` addresses {head!r}, which alignment does not read or write yet. "
            f"Expected one of {_EXAMPLES}."
        )
    raise ValueError(f"`{name}={path!r}` does not start with a known collection. Expected one of {_EXAMPLES}.")


def _table(container: AnnData | SpatialData, path: DataPath, *, name: str) -> AnnData:
    """Resolve the AnnData a point path refers to."""
    if isinstance(container, AnnData):
        if path.element is not None:
            raise ValueError(
                f"`{name}={path.raw!r}` names a SpatialData table, but the container is an AnnData. "
                f"Use `obsm/{path.key}`."
            )
        return container

    if not isinstance(container, SpatialData):
        raise TypeError(f"Expected an AnnData or SpatialData, got {type(container).__name__}.")
    if path.element is None:
        raise ValueError(
            f"`{name}={path.raw!r}` is ambiguous for a SpatialData, which may hold several tables. "
            f"Use `tables/<table>/obsm/{path.key}`."
        )
    if path.element not in container.tables:
        raise KeyError(f"`{name}={path.raw!r}`: no table {path.element!r}. Available: {sorted(container.tables)}.")
    return container.tables[path.element]


def read_path(container: AnnData | SpatialData, path: DataPath, *, name: str) -> np.ndarray:
    """Read the array a path points at.

    Returns an ``(N, 2)`` ``(x, y)`` array for ``obs`` and ``landmarks`` paths, or a
    channels-first ``(C, H, W)`` array for ``images`` paths.
    """
    if path.coordinate_system:
        raise ValueError(
            f"`{name}={path.raw!r}` names a coordinate system, which holds transformations "
            f"rather than data to align. It is only valid as an `out`."
        )

    if path.modality == "obs":
        adata = _table(container, path, name=name)
        if path.key not in adata.obsm:
            raise KeyError(f"`{name}={path.raw!r}`: no `obsm[{path.key!r}]`. Available: {sorted(adata.obsm)}.")
        coords = np.asarray(adata.obsm[path.key])
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"`{name}={path.raw!r}` must be an (N, 2) array, found shape {coords.shape}.")
        return coords

    if path.modality == "landmarks":
        return _read_landmarks(container, path, name=name)

    return _read_image(container, path, name=name)


def _read_landmarks(container: AnnData | SpatialData, path: DataPath, *, name: str) -> np.ndarray:
    """Read ``(N, 2)`` ``(x, y)`` landmark coordinates from a shapes element.

    This is the layout napari-spatialdata writes when landmarks are picked interactively,
    so annotations made in the viewer are usable here without conversion.
    """
    if not isinstance(container, SpatialData):
        raise TypeError(f"`{name}={path.raw!r}` names a shapes element, which only a SpatialData holds.")
    if path.element not in container.shapes:
        raise KeyError(f"`{name}={path.raw!r}`: no shapes {path.element!r}. Available: {sorted(container.shapes)}.")

    shapes = container.shapes[path.element]
    geometry = shapes.geometry
    coords = np.column_stack([geometry.x.to_numpy(), geometry.y.to_numpy()])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"`{name}={path.raw!r}` must hold point or circle geometries, found {geometry.geom_type[0]!r}."
        )
    return coords


def _read_image(container: AnnData | SpatialData, path: DataPath, *, name: str) -> np.ndarray:
    if not isinstance(container, SpatialData):
        raise TypeError(f"`{name}={path.raw!r}` names an image, which only a SpatialData holds.")
    if path.element not in container.images:
        raise KeyError(f"`{name}={path.raw!r}`: no image {path.element!r}. Available: {sorted(container.images)}.")

    element = container.images[path.element]
    # Multiscale images are a DataTree; the full-resolution level is the first scale.
    if not hasattr(element, "dims"):
        element = next(iter(element.values()))
        element = element[next(iter(element.data_vars))]

    array = np.asarray(element.data)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"`{name}={path.raw!r}` must be a 2D or (c, y, x) image, found shape {array.shape}.")
    return array


def write_path(
    container: AnnData | SpatialData,
    path: DataPath,
    value: np.ndarray,
    *,
    name: str = "out",
) -> None:
    """Write ``value`` at ``path``, mutating ``container`` in place.

    Copy semantics belong to the caller: it decides what to duplicate before calling.
    """
    if path.modality == "obs":
        _table(container, path, name=name).obsm[path.key] = np.asarray(value)
        return

    if not isinstance(container, SpatialData):
        raise TypeError(f"`{name}={path.raw!r}` names an image, which only a SpatialData holds.")

    from spatialdata.models import Image2DModel

    array = np.asarray(value)
    if array.ndim == 2:
        array = array[None]
    container.images[path.element] = Image2DModel.parse(array, dims=("c", "y", "x"))
