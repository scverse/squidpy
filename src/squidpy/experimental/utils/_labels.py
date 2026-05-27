"""Shared internal helpers for working with labels elements.

Not part of the public API - symbols here are private and may change
without notice.
"""

from __future__ import annotations

import spatialdata as sd
import xarray as xr
from spatialdata._logging import logger as logg


def resolve_labels_array(sdata: sd.SpatialData, labels_key: str, scale: str | None) -> xr.DataArray:
    """Resolve a labels element to its 2-D ``xarray.DataArray``.

    Single-scale elements pass through; multi-scale (``xarray.DataTree``)
    elements require an explicit ``scale``.  Passing ``scale`` for a
    single-scale element logs a warning and is ignored.

    A strict variant of :func:`squidpy.experimental.im._utils.get_element_data`:
    that helper falls back to a default scale and warns; this one raises
    so callers cannot silently process the wrong resolution.
    """
    node = sdata.labels[labels_key]
    if isinstance(node, xr.DataTree):
        if scale is None:
            raise ValueError(f"Labels `{labels_key}` is multi-scale; pass `scale` (e.g. 'scale0').")
        return node[scale].ds["image"]
    if scale is not None:
        logg.warning(f"`scale={scale!r}` ignored: labels at `{labels_key}` are single-scale.")
    return node
