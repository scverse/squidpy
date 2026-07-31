"""Shared internal helpers for working with labels elements.

Not part of the public API - symbols here are private and may change
without notice.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import spatialdata as sd
import xarray as xr
from skimage.measure import regionprops
from spatialdata._logging import logger as logg


def iter_chunked_regionprops(
    labels: xr.DataArray | np.ndarray,
    chunk_size: int = 4096,
    label_subset: Iterable[int] | None = None,
) -> Iterator[tuple[int, Any, int, int]]:
    """Yield ``(label_id, region, y0, x0)`` over chunked ``regionprops`` of a labels array.

    Works on a plain :class:`numpy.ndarray` (a single chunk) or a possibly
    dask-backed 2-D :class:`xarray.DataArray`, reading at most ``chunk_size`` x
    ``chunk_size`` at a time so memory stays bounded for very large images.

    ``region`` is a :class:`skimage.measure.RegionProperties` whose coordinates
    are LOCAL to the chunk; add ``y0`` / ``x0`` for global coordinates.  When
    ``label_subset`` is given, only regions with those label ids are yielded.
    Background (label 0) is never yielded (``regionprops`` skips it).
    """
    subset = None if label_subset is None else {int(x) for x in label_subset}

    if isinstance(labels, np.ndarray):
        for region in regionprops(labels):
            lid = int(region.label)
            if subset is None or lid in subset:
                yield lid, region, 0, 0
        return

    h = int(labels.sizes.get("y", labels.shape[-2]))
    w = int(labels.sizes.get("x", labels.shape[-1]))
    for y0 in range(0, h, chunk_size):
        y1 = min(y0 + chunk_size, h)
        for x0 in range(0, w, chunk_size):
            x1 = min(x0 + chunk_size, w)
            chunk = labels.isel(y=slice(y0, y1), x=slice(x0, x1)).values
            while chunk.ndim > 2:
                chunk = chunk.squeeze(0)
            for region in regionprops(chunk):
                lid = int(region.label)
                if subset is None or lid in subset:
                    yield lid, region, y0, x0


def resolve_labels_array(sdata: sd.SpatialData, labels_key: str, scale: str | None) -> xr.DataArray:
    """Resolve a labels element to its 2-D ``xarray.DataArray``.

    Single-scale elements pass through; multi-scale (``xarray.DataTree``)
    elements require an explicit ``scale`` and raise otherwise.  Passing
    ``scale`` for a single-scale element logs a warning and is ignored.
    """
    node = sdata.labels[labels_key]
    if isinstance(node, xr.DataTree):
        if scale is None:
            raise ValueError(f"Labels `{labels_key}` is multi-scale; pass `scale` (e.g. 'scale0').")
        return node[scale].ds["image"]
    if scale is not None:
        logg.warning(f"`scale={scale!r}` ignored: labels at `{labels_key}` are single-scale.")
    return node
