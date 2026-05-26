"""Shared internal helpers for working with labels elements.

Not part of the public API — symbols here are private and may change
without notice.
"""

from __future__ import annotations

import spatialdata as sd
import xarray as xr


def resolve_labels_array(sdata: sd.SpatialData, labels_key: str, scale: str | None) -> xr.DataArray:
    """Resolve a labels element to its 2-D ``xarray.DataArray``.

    Single-scale elements pass through; multi-scale (``xarray.DataTree``)
    elements require a ``scale`` selector.

    Used by tiling QC and the stitch helpers so the multi-scale branch
    stays consistent across the two pipelines.
    """
    node = sdata.labels[labels_key]
    if isinstance(node, xr.DataTree):
        if scale is None:
            raise ValueError(f"Labels '{labels_key}' is multi-scale; pass `scale` (e.g. 'scale0').")
        return node[scale].ds["image"]
    return node
