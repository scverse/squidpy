from __future__ import annotations

from typing import Any

from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from spatialdata import SpatialData

from ._adapter import _make_tmp_sdata
from ._capture import capture_scatter_intent, capture_segment_intent
from ._render import _render_from_intent


def _resolve_use_raw(adata: AnnData, use_raw: bool | None) -> AnnData:
    """Swap adata.X with adata.raw.X when use_raw=True, preserving obs/obsm/uns."""
    if not use_raw:
        return adata
    if adata.raw is None:
        raise ValueError("use_raw=True but adata.raw is None.")
    raw = adata.raw.to_adata()
    raw.obs = adata.obs.copy()
    raw.obsm = adata.obsm.copy() if adata.obsm is not None else None
    raw.uns = dict(adata.uns)
    return raw


def _spatial_scatter_via_sdata_plot(
    input_obj: AnnData | SpatialData,
    **kwargs: Any,
) -> Figure | Axes | list[Axes] | None:
    """Internal entrypoint for spatial_scatter delegation (Paths 1+2).

    Routes a squidpy-style spatial_scatter call through the
    capture-intent -> adapter -> spatialdata-plot pipeline. Not wired into the
    public `sq.pl.spatial_scatter` yet — callable from tests while we verify
    feature parity on the happy paths.
    """
    if isinstance(input_obj, SpatialData):
        raise NotImplementedError("SpatialData input path lands in Stage 2 follow-up.")
    if not isinstance(input_obj, AnnData):
        raise TypeError(f"Expected AnnData or SpatialData, got {type(input_obj).__name__}.")

    intent = capture_scatter_intent(input_obj, **kwargs)
    resolved_adata = _resolve_use_raw(input_obj, intent.data.use_raw)
    sdata = _make_tmp_sdata(resolved_adata, intent)
    return _render_from_intent(sdata, intent)


def _spatial_segment_via_sdata_plot(
    input_obj: AnnData | SpatialData,
    **kwargs: Any,
) -> Figure | Axes | list[Axes] | None:
    """Internal entrypoint for spatial_segment delegation (Path 3).

    Routes a squidpy-style spatial_segment call through the labels-flavoured
    capture-intent -> adapter -> spatialdata-plot pipeline.
    """
    if isinstance(input_obj, SpatialData):
        raise NotImplementedError("SpatialData input path lands in Stage 2 follow-up.")
    if not isinstance(input_obj, AnnData):
        raise TypeError(f"Expected AnnData or SpatialData, got {type(input_obj).__name__}.")

    intent = capture_segment_intent(input_obj, **kwargs)
    resolved_adata = _resolve_use_raw(input_obj, intent.data.use_raw)
    sdata = _make_tmp_sdata(resolved_adata, intent)
    return _render_from_intent(sdata, intent)


__all__ = ["_spatial_scatter_via_sdata_plot", "_spatial_segment_via_sdata_plot"]
