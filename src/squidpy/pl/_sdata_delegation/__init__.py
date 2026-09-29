from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from spatialdata import SpatialData

from ._adapter import _make_tmp_sdata
from ._capture import capture_scatter_intent, capture_segment_intent
from ._intent import Intent
from ._render import _render_from_intent

_ANNDATA_DEPRECATION = (
    "Passing an AnnData to squidpy spatial plotting is deprecated and will be removed in "
    "squidpy v2.0; pass a SpatialData object instead."
)


def _warn_anndata_input() -> None:
    warnings.warn(_ANNDATA_DEPRECATION, DeprecationWarning, stacklevel=3)


def _resolve_use_raw(adata: AnnData, use_raw: bool | None, layer: str | None = None) -> AnnData:
    """Swap adata.X with adata.raw.X when use_raw resolves True, preserving obs/obsm/uns.

    Matches legacy squidpy/scanpy semantics: ``use_raw=None`` resolves to True when no
    layer is requested and ``adata.raw`` exists. Without this, flipping the delegation
    flag would silently plot ``.X`` where the legacy path plotted raw counts.
    """
    if use_raw is None:
        use_raw = layer is None and adata.raw is not None
    if not use_raw:
        return adata
    if adata.raw is None:
        raise ValueError("use_raw=True but adata.raw is None.")
    raw = adata.raw.to_adata()
    raw.obs = adata.obs.copy()
    raw.obsm = adata.obsm.copy()
    raw.uns = dict(adata.uns)
    return raw


def _delegate(
    input_obj: AnnData | SpatialData,
    capture: Callable[..., Intent],
    **kwargs: Any,
) -> Figure | Axes | list[Axes] | None:
    """Shared input dispatch for the delegation entrypoints.

    SpatialData renders directly; AnnData goes through the transient-sdata shim
    (deprecated) after resolving ``use_raw``. ``capture`` is the per-mode intent builder.
    """
    if isinstance(input_obj, SpatialData):
        if kwargs.get("use_raw"):
            raise ValueError("`use_raw` is AnnData-only; SpatialData has no `.raw`.")
        return _render_from_intent(input_obj, capture(input_obj, **kwargs))
    if not isinstance(input_obj, AnnData):
        raise TypeError(f"Expected AnnData or SpatialData, got {type(input_obj).__name__}.")

    _warn_anndata_input()
    intent = capture(input_obj, **kwargs)
    resolved_adata = _resolve_use_raw(input_obj, intent.data.use_raw, intent.data.layer)
    return _render_from_intent(_make_tmp_sdata(resolved_adata, intent), intent)


def _spatial_scatter_via_sdata_plot(
    input_obj: AnnData | SpatialData,
    **kwargs: Any,
) -> Figure | Axes | list[Axes] | None:
    """spatial_scatter delegation (Paths 1+2): AnnData (shim, deprecated) or SpatialData."""
    return _delegate(input_obj, capture_scatter_intent, **kwargs)


def _spatial_segment_via_sdata_plot(
    input_obj: AnnData | SpatialData,
    **kwargs: Any,
) -> Figure | Axes | list[Axes] | None:
    """spatial_segment delegation (Path 3): AnnData (shim, deprecated) or SpatialData."""
    return _delegate(input_obj, capture_segment_intent, **kwargs)


__all__ = ["_spatial_scatter_via_sdata_plot", "_spatial_segment_via_sdata_plot"]
