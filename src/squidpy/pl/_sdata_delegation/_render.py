from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import spatialdata_plot  # noqa: F401 -- registers .pl accessor
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from spatialdata import SpatialData

from ._adapter import _image_name, _labels_name, _points_name, _shapes_name, _table_name
from ._intent import Intent, PanelIntent


def _make_grid(
    n_panels: int,
    ncols: int,
    figsize: tuple[float, float] | None,
    dpi: int | None,
    fig: Figure | None,
    ax: tuple[Axes, ...] | None,
) -> tuple[Figure, list[Axes]]:
    if ax is not None:
        axes = list(ax)
        owning_fig = fig if fig is not None else axes[0].get_figure()
        return owning_fig, axes
    cols = min(ncols, n_panels)
    rows = math.ceil(n_panels / cols)
    if figsize is None:
        figsize = (4.0 * cols, 4.0 * rows)
    if fig is None:
        new_fig, new_axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, squeeze=False)
    else:
        new_fig = fig
        new_axes = fig.subplots(rows, cols, squeeze=False)
    flat = list(new_axes.ravel())
    for blank in flat[n_panels:]:
        blank.set_axis_off()
    return new_fig, flat[:n_panels]


def _color_kwargs(panel: PanelIntent, intent: Intent) -> dict[str, Any]:
    """Build the color/cmap/palette/groups/table_* kwargs shared across render_* calls."""
    return {
        "color": panel.color,
        "palette": intent.render.palette,
        "cmap": intent.render.cmap,
        "norm": intent.render.norm,
        "na_color": intent.render.na_color,
        "groups": list(intent.render.groups) if intent.render.groups else None,
        "table_name": _table_name(panel.library_id),
        "table_layer": intent.data.layer,
        "gene_symbols": intent.data.alt_var,
    }


def _draw_panel(chain: SpatialData, panel: PanelIntent, intent: Intent) -> SpatialData:
    """Compose render_* calls for one panel.

    Z-order: render_images (bottom) -> render_graph -> render_shapes / render_labels /
    render_points (top). Edges drawn before points so points sit on top, matching
    squidpy's legacy order at _spatial.py:267-277.
    """
    color_kw = _color_kwargs(panel, intent)

    if intent.data.needs_image:
        chain = chain.pl.render_images(_image_name(panel.library_id))

    kind = intent.data.element_kind

    if intent.data.needs_graph and intent.data.graph_layer is not None:
        element_name = _shapes_name(panel.library_id) if kind == "shapes" else _points_name(panel.library_id)
        chain = chain.pl.render_graph(
            element_name,
            color=intent.render.edges_color if isinstance(intent.render.edges_color, str) else "grey",
            connectivity_key=intent.data.graph_layer,
            edge_width=intent.render.edges_width,
            table_name=_table_name(panel.library_id),
        )

    if kind == "shapes":
        kw = dict(color_kw)
        kw["shape"] = intent.render.shape
        kw["fill_alpha"] = intent.render.alpha
        if panel.size is not None:
            kw["scale"] = float(panel.size)
        if intent.render.outline:
            bg_color, gap_color = intent.render.outline_color
            bg_width, gap_width = intent.render.outline_width
            # sdata-plot v0.3.4 tuple-outline: nested rings rendered in one pass.
            kw["outline_color"] = (bg_color, gap_color)
            kw["outline_width"] = (bg_width + gap_width, gap_width)
            kw["outline_alpha"] = (1.0, 1.0)
        chain = chain.pl.render_shapes(_shapes_name(panel.library_id), **kw)
    elif kind == "labels":
        kw = dict(color_kw)
        kw["fill_alpha"] = intent.render.alpha
        kw["contour_px"] = intent.render.contour_px
        kw["outline_alpha"] = intent.render.outline_alpha
        chain = chain.pl.render_labels(_labels_name(panel.library_id), **kw)
    else:  # points
        kw = dict(color_kw)
        kw["alpha"] = intent.render.alpha
        chain = chain.pl.render_points(_points_name(panel.library_id), **kw)

    return chain


def _apply_post(panel: PanelIntent, intent: Intent, ax: Axes) -> None:
    if panel.title is not None:
        ax.set_title(panel.title)
    if intent.layout.frameon is False:
        ax.set_frame_on(False)
    if panel.crop_coord is not None:
        x0, x1, y0, y1 = panel.crop_coord
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # image y-axis is top-down


def _render_from_intent(sdata: SpatialData, intent: Intent) -> Figure | Axes | Sequence[Axes] | None:
    panels = intent.panels
    owning_fig, axes = _make_grid(
        n_panels=len(panels),
        ncols=intent.layout.ncols,
        figsize=intent.layout.figsize,
        dpi=intent.layout.dpi,
        fig=intent.layout.fig,
        ax=intent.layout.ax,
    )

    for panel, ax in zip(panels, axes, strict=True):
        chain = _draw_panel(sdata, panel, intent)
        show_kw: dict[str, Any] = {
            "ax": ax,
            "coordinate_systems": panel.library_id,
            "return_ax": False,
        }
        if panel.scalebar_dx is not None:
            show_kw["scalebar_dx"] = panel.scalebar_dx
        if panel.scalebar_units is not None:
            show_kw["scalebar_units"] = panel.scalebar_units
        chain.pl.show(**show_kw)
        _apply_post(panel, intent, ax)

    if intent.layout.return_ax:
        return axes[0] if len(axes) == 1 else axes
    return owning_fig
