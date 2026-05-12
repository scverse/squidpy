from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DataIntent:
    needs_shapes: bool = False
    needs_labels: bool = False
    needs_points: bool = False
    needs_image: bool = False
    needs_graph: bool = False
    library_ids: tuple[str, ...] = ()
    library_key: str | None = None
    coordinate_system: str | None = None
    img_res_key: str | None = None
    img_channel: int | tuple[int, ...] | None = None
    color: tuple[str, ...] = ()
    use_raw: bool | None = None
    layer: str | None = None
    alt_var: str | None = None
    size_key: str | None = None
    seg_cell_id: str | None = None
    shapes_layer: str | None = None
    labels_layer: str | None = None
    image_layer: str | None = None
    points_layer: str | None = None
    graph_layer: str | None = None


@dataclass(frozen=True, slots=True)
class RenderIntent:
    shape: str | None = None
    cmap: Any = None
    norm: Any = None
    palette: Any = None
    alpha: float = 1.0
    na_color: Any = (0.0, 0.0, 0.0, 0.0)
    groups: tuple[str, ...] | None = None
    img_alpha: float | None = None
    img_cmap: Any = None
    contour_px: int | None = None
    outline_alpha: float | None = None
    outline: bool = False
    outline_color: tuple[str, str] = ("black", "white")
    outline_width: tuple[float, float] = (0.3, 0.05)
    edges_width: float = 1.0
    edges_color: Any = "grey"
    edges_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LayoutIntent:
    ncols: int = 4
    library_first: bool = True
    wspace: float | None = None
    hspace: float = 0.25
    figsize: tuple[float, float] | None = None
    dpi: int | None = None
    frameon: bool | None = None
    return_ax: bool = False
    fig: Any = None
    ax: Any = None


@dataclass(frozen=True, slots=True)
class PostRenderIntent:
    title: tuple[str, ...] | None = None
    axis_label: tuple[str, ...] | None = None
    legend_loc: str | None = "right margin"
    legend_fontsize: Any = None
    legend_fontweight: Any = "bold"
    legend_fontoutline: int | None = None
    legend_na: bool = True
    colorbar: bool = True
    scalebar_dx: tuple[float, ...] | None = None
    scalebar_units: tuple[str, ...] | None = None
    scalebar_kwargs: dict[str, Any] = field(default_factory=dict)
    save: str | None = None


@dataclass(frozen=True, slots=True)
class PanelIntent:
    library_id: str
    color: str | None
    size: float | None = None
    scale_factor: float | None = None
    crop_coord: tuple[float, float, float, float] | None = None
    scalebar_dx: float | None = None
    scalebar_units: str | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class Intent:
    mode: str
    data: DataIntent
    render: RenderIntent
    layout: LayoutIntent
    post: PostRenderIntent
    panels: tuple[PanelIntent, ...]
