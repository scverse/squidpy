from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from anndata import AnnData
from matplotlib.colors import Normalize, TwoSlopeNorm

from squidpy._constants._pkg_constants import Key

from ._intent import (
    DataIntent,
    Intent,
    LayoutIntent,
    PanelIntent,
    PostRenderIntent,
    RenderIntent,
)


def _build_norm(
    vmin: float | None,
    vmax: float | None,
    vcenter: float | None,
    norm: Normalize | None,
) -> Normalize | None:
    """Fold vmin/vmax/vcenter into a matplotlib Normalize.

    sdata-plot v0.3.4 dropped vmin/vmax kwargs (#652); the wrapper builds
    the Normalize and passes it through `norm=`.
    """
    if norm is not None:
        if any(v is not None for v in (vmin, vmax, vcenter)):
            raise ValueError("Pass either `norm=` or `vmin`/`vmax`/`vcenter`, not both.")
        return norm
    if all(v is None for v in (vmin, vmax, vcenter)):
        return None
    if vcenter is not None:
        return TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    return Normalize(vmin=vmin, vmax=vmax)


def _normalize_library_ids(adata: AnnData, library_key: str | None, library_id: Any) -> tuple[str, ...]:
    if library_id is not None:
        ids = (library_id,) if isinstance(library_id, str) else tuple(library_id)
    elif library_key is not None:
        ids = tuple(map(str, adata.obs[library_key].cat.categories))
    elif Key.uns.spatial in adata.uns:
        ids = tuple(adata.uns[Key.uns.spatial].keys())
    else:
        raise ValueError("No library_id or library_key provided and no 'spatial' key in adata.uns.")
    return ids


def _normalize_color(color: str | Sequence[str] | None) -> tuple[str, ...]:
    if isinstance(color, str):
        return (color,)
    if color is None:
        return ()
    return tuple(color)


def _normalize_groups(groups: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if groups is None:
        return None
    if isinstance(groups, str):
        return (groups,)
    return tuple(groups)


def _per_library(value: Any, library_ids: tuple[str, ...], name: str) -> tuple[Any, ...]:
    """Broadcast a scalar or validate a sequence to library count.

    Disambiguates a crop tuple (2 or 4 ints/floats, single value) from a sequence
    of per-library values. For ambiguous cases, prefer the broadcast interpretation
    only when the tuple has exactly 2 or 4 numeric elements.
    """
    if value is None:
        return tuple(None for _ in library_ids)
    if isinstance(value, (list, tuple)) and not (
        len(value) in (2, 4) and all(isinstance(v, (int, float)) for v in value)
    ):
        if len(value) != len(library_ids):
            raise ValueError(f"`{name}` length {len(value)} != number of libraries {len(library_ids)}.")
        return tuple(value)
    return tuple(value for _ in library_ids)


def _per_library_scalar(value: Any, library_ids: tuple[str, ...], name: str) -> tuple[Any, ...]:
    """Broadcast a scalar to all libraries, or validate a sequence per library.

    For kwargs like `size` where a sequence is always per-library (never a single tuple).
    """
    if value is None:
        return tuple(None for _ in library_ids)
    if isinstance(value, (list, tuple)):
        if len(value) != len(library_ids):
            raise ValueError(f"`{name}` length {len(value)} != number of libraries {len(library_ids)}.")
        return tuple(value)
    return tuple(value for _ in library_ids)


def _resolve_palette(palette: Any) -> tuple[Any, Any, tuple[str, ...] | None]:
    """Route a squidpy `palette` value to the right sdata-plot slot.

    Returns (palette, cmap, groups). sdata-plot's render_shapes rejects `palette` without
    `groups`, but accepts `Colormap` via `cmap` even for categorical color (the renderer
    samples it by category index internally). So:
      - dict {category: color} -> palette + groups from keys
      - Colormap / ListedColormap -> route to cmap (no groups needed)
      - list of color strings -> wrap as ListedColormap -> cmap
      - str (single color, palette name) or None -> passthrough
    """
    from matplotlib.colors import Colormap, ListedColormap

    if palette is None:
        return None, None, None
    if isinstance(palette, dict):
        return palette, None, tuple(palette.keys())
    if isinstance(palette, Colormap):
        return None, palette, None
    if isinstance(palette, (list, tuple)):
        return None, ListedColormap(list(palette)), None
    return palette, None, None


def _expand_panels(
    library_ids: tuple[str, ...],
    color_tuple: tuple[str, ...],
    library_first: bool,
    crop_coord_per_lib: tuple[Any, ...],
    scalebar_dx_per_lib: tuple[Any, ...],
    scalebar_units_per_lib: tuple[Any, ...],
    size_per_lib: tuple[Any, ...],
    title: str | Sequence[str] | None,
) -> tuple[PanelIntent, ...]:
    """Flatten (library x color) into a panel list with the requested iteration order."""
    colors = color_tuple if color_tuple else (None,)
    if library_first:
        pairs = list(itertools.product(library_ids, colors))
    else:
        pairs = [(lib, col) for col, lib in itertools.product(colors, library_ids)]

    if isinstance(title, str):
        titles = [title] * len(pairs)
    elif title is None:
        titles = [None] * len(pairs)
    else:
        titles_seq = tuple(title)
        if len(titles_seq) != len(pairs):
            raise ValueError(f"`title` length {len(titles_seq)} != number of panels {len(pairs)}.")
        titles = list(titles_seq)

    lib_index = {lib: i for i, lib in enumerate(library_ids)}
    panels = []
    for (lib, col), t in zip(pairs, titles, strict=True):
        i = lib_index[lib]
        panels.append(
            PanelIntent(
                library_id=lib,
                color=col,
                size=size_per_lib[i],
                crop_coord=crop_coord_per_lib[i],
                scalebar_dx=scalebar_dx_per_lib[i],
                scalebar_units=scalebar_units_per_lib[i],
                title=t,
            )
        )
    return tuple(panels)


def _validate_ax(ax: Any, n_panels: int) -> tuple[Any, ...] | None:
    """Normalize user-supplied `ax` into a tuple matching panel count."""
    if ax is None:
        return None
    from matplotlib.axes import Axes

    if isinstance(ax, Axes):
        ax_seq = (ax,)
    else:
        ax_seq = tuple(ax)
    if len(ax_seq) != n_panels:
        raise ValueError(f"`ax` has {len(ax_seq)} axes but {n_panels} panels are required.")
    return ax_seq


def capture_scatter_intent(
    adata: AnnData,
    *,
    shape: str | None = "circle",
    color: str | Sequence[str] | None = None,
    groups: str | Sequence[str] | None = None,
    img: bool = True,
    img_res_key: str = Key.uns.image_res_key,
    library_key: str | None = None,
    library_id: str | Sequence[str] | None = None,
    spatial_key: str = Key.obsm.spatial,
    size_key: str = Key.uns.size_key,
    palette: Any = None,
    cmap: Any = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    alpha: float = 1.0,
    na_color: Any = (0.0, 0.0, 0.0, 0.0),
    use_raw: bool | None = None,
    layer: str | None = None,
    alt_var: str | None = None,
    outline: bool = False,
    outline_color: tuple[str, str] = ("black", "white"),
    outline_width: tuple[float, float] = (0.3, 0.05),
    size: float | Sequence[float] | None = None,
    connectivity_key: str | None = None,
    edges_width: float = 1.0,
    edges_color: str | Sequence[str] = "grey",
    edges_kwargs: Any = None,
    img_alpha: float | None = None,
    img_cmap: Any = None,
    img_channel: int | tuple[int, ...] | None = None,
    crop_coord: tuple[float, float, float, float] | Sequence[tuple[float, float, float, float]] | None = None,
    scalebar_dx: float | Sequence[float] | None = None,
    scalebar_units: str | Sequence[str] | None = None,
    scalebar_kwargs: Any = None,
    title: str | Sequence[str] | None = None,
    axis_label: str | Sequence[str] | None = None,
    frameon: bool | None = None,
    colorbar: bool = True,
    legend_loc: str | None = "right margin",
    legend_fontsize: Any = None,
    legend_fontweight: Any = "bold",
    legend_fontoutline: int | None = None,
    legend_na: bool = True,
    ncols: int = 4,
    library_first: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    fig: Any = None,
    ax: Any = None,
    save: str | None = None,
    return_ax: bool = False,
    **unsupported: Any,
) -> Intent:
    """Capture squidpy spatial_scatter kwargs into an Intent.

    Covers Paths 1+2 plus the stress-test parity surface. Kwargs still outside
    scope (connectivity_key/edges, legend_loc='on data', spatial_key override)
    raise NotImplementedError.
    """
    if unsupported:
        offenders = sorted(unsupported)
        raise NotImplementedError(f"spatial_scatter via spatialdata-plot does not yet support kwargs: {offenders}.")
    if legend_loc == "on data":
        import warnings

        warnings.warn(
            "legend_loc='on data' is deprecated for spatial plots: known to be unreliable "
            "in coordinate space and slated for removal. Use the default 'right margin' or pass "
            "legend_loc=None to hide.",
            DeprecationWarning,
            stacklevel=3,
        )
        legend_loc = "right margin"

    if shape is not None and shape not in {"circle", "hex", "square", "visium_hex"}:
        raise ValueError(f"shape must be None or one of {{'circle','hex','square','visium_hex'}}; got {shape!r}.")
    use_points = shape is None

    color_tuple = _normalize_color(color)
    library_ids = _normalize_library_ids(adata, library_key, library_id)

    crop_per_lib = _per_library(crop_coord, library_ids, "crop_coord")
    scalebar_dx_per_lib = _per_library(scalebar_dx, library_ids, "scalebar_dx")
    scalebar_units_per_lib = _per_library(scalebar_units, library_ids, "scalebar_units")
    size_per_lib = _per_library_scalar(size, library_ids, "size")

    panels = _expand_panels(
        library_ids,
        color_tuple,
        library_first,
        crop_per_lib,
        scalebar_dx_per_lib,
        scalebar_units_per_lib,
        size_per_lib,
        title,
    )

    ax_seq = _validate_ax(ax, len(panels))

    data = DataIntent(
        needs_shapes=not use_points,
        needs_points=use_points,
        needs_image=bool(img),
        needs_graph=connectivity_key is not None,
        library_ids=library_ids,
        library_key=library_key,
        img_res_key=img_res_key if img else None,
        img_channel=img_channel,
        color=color_tuple,
        use_raw=use_raw,
        layer=layer,
        alt_var=alt_var,
        size_key=size_key,
        graph_layer=connectivity_key,
    )

    resolved_norm = _build_norm(vmin=vmin, vmax=vmax, vcenter=vcenter, norm=norm)
    resolved_palette, palette_cmap, inferred_groups = _resolve_palette(palette)
    resolved_cmap = palette_cmap if cmap is None else cmap
    groups_tuple = _normalize_groups(groups) or inferred_groups

    render = RenderIntent(
        shape=shape,
        palette=resolved_palette,
        cmap=resolved_cmap,
        norm=resolved_norm,
        alpha=alpha,
        na_color=na_color,
        groups=groups_tuple,
        outline=outline,
        outline_color=outline_color,
        outline_width=outline_width,
        img_alpha=img_alpha,
        img_cmap=img_cmap,
        edges_width=edges_width,
        edges_color=edges_color,
        edges_kwargs=edges_kwargs or {},
    )

    layout = LayoutIntent(
        ncols=ncols,
        library_first=library_first,
        figsize=figsize,
        dpi=dpi,
        frameon=frameon,
        return_ax=return_ax,
        fig=fig,
        ax=ax_seq,
    )

    post = PostRenderIntent()

    return Intent(
        mode="scatter",
        data=data,
        render=render,
        layout=layout,
        post=post,
        panels=panels,
    )


def capture_segment_intent(
    adata: AnnData,
    *,
    seg_cell_id: str,
    color: str | Sequence[str] | None = None,
    groups: str | Sequence[str] | None = None,
    seg_key: str = Key.uns.image_seg_key,
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    img: bool = True,
    img_res_key: str = Key.uns.image_res_key,
    library_key: str | None = None,
    library_id: str | Sequence[str] | None = None,
    spatial_key: str = Key.obsm.spatial,
    palette: Any = None,
    cmap: Any = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    alpha: float = 1.0,
    na_color: Any = (0.0, 0.0, 0.0, 0.0),
    use_raw: bool | None = None,
    layer: str | None = None,
    alt_var: str | None = None,
    img_alpha: float | None = None,
    img_cmap: Any = None,
    img_channel: int | tuple[int, ...] | None = None,
    crop_coord: tuple[float, float, float, float] | Sequence[tuple[float, float, float, float]] | None = None,
    scalebar_dx: float | Sequence[float] | None = None,
    scalebar_units: str | Sequence[str] | None = None,
    scalebar_kwargs: Any = None,
    title: str | Sequence[str] | None = None,
    axis_label: str | Sequence[str] | None = None,
    frameon: bool | None = None,
    colorbar: bool = True,
    legend_loc: str | None = "right margin",
    legend_fontsize: Any = None,
    legend_fontweight: Any = "bold",
    legend_fontoutline: int | None = None,
    legend_na: bool = True,
    ncols: int = 4,
    library_first: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    fig: Any = None,
    ax: Any = None,
    save: str | None = None,
    return_ax: bool = False,
    **unsupported: Any,
) -> Intent:
    """Capture squidpy spatial_segment kwargs into an Intent.

    Routes through sdata-plot's render_labels at execution time.
    """
    if unsupported:
        offenders = sorted(unsupported)
        raise NotImplementedError(f"spatial_segment via spatialdata-plot does not yet support kwargs: {offenders}.")
    if legend_loc == "on data":
        import warnings

        warnings.warn(
            "legend_loc='on data' is deprecated for spatial plots: known to be unreliable "
            "in coordinate space and slated for removal. Use the default 'right margin' or pass "
            "legend_loc=None to hide.",
            DeprecationWarning,
            stacklevel=3,
        )
        legend_loc = "right margin"

    if seg_contourpx == 1:
        raise ValueError("seg_contourpx=1 is rejected by spatialdata-plot v0.3.4 (PR #645). Use >= 2 or None.")

    color_tuple = _normalize_color(color)
    library_ids = _normalize_library_ids(adata, library_key, library_id)

    crop_per_lib = _per_library(crop_coord, library_ids, "crop_coord")
    scalebar_dx_per_lib = _per_library(scalebar_dx, library_ids, "scalebar_dx")
    scalebar_units_per_lib = _per_library(scalebar_units, library_ids, "scalebar_units")
    size_per_lib = tuple(None for _ in library_ids)  # spatial_segment has no size kwarg

    panels = _expand_panels(
        library_ids,
        color_tuple,
        library_first,
        crop_per_lib,
        scalebar_dx_per_lib,
        scalebar_units_per_lib,
        size_per_lib,
        title,
    )

    ax_seq = _validate_ax(ax, len(panels))

    data = DataIntent(
        needs_labels=True,
        needs_image=bool(img),
        library_ids=library_ids,
        library_key=library_key,
        img_res_key=img_res_key if img else None,
        img_channel=img_channel,
        color=color_tuple,
        use_raw=use_raw,
        layer=layer,
        alt_var=alt_var,
        seg_cell_id=seg_cell_id,
    )

    resolved_norm = _build_norm(vmin=vmin, vmax=vmax, vcenter=vcenter, norm=norm)
    outline_alpha = 1.0 if seg_outline else 0.0
    resolved_palette, palette_cmap, inferred_groups = _resolve_palette(palette)
    resolved_cmap = palette_cmap if cmap is None else cmap
    groups_tuple = _normalize_groups(groups) or inferred_groups

    render = RenderIntent(
        cmap=resolved_cmap,
        norm=resolved_norm,
        palette=resolved_palette,
        alpha=alpha,
        na_color=na_color,
        contour_px=seg_contourpx,
        outline_alpha=outline_alpha,
        groups=groups_tuple,
        img_alpha=img_alpha,
        img_cmap=img_cmap,
    )

    layout = LayoutIntent(
        ncols=ncols,
        library_first=library_first,
        figsize=figsize,
        dpi=dpi,
        frameon=frameon,
        return_ax=return_ax,
        fig=fig,
        ax=ax_seq,
    )

    post = PostRenderIntent()

    return Intent(
        mode="segment",
        data=data,
        render=render,
        layout=layout,
        post=post,
        panels=panels,
    )


capture_scatter_intent_path1 = capture_scatter_intent
