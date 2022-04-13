from __future__ import annotations

from types import MappingProxyType
from typing import Union  # noqa: F401
from typing import Any, Tuple, Mapping, Callable, Optional, Sequence
from pathlib import Path
from functools import partial
import itertools

from anndata import AnnData

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from squidpy._docs import d, inject_docs
from squidpy.gr._utils import _assert_spatial_basis
from squidpy.pl._utils import save_fig, sanitize_anndata
from squidpy.pl._spatial_utils import (
    _subs,
    _SeqStr,
    _FontSize,
    _SeqArray,
    _SeqFloat,
    Palette_t,
    _Normalize,
    _CoordTuple,
    _FontWeight,
    _plot_edges,
    _AvailShapes,
    _set_outline,
    _decorate_axs,
    _plot_scatter,
    _plot_segment,
    _set_ax_title,
    _set_coords_crops,
    _prepare_args_plot,
    _image_spatial_attrs,
    _prepare_params_plot,
    _set_color_source_vec,
)
from squidpy._constants._constants import ScatterShape
from squidpy._constants._pkg_constants import Key


@d.dedent
def _spatial_plot(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    shape: _AvailShapes | None = None,
    color: Sequence[str | None] | str | None = None,
    groups: _SeqStr | None = None,
    library_id: _SeqStr | None = None,
    library_key: str | None = None,
    # image
    img: _SeqArray | bool | None = None,
    img_res_key: str | None = Key.uns.image_res_key,
    img_alpha: float | None = None,
    img_cmap: Colormap | str | None = None,
    img_channel: int | None = None,
    # segment
    seg: _SeqArray | bool | None = None,
    seg_key: str | None = Key.uns.image_seg_key,
    seg_cell_id: str | None = None,
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    # features
    use_raw: bool | None = None,
    layer: str | None = None,
    alt_var: str | None = None,
    # size, coords, cmap, palette
    size: _SeqFloat | None = None,
    size_key: str | None = Key.uns.size_key,
    scale_factor: _SeqFloat | None = None,
    crop_coord: Sequence[_CoordTuple] | _CoordTuple | None = None,
    cmap: Colormap | str | None = None,
    palette: Palette_t = None,
    alpha: float = 1.0,
    norm: _Normalize | None = None,
    na_color: str | Tuple[float, ...] = (0, 0, 0, 0),
    # edges
    connectivity_key: str | None = None,
    edges_width: float = 1,
    edges_color: str | Sequence[str] | Sequence[float] = "grey",
    # panels
    library_first: bool = True,
    frameon: Optional[bool] = None,
    wspace: float | None = None,
    hspace: float = 0.25,
    ncols: int = 4,
    # outline
    outline: bool = False,
    outline_color: Tuple[str, str] = ("black", "white"),
    outline_width: Tuple[float, float] = (0.3, 0.05),
    # legend
    legend_loc: str = "right margin",
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_fontoutline: Optional[int] = None,
    legend_na: bool = True,
    # scalebar
    scalebar_dx: _SeqFloat | None = None,
    scalebar_units: _SeqStr | None = None,
    # title and axis
    title: _SeqStr | None = None,
    axis_label: _SeqStr | None = None,
    fig: Figure | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    figsize: Tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    # kwargs
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> None:
    """
    Plot spatial omics data saved in AnnData.

    This is a general purpose function for plotting scatterplots, images and segmentation masks.
    Consider using :func:`squidpy.pl.spatial_scatter` or :func:`squidpy.pl.spatial_segment` for
    specific use cases.

    %(plotting_general_summary)s

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(shape)s
    %(color)s
    %(groups)s
    %(library_id)s
    %(library_key)s
    %(plotting_image)s
    %(plotting_segment)s
    %(plotting_features)s
    scalebar_kwargs
        Keyword arguments for :func:`matplotlib_scalebar.ScaleBar`.
    edges_kwargs
        Keyword arguments for :func:`networkx.draw_networkx_edges`.
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.scatter` or :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    %(plotting_returns)s
    """
    sanitize_anndata(adata)
    _assert_spatial_basis(adata, spatial_key)

    scalebar_kwargs = dict(scalebar_kwargs)
    edges_kwargs = dict(edges_kwargs)

    color_params = _prepare_args_plot(
        adata=adata,
        shape=shape,
        color=color,
        groups=groups,
        alpha=alpha,
        img_alpha=img_alpha,
        use_raw=use_raw,
        layer=layer,
        palette=palette,
    )

    spatial_params = _image_spatial_attrs(
        adata=adata,
        shape=shape,
        spatial_key=spatial_key,
        library_id=library_id,
        library_key=library_key,
        img=img,
        img_res_key=img_res_key,
        img_channel=img_channel,
        seg=seg,
        seg_key=seg_key,
        cell_id_key=seg_cell_id,
        scale_factor=scale_factor,
        size=size,
        size_key=size_key,
        img_cmap=img_cmap,
    )

    coords, crops = _set_coords_crops(
        adata=adata,
        spatial_params=spatial_params,
        spatial_key=spatial_key,
        library_key=library_key,
        crop_coord=crop_coord,
    )

    fig_params, cmap_params, scalebar_params, kwargs = _prepare_params_plot(
        color_params=color_params,
        spatial_params=spatial_params,
        spatial_key=spatial_key,
        wspace=wspace,
        hspace=hspace,
        ncols=ncols,
        cmap=cmap,
        norm=norm,
        library_first=library_first,
        img_cmap=img_cmap,
        frameon=frameon,
        na_color=na_color,
        title=title,
        axis_label=axis_label,
        scalebar_dx=scalebar_dx,
        scalebar_units=scalebar_units,
        dpi=dpi,
        figsize=figsize,
        fig=fig,
        ax=ax,
        **kwargs,
    )

    _subset: Callable[[AnnData, str | None, str | None], AnnData] = (
        partial(_subs, library_key=library_key) if library_key is not None else lambda _, **__: _  # type: ignore
    )

    for count, (_lib_count, value_to_plot) in enumerate(itertools.product(*fig_params.iter_panels)):
        if not library_first:
            _lib_count, value_to_plot = value_to_plot, _lib_count

        _size = spatial_params.size[_lib_count]
        _img = spatial_params.img[_lib_count]
        _seg = spatial_params.segment[_lib_count]
        _cell_id = spatial_params.cell_id[_lib_count]
        _crops = crops[_lib_count]
        _lib = spatial_params.library_id[_lib_count]
        _coords = coords[_lib_count]  # TODO: do we want to order points? for now no, skip

        color_source_vector, color_vector, categorical = _set_color_source_vec(
            _subset(adata, library_id=_lib),  # type: ignore
            value_to_plot,
            layer=layer,
            use_raw=color_params.use_raw,
            alt_var=alt_var,
            groups=color_params.groups,
            palette=palette,
            na_color=na_color,
        )

        # set frame and title
        ax = _set_ax_title(fig_params, count, value_to_plot)

        # plot edges and arrows if needed. Do it here cause otherwise image is on top.
        if connectivity_key is not None:
            _cedge = _plot_edges(
                _subset(adata, library_id=_lib),  # type: ignore
                _coords,
                ax,
                edges_width,
                edges_color,
                connectivity_key,
                **edges_kwargs,
            )
            ax.add_collection(_cedge)

        if _seg is None and _cell_id is None:
            outline_params, kwargs = _set_outline(
                size=_size, outline=outline, outline_width=outline_width, outline_color=outline_color, **kwargs
            )

            ax, cax = _plot_scatter(
                coords=_coords,
                ax=ax,
                outline_params=outline_params,
                cmap_params=cmap_params,
                color_params=color_params,
                size=_size,
                color_vector=color_vector,
                na_color=na_color,
                **kwargs,
            )
        elif _seg is not None and _cell_id is not None:
            ax, cax = _plot_segment(
                seg=_seg,
                cell_id=_cell_id,
                color_vector=color_vector,
                color_source_vector=color_source_vector,
                seg_contourpx=seg_contourpx,
                seg_outline=seg_outline,
                na_color=na_color,
                ax=ax,
                cmap_params=cmap_params,
                color_params=color_params,
                categorical=categorical,
                **kwargs,
            )

        ax = _decorate_axs(
            ax=ax,
            cax=cax,
            lib_count=_lib_count,
            fig_params=fig_params,
            adata=adata,
            coords=_coords,
            value_to_plot=value_to_plot,
            color_source_vector=color_source_vector,
            crops=_crops,
            img=_img,
            img_cmap=cmap_params.img_cmap,
            img_alpha=color_params.img_alpha,
            palette=palette,
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_loc=legend_loc,
            legend_fontoutline=legend_fontoutline,
            na_color=na_color,
            na_in_legend=legend_na,
            scalebar_dx=scalebar_params.scalebar_dx,
            scalebar_units=scalebar_params.scalebar_units,
            scalebar_kwargs=scalebar_kwargs,
        )

    if fig_params.fig is not None and save is not None:
        save_fig(fig_params.fig, path=save)


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn())
def spatial_scatter(
    adata: AnnData,
    shape: _AvailShapes | None = ScatterShape.CIRCLE,  # type: ignore[assignment]
    img: _SeqArray | bool | None = True,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Any:
    """
    Plot scatter plot of shapes in spatial coordinates.

    %(plotting_shape_summary)s

    %(plotting_general_summary)s

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(shape)s
    %(color)s
    %(groups)s
    %(library_id)s
    %(library_key)s
    %(plotting_image)s
    %(plotting_features)s
    scalebar_kwargs
        Keyword arguments for :func:`matplotlib_scalebar.ScaleBar`.
    edges_kwargs
        Keyword arguments for :func:`networkx.draw_networkx_edges`.
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.scatter` or :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    %(plotting_returns)s
    """
    return _spatial_plot(
        adata,
        shape=shape,
        img=img,
        seg_key=None,
        scalebar_kwargs=scalebar_kwargs,
        edges_kwargs=edges_kwargs,
        **kwargs,
    )


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn())
def spatial_segment(
    adata: AnnData,
    seg_cell_id: str,
    seg_key: str = Key.uns.image_seg_key,
    img: _SeqArray | bool | None = True,
    seg: _SeqArray | bool | None = True,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Any:
    """
    Plot segmentation masks in spatial coordinates.

    %(plotting_segment_summary)s

    %(plotting_general_summary)s

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(color)s
    %(groups)s
    %(library_id)s
    %(library_key)s
    %(plotting_image)s
    %(plotting_segment)s
    %(plotting_features)s
    scalebar_kwargs
        Keyword arguments for :func:`matplotlib_scalebar.ScaleBar`.
    edges_kwargs
        Keyword arguments for :func:`networkx.draw_networkx_edges`.
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.scatter` or :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    %(plotting_returns)s
    """
    return _spatial_plot(
        adata,
        img=img,
        seg=seg,
        seg_key=seg_key,
        seg_cell_id=seg_cell_id,
        scalebar_kwargs=scalebar_kwargs,
        edges_kwargs=edges_kwargs,
        **kwargs,
    )
