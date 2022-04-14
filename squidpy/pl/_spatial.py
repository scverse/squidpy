from __future__ import annotations

from types import MappingProxyType
from typing import Union  # noqa: F401
from typing import Any, Tuple, Mapping, Optional, Sequence
from pathlib import Path
from functools import wraps
from typing_extensions import Literal
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


@d.get_sections(base="spatial_plot", sections=["Returns"])
@d.get_extended_summary(base="spatial_plot")
@d.get_summary(base="spatial_plot")
@d.dedent
def _spatial_plot(
    adata: AnnData,
    shape: _AvailShapes | None = None,
    color: Sequence[str | None] | str | None = None,
    groups: _SeqStr | None = None,
    library_id: _SeqStr | None = None,
    library_key: str | None = None,
    spatial_key: str = Key.obsm.spatial,
    # image
    img: _SeqArray | bool | None = True,
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
    legend_loc: str | None = "right margin",
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_fontoutline: Optional[int] = None,
    legend_na: bool = True,
    colorbar: bool = True,
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

    Use the parameter ``library_id`` to select the image.
    If multiple ``library_id`` are available, use ``library_key`` to plot subsets of :class:`anndata.AnnData` object.
    Use ``crop_coord`` to crop the spatial plot based on coordinate boundaries.

    This function has few key assumptions about how coordinates and libraries are handled:

        - The arguments ``library_key`` and ``library_id`` control which dataset is plotted.
          If multiple libraries are present, specifying solely ``library_key`` will suffice, and
          all unique libraries will be plotted sequentially. To select specific libraries, use
          the ``library_id`` argument.
        - The arguments ``color`` controls which features in obs/var are plotted. They are plotted
          for all available/specified libraries. The argument ``groups`` can be used to select
          categories to be plotted (valid only for categorical features in :attr:`adata.obs`).
        - If multiple ``library_id`` are available, arguments such as ``size`` and ``crop_coord``
          accept lists, to selectively customize different ``library_id`` plots. This requires that
          the length of such lists match the number of unique libraries in the dataset.
        - Coordinates are in the pixel space of the source image, so an equal aspect ratio is assumed.
        - The origin (e.g `(0, 0)`) is at the top left - as is common convention with image data.
        - The plotted points (dots) do not have a real "size" but only relative to their
          coordinate/pixel space. This does not hold if no image is plotted, then size correspond
          to points size passed to :meth:`matplotlib.axes.Axes.scatter`.

    If your anndata object has a `"spatial"` entry in :attr:`anndata.AnnData.uns`,
    use ``img_key``, ``seg_key`` and ``size_key`` parameters to find values
    for ``img``, ``seg`` and ``size``.
    Alternatively, these values can be passed directly.

    Parameters
    ----------
    %(adata)s
    %(shape)s
    %(color)s
    %(groups)s
    %(library_id)s
    %(library_key)s
    %(spatial_key)s
    %(plotting_image)s
    %(plotting_segment)s
    %(plotting_features)s

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
        adata_sub, coords_sub, image_sub = _subs(
            adata,
            _coords,
            _img,
            library_key=library_key,
            library_id=_lib,
            crop_coords=_crops,
            groups_key=value_to_plot,
            groups=color_params.groups,
        )
        color_source_vector, color_vector, categorical = _set_color_source_vec(
            adata_sub,
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
            _plot_edges(
                adata_sub,
                coords_sub,
                connectivity_key,
                ax=ax,
                edges_width=edges_width,
                edges_color=edges_color,
                **edges_kwargs,
            )

        if _seg is None and _cell_id is None:
            outline_params, kwargs = _set_outline(
                size=_size, outline=outline, outline_width=outline_width, outline_color=outline_color, **kwargs
            )

            ax, cax = _plot_scatter(
                coords=coords_sub,
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

        _ = _decorate_axs(
            ax=ax,
            cax=cax,
            lib_count=_lib_count,
            fig_params=fig_params,
            adata=adata_sub,
            coords=coords_sub,
            value_to_plot=value_to_plot,
            color_source_vector=color_source_vector,
            img=image_sub,
            img_cmap=cmap_params.img_cmap,
            img_alpha=color_params.img_alpha,
            palette=palette,
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_loc=legend_loc,
            legend_fontoutline=legend_fontoutline,
            na_color=na_color,
            na_in_legend=legend_na,
            colorbar=colorbar,
            scalebar_dx=scalebar_params.scalebar_dx,
            scalebar_units=scalebar_params.scalebar_units,
            scalebar_kwargs=scalebar_kwargs,
        )

    if fig_params.fig is not None and save is not None:
        save_fig(fig_params.fig, path=save)


def _wrap_signature(key: Literal["spatial_scatter", "spatial_segment"]) -> Any:
    def _wrap_plot(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect

            params = inspect.signature(_spatial_plot).parameters.copy()
            wrapper_sig = inspect.signature(func)
            wrapper_params = wrapper_sig.parameters.copy()

            if key == "spatial_scatter":
                params.pop("seg")
                params.pop("seg_key")
                params.pop("seg_cell_id")
                params.pop("seg_contourpx")
                params.pop("seg_outline")
                wrapper_params.pop("adata")
                wrapper_params.pop("shape")
            elif key == "spatial_segment":
                params.pop("shape")
                params.pop("size")
                params.pop("size_key")
                params.pop("scale_factor")
                wrapper_params.pop("adata")
                wrapper_params.pop("seg_cell_id")
                wrapper_params.pop("seg")
                wrapper_params.pop("seg_key")
                wrapper_params.pop("seg_contourpx")
                wrapper_params.pop("seg_outline")
            else:
                raise NotImplementedError("Function signature not implemented.")

            params.update(wrapper_params)
            annotations = {k: v.annotation for k, v in params.items() if v.annotation != inspect.Parameter.empty}
            if wrapper_sig.return_annotation is not inspect.Signature.empty:
                annotations["return"] = wrapper_sig.return_annotation

            func.__signature__ = inspect.Signature(
                list(params.values()), return_annotation=wrapper_sig.return_annotation
            )
            func.__annotations__ = annotations

            return func(*args, **kwargs)

        return wrapper

    return _wrap_plot


@_wrap_signature(key="spatial_scatter")
@d.dedent
def spatial_scatter(
    adata: AnnData,
    shape: _AvailShapes | None = ScatterShape.CIRCLE,  # type: ignore[assignment]
    color: Sequence[str | None] | str | None = None,
    groups: _SeqStr | None = None,
    library_id: _SeqStr | None = None,
    library_key: str | None = None,
    spatial_key: str = Key.obsm.spatial,
    # image
    img: _SeqArray | bool | None = True,
    img_res_key: str | None = Key.uns.image_res_key,
    img_alpha: float | None = None,
    img_cmap: Colormap | str | None = None,
    img_channel: int | None = None,
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
    legend_loc: str | None = "right margin",
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_fontoutline: Optional[int] = None,
    legend_na: bool = True,
    colorbar: bool = True,
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
) -> Any:
    """
    %(spatial_plot.summary)s

    This function allows overlaying data on top of images.
    The plotted shapes (circles, squares or hexagons) have a real "size" with respect to their
    coordinate space, which can be specified via the ``size`` or ``size_key`` parameter.

        - Use the parameter ``img_key`` to see the image in the background.
        - Use the parameter ``library_id`` to select the image. By default, ``'hires'`` key is attempted.
        - Use ``img_alpha``, ``img_cmap`` or ``img_channel`` to control how it is displayed.
        - Use ``size`` to scale the size of the shapes plotted on top.

    If no image is present or plotted, it will defaults to a scatter plot,
    see :func:`matplotlib.axes.Axes.scatter`.

    %(spatial_plot.summary_ext)s

    Parameters
    ----------
    %(adata)s
    %(shape)s
    %(color)s
    %(groups)s
    %(library_id)s
    %(library_key)s
    %(spatial_key)s
    %(plotting_image)s
    %(plotting_features)s

    Returns
    -------
    %(spatial_plot.returns)s
    """  # noqa: D400
    locs = locals()
    kwargs = locs.pop("kwargs", {})
    return _spatial_plot(**locs, **kwargs)


@_wrap_signature(key="spatial_segment")
@d.dedent
@inject_docs(key=Key.obsp.spatial_conn())
def spatial_segment(
    adata: AnnData,
    seg_cell_id: str,
    seg: _SeqArray | bool | None = True,
    seg_key: str = Key.uns.image_seg_key,
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    # colors
    color: Sequence[str | None] | str | None = None,
    groups: _SeqStr | None = None,
    library_id: _SeqStr | None = None,
    library_key: str | None = None,
    spatial_key: str = Key.obsm.spatial,
    # image
    img: _SeqArray | bool | None = True,
    img_res_key: str | None = Key.uns.image_res_key,
    img_alpha: float | None = None,
    img_cmap: Colormap | str | None = None,
    img_channel: int | None = None,
    # features
    use_raw: bool | None = None,
    layer: str | None = None,
    alt_var: str | None = None,
    # size, coords, cmap, palette
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
    legend_loc: str | None = "right margin",
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_fontoutline: Optional[int] = None,
    legend_na: bool = True,
    colorbar: bool = True,
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
) -> Any:
    """
    %(spatial_plot.summary)s

    This function allows overlaying segmentation masks on top of images. ``seg_cell_id`` is a mandatory argument
    in :attr:`anndata.AnnData.obs` to control unique segmentation masks's ids to be plotted.
    By default, ``'segmentation'`` ``seg_key`` is attempted and ``'hires'`` image key is attempted.

        - Use the parameter ``seg_key`` to see the image in the background.
        - Use ``seg_contourpx`` or ``seg_outline`` to control how the segmentation mask is displayed.

    %(spatial_plot.summary_ext)s

    Parameters
    ----------
    %(adata)s
    %(plotting_segment)s
    %(color)s
    %(groups)s
    %(library_id)s
    %(library_key)s
    %(spatial_key)s
    %(plotting_image)s
    %(plotting_features)s

    Returns
    -------
    %(spatial_plot.returns)s
    """  # noqa: D400
    locs = locals()
    kwargs = locs.pop("kwargs", {})
    return _spatial_plot(**locs, **kwargs)
