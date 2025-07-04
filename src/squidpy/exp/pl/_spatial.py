from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import spatialdata as sd
import spatialdata_plot
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from squidpy._constants._constants import ScatterShape
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d
from squidpy.exp.pl._spatial_utils import (
    Palette_t,
    _AvailShapes,
    _CoordTuple,
    _decorate_axs,
    _FontSize,
    _FontWeight,
    _get_image,
    _get_library_id,
    _get_title_axlabels,
    _image_spatial_attrs,
    _Normalize,
    _plot_edges,
    _plot_scatter,
    _plot_segment,
    _prepare_args_plot,
    _prepare_params_plot,
    _SeqArray,
    _SeqFloat,
    _SeqStr,
    _set_ax_title,
    _set_color_source_vec,
    _set_coords_crops,
    _set_outline,
    _subs,
)
from squidpy.pl._utils import save_fig


@d.get_sections(base="spatial_plot", sections=["Returns"])
@d.get_extended_summary(base="spatial_plot")
@d.dedent
def _spatial_plot(
    sdata: sd.SpatialData,
    shape: _AvailShapes | None = None,
    color: str | Sequence[str | None] | None = None,
    groups: _SeqStr | None = None,
    library_id: _SeqStr | None = None,
    library_key: str | None = None,
    spatial_key: str = Key.obsm.spatial,
    # image
    img: bool | _SeqArray | None = True,
    img_res_key: str | None = Key.uns.image_res_key,
    img_alpha: float | None = None,
    img_cmap: Colormap | str | None = None,
    img_channel: int | list[int] | None = None,
    # segment
    seg: bool | _SeqArray | None = None,
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
    crop_coord: _CoordTuple | Sequence[_CoordTuple] | None = None,
    cmap: str | Colormap | None = None,
    palette: Palette_t = None,
    alpha: float = 1.0,
    norm: _Normalize | None = None,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
    # edges
    connectivity_key: str | None = None,
    edges_width: float = 1.0,
    edges_color: str | Sequence[str] | Sequence[float] = "grey",
    # panels
    library_first: bool = True,
    frameon: bool | None = None,
    wspace: float | None = None,
    hspace: float = 0.25,
    ncols: int = 4,
    # outline
    outline: bool = False,
    outline_color: tuple[str, str] = ("black", "white"),
    outline_width: tuple[float, float] = (0.3, 0.05),
    # legend
    legend_loc: str | None = "right margin",
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_fontoutline: int | None = None,
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
    return_ax: bool = False,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    # kwargs
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Axes | Sequence[Axes] | None:
    """
    Plot spatial omics data.

    Use ``library_id`` to select the image. If multiple ``library_id`` are available, use ``library_key`` in
    :attr:`anndata.AnnData.obs` to plot the subsets.
    Use ``crop_coord`` to crop the spatial plot based on coordinate boundaries.

    This function has few key assumptions about how coordinates and libraries are handled:

        - The arguments ``library_key`` and ``library_id`` control which dataset is plotted.
          If multiple libraries are present, specifying solely ``library_key`` will suffice, and all unique libraries
          will be plotted sequentially. To select specific libraries, use the ``library_id`` argument.
        - The argument ``color`` controls which features in obs/var are plotted. They are plotted for all
          available/specified libraries. The argument ``groups`` can be used to select categories to be plotted.
          This is valid only for categorical features in :attr:`anndata.AnnData.obs`.
        - If multiple ``library_id`` are available, arguments such as ``size`` and ``crop_coord`` accept lists to
          selectively customize different ``library_id`` plots. This requires that the length of such lists matches
          the number of unique libraries in the dataset.
        - Coordinates are in the pixel space of the source image, so an equal aspect ratio is assumed.
        - The origin *(0, 0)* is on the top left, as is common convention with image data.
        - The plotted points (dots) do not have a real "size" but it is relative to their coordinate/pixel space.
          This does not hold if no image is plotted, then the size corresponds to points size passed to
          :meth:`matplotlib.axes.Axes.scatter`.

    If :attr:`anndata.AnnData.uns` ``['spatial']`` is present, use ``img_key``, ``seg_key`` and
    ``size_key`` arguments to find values for ``img``, ``seg`` and ``size``.
    Alternatively, these values can be passed directly via ``img``.

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
    scalebar_kwargs = dict(scalebar_kwargs)
    edges_kwargs = dict(edges_kwargs)

    if color is not None and not isinstance(color, str):  #  TODO: different condition ?! (multiple library ids?)
        # > 1 plot to do (multiple panels), need multiple calls
        # TODO
        pass
    else:  # TODO: make a wrapper function for this whole part (single panel)
        # crop image if necessary
        if crop_coord is not None:  # TODO: what if this is a sequence of coords? only for multiple plots though
            sdata = sd.bounding_box_query(
                sdata,
                axes=("x", "y"),  # TODO: change?
                min_coordinate=[crop_coord[0], crop_coord[1]],
                max_coordinate=[crop_coord[2], crop_coord[3]],
                target_coordinate_system="global",  # TODO: change?
            )
        # render image
        # TODO: consider img argument!!! (can be false, or an image itself!)
        sdata = sdata.pl.render_images(
            # element, TODO: specify?
            channel=img_channel,
            cmap=img_cmap,
            # norm=None, TODO: I think squidpy doesn't support this
            # na_color="default", TODO: not supported by squidpy, could use na_color (default: 0, 0, 0, 0)
            # palette=None, not supported by squidpy
            alpha=img_alpha,
            # scale: str | None = None, TODO: not really supported by squidpy (maybe img_res_key refactor?)
            # **kwargs: Additional args passed to cmap, norm, ...
        )
        # TODO: not used so far: img_res_key, would be needed if we had anndata which is not the case
        # TODO: points, labels ???
        sdata = sdata.pl.render_shapes(
            # element, TODO: specify?
            color=color,
            fill_alpha=alpha,  # TODO: can be None, convert?
            groups=groups,
            palette=palette,
            na_color=na_color,
            outline_width=outline_width[0],  # TODO: double in squidpy, diff defaults
            outline_color=outline_color[0],  # TODO: double in squidpy
            outline_alpha=1.0 if outline else 0.0,  # TODO: alpha not supported in squidpy
            cmap=cmap,
            norm=norm,
            scale=1.0 if scale_factor is None else scale_factor,  # NOTE: problem if default in sdata-plot changes!
            # method, not supported by squidpy, we already have a heuristic
            # table_name, not supported by squidpy
            # table_layer, not supported by squidpy
            # **kwargs, TODO: includes datashader_reduction, set that to e.g. max?
        )

    # finally: call pl.show()
    result = sdata.pl.show(
        # coordinate_systems, TODO: needs to be an input argument???
        legend_fontsize=legend_fontsize,
        legend_fontweight=legend_fontweight,
        legend_loc=legend_loc,
        legend_fontoutline=legend_fontoutline,
        na_in_legend=legend_na,
        colorbar=colorbar,
        wspace=wspace,
        hspace=hspace,
        ncols=ncols,
        frameon=frameon,
        figsize=figsize,
        dpi=dpi,
        fig=fig,
        title=title,  # TODO: per default, set to color if not None?
        # share_extent: bool = True, TODO: extra input?
        # pad_extent: int | float = 0, TODO: "
        ax=ax,
        return_ax=True,
        save=save,
    )

    # set title and axis labels (potential TODO: outsource to spatialdata-plot?)
    title, ax_labels = _get_title_axlabels(title, axis_label, spatial_key, n_plots=1)  # TODO: n_plots!!!
    if title is not None:
        result.set_title(title)
    elif color is not None and isinstance(color, str):
        result.set_title(color)  # TODO: adapt for multiple panels...
    else:
        result.set_title(None)
    result.set_xlabel(ax_labels[0])
    result.set_ylabel(ax_labels[1])
    # TODO: possibly remove later, squidpy originally doesn't show ticks
    result.set_xticks([])
    result.set_yticks([])

    if return_ax:
        return result


def _wrap_signature(wrapper: Callable[[Any], Any]) -> Callable[[Any], Any]:
    import inspect

    name = wrapper.__name__
    params = inspect.signature(_spatial_plot).parameters.copy()
    wrapper_sig = inspect.signature(wrapper)
    wrapper_params = wrapper_sig.parameters.copy()

    if name == "spatial_scatter":
        params_remove = [
            "seg",
            "seg_cell_id",
            "seg_key",
            "seg_contourpx",
            "seg_outline",
        ]
        wrapper_remove = ["shape"]
    elif name == "spatial_segment":
        params_remove = ["shape", "size", "size_key", "scale_factor"]
        wrapper_remove = [
            "seg_cell_id",
            "seg",
            "seg_key",
            "seg_contourpx",
            "seg_outline",
        ]
    else:
        raise NotImplementedError(f"Docstring interpolation not implemented for `{name}`.")

    for key in params_remove:
        params.pop(key)
    for key in wrapper_remove:
        wrapper_params.pop(key)

    params.update(wrapper_params)
    annotations = {k: v.annotation for k, v in params.items() if v.annotation != inspect.Parameter.empty}
    if wrapper_sig.return_annotation is not inspect.Signature.empty:
        annotations["return"] = wrapper_sig.return_annotation

    wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        list(params.values()), return_annotation=wrapper_sig.return_annotation
    )
    wrapper.__annotations__ = annotations

    return wrapper


@d.dedent
@_wrap_signature
def spatial_scatter(
    sdata: sd.SpatialData,
    shape: _AvailShapes | None = ScatterShape.CIRCLE.v,
    **kwargs: Any,
) -> Axes | Sequence[Axes] | None:
    """
    Plot spatial omics data with data overlayed on top.

    The plotted shapes (circles, squares or hexagons) have a real "size" with respect to their
    coordinate space, which can be specified via the ``size`` or ``size_key`` argument.

        - Use ``img_key`` to display the image in the background.
        - Use ``library_id`` to select the image. By default, ``'hires'`` is attempted.
        - Use ``img_alpha``, ``img_cmap`` and ``img_channel`` to control how it is displayed.
        - Use ``size`` to scale the size of the shapes plotted on top.

    If no image is plotted, it defaults to a scatter plot, see :meth:`matplotlib.axes.Axes.scatter`.

    %(spatial_plot.summary_ext)s

    .. seealso::
        - :func:`squidpy.pl.spatial_segment` on how to plot spatial data with segmentation masks on top.

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
    """
    return _spatial_plot(sdata, shape=shape, seg=None, seg_key=None, **kwargs)


@d.dedent
@_wrap_signature  # type: ignore[arg-type]
def spatial_segment(
    sdata: sd.SpatialData,
    seg_cell_id: str,
    seg: bool | _SeqArray | None = True,
    seg_key: str = Key.uns.image_seg_key,
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    **kwargs: Any,
) -> Axes | Sequence[Axes] | None:
    """
    Plot spatial omics data with segmentation masks on top.

    Argument ``seg_cell_id`` in :attr:`anndata.AnnData.obs` controls unique segmentation mask's ids to be plotted.
    By default, ``'segmentation'``, ``seg_key`` for the segmentation and ``'hires'`` for the image is attempted.

        - Use ``seg_key`` to display the image in the background.
        - Use ``seg_contourpx`` or ``seg_outline`` to control how the segmentation mask is displayed.

    %(spatial_plot.summary_ext)s

    .. seealso::
        - :func:`squidpy.pl.spatial_scatter` on how to plot spatial data with overlayed data on top.

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
    """
    return _spatial_plot(
        sdata,
        seg=seg,
        seg_key=seg_key,
        seg_cell_id=seg_cell_id,
        seg_contourpx=seg_contourpx,
        seg_outline=seg_outline,
        **kwargs,
    )
