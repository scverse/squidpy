from __future__ import annotations

from copy import copy
from types import MappingProxyType
from typing import Any, Tuple, Mapping, Optional, Sequence, TYPE_CHECKING
from pathlib import Path
from functools import partial
import itertools

from anndata import AnnData
from scanpy._settings import settings as sc_settings
from scanpy.plotting._utils import _FontSize, _FontWeight, check_colornorm
from scanpy.plotting._tools.scatterplots import _panel_grid, _get_vboundnorm

from pandas.core.dtypes.common import is_categorical_dtype
import numpy as np

from matplotlib import pyplot as pl, rcParams
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from squidpy.gr._utils import _assert_spatial_basis
from squidpy.pl._graph import _maybe_set_colors
from squidpy.pl._utils import save_fig, _sanitize_anndata, _assert_value_in_obs
from squidpy.pl._spatial_utils import (
    _subs,
    _SeqStr,
    _VBound,
    _SeqArray,
    _SeqFloat,
    Palette_t,
    _Normalize,
    _CoordTuple,
    _plot_edges,
    _AvailShapes,
    _decorate_axs,
    _get_scalebar,
    _get_color_vec,
    _map_color_seg,
    _get_source_vec,
    _shaped_scatter,
    _get_cmap_params,
    _get_coords_crops,
    _get_title_axlabels,
    _image_spatial_attrs,
)
from squidpy._constants._constants import ScatterShape
from squidpy._constants._pkg_constants import Key


def spatial_point(
    adata: AnnData,
    **kwargs: Any,
) -> Any:
    """Spatial point."""
    _spatial_plot(adata, shape=None, **kwargs)


def spatial_shape(
    adata: AnnData,
    **kwargs: Any,
) -> Any:
    """Spatial shape."""
    _spatial_plot(adata, **kwargs)


def spatial_segment(
    adata: AnnData,
    seg_key: str = Key.uns.seg_key,
    na_color: str | Tuple[float, ...] = (0, 0, 0, 0),  # "lightgray",
    **kwargs: Any,
) -> Any:
    """Spatial shape."""
    _spatial_plot(adata, shape=None, na_color=na_color, seg_key=seg_key, **kwargs)


def _spatial_plot(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    shape: _AvailShapes | None = ScatterShape.CIRCLE.s,  # type: ignore[assignment]
    color: Sequence[str | None] | str | None = None,
    groups: _SeqStr = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    alt_var: str | None = None,
    library_id: _SeqStr = None,
    library_key: str | None = None,
    library_first: bool = True,
    cmap: Colormap | str | None = None,
    cmap_kwargs: Mapping[str, _VBound | _Normalize] | None = None,
    palette: Palette_t = None,
    alpha: Optional[float] = None,
    scale_factor: _SeqFloat = None,
    size: _SeqFloat = None,
    size_key: str = Key.uns.size_key,
    crop_coord: Sequence[_CoordTuple] | _CoordTuple | None = None,
    img: _SeqArray = None,
    img_key: str | None = None,
    img_alpha: Optional[float] = None,
    img_cmap: Colormap | str | None = None,
    img_channel: int | None = None,
    seg_key: str = Key.uns.seg_key,
    cell_id_key: str | None = None,
    seg_boundaries: bool = False,
    bw: bool = False,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: str | Sequence[float] | Sequence[str] = "grey",
    connectivity_key: str = Key.obsp.spatial_conn(),
    frameon: Optional[bool] = None,
    title: _SeqStr = None,
    axis_label: _SeqStr = None,
    wspace: float | None = None,
    hspace: float = 0.25,
    ncols: int = 4,
    add_outline: bool = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ("black", "white"),
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str = "right margin",
    legend_fontoutline: Optional[int] = None,
    na_color: str | Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    na_in_legend: bool = True,
    scalebar_dx: _SeqFloat = None,
    scalebar_units: _SeqStr = None,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
    dpi: int | None = None,
    ax: Axes | None = None,
    save: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """
    Spatial plotting for squidpy.

    Parameters
    ----------
    adata
    spatial_key
    library_id
    library_key
        If multiple library_id then library_key necessary in order to subset adata[adata.obs[library_key]==_library_id]
    shape
        This is the master argument for visium v. non-visium. If None, then it's just scatter and args are overridden.
    img
        To pass images not saved in anndata
    img_key
        To get images saved in anndata
    scale_factor
    size
        If shape is not None (e.g. visium) then it functions as a scaling factor otherwise it's true size of point.
    size_key
        To get e.g. diameter in Visium, otherwise manually passed.
    crop_coord
    bw
        for black and white of image.
    img_alpha
        for alpha of image
    color
    groups
    use_raw
    layer
    alt_var
        This is equivalent ot gene_symbol in embedding. See scanpy docs.
    edges
    edges_width
    edges_color
    neighbors_key
    palette
    cmap
    na_color
    frameon
    title
    axis_label
        to change something else from spatial1,2
    wspace
    hspace
    ncols
    vmin
    vmax
    vcenter
    norm
    add_outline
    ...
    scalebar_dx
        px_to_scale factor for scalebar
    scalebar_units
        units, by default is "um".


    Returns
    -------
    plotting_returns
    """
    _sanitize_anndata(adata)
    _assert_spatial_basis(adata, spatial_key)

    scalebar_kwargs = dict(scalebar_kwargs)
    edges_kwargs = dict(edges_kwargs)

    alpha = 1.0 if alpha is None else alpha
    img_alpha = 1.0 if img_alpha is None else img_alpha

    # make colors and groups as list
    groups = [groups] if isinstance(groups, str) else groups
    color = [color] if isinstance(color, str) or color is None else color

    # set palette if missing
    for c in color:
        if c in adata.obs.columns and is_categorical_dtype(adata.obs[c]) and c is not None:
            _maybe_set_colors(source=adata, target=adata, key=c, palette=palette)

    # check raw
    if use_raw is None:
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            "Cannot use both a layer and the raw representation. Was passed:" f"use_raw={use_raw}, layer={layer}."
        )
    if adata.raw is None and use_raw:
        raise ValueError(f"`use_raw={use_raw}` but AnnData object does not have raw.")

    # set wspace
    wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02 if wspace is None else wspace

    # logic for image v. non-image data is handled here
    shape = ScatterShape(shape) if shape is not None else shape  # type: ignore
    if TYPE_CHECKING:
        assert isinstance(shape, ScatterShape)

    spatial_params = _image_spatial_attrs(
        adata=adata,
        shape=shape,
        spatial_key=spatial_key,
        library_id=library_id,
        library_key=library_key,
        img=img,
        img_key=img_key,
        img_channel=img_channel,
        scale_factor=scale_factor,
        size=size,
        size_key=size_key,
        bw=bw,
        seg_key=seg_key,
        cell_id_key=cell_id_key,
    )

    # assert library_key exists and follows logic
    if library_key is not None:
        _assert_value_in_obs(adata, key=library_key, val=spatial_params.library_id)
    else:
        if len(spatial_params.library_id) > 1:
            raise ValueError(
                f"Multiple `library_ids: {spatial_params.library_id}` found but no `library_key` specified. \
                    Please specify `library_key`."
            )

    # set coords
    coords, crops = _get_coords_crops(
        adata=adata,
        spatial_params=spatial_params,
        spatial_key=spatial_key,
        library_key=library_key,
        crop_coord=crop_coord,
    )

    # partial init subset
    _subset = partial(_subs, library_key=library_key) if library_key is not None else lambda _, **__: _

    # initialize axis
    if (not isinstance(color, str) and isinstance(color, Sequence) and len(color) > 1) or (
        len(spatial_params.library_id) > 1
    ):
        if ax is not None:
            raise ValueError("Cannot specify `ax` when plotting multiple panels ")

        # each plot needs to be its own panel
        num_panels = len(color) * len(spatial_params.library_id)
        fig, grid = _panel_grid(hspace, wspace, ncols, num_panels)
    else:
        grid = None
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)

    # init axs list and vparams
    axs: Any = []
    _cmap_kwargs = _get_cmap_params(cmap_kwargs)

    if library_first:
        iter_panels = (range(len(spatial_params.library_id)), color)
    else:
        iter_panels = (color, range(len(spatial_params.library_id)))
    n_plots = len(list(itertools.product(*iter_panels)))

    # set title and axis labels
    title, axis_labels = _get_title_axlabels(title, axis_label, spatial_key, n_plots)
    # set cmap
    cmap = copy(get_cmap(cmap))
    cmap.set_bad("lightgray" if na_color is None else na_color)
    kwargs["cmap"] = cmap

    # set img_cmap
    img_cmap = "gray" if bw else img_cmap if cmap is not None else None

    # set scalebar
    if scalebar_dx is not None:
        scalebar_dx, scalebar_units = _get_scalebar(scalebar_dx, scalebar_units, len(spatial_params.library_id))
    # make plots
    for count, (first, second) in enumerate(itertools.product(*iter_panels)):
        _lib_count = first if library_first else second
        value_to_plot = second if library_first else first

        _size = spatial_params.size[_lib_count]
        _img = spatial_params.img[_lib_count]
        _crops = crops[_lib_count]
        _lib = spatial_params.library_id[_lib_count]
        _cell_id = spatial_params.cell_id[_lib_count]

        color_source_vector = _get_source_vec(
            _subset(adata, library_id=_lib),
            value_to_plot,
            layer=layer,
            use_raw=use_raw,
            alt_var=alt_var,
            groups=groups,
        )
        color_vector, categorical = _get_color_vec(
            _subset(adata, library_id=_lib),
            value_to_plot,
            color_source_vector,
            palette=palette,
            na_color=na_color,
        )

        if seg_key is not None:
            _img = _map_color_seg(_img, _cell_id, color_vector, seg_boundaries)

        # TODO: do we want to order points? for now no, skip
        _coords = coords[_lib_count]

        # set frame
        if grid:
            ax = pl.subplot(grid[count])
            axs.append(ax)
        if not (sc_settings._frameon if frameon is None else frameon):
            ax.axis("off")

        # set title
        if title is None:
            ax.set_title(value_to_plot)
        else:
            ax.set_title(title[count])

        if not categorical:
            vmin_float, vmax_float, vcenter_float, norm_obj = _get_vboundnorm(*_cmap_kwargs, count, color_vector)
            normalize = check_colornorm(
                vmin_float,
                vmax_float,
                vcenter_float,
                norm_obj,
            )
        else:
            normalize = None

        # plot edges and arrows if needed. Do it here cause otherwise image is on top.
        if seg_key is None:
            if edges:
                _cedge = _plot_edges(
                    _subset(adata, library_id=_lib), _coords, edges_width, edges_color, connectivity_key, edges_kwargs
                )
                ax.add_collection(_cedge)

            scatter = (
                partial(ax.scatter, marker=".", alpha=alpha, plotnonfinite=True)
                if shape is None
                else partial(_shaped_scatter, shape=shape, alpha=alpha)
            )

            if add_outline:
                bg_width, gap_width = outline_width
                point = np.sqrt(_size)
                gap_size = (point + (point * gap_width) * 2) ** 2
                bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
                # the default black and white colors can be changes using the contour_config parameter
                bg_color, gap_color = outline_color

                kwargs.pop("edgecolor", None)  # remove edge from kwargs if present
                alpha = kwargs.pop("alpha") if "alpha" in kwargs else None  # remove alpha for outline

                _cax = scatter(
                    _coords[:, 0],
                    _coords[:, 1],
                    s=bg_size,
                    c=bg_color,
                    rasterized=sc_settings._vector_friendly,
                    norm=normalize,
                    **kwargs,
                )
                ax.add_collection(_cax)
                _cax = scatter(
                    _coords[:, 0],
                    _coords[:, 1],
                    s=gap_size,
                    c=gap_color,
                    rasterized=sc_settings._vector_friendly,
                    norm=normalize,
                    **kwargs,
                )
                ax.add_collection(_cax)

                # if user did not set alpha, set alpha to 0.7
                alpha = 0.7 if alpha is None else alpha

            _cax = scatter(
                _coords[:, 0],
                _coords[:, 1],
                c=color_vector,
                s=_size,
                rasterized=sc_settings._vector_friendly,
                norm=normalize,
                alpha=alpha,
                **kwargs,
            )
            cax = ax.add_collection(_cax)
        else:
            _cax = ax.imshow(
                _img,
                rasterized=sc_settings._vector_friendly,
                norm=normalize,
                alpha=img_alpha,
                origin="lower",
                **kwargs,
            )
            cax = ax.add_collection(_cax, autolim=False)

        ax = _decorate_axs(
            ax=ax,
            cax=cax,
            lib_count=_lib_count,
            grid=grid,
            adata=adata,
            coords=_coords,
            img=_img,
            img_cmap=img_cmap,
            img_alpha=img_alpha,
            value_to_plot=value_to_plot,
            color_source_vector=color_source_vector,
            seg_key=seg_key,
            axis_labels=axis_labels,
            crops=_crops,
            categorical=categorical,
            palette=palette,
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_loc=legend_loc,
            legend_fontoutline=legend_fontoutline,
            na_color=na_color,
            na_in_legend=na_in_legend,
            scalebar_dx=scalebar_dx,
            scalebar_units=scalebar_units,
            scalebar_kwargs=scalebar_kwargs,
        )

    axs = axs if grid else ax

    if fig is not None and save is not None:
        save_fig(fig, path=save)

    return axs
