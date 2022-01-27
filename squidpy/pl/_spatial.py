from __future__ import annotations

from copy import copy
from types import MappingProxyType
from typing import (
    Any,
    Tuple,
    Union,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
)
from pathlib import Path
from functools import partial
from collections import namedtuple
from matplotlib_scalebar.scalebar import ScaleBar
import itertools

from scanpy import logging as logg
from anndata import AnnData
from scanpy._settings import settings as sc_settings
from scanpy.plotting._utils import VBound, _FontSize, _FontWeight, check_colornorm
from scanpy.plotting._tools.scatterplots import (
    _panel_grid,
    _get_vboundnorm,
    _add_categorical_legend,
)

from pandas.core.dtypes.common import is_categorical_dtype
import numpy as np
import pandas as pd

from matplotlib import colors, pyplot as pl, rcParams, patheffects
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize, ListedColormap
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection

from squidpy._utils import NDArrayA
from squidpy.gr._utils import _assert_spatial_basis
from squidpy.pl._graph import _get_palette, _maybe_set_colors
from squidpy.pl._utils import save_fig, _sanitize_anndata, _assert_value_in_obs
from squidpy.im._coords import CropCoords
from squidpy._constants._constants import ScatterShape
from squidpy._constants._pkg_constants import Key

_AvailShapes = Literal["circle", "square", "hex"]
Palette_t = Optional[Union[str, ListedColormap]]
_VBound = Union[VBound, Sequence[VBound]]
_Normalize = Union[Normalize, Sequence[Normalize]]
_SeqStr = Union[Sequence[str], str, None]
CmapParams = namedtuple("CmapParams", ["vmin", "vmax", "vcenter", "norm"])


def spatial(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    library_id: Sequence[str] | str | None = None,
    library_key: str | None = None,
    shape: _AvailShapes | None = ScatterShape.CIRCLE.s,  # type: ignore[assignment]
    img: Sequence[NDArrayA] | NDArrayA | None = None,
    img_key: str | None = None,
    scale_factor: Sequence[float] | float | None = None,
    size: Sequence[float] | float | None = None,
    size_key: str = Key.uns.size_key,
    crop_coord: Sequence[Tuple[int, int, int, int]] | Tuple[int, int, int, int] | None = None,
    bw: bool = False,
    alpha_img: float = 1.0,
    color: Sequence[str | None] | str | None = None,
    groups: Sequence[str] | str | None = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    alt_var: str | None = None,
    projection: Literal["2d", "3d"] = "2d",
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: str | Sequence[float] | Sequence[str] = "grey",
    neighbors_key: str | None = None,
    palette: Palette_t = None,
    cmap: Colormap | str | None = None,
    na_color: str | Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    frameon: Optional[bool] = None,
    title: str | Sequence[str] | None = None,
    axis_label: Sequence[str] | str | None = None,
    wspace: float | None = None,
    hspace: float = 0.25,
    ncols: int = 4,
    cmap_kwargs: Mapping[str, _VBound | _Normalize] | None = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ("black", "white"),
    ax: Axes | None = None,
    save: str | Path | None = None,
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str = "right margin",
    legend_fontoutline: Optional[int] = None,
    na_in_legend: bool = True,
    scalebar_dx: Sequence[float] | float | None = None,
    scalebar_units: Sequence[str] | str | None = None,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
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
    alpha_img
        for alpha of image
    color
    groups
    use_raw
    layer
    alt_var
        This is equivalent ot gene_symbol in embedding. See scanpy docs.
    projection
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
    # get projection
    subplot_kwargs = {"projection": "3d"} if projection == "3d" else {}

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

    if shape is not None:
        # get spatial attrs if shape is not None
        _library_id, scale_factor, size, img = _image_spatial_attrs(
            adata, spatial_key, library_id, img, img_key, scale_factor, size, size_key, bw
        )
    else:  # handle library_id logic for non-image data and spatial attributes
        _library_id, scale_factor, size, img = _spatial_attrs(adata, library_id, library_key, scale_factor, size)

    # set crops
    if crop_coord is None:
        crops: Union[list[Tuple[float, ...]], Tuple[None, ...]] = tuple(None for _ in _library_id)
    else:
        crop_coord = _get_list(crop_coord, tuple, _library_id)
        crops = [CropCoords(*cr) * sf for cr, sf in zip(crop_coord, scale_factor)]

    # assert library_key exists and follows logic
    if library_key is not None:
        _assert_value_in_obs(adata, key=library_key, val=_library_id)
    else:
        if len(_library_id) > 1:
            raise ValueError(
                f"Multiple `library_ids: {_library_id}` found but no `library_key` specified. \
                    Please specify `library_key`."
            )

    # set coords
    coords = _get_coords(adata, spatial_key, _library_id, scale_factor, library_key)

    # partial init subset
    _subset = partial(_subs, library_key=library_key) if library_key is not None else lambda _, **__: _

    # initialize axis
    if (not isinstance(color, str) and isinstance(color, Sequence) and len(color) > 1) or (len(_library_id) > 1):
        if ax is not None:
            raise ValueError("Cannot specify `ax` when plotting multiple panels ")

        # each plot needs to be its own panel
        num_panels = len(color) * len(_library_id)
        fig, grid = _panel_grid(hspace, wspace, ncols, num_panels)
    else:
        grid = None
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111, **subplot_kwargs)

    # init axs list and vparams
    axs: Any = []
    _cmap_kwargs = _get_cmap_params(cmap_kwargs)

    library_idx = range(len(_library_id))
    n_plots = len(list(itertools.product(color, library_idx)))

    # set title and axis labels
    title, axis_labels = _get_title_axlabels(title, axis_label, spatial_key, n_plots, subplot_kwargs)
    # set cmap
    cmap = copy(get_cmap(cmap))
    cmap.set_bad("lightgray" if na_color is None else na_color)
    kwargs["cmap"] = cmap

    # set cmap_img
    cmap_img = "gray" if bw else None

    # set scalebar
    if scalebar_dx is not None:
        _scalebar_dx, _scalebar_units = _get_scalebar(scalebar_dx, scalebar_units, _library_id)
    # make plots
    for count, (value_to_plot, _lib_count) in enumerate(itertools.product(color, library_idx)):
        _size = size[_lib_count]
        _img = img[_lib_count]
        _crops = crops[_lib_count]
        _lib = _library_id[_lib_count]

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

        # TODO: do we want to order points? for now no, skip
        _coords = coords[_lib_count]

        # set frame
        if grid:
            ax = pl.subplot(grid[count], **subplot_kwargs)
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
        if edges:
            _cedge = _plot_edges(
                _subset(adata, library_id=_lib), _coords, edges_width, edges_color, neighbors_key, edges_kwargs
            )
            ax.add_collection(_cedge)

        # make scatter
        if projection == "3d":
            cax = ax.scatter(
                _coords[:, 0],
                _coords[:, 1],
                _coords[:, 2],
                marker=".",
                c=color_vector,
                rasterized=sc_settings._vector_friendly,
                norm=normalize,
                s=_size,
                **kwargs,
            )
        else:

            scatter = (
                partial(ax.scatter, marker=".", plotnonfinite=True)
                if shape is None
                else partial(_shaped_scatter, shape=shape)
            )

            if add_outline:
                bg_width, gap_width = outline_width
                point = np.sqrt(_size)
                gap_size = (point + (point * gap_width) * 2) ** 2
                bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
                # the default black and white colors can be changes using the contour_config parameter
                bg_color, gap_color = outline_color

                kwargs["edgecolor"] = "none"  # remove edge from kwargs if present
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
                kwargs["alpha"] = 0.7 if alpha is None else alpha

            _cax = scatter(
                _coords[:, 0],
                _coords[:, 1],
                c=color_vector,
                s=_size,
                rasterized=sc_settings._vector_friendly,
                norm=normalize,
                **kwargs,
            )
            cax = ax.add_collection(_cax)

        ax.set_yticks([])
        ax.set_xticks([])
        if projection == "3d":
            ax.set_zticks([])

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if projection == "3d":
            # shift the label closer to the axis
            ax.set_zlabel(axis_labels[2], labelpad=-7)

        ax.autoscale_view()

        if value_to_plot is None:
            # if only dots were plotted without an associated value
            # there is not need to plot a legend or a colorbar
            continue

        if legend_fontoutline is not None:
            path_effect = [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
        else:
            path_effect = []

        # Adding legends
        if categorical:
            clusters = adata.obs[value_to_plot].cat.categories
            _palette = _get_palette(adata, cluster_key=value_to_plot, categories=clusters, palette=palette)
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=_palette,
                scatter_array=_coords,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=na_color,
                na_in_legend=na_in_legend,
                multi_panel=bool(grid),
            )
        else:
            # TODO: na_in_legend should have some effect here
            pl.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30)

        cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
        if _img is not None:
            ax.imshow(_img, cmap=cmap_img, alpha=alpha_img)
        else:
            ax.set_aspect("equal")
            ax.invert_yaxis()

        if _crops is not None:
            ax.set_xlim(_crops.to_tuple()[0], _crops.to_tuple()[1])
            ax.set_ylim(_crops.to_tuple()[3], _crops.to_tuple()[2])
        else:
            ax.set_xlim(cur_coords[0], cur_coords[1])
            ax.set_ylim(cur_coords[3], cur_coords[2])

        if isinstance(_scalebar_dx, list):
            scalebar = ScaleBar(_scalebar_dx[_lib_count], units=_scalebar_units[_lib_count], **scalebar_kwargs)
            ax.add_artist(scalebar)

    axs = axs if grid else ax

    if fig is not None and save is not None:
        save_fig(fig, path=save)

    return axs


def _spatial_attrs(
    adata: AnnData,
    library_id: Sequence[str] | None = None,
    library_key: str | None = None,
    scale_factor: Sequence[float] | float | None = None,
    size: Sequence[float] | float | None = None,
) -> Tuple[Sequence[str], Sequence[float], Sequence[float], Sequence[None]]:

    if library_id is None and library_key is not None:  # try to assign library_id
        try:
            _library_id = adata.obs[library_key].cat.categories.tolist()
        except IndexError:
            raise IndexError(f"`library_key: {library_key}` not in `adata.obs`.")
    elif library_id is None and library_key is None:  # create dummy library_id
        logg.warning(
            "Please specify a valid `library_id` or set it permanently in `adata.uns['spatial'][<library_id>]`"
        )
        _library_id = [""]
    elif isinstance(library_id, list):  # get library_id from arg
        _library_id = library_id
    else:
        raise ValueError(f"Could not set library_id: `{library_id}`")

    size = 120000 / adata.shape[0] if size is None else size
    size = _get_list(size, float, _library_id)

    scale_factor = 1.0 if scale_factor is None else scale_factor
    scale_factor = _get_list(scale_factor, float, _library_id)
    img = [None for _ in _library_id]

    return _library_id, scale_factor, size, img


def _image_spatial_attrs(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    library_id: Sequence[str] | None = None,
    img: Sequence[NDArrayA] | NDArrayA | None = None,
    img_key: str | None = None,
    scale_factor: Sequence[float] | float | None = None,
    size: Sequence[float] | float | None = None,
    size_key: str = Key.uns.size_key,
    bw: bool = False,
) -> Tuple[Sequence[str], Sequence[float], Sequence[float], Sequence[NDArrayA]]:
    """Return lists of image attributes saved in adata for plotting."""
    library_id = Key.uns.library_id(adata, spatial_key, library_id, return_all=True)
    if library_id is None:
        raise ValueError(f"Could not fetch `library_id`, check that `spatial_key: {spatial_key}` is correct.")

    image_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.image_key, library_id)
    scalefactor_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.scalefactor_key, library_id)
    scalefactors = _get_unique_map(scalefactor_mapping)

    if not (image_mapping.keys() == scalefactor_mapping.keys()):  # check that keys match
        raise KeyError(
            f"Image keys: `{image_mapping.keys()}` and scalefactor keys: `{scalefactor_mapping.keys()}` are not equal."
        )

    if img_key is None:
        img_key = _get_unique_map(image_mapping)  # get intersection of image_mapping.values()
        img_key = img_key[0]  # get first of set
    else:
        if img_key not in image_mapping.values():
            raise ValueError(f"Image key: `{img_key}` does not exist. Available image keys: `{image_mapping.values()}`")

    if img is None:
        img = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][img_key] for i in library_id]
    else:  # handle case where img is ndarray or list
        img = _get_list(img, np.ndarray, library_id)

    if bw:
        img = [np.dot(im[..., :3], [0.2989, 0.5870, 0.1140]) for im in img]

    if scale_factor is None:  # get intersection of scale_factor and match to img_key
        scale_factor_key = [i for i in scalefactors if img_key in i]
        if len(scale_factor_key) == 0:
            raise ValueError(f"No `scale_factor` found that could match `img_key`: {img_key}.")
        _scale_factor_key = scale_factor_key[0]  # get first scale_factor
        scale_factor = [adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][_scale_factor_key] for i in library_id]
    else:  # handle case where scale_factor is float or list
        scale_factor = _get_list(scale_factor, float, library_id)

    # size_key = [i for i in scalefactors if size_key in i][0]
    if size_key not in scalefactors and size is None:
        raise ValueError(
            f"Specified `size_key: {size_key}` does not exist and size is `None`,\
            available keys are: `{scalefactors}`.\n Specify valid `size_key` or `size`."
        )
    if size is None:
        size = 1.0
    size = _get_list(size, float, library_id)
    size = [
        adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][size_key] * s * sf * 0.5
        for i, s, sf in zip(library_id, size, scale_factor)
    ]

    return library_id, scale_factor, size, img


def _get_coords(
    adata: AnnData,
    spatial_key: str,
    library_id: Sequence[str],
    scale_factor: Sequence[float],
    library_key: str | None = None,
) -> Sequence[NDArrayA]:

    coords = adata.obsm[spatial_key]
    if library_key is None:
        return [coords * sf for sf in scale_factor]
    else:
        if len(library_id) != len(scale_factor):
            raise ValueError(
                f"Length `library_id: {len(library_id)}` is not equal to length `scale_factor: {len(scale_factor)}`."
            )
        return [coords[adata.obs[library_key] == lib, :] * sf for lib, sf in zip(library_id, scale_factor)]


def _subs(adata: AnnData, library_key: str | None = None, library_id: str | None = None) -> AnnData:
    try:
        return adata[adata.obs[library_key] == library_id].copy()
    except KeyError:
        raise KeyError(
            f"Cannot subset adata. Either `library_key: {library_key}` or `library_id: {library_id}` is invalid."
        )


def _get_unique_map(dic: Mapping[str, Any]) -> Any:
    """Get intersection of dict values."""
    return sorted(set.intersection(*map(set, dic.values())))


def _get_list(var: Any, _type: Any, ref: Sequence[Any] | None = None) -> Sequence[Any] | Any:
    if isinstance(var, _type):
        if ref is None:
            return [var]
        else:
            return [var for _ in ref]
    else:
        if isinstance(var, list):
            if (ref is not None) and (len(ref) != len(var)):
                raise ValueError(f"Var len: {len(var)} is not equal to ref len: {len(ref)}. Please Check.")
            else:
                return var
        else:
            raise ValueError(f"Can't make list from var: `{var}`")


def _get_cmap_params(
    cmap_kwargs: Mapping[str, _VBound | _Normalize] | None = None,
) -> CmapParams:
    if cmap_kwargs is not None:
        cmap_params = []
        for f in CmapParams._fields:
            if f in cmap_kwargs.keys():
                v = cmap_kwargs[f]
            else:
                v = None
            if isinstance(v, str) or not isinstance(v, Sequence):
                v = [v]
            cmap_params.append(v)
    else:
        cmap_params = [[None] for _ in CmapParams._fields]

    return CmapParams(*cmap_params)


def _get_source_vec(
    adata: AnnData,
    value_to_plot: str | None,
    use_raw: bool | None = None,
    alt_var: str | None = None,
    layer: str | None = None,
    groups: Sequence[str] | str | None = None,
) -> NDArrayA | pd.Series:

    if value_to_plot is None:
        return np.full(np.nan, adata.n_obs)
    if alt_var is not None and value_to_plot not in adata.obs.columns and value_to_plot not in adata.var_names:
        value_to_plot = adata.var_names[adata.var[alt_var] == value_to_plot][0]
    if use_raw and value_to_plot not in adata.obs.columns:
        values = adata.raw.obs_vector(value_to_plot)
    else:
        values = adata.obs_vector(value_to_plot, layer=layer)
    if groups is not None and is_categorical_dtype(values):
        values = values.replace(values.categories.difference(groups), np.nan)
    return values


def _get_color_vec(
    adata: AnnData,
    value_to_plot: str | None,
    values: NDArrayA | pd.Series,
    palette: Palette_t = None,
    na_color: str | Tuple[float, ...] | None = None,
) -> Tuple[NDArrayA, bool]:
    to_hex = partial(colors.to_hex, keep_alpha=True)
    if value_to_plot is None:
        return np.full(to_hex(na_color), adata.n_obs), False
    if not is_categorical_dtype(values):
        return values, False
    else:
        # use scanpy _get_palette to set palette if not present
        clusters = adata.obs[value_to_plot].cat.categories
        color_map = _get_palette(adata, cluster_key=value_to_plot, categories=clusters, palette=palette)
        color_vector = values.rename_categories(color_map)  # type: ignore
        # Set color to 'missing color' for all missing values
        if color_vector.isna().any():
            color_vector = color_vector.add_categories([to_hex(na_color)])
            color_vector = color_vector.fillna(to_hex(na_color))
        return color_vector, True


def _shaped_scatter(
    x: NDArrayA,
    y: NDArrayA,
    s: float,
    c: NDArrayA,
    shape: _AvailShapes | None = ScatterShape.CIRCLE.s,  # type: ignore[assignment]
    vmin: VBound | Sequence[VBound] | None = None,
    vmax: VBound | Sequence[VBound] | None = None,
    **kwargs: Any,
) -> PatchCollection:
    """
    Get shapes for scatterplot.

    Adapted from here: https://gist.github.com/syrte/592a062c562cd2a98a83 .
    This code is under [The BSD 3-Clause License](http://opensource.org/licenses/BSD-3-Clause)
    """
    shape = ScatterShape(shape)  # type: ignore[assignment]
    if TYPE_CHECKING:
        assert isinstance(shape, ScatterShape)
    shape = shape.s

    if shape == ScatterShape.CIRCLE.s:
        zipped = np.broadcast(x, y, s)
        patches = [Circle((x_, y_), s_) for x_, y_, s_ in zipped]
    else:
        n = 4 if shape == ScatterShape.SQUARE.s else 6
        r: float = s / (2 * np.sin(np.pi / n))
        polys = np.stack([_make_poly(x, y, r, n, i) for i in range(n)], 1).swapaxes(0, 2)
        patches = [Polygon(p, False) for p in polys]
    collection = PatchCollection(patches, **kwargs)

    if isinstance(c, np.ndarray) and np.issubdtype(c.dtype, np.number):
        collection.set_array(np.ma.masked_invalid(c))
        collection.set_clim(vmin, vmax)
    else:
        collection.set_facecolor(c)

    return collection


def _make_poly(x: NDArrayA, y: NDArrayA, r: float, n: int, i: int) -> Tuple[NDArrayA, NDArrayA]:
    x_i = x + r * np.sin((np.pi / n) * (1 + 2 * i))
    y_i = y + r * np.cos((np.pi / n) * (1 + 2 * i))
    return x_i, y_i


def _plot_edges(
    adata: AnnData,
    coords: NDArrayA,
    edges_width: float = 0.1,
    edges_color: str | Sequence[float] | Sequence[str] = "grey",
    neighbors_key: str | None = None,
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Any:
    """Graph plotting."""
    from networkx import Graph
    from networkx.drawing.nx_pylab import draw_networkx_edges

    neighbors_key = Key.obsp.spatial_conn(neighbors_key)
    if neighbors_key not in adata.obsp:
        raise KeyError(
            f"Unable to find `neighbors_key: {neighbors_key}` in `adata.obsp`.\
             Please set `neighbors_key`."
        )

    g = Graph(adata.obsp[neighbors_key])
    edge_collection = draw_networkx_edges(
        g, coords, width=edges_width, edge_color=edges_color, arrows=False, **edges_kwargs
    )
    edge_collection.set_rasterized(sc_settings._vector_friendly)

    return edge_collection


def _get_title_axlabels(
    title: _SeqStr, axis_label: _SeqStr, spatial_key: str, n_plots: int, subplot_kwargs: Mapping[str, Any]
) -> Tuple[_SeqStr, _SeqStr]:

    # handle title
    if title is not None:
        if isinstance(title, list) and len(title) != n_plots:
            raise ValueError("Title list is shorter than number of plots.")
    elif isinstance(title, str):
        title = [title] * n_plots
    else:
        title = None

    # handle axis labels
    axis_label = spatial_key if axis_label is None else axis_label

    if isinstance(axis_label, list):
        if "3d" in subplot_kwargs.values() and len(axis_label) != 3:
            raise ValueError(f"Invalid `len(axis_label)={len(axis_label)}` for `projection='3d'`.")
        elif len(axis_label) != 2:
            raise ValueError("Invalid `len(axis_label)={len(axis_label)}` for `projection='2d'`.")
    elif isinstance(axis_label, str):
        if "3d" in subplot_kwargs.values():
            axis_labels: Sequence[str] = [axis_label + str(x + 1) for x in range(3)]
        else:
            axis_labels = [axis_label + str(x + 1) for x in range(2)]
    else:
        axis_labels = axis_label

    return title, axis_labels


def _get_scalebar(
    scalebar_dx: Sequence[float] | float | None = None,
    scalebar_units: Sequence[str] | str | None = None,
    library_id: Sequence[str] | str | None = None,
) -> Any:
    if scalebar_dx is not None:
        _scalebar_dx = _get_list(scalebar_dx, float, library_id)
        scalebar_units = "um" if scalebar_units is None else scalebar_units
        _scalebar_units = _get_list(scalebar_units, str, library_id)
    else:
        _scalebar_dx = None

    return _scalebar_dx, _scalebar_units
