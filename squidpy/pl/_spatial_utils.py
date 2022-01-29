from __future__ import annotations

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
from functools import partial
from collections import namedtuple
from matplotlib_scalebar.scalebar import ScaleBar

from scanpy import logging as logg
from anndata import AnnData
from scanpy._settings import settings as sc_settings
from scanpy.plotting._utils import VBound, _FontSize, _FontWeight
from scanpy.plotting._tools.scatterplots import _add_categorical_legend

from pandas.core.dtypes.common import is_categorical_dtype
import numpy as np
import pandas as pd

from matplotlib import colors, pyplot as pl, patheffects
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Circle, Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PatchCollection

from squidpy._utils import NDArrayA
from squidpy.pl._graph import _get_palette
from squidpy.im._coords import CropCoords
from squidpy._constants._constants import ScatterShape
from squidpy._constants._pkg_constants import Key

_AvailShapes = Literal["circle", "square", "hex"]
Palette_t = Optional[Union[str, ListedColormap]]
_VBound = Union[VBound, Sequence[VBound]]
_Normalize = Union[Normalize, Sequence[Normalize]]
_SeqStr = Union[str, Sequence[str], None]
_SeqFloat = Union[float, Sequence[float], None]
_CoordTuple = Tuple[int, int, int, int]
# named tuples
CmapParams = namedtuple("CmapParams", ["vmin", "vmax", "vcenter", "norm"])


def _spatial_attrs(
    adata: AnnData,
    library_id: Sequence[str] | None = None,
    library_key: str | None = None,
    scale_factor: _SeqFloat = None,
    size: _SeqFloat = None,
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
    size = _get_list(size, float, len(_library_id))

    scale_factor = 1.0 if scale_factor is None else scale_factor
    scale_factor = _get_list(scale_factor, float, len(_library_id))
    img = [None for _ in _library_id]

    return _library_id, scale_factor, size, img


def _image_spatial_attrs(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    library_id: Sequence[str] | None = None,
    img: NDArrayA | Sequence[NDArrayA] | None = None,
    img_key: str | None = None,
    img_channel: int | None = None,
    scale_factor: _SeqFloat = None,
    size: _SeqFloat = None,
    size_key: str = Key.uns.size_key,
    bw: bool = False,
) -> Tuple[Sequence[str], Sequence[float], Sequence[float], Sequence[NDArrayA]]:
    """Return lists of image attributes saved in adata for plotting."""
    library_id = Key.uns.library_id(adata, spatial_key, library_id, return_all=True)
    if library_id is None:
        raise ValueError(f"Could not fetch `library_id`, check that `spatial_key: {spatial_key}` is correct.")

    image_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.image_key, library_id)
    scalefactor_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.scalefactor_key, library_id)

    if image_mapping.keys() != scalefactor_mapping.keys():  # check that keys match
        raise KeyError(
            f"Image keys: `{image_mapping.keys()}` and scalefactor keys: `{scalefactor_mapping.keys()}` are not equal."
        )

    scalefactors = _get_unique_map(scalefactor_mapping)

    if img_key is None:
        img_key = _get_unique_map(image_mapping)  # get intersection of image_mapping.values()
        img_key = img_key[0]  # get first of set
    else:
        if img_key not in image_mapping.values():
            raise ValueError(f"Image key: `{img_key}` does not exist. Available image keys: `{image_mapping.values()}`")

    _img_channel = [0, 1, 2] if img_channel is None else [img_channel]
    if max(_img_channel) > 2:
        raise ValueError(f"Invalid value for `img_channel: {_img_channel}`.")
    if img is None:
        img = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][img_key][..., _img_channel] for i in library_id]
    else:  # handle case where img is ndarray or list
        img = _get_list(img, np.ndarray, len(library_id))
        img = [im[..., _img_channel] for im in img]

    if bw:
        img = [np.dot(im[..., :3], [0.2989, 0.5870, 0.1140]) for im in img]

    if scale_factor is None:  # get intersection of scale_factor and match to img_key
        scale_factor_key = [i for i in scalefactors if img_key in i]
        if not len(scale_factor_key):
            raise ValueError(f"No `scale_factor` found that could match `img_key`: {img_key}.")
        _scale_factor_key = scale_factor_key[0]  # get first scale_factor
        scale_factor = [adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][_scale_factor_key] for i in library_id]
    else:  # handle case where scale_factor is float or list
        scale_factor = _get_list(scale_factor, float, len(library_id))

    # size_key = [i for i in scalefactors if size_key in i][0]
    if size_key not in scalefactors and size is None:
        raise ValueError(
            f"Specified `size_key: {size_key}` does not exist and size is `None`,\
            available keys are: `{scalefactors}`.\n Specify valid `size_key` or `size`."
        )
    if size is None:
        size = 1.0
    size = _get_list(size, float, len(library_id))
    if not (len(size) == len(library_id) == len(scale_factor)):
        raise ValueError("Len of `size`, `library_id` and `scale_factor` do not match.")
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


def _get_list(var: Any, _type: Any, ref_len: int | None = None) -> Sequence[Any] | Any:
    if isinstance(var, _type):
        if ref_len is None:
            return [var]
        return [var] * ref_len
    elif isinstance(var, list):
        if (ref_len is not None) and (ref_len != len(var)):
            raise ValueError(f"Var len: {len(var)} is not equal to ref len: {ref_len}. Please Check.")
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
    groups: _SeqStr = None,
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
    connectivity_key: str = Key.obsp.spatial_conn(),
    edges_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Any:
    """Graph plotting."""
    from networkx import Graph
    from networkx.drawing.nx_pylab import draw_networkx_edges

    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"Unable to find `connectivity_key: {connectivity_key}` in `adata.obsp`.\
             Please set `connectivity_key`."
        )

    g = Graph(adata.obsp[connectivity_key])
    edge_collection = draw_networkx_edges(
        g, coords, width=edges_width, edge_color=edges_color, arrows=False, **edges_kwargs
    )
    edge_collection.set_rasterized(sc_settings._vector_friendly)

    return edge_collection


def _get_title_axlabels(
    title: _SeqStr, axis_label: _SeqStr, spatial_key: str, n_plots: int
) -> Tuple[_SeqStr, Sequence[str]]:

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
        if len(axis_label) != 2:
            raise ValueError("Invalid `len(axis_label)={len(axis_label)}`.")
        axis_labels = axis_label
    elif isinstance(axis_label, str):
        axis_labels = [axis_label + str(x + 1) for x in range(2)]

    return title, axis_labels


def _get_scalebar(
    scalebar_dx: _SeqFloat = None,
    scalebar_units: _SeqStr = None,
    len_lib: int | None = None,
) -> Tuple[Sequence[float] | None, Sequence[str] | None]:
    if scalebar_dx is not None:
        _scalebar_dx = _get_list(scalebar_dx, float, len_lib)
        scalebar_units = "um" if scalebar_units is None else scalebar_units
        _scalebar_units = _get_list(scalebar_units, str, len_lib)
    else:
        _scalebar_dx = None
        _scalebar_units = None

    return _scalebar_dx, _scalebar_units


def _decorate_axs(
    ax: Axes,
    cax: PatchCollection,
    lib_count: int,
    grid: GridSpec,
    adata: AnnData,
    coords: NDArrayA,
    img: NDArrayA,
    img_cmap: str,
    img_alpha: float,
    value_to_plot: str,
    color_source_vector: NDArrayA | pd.Series,
    axis_labels: Sequence[str],
    crops: CropCoords,
    categorical: bool,
    palette: Palette_t = None,
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str = "right margin",
    legend_fontoutline: Optional[int] = None,
    na_color: str | Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    na_in_legend: bool = True,
    scalebar_dx: Sequence[float] | None = None,
    scalebar_units: Sequence[str] | None = None,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes:

    ax.set_yticks([])
    ax.set_xticks([])

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    ax.autoscale_view()

    if value_to_plot is None:
        # if only dots were plotted without an associated value
        # there is not need to plot a legend or a colorbar
        return ax

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
            scatter_array=coords,
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
    if img is not None:
        ax.imshow(img, cmap=img_cmap, alpha=img_alpha)
    else:
        ax.set_aspect("equal")
        ax.invert_yaxis()

    if crops is not None:
        ax.set_xlim(crops.to_tuple()[0], crops.to_tuple()[1])
        ax.set_ylim(crops.to_tuple()[3], crops.to_tuple()[2])
    else:
        ax.set_xlim(cur_coords[0], cur_coords[1])
        ax.set_ylim(cur_coords[3], cur_coords[2])

    if isinstance(scalebar_dx, list) and isinstance(scalebar_units, list):
        scalebar = ScaleBar(scalebar_dx[lib_count], units=scalebar_units[lib_count], **scalebar_kwargs)
        ax.add_artist(scalebar)

    return ax
