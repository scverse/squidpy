from __future__ import annotations

from copy import copy
from types import MappingProxyType
from typing import (
    Any,
    List,
    Type,
    Tuple,
    Union,
    Mapping,
    Optional,
    Sequence,
    NamedTuple,
    TYPE_CHECKING,
)
from numbers import Number
from functools import partial
from typing_extensions import Literal
from matplotlib_scalebar.scalebar import ScaleBar
import itertools

from scanpy import logging as logg
from anndata import AnnData
from scanpy._settings import settings as sc_settings
from scanpy.plotting._utils import _FontSize, _FontWeight
from scanpy.plotting._tools.scatterplots import _add_categorical_legend

from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.common import is_categorical_dtype
import numpy as np
import pandas as pd

from matplotlib import colors, pyplot as plt, rcParams, patheffects
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.collections import Collection, PatchCollection

from skimage.util import map_array
from skimage.color import label2rgb
from skimage.morphology import square, erosion
from skimage.segmentation import find_boundaries

from squidpy._utils import NDArrayA
from squidpy.pl._utils import _assert_value_in_obs
from squidpy.im._coords import CropCoords, TupleSerializer
from squidpy.pl._color_utils import _get_palette, _maybe_set_colors
from squidpy._constants._constants import ScatterShape
from squidpy._constants._pkg_constants import Key

_AvailShapes = Literal["circle", "square", "hex"]
Palette_t = Optional[Union[str, ListedColormap]]
_Normalize = Union[Normalize, Sequence[Normalize], None]
_SeqStr = Union[str, Sequence[str], None]
_SeqFloat = Union[float, Sequence[float], None]
_SeqArray = Union[NDArrayA, Sequence[NDArrayA], None]
_CoordTuple = Tuple[int, int, int, int]


# named tuples
class FigParams(NamedTuple):
    """Figure params."""

    fig: Figure
    ax: Axes
    axs: Sequence[Axes] | None
    iter_panels: Tuple[Any, Any]
    title: _SeqStr
    ax_labels: Sequence[str]
    frameon: bool | None


class CmapParams(NamedTuple):
    """Cmap params."""

    cmap: Colormap
    img_cmap: Colormap
    norm: Normalize


class OutlineParams(NamedTuple):
    """Outline params."""

    outline: bool
    gap_size: float
    gap_color: str
    bg_size: float
    bg_color: str


class ScalebarParams(NamedTuple):
    """Scalebar params."""

    scalebar_dx: Sequence[float] | None
    scalebar_units: _SeqStr


class ColorParams(NamedTuple):
    """Color params."""

    shape: _AvailShapes | None
    color: Sequence[str | None]
    groups: Sequence[str] | None
    alpha: float
    img_alpha: float
    use_raw: bool


class SpatialParams(NamedTuple):
    """Color params."""

    library_id: Sequence[str]
    scale_factor: Sequence[float]
    size: Sequence[float]
    img: Sequence[NDArrayA] | Tuple[None, ...]
    segment: Sequence[NDArrayA] | Tuple[None, ...]
    cell_id: Sequence[NDArrayA] | Tuple[None, ...]


def _get_library_id(
    adata: AnnData,
    shape: _AvailShapes | None,
    spatial_key: str = Key.uns.spatial,
    library_id: Sequence[str] | None = None,
    library_key: str | None = None,
) -> Sequence[str]:
    if shape is not None:
        library_id = Key.uns.library_id(adata, spatial_key, library_id, return_all=True)
        if library_id is None:
            raise ValueError(f"Could not fetch `library_id`, check that `spatial_key: {spatial_key}` is correct.")
        return library_id
    if library_key is not None:
        if library_key not in adata.obs.columns:
            raise KeyError(f"`library_key: {library_key}` not in `adata.obs`.")
        if library_id is None:
            library_id = adata.obs[library_key].cat.categories.tolist()
        _assert_value_in_obs(adata, key=library_key, val=library_id)
        if isinstance(library_id, str):
            library_id = [library_id]
        return library_id
    if library_id is None:
        logg.warning(
            "Please specify a valid `library_id` or set it permanently in `adata.uns['spatial'][<library_id>]`"
        )
        library_id = [""]
    elif isinstance(library_id, list):  # get library_id from arg
        library_id = library_id
    elif isinstance(library_id, str):
        library_id = [library_id]
    else:
        raise ValueError(f"Invalid `library_id`: {library_id}.")
    return library_id


def _get_image(
    adata: AnnData,
    library_id: Sequence[str],
    spatial_key: str = Key.obsm.spatial,
    img: _SeqArray | bool = None,
    img_res_key: str | None = None,
    img_channel: int | None = None,
    img_cmap: Colormap | str | None = None,
) -> Union[Sequence[NDArrayA], Tuple[None, ...]]:
    image_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.image_key, library_id)
    if img_res_key is None:
        _img_res_key = _get_unique_map(image_mapping)  # get intersection of image_mapping.values()
        _img_res_key = _img_res_key[0]  # get first of set
    else:
        if img_res_key not in _get_unique_map(image_mapping):
            raise ValueError(
                f"Image key: `{img_res_key}` does not exist. Available image keys: `{image_mapping.values()}`"
            )

    _img_channel = [0, 1, 2] if img_channel is None else [img_channel]
    if max(_img_channel) > 2:
        raise ValueError(f"Invalid value for `img_channel: {_img_channel}`.")
    if isinstance(img, np.ndarray) or isinstance(img, list):
        img = _get_list(img, np.ndarray, len(library_id), "img")
        img = [im[..., _img_channel] for im in img]
    else:
        img = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][_img_res_key][..., _img_channel] for i in library_id]
    if img_cmap == "gray":
        img = [np.dot(im[..., :3], [0.2989, 0.5870, 0.1140]) for im in img]
    return img


def _get_segment(
    adata: AnnData,
    library_id: Sequence[str],
    cell_id_key: str | None = None,
    library_key: str | None = None,
    seg: _SeqArray | bool = None,
    seg_key: str | None = None,
) -> Tuple[Sequence[NDArrayA], Sequence[NDArrayA]] | Tuple[Tuple[None, ...], Tuple[None, ...]]:
    if cell_id_key not in adata.obs.columns:
        raise ValueError(f"`cell_id_key: {cell_id_key}` not in `adata.obs`.")
    cell_id_vec = adata.obs[cell_id_key].values

    if library_key not in adata.obs.columns:
        raise ValueError(f"`library_key: {library_key}` not in `adata.obs`.")
    if cell_id_vec.dtype != np.int_:
        raise ValueError(f"Invalid type {cell_id_vec.dtype} for `adata.obs[{cell_id_key}]`.")
    cell_id_vec = [cell_id_vec[adata.obs[library_key] == lib] for lib in library_id]

    if isinstance(seg, np.ndarray) or isinstance(seg, list):
        img_seg = _get_list(seg, np.ndarray, len(library_id), "img_seg")
    else:
        img_seg = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][seg_key] for i in library_id]
    return img_seg, cell_id_vec


def _get_scalefactor_size(
    adata: AnnData,
    library_id: Sequence[str],
    spatial_key: str = Key.obsm.spatial,
    img_res_key: str | None = None,
    scale_factor: _SeqFloat = None,
    size: _SeqFloat = None,
    size_key: str | None = Key.uns.size_key,
) -> Tuple[Sequence[float], Sequence[float]]:
    try:
        scalefactor_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.scalefactor_key, library_id)
        scalefactors = _get_unique_map(scalefactor_mapping)
    except KeyError:
        logg.debug("`scalefactors` set to `None`.")
        scalefactors = None
    if scalefactors is not None and img_res_key is not None:
        if scale_factor is None:  # get intersection of scale_factor and match to img_res_key
            scale_factor_key = [i for i in scalefactors if img_res_key in i]
            if not len(scale_factor_key):
                raise ValueError(f"No `scale_factor` found that could match `img_res_key`: {img_res_key}.")
            _scale_factor_key = scale_factor_key[0]  # get first scale_factor
            scale_factor = [
                adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][_scale_factor_key] for i in library_id
            ]
        else:  # handle case where scale_factor is float or list
            scale_factor = _get_list(scale_factor, float, len(library_id), "scale_factor")

        if size_key not in scalefactors and size is None:
            raise ValueError(
                f"Specified `size_key: {size_key}` does not exist and size is `None`, "
                f"available keys are: `{scalefactors}`.\n Specify valid `size_key` or `size`."
            )
        if size is None:
            size = 1.0
        size = _get_list(size, float, len(library_id), "size")
        if not (len(size) == len(library_id) == len(scale_factor)):
            raise ValueError("Len of `size`, `library_id` and `scale_factor` do not match.")
        size = [
            adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][size_key] * s * sf * 0.5
            for i, s, sf in zip(library_id, size, scale_factor)
        ]
        return scale_factor, size
    else:
        scale_factor = 1.0 if scale_factor is None else scale_factor
        scale_factor = _get_list(scale_factor, float, len(library_id), "scale_factor")

        size = 120000 / adata.shape[0] if size is None else size
        size = _get_list(size, Number, len(library_id), "size")
        return scale_factor, size


def _image_spatial_attrs(
    adata: AnnData,
    shape: _AvailShapes | None = None,
    spatial_key: str = Key.obsm.spatial,
    library_id: Sequence[str] | None = None,
    library_key: str | None = None,
    img: _SeqArray | bool = None,
    img_res_key: str | None = Key.uns.image_res_key,
    img_channel: int | None = None,
    seg: _SeqArray | bool = None,
    seg_key: str | None = None,
    cell_id_key: str | None = None,
    scale_factor: _SeqFloat = None,
    size: _SeqFloat = None,
    size_key: str | None = Key.uns.size_key,
    img_cmap: Colormap | str | None = None,
) -> SpatialParams:
    library_id = _get_library_id(
        adata=adata, shape=shape, spatial_key=spatial_key, library_id=library_id, library_key=library_key
    )
    if len(library_id) > 1 and library_key is None:
        raise ValueError(
            f"Found `library_id: `{library_id} but no `library_key` specified. Please specify `library_key`."
        )

    scale_factor, size = _get_scalefactor_size(
        adata=adata,
        spatial_key=spatial_key,
        library_id=library_id,
        img_res_key=img_res_key,
        scale_factor=scale_factor,
        size=size,
        size_key=size_key,
    )

    if (img and seg) or (img and shape is not None):
        _img = _get_image(
            adata=adata,
            spatial_key=spatial_key,
            library_id=library_id,
            img=img,
            img_res_key=img_res_key,
            img_channel=img_channel,
            img_cmap=img_cmap,
        )
    else:
        _img = tuple(None for _ in library_id)

    if seg:
        _seg, _cell_vec = _get_segment(
            adata=adata,
            library_id=library_id,
            cell_id_key=cell_id_key,
            library_key=library_key,
            seg=seg,
            seg_key=seg_key,
        )
    else:
        _seg = tuple(None for _ in library_id)
        _cell_vec = tuple(None for _ in library_id)

    return SpatialParams(library_id, scale_factor, size, _img, _seg, _cell_vec)


def _set_coords_crops(
    adata: AnnData,
    spatial_params: SpatialParams,
    spatial_key: str,
    library_key: str | None = None,
    crop_coord: Sequence[_CoordTuple] | _CoordTuple | None = None,
) -> Tuple[Sequence[NDArrayA], List[TupleSerializer] | Tuple[None, ...]]:
    # set crops
    if crop_coord is None:
        crops: Union[List[TupleSerializer], Tuple[None, ...]] = tuple(None for _ in spatial_params.library_id)
    else:
        crop_coord = _get_list(crop_coord, tuple, len(spatial_params.library_id), "crop_coord")
        crops = [CropCoords(*cr) * sf for cr, sf in zip(crop_coord, spatial_params.scale_factor)]

    coords = adata.obsm[spatial_key]
    if library_key is None:
        return [coords * sf for sf in spatial_params.scale_factor], crops
    else:
        return [
            coords[adata.obs[library_key] == lib, :] * sf  # TODO: check that library_key is asserted upstream
            for lib, sf in zip(spatial_params.library_id, spatial_params.scale_factor)
        ], crops


def _subs(adata: AnnData, library_key: str | None = None, library_id: str | None = None) -> AnnData:
    try:
        if not adata[adata.obs[library_key] == library_id].shape[0]:
            raise ValueError("Subset is empty.")
        return adata[adata.obs[library_key] == library_id]
    except KeyError:
        raise KeyError(
            f"Cannot subset adata. Either `library_key: {library_key}` or `library_id: {library_id}` is invalid."
        )


def _get_unique_map(dic: Mapping[str, Any]) -> Sequence[Any]:
    """Get intersection of dict values."""
    return sorted(set.intersection(*map(set, dic.values())))


def _get_list(
    var: Any,
    _type: Type[Any],
    ref_len: int | None = None,
    name: str | None = None,
) -> List[Any]:
    if isinstance(var, _type):
        if ref_len is None:
            return [var]
        return [var] * ref_len
    elif isinstance(var, list):
        if (ref_len is not None) and (ref_len != len(var)):
            raise ValueError(
                f"Variable: `{name}` has length: {len(var)}, which is not equal to reference length: {ref_len}."
            )
        return var
    else:
        raise ValueError(f"Can't make list from variable: `{var}`")


def _set_color_source_vec(
    adata: AnnData,
    value_to_plot: str | None,
    use_raw: bool | None = None,
    alt_var: str | None = None,
    layer: str | None = None,
    groups: _SeqStr = None,
    palette: Palette_t = None,
    na_color: str | Tuple[float, ...] | None = None,
) -> Tuple[NDArrayA | pd.Series | None, NDArrayA, bool]:
    to_hex = partial(colors.to_hex, keep_alpha=True)

    if value_to_plot is None:
        return np.full(adata.n_obs, to_hex(na_color)), np.broadcast_to(np.nan, adata.n_obs), False
    if alt_var is not None and value_to_plot not in adata.obs.columns and value_to_plot not in adata.var_names:
        value_to_plot = adata.var_names[adata.var[alt_var] == value_to_plot][0]
    if use_raw and value_to_plot not in adata.obs.columns:
        color_source_vector = adata.raw.obs_vector(value_to_plot)
    else:
        color_source_vector = adata.obs_vector(value_to_plot, layer=layer)
    if groups is not None and is_categorical_dtype(color_source_vector):
        color_source_vector = color_source_vector.replace(color_source_vector.categories.difference(groups), np.nan)

    if not is_categorical_dtype(color_source_vector):
        return None, color_source_vector, False
    else:
        clusters = color_source_vector.categories
        color_map = _get_palette(adata, cluster_key=value_to_plot, categories=clusters, palette=palette)  # type: ignore
        color_vector = color_source_vector.rename_categories(color_map)
        # Set color to 'missing color' for all missing values
        if color_vector.isna().any():
            color_vector = color_vector.add_categories([to_hex(na_color)])
            color_vector = color_vector.fillna(to_hex(na_color))
        return color_source_vector, color_vector, True


def _shaped_scatter(
    x: NDArrayA,
    y: NDArrayA,
    s: float,
    c: NDArrayA,
    shape: _AvailShapes | ScatterShape | None = ScatterShape.CIRCLE,
    norm: _Normalize = None,
    **kwargs: Any,
) -> PatchCollection:
    """
    Get shapes for scatterplot.

    Adapted from here: https://gist.github.com/syrte/592a062c562cd2a98a83 .
    This code is under [The BSD 3-Clause License](http://opensource.org/licenses/BSD-3-Clause)
    """
    shape = ScatterShape(shape)
    if TYPE_CHECKING:
        assert isinstance(shape, ScatterShape)
    shape = ScatterShape(shape)

    if shape == ScatterShape.CIRCLE:
        patches = [Circle((x, y), radius=s) for x, y, s in np.broadcast(x, y, s)]
    elif shape == ScatterShape.SQUARE:
        patches = [Rectangle((x - s, y - s), width=2 * s, height=2 * s) for x, y, s in np.broadcast(x, y, s)]
    elif shape == ScatterShape.HEX:
        n = 6
        r = s / (2 * np.sin(np.pi / n))
        polys = np.stack([_make_poly(x, y, r, n, i) for i in range(n)], 1).swapaxes(0, 2)
        patches = [Polygon(p, closed=False) for p in polys]
    else:
        raise NotImplementedError(f"Shape `{shape}` is not yet implemented.")
    collection = PatchCollection(patches, snap=False, **kwargs)

    if isinstance(c, np.ndarray) and np.issubdtype(c.dtype, np.number):
        collection.set_array(np.ma.masked_invalid(c))
        collection.set_norm(norm)
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
    ax: Axes,
    edges_width: float = 0.1,
    edges_color: str | Sequence[float] | Sequence[str] = "grey",
    connectivity_key: str = Key.obsp.spatial_conn(),
    **kwargs: Any,
) -> Any:
    """Graph plotting."""
    from networkx import Graph
    from networkx.drawing.nx_pylab import draw_networkx_edges

    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"Unable to find `connectivity_key: {connectivity_key}` in `adata.obsp`. " "Please set `connectivity_key`."
        )

    g = Graph(adata.obsp[connectivity_key])
    edge_collection = draw_networkx_edges(
        g, coords, width=edges_width, edge_color=edges_color, arrows=False, ax=ax, **kwargs
    )
    edge_collection.set_rasterized(sc_settings._vector_friendly)

    return edge_collection


def _get_title_axlabels(
    title: _SeqStr, axis_label: _SeqStr, spatial_key: str, n_plots: int
) -> Tuple[_SeqStr, Sequence[str]]:
    # handle title
    if title is not None:
        if isinstance(title, (tuple, list)) and len(title) != n_plots:
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
        _scalebar_dx = _get_list(scalebar_dx, float, len_lib, "scalebar_dx")
        scalebar_units = "um" if scalebar_units is None else scalebar_units
        _scalebar_units = _get_list(scalebar_units, str, len_lib, "scalebar_units")
    else:
        _scalebar_dx = None
        _scalebar_units = None

    return _scalebar_dx, _scalebar_units


def _decorate_axs(
    ax: Axes,
    cax: PatchCollection,
    lib_count: int,
    fig_params: FigParams,
    adata: AnnData,
    coords: NDArrayA,
    value_to_plot: str,
    color_source_vector: pd.Series[CategoricalDtype],
    crops: TupleSerializer | None = None,
    img: NDArrayA | None = None,
    img_cmap: str | None = None,
    img_alpha: float | None = None,
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

    ax.set_xlabel(fig_params.ax_labels[0])
    ax.set_ylabel(fig_params.ax_labels[1])

    ax.autoscale_view()

    if value_to_plot is not None:
        # if only dots were plotted without an associated value
        # there is not need to plot a legend or a colorbar

        if legend_fontoutline is not None:
            path_effect = [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
        else:
            path_effect = []

        # Adding legends
        if is_categorical_dtype(color_source_vector):
            clusters = color_source_vector.categories
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
                na_color=[na_color],
                na_in_legend=na_in_legend,
                multi_panel=True if fig_params.axs is not None else False,
            )
        else:
            # TODO: na_in_legend should have some effect here
            plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30)

    cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])

    if img is not None:
        ax.imshow(img, cmap=img_cmap, alpha=img_alpha)
    else:
        ax.set_aspect("equal")
        ax.invert_yaxis()

    if crops is not None:
        crops_tup = crops.to_tuple()
        ax.set_xlim(crops_tup[0], crops_tup[1])
        ax.set_ylim(crops_tup[3], crops_tup[2])
    else:
        ax.set_xlim(cur_coords[0], cur_coords[1])
        ax.set_ylim(cur_coords[3], cur_coords[2])

    if isinstance(scalebar_dx, list) and isinstance(scalebar_units, list):
        scalebar = ScaleBar(scalebar_dx[lib_count], units=scalebar_units[lib_count], **scalebar_kwargs)
        ax.add_artist(scalebar)

    return ax


def _map_color_seg(
    seg: NDArrayA,
    cell_id: NDArrayA,
    color_vector: NDArrayA | pd.Series[CategoricalDtype],
    color_source_vector: pd.Series[CategoricalDtype],
    cmap_params: CmapParams,
    seg_erosionpx: int | None = None,
    seg_boundaries: bool = False,
    na_color: str | Tuple[float, ...] = (0, 0, 0, 0),
) -> NDArrayA:
    cell_id = np.array(cell_id)

    if is_categorical_dtype(color_vector):
        if isinstance(na_color, tuple) and len(na_color) == 4 and np.any(color_source_vector.isna()):
            cell_id[color_source_vector.isna()] = 0
        val_im: NDArrayA = map_array(seg, cell_id, color_vector.codes + 1)  # type: ignore
        cols = colors.to_rgba_array(color_vector.categories)  # type: ignore
    else:
        val_im = map_array(seg, cell_id, cell_id)  # replace with same seg id to remove missing segs
        cols = cmap_params.cmap(cmap_params.norm(color_vector))

    if seg_erosionpx is not None:
        val_im[val_im == erosion(val_im, square(seg_erosionpx))] = 0

    seg_im: NDArrayA = label2rgb(
        label=val_im,
        colors=cols,
        bg_label=0,
        bg_color=(1, 1, 1),  # transparency doesn't really work
    )

    if seg_boundaries:
        seg_bound: NDArrayA = np.clip(seg_im - find_boundaries(seg)[:, :, None], 0, 1)
        seg_bound = np.dstack((seg_bound, np.where(val_im > 0, 1, 0)))  # add transparency here
        return seg_bound
    else:
        seg_im = np.dstack((seg_im, np.where(val_im > 0, 1, 0)))  # add transparency here
        return seg_im


def _prepare_args_plot(
    adata: AnnData,
    shape: _AvailShapes | None = None,
    color: Sequence[str | None] | str | None = None,
    groups: _SeqStr = None,
    img_alpha: Optional[float] = None,
    alpha: Optional[float] = None,
    use_raw: bool | None = None,
    layer: str | None = None,
    palette: Palette_t = None,
) -> ColorParams:
    alpha = 1.0 if alpha is None else alpha
    img_alpha = 1.0 if img_alpha is None else img_alpha

    # make colors and groups as list
    groups = [groups] if isinstance(groups, str) else groups
    color = [color] if isinstance(color, str) or color is None else color

    # set palette if missing
    for c in color:
        if c is not None and c in adata.obs.columns and is_categorical_dtype(adata.obs[c]):
            _maybe_set_colors(source=adata, target=adata, key=c, palette=palette)

    # check raw
    if use_raw is None:
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            f"Cannot use both a layer and the raw representation. Was passed: use_raw={use_raw}, layer={layer}."
        )
    if adata.raw is None and use_raw:
        raise ValueError(f"`use_raw={use_raw}` but AnnData object does not have raw.")

    # logic for image v. non-image data is handled here
    shape = ScatterShape(shape) if shape is not None else shape  # type: ignore
    if TYPE_CHECKING:
        assert isinstance(shape, ScatterShape) or shape is None

    return ColorParams(shape, color, groups, alpha, img_alpha, use_raw)


def _prepare_params_plot(
    color_params: ColorParams,
    spatial_params: SpatialParams,
    spatial_key: str = Key.obsm.spatial,
    wspace: float | None = None,
    hspace: float = 0.25,
    ncols: int = 4,
    cmap: Colormap | str | None = None,
    norm: _Normalize = None,
    library_first: bool = True,
    img_cmap: Colormap | str | None = None,
    frameon: Optional[bool] = None,
    na_color: str | Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    title: _SeqStr = None,
    axis_label: _SeqStr = None,
    scalebar_dx: _SeqFloat = None,
    scalebar_units: _SeqStr = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    fig: Figure | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    **kwargs: Any,
) -> Tuple[FigParams, CmapParams, ScalebarParams, Any]:

    if library_first:
        iter_panels: Tuple[range | Sequence[str | None], range | Sequence[str | None]] = (
            range(len(spatial_params.library_id)),
            color_params.color,
        )
    else:
        iter_panels = (color_params.color, range(len(spatial_params.library_id)))
    num_panels = len(list(itertools.product(*iter_panels)))

    wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02 if wspace is None else wspace
    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    dpi = rcParams["figure.dpi"] if dpi is None else dpi
    if num_panels > 1 and ax is None:
        fig, grid = _panel_grid(
            num_panels=num_panels, hspace=hspace, wspace=wspace, ncols=ncols, dpi=dpi, figsize=figsize
        )
        axs: Union[Sequence[Axes], None] = [plt.subplot(grid[c]) for c in range(num_panels)]
    elif num_panels > 1 and ax is not None:
        if len(ax) != num_panels:
            raise ValueError(f"Len of `ax`: {len(ax)} is not equal to number of panels: {num_panels}.")
        if fig is None:
            raise ValueError(
                f"Invalid value of `fig`: {fig}. If a list of `Axes` is passed, a `Figure` must also be specified."
            )
        axs = ax
    else:
        axs = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # set cmap and norm
    cmap = copy(get_cmap(cmap))
    cmap.set_bad("lightgray" if na_color is None else na_color)

    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    vcenter = kwargs.pop("vcenter", None)

    if norm is not None:
        if (vmin is not None) or (vmax is not None) or (vcenter is not None):
            raise ValueError("Passing both norm and vmin/vmax/vcenter is not allowed.")
    else:
        if vcenter is not None:
            norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

    # set title and axis labels
    title, ax_labels = _get_title_axlabels(title, axis_label, spatial_key, num_panels)

    # set scalebar
    if scalebar_dx is not None:
        scalebar_dx, scalebar_units = _get_scalebar(scalebar_dx, scalebar_units, len(spatial_params.library_id))

    fig_params = FigParams(fig, ax, axs, iter_panels, title, ax_labels, frameon)
    cmap_params = CmapParams(cmap, img_cmap, norm)
    scalebar_params = ScalebarParams(scalebar_dx, scalebar_units)

    return fig_params, cmap_params, scalebar_params, kwargs


def _panel_grid(
    num_panels: int,
    hspace: float,
    wspace: float,
    ncols: int,
    figsize: tuple[float, float],
    dpi: int | None = None,
) -> Tuple[Figure, GridSpec]:
    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)

    fig = plt.figure(
        figsize=(figsize[0] * n_panels_x * (1 + wspace), figsize[1] * n_panels_y),
        dpi=dpi,
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def _set_ax_title(fig_params: FigParams, count: int, value_to_plot: str | None = None) -> Axes:
    if fig_params.axs is not None:
        ax = fig_params.axs[count]
    else:
        ax = fig_params.ax
    if not (sc_settings._frameon if fig_params.frameon is None else fig_params.frameon):
        ax.axis("off")

    # set title
    if fig_params.title is None:
        ax.set_title(value_to_plot)
    else:
        ax.set_title(fig_params.title[count])
    return ax


def _set_outline(
    size: float,
    outline: bool = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ("black", "white"),
    **kwargs: Any,
) -> Tuple[OutlineParams, Any]:

    bg_width, gap_width = outline_width
    point = np.sqrt(size)
    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
    # the default black and white colors can be changes using the contour_config parameter
    bg_color, gap_color = outline_color

    if outline:
        kwargs.pop("edgecolor", None)  # remove edge from kwargs if present
        kwargs.pop("alpha", None)  # remove alpha from kwargs if present

    return OutlineParams(outline, gap_size, gap_color, bg_size, bg_color), kwargs


def _plot_scatter(
    coords: NDArrayA,
    ax: Axes,
    outline_params: OutlineParams,
    cmap_params: CmapParams,
    color_params: Colormap,
    size: float,
    color_vector: NDArrayA,
    **kwargs: Any,
) -> Tuple[Axes, Collection | PatchCollection]:

    if color_params.shape is not None:
        scatter = partial(_shaped_scatter, shape=color_params.shape, alpha=color_params.alpha)
    else:
        scatter = partial(ax.scatter, marker=".", alpha=color_params.alpha, plotnonfinite=True)

    if outline_params.outline:
        _cax = scatter(
            coords[:, 0],
            coords[:, 1],
            s=outline_params.bg_size,
            c=outline_params.bg_color,
            rasterized=sc_settings._vector_friendly,
            cmap=cmap_params.cmap,
            norm=cmap_params.norm,
            **kwargs,
        )
        ax.add_collection(_cax)
        _cax = scatter(
            coords[:, 0],
            coords[:, 1],
            s=outline_params.gap_size,
            c=outline_params.gap_color,
            rasterized=sc_settings._vector_friendly,
            cmap=cmap_params.cmap,
            norm=cmap_params.norm,
            **kwargs,
        )
        ax.add_collection(_cax)

    _cax = scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_vector,
        s=size,
        rasterized=sc_settings._vector_friendly,
        cmap=cmap_params.cmap,
        norm=cmap_params.norm,
        **kwargs,
    )
    cax = ax.add_collection(_cax)

    return ax, cax


def _plot_segment(
    seg: NDArrayA,
    cell_id: NDArrayA,
    color_vector: NDArrayA | pd.Series[CategoricalDtype],
    color_source_vector: pd.Series[CategoricalDtype],
    ax: Axes,
    cmap_params: CmapParams,
    color_params: Colormap,
    categorical: bool,
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    na_color: str | Tuple[float, ...] = (0, 0, 0, 0),
    **kwargs: Any,
) -> Tuple[Axes, Collection]:

    img = _map_color_seg(
        seg=seg,
        cell_id=cell_id,
        color_vector=color_vector,
        color_source_vector=color_source_vector,
        cmap_params=cmap_params,
        seg_erosionpx=seg_contourpx,
        seg_boundaries=seg_outline,
        na_color=na_color,
    )

    _cax = ax.imshow(
        img,
        rasterized=True,
        cmap=cmap_params.cmap if not categorical else None,
        norm=cmap_params.norm if not categorical else None,
        alpha=color_params.alpha,
        origin="lower",
        zorder=3,
        **kwargs,
    )
    cax = ax.add_collection(_cax, autolim=False)

    return ax, cax
