from __future__ import annotations

import itertools
from copy import copy
from functools import partial
from numbers import Number
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import colors, patheffects, rcParams
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.collections import Collection, PatchCollection
from matplotlib.colors import (
    ColorConverter,
    Colormap,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
from pandas import CategoricalDtype
from pandas.core.dtypes.common import is_categorical_dtype
from scanpy import logging as logg
from scanpy._settings import settings as sc_settings
from scanpy.plotting._tools.scatterplots import _add_categorical_legend
from skimage.color import label2rgb
from skimage.morphology import erosion, square
from skimage.segmentation import find_boundaries
from skimage.util import map_array

from squidpy._constants._constants import ScatterShape
from squidpy._constants._pkg_constants import Key
from squidpy._utils import NDArrayA
from squidpy.im._coords import CropCoords
from squidpy.pl._color_utils import _get_palette, _maybe_set_colors
from squidpy.pl._utils import _assert_value_in_obs

_AvailShapes = Literal["circle", "square", "hex"]
Palette_t = Optional[Union[str, ListedColormap]]
_Normalize = Union[Normalize, Sequence[Normalize]]
_SeqStr = Union[str, Sequence[str]]
_SeqFloat = Union[float, Sequence[float]]
_SeqArray = Union[NDArrayA, Sequence[NDArrayA]]
_CoordTuple = Tuple[int, int, int, int]
_FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]


# named tuples
class FigParams(NamedTuple):
    """Figure params."""

    fig: Figure
    ax: Axes
    axs: Sequence[Axes] | None
    iter_panels: tuple[Sequence[Any], Sequence[Any]]
    title: _SeqStr | None
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
    scalebar_units: _SeqStr | None


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
    img: Sequence[NDArrayA] | tuple[None, ...]
    segment: Sequence[NDArrayA] | tuple[None, ...]
    cell_id: Sequence[NDArrayA] | tuple[None, ...]


to_hex = partial(colors.to_hex, keep_alpha=True)


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
        if library_key not in adata.obs:
            raise KeyError(f"`library_key: {library_key}` not in `adata.obs`.")
        if library_id is None:
            library_id = adata.obs[library_key].cat.categories.tolist()
        _assert_value_in_obs(adata, key=library_key, val=library_id)
        if isinstance(library_id, str):
            library_id = [library_id]
        return library_id
    if library_id is None:
        logg.warning("Please specify a valid `library_id` or set it permanently in `adata.uns['spatial']`")
        library_id = [""]  # dummy value to maintain logic of number of plots (nplots=library_id*color)
    elif isinstance(library_id, list):  # get library_id from arg
        pass
    elif isinstance(library_id, str):
        library_id = [library_id]
    else:
        raise TypeError(f"Invalid `library_id`: {library_id}.")
    return library_id


def _get_image(
    adata: AnnData,
    library_id: Sequence[str],
    spatial_key: str = Key.obsm.spatial,
    img: bool | _SeqArray | None = None,
    img_res_key: str | None = None,
    img_channel: int | list[int] | None = None,
    img_cmap: Colormap | str | None = None,
) -> Sequence[NDArrayA] | tuple[None, ...]:
    from squidpy.pl._utils import _to_grayscale

    if isinstance(img, (list, np.ndarray, da.Array)):
        img = _get_list(img, _type=(np.ndarray, da.Array), ref_len=len(library_id), name="img")
    else:
        image_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.image_key, library_id)
        if img_res_key is None:
            img_res_key = _get_unique_map(image_mapping)[0]
        elif img_res_key not in _get_unique_map(image_mapping):
            raise KeyError(
                f"Image key: `{img_res_key}` does not exist. Available image keys: `{image_mapping.values()}`"
            )
        img = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][img_res_key] for i in library_id]

    if img_channel is None:
        img = [im[..., :3] for im in img]
    elif isinstance(img_channel, int):
        img = [im[..., [img_channel]] for im in img]
    elif isinstance(img_channel, list):
        img = [im[..., img_channel] for im in img]
    else:
        raise TypeError(f"Expected image channel to be either `int` or `None`, found `{type(img_channel).__name__}`.")

    if img_cmap == "gray":
        img = [_to_grayscale(im) for im in img]
    return img


def _get_segment(
    adata: AnnData,
    library_id: Sequence[str],
    seg_cell_id: str | None = None,
    library_key: str | None = None,
    seg: _SeqArray | bool | None = None,
    seg_key: str | None = None,
) -> tuple[Sequence[NDArrayA], Sequence[NDArrayA]] | tuple[tuple[None, ...], tuple[None, ...]]:
    if seg_cell_id not in adata.obs:
        raise ValueError(f"Cell id `{seg_cell_id!r}` not found in `adata.obs`.")
    cell_id_vec = adata.obs[seg_cell_id].values

    if library_key not in adata.obs:
        raise ValueError(f"Library key `{library_key}` not found in `adata.obs`.")
    if not np.issubdtype(cell_id_vec.dtype, np.integer):
        raise ValueError(f"Invalid type `{cell_id_vec.dtype}` for `adata.obs[{seg_cell_id!r}]`.")
    cell_id_vec = [cell_id_vec[adata.obs[library_key] == lib] for lib in library_id]

    if isinstance(seg, (list, np.ndarray, da.Array)):
        img_seg = _get_list(seg, _type=(np.ndarray, da.Array), ref_len=len(library_id), name="img_seg")
    else:
        img_seg = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][seg_key] for i in library_id]
    return img_seg, cell_id_vec


def _get_scalefactor_size(
    adata: AnnData,
    library_id: Sequence[str],
    spatial_key: str = Key.obsm.spatial,
    img_res_key: str | None = None,
    scale_factor: _SeqFloat | None = None,
    size: _SeqFloat | None = None,
    size_key: str | None = Key.uns.size_key,
) -> tuple[Sequence[float], Sequence[float]]:
    try:
        scalefactor_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.scalefactor_key, library_id)
        scalefactors = _get_unique_map(scalefactor_mapping)
    except KeyError as e:
        scalefactors = None
        logg.debug(f"Setting `scalefactors={scalefactors}`, reason: `{e}`")

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
            scale_factor = _get_list(scale_factor, _type=float, ref_len=len(library_id), name="scale_factor")

        if size_key not in scalefactors and size is None:
            raise ValueError(
                f"Specified `size_key: {size_key}` does not exist and size is `None`, "
                f"available keys are: `{scalefactors}`. Specify a valid `size_key` or `size`."
            )
        if size is None:
            size = 1.0
        size = _get_list(size, _type=Number, ref_len=len(library_id), name="size")
        if not (len(size) == len(library_id) == len(scale_factor)):
            raise ValueError("Len of `size`, `library_id` and `scale_factor` do not match.")
        size = [
            adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][size_key] * s * sf * 0.5
            for i, s, sf in zip(library_id, size, scale_factor)
        ]
        return scale_factor, size

    scale_factor = 1.0 if scale_factor is None else scale_factor
    scale_factor = _get_list(scale_factor, _type=float, ref_len=len(library_id), name="scale_factor")

    size = 120000 / adata.shape[0] if size is None else size
    size = _get_list(size, _type=Number, ref_len=len(library_id), name="size")
    return scale_factor, size


def _image_spatial_attrs(
    adata: AnnData,
    shape: _AvailShapes | None = None,
    spatial_key: str = Key.obsm.spatial,
    library_id: Sequence[str] | None = None,
    library_key: str | None = None,
    img: bool | _SeqArray | None = None,
    img_res_key: str | None = Key.uns.image_res_key,
    img_channel: int | list[int] | None = None,
    seg: _SeqArray | bool | None = None,
    seg_key: str | None = None,
    cell_id_key: str | None = None,
    scale_factor: _SeqFloat | None = None,
    size: _SeqFloat | None = None,
    size_key: str | None = Key.uns.size_key,
    img_cmap: Colormap | str | None = None,
) -> SpatialParams:
    def truthy(img: bool | NDArrayA | _SeqArray | None) -> bool:
        if img is None or img is False:
            return False
        return img is True or len(img)  # type: ignore

    library_id = _get_library_id(
        adata=adata, shape=shape, spatial_key=spatial_key, library_id=library_id, library_key=library_key
    )
    if len(library_id) > 1 and library_key is None:
        raise ValueError(
            f"Found `library_id: `{library_id} but no `library_key` was specified. Please specify `library_key`."
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

    if (truthy(img) and truthy(seg)) or (truthy(img) and shape is not None):
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
        _img = (None,) * len(library_id)

    if truthy(seg):
        _seg, _cell_vec = _get_segment(
            adata=adata,
            library_id=library_id,
            seg_cell_id=cell_id_key,
            library_key=library_key,
            seg=seg,
            seg_key=seg_key,
        )
    else:
        _seg = (None,) * len(library_id)
        _cell_vec = (None,) * len(library_id)

    return SpatialParams(library_id, scale_factor, size, _img, _seg, _cell_vec)


def _set_coords_crops(
    adata: AnnData,
    spatial_params: SpatialParams,
    spatial_key: str,
    crop_coord: Sequence[_CoordTuple] | _CoordTuple | None = None,
) -> tuple[list[NDArrayA], list[CropCoords] | list[None]]:
    if crop_coord is None:
        crops = [None] * len(spatial_params.library_id)
    else:
        crop_coord = _get_list(crop_coord, _type=tuple, ref_len=len(spatial_params.library_id), name="crop_coord")
        crops = [CropCoords(*cr) * sf for cr, sf in zip(crop_coord, spatial_params.scale_factor)]  # type: ignore[misc]

    coords = adata.obsm[spatial_key]
    return [coords * sf for sf in spatial_params.scale_factor], crops  # TODO(giovp): refactor with _subs


def _subs(
    adata: AnnData,
    coords: NDArrayA,
    img: NDArrayA | None = None,
    library_key: str | None = None,
    library_id: str | None = None,
    crop_coords: CropCoords | None = None,
    groups_key: str | None = None,
    groups: Sequence[Any] | None = None,
) -> AnnData:
    def assert_notempty(adata: AnnData, *, msg: str) -> AnnData:
        if not adata.n_obs:
            raise ValueError(f"Empty AnnData, reason: {msg}.")
        return adata

    def subset_by_key(
        adata: AnnData,
        coords: NDArrayA,
        key: str | None,
        values: Sequence[Any] | None,
    ) -> tuple[AnnData, NDArrayA]:
        if key is None or values is None:
            return adata, coords
        if key not in adata.obs or not is_categorical_dtype(adata.obs[key]):
            return adata, coords
        try:
            mask = adata.obs[key].isin(values).values
            msg = f"None of `adata.obs[{key}]` are in `{values}`"
            return assert_notempty(adata[mask], msg=msg), coords[mask]
        except KeyError:
            raise KeyError(f"Unable to find `{key!r}` in `adata.obs`.") from None

    def subset_by_coords(
        adata: AnnData, coords: NDArrayA, img: NDArrayA | None, crop_coords: CropCoords | None
    ) -> tuple[AnnData, NDArrayA, NDArrayA | None]:
        if crop_coords is None:
            return adata, coords, img

        mask = (
            (coords[:, 0] >= crop_coords.x0)
            & (coords[:, 0] <= crop_coords.x1)
            & (coords[:, 1] >= crop_coords.y0)
            & (coords[:, 1] <= crop_coords.y1)
        )
        adata = assert_notempty(adata[mask, :], msg=f"Invalid crop coordinates `{crop_coords}`")
        coords = coords[mask]
        coords[:, 0] -= crop_coords.x0
        coords[:, 1] -= crop_coords.y0
        if img is not None:
            img = img[crop_coords.slice]
        return adata, coords, img

    adata, coords, img = subset_by_coords(adata, coords=coords, img=img, crop_coords=crop_coords)
    adata, coords = subset_by_key(adata, coords=coords, key=library_key, values=[library_id])
    adata, coords = subset_by_key(adata, coords=coords, key=groups_key, values=groups)
    return adata, coords, img


def _get_unique_map(dic: Mapping[str, Any]) -> Sequence[Any]:
    """Get intersection of dict values."""
    return sorted(set.intersection(*map(set, dic.values())))


def _get_list(
    var: Any,
    _type: type[Any] | tuple[type[Any], ...],
    ref_len: int | None = None,
    name: str | None = None,
) -> list[Any]:
    if isinstance(var, _type):
        return [var] if ref_len is None else ([var] * ref_len)
    if isinstance(var, list):
        if ref_len is not None and ref_len != len(var):
            raise ValueError(
                f"Variable: `{name}` has length: {len(var)}, which is not equal to reference length: {ref_len}."
            )
        for v in var:
            if not isinstance(v, _type):
                raise ValueError(f"Variable: `{name}` has invalid type: {type(v)}, expected: {_type}.")
        return var

    raise ValueError(f"Can't make a list from variable: `{var}`")


def _set_color_source_vec(
    adata: AnnData,
    value_to_plot: str | None,
    use_raw: bool | None = None,
    alt_var: str | None = None,
    layer: str | None = None,
    groups: _SeqStr | None = None,
    palette: Palette_t = None,
    na_color: str | tuple[float, ...] | None = None,
    alpha: float = 1.0,
) -> tuple[NDArrayA | pd.Series | None, NDArrayA, bool]:
    if value_to_plot is None:
        color = np.full(adata.n_obs, to_hex(na_color))
        return color, color, False

    if alt_var is not None and value_to_plot not in adata.obs and value_to_plot not in adata.var_names:
        value_to_plot = adata.var_names[adata.var[alt_var] == value_to_plot][0]
    if use_raw and value_to_plot not in adata.obs:
        color_source_vector = adata.raw.obs_vector(value_to_plot)
    else:
        color_source_vector = adata.obs_vector(value_to_plot, layer=layer)

    if not is_categorical_dtype(color_source_vector):
        return None, color_source_vector, False

    color_source_vector = pd.Categorical(color_source_vector)  # convert, e.g., `pd.Series`
    categories = color_source_vector.categories
    if groups is not None:
        color_source_vector = color_source_vector.remove_categories(categories.difference(groups))

    color_map = _get_palette(adata, cluster_key=value_to_plot, categories=categories, palette=palette, alpha=alpha)
    if color_map is None:
        raise ValueError("Unable to create color palette.")
    # do not rename categories, as colors need not be unique
    color_vector = color_source_vector.map(color_map)
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
    norm: _Normalize | None = None,
    **kwargs: Any,
) -> PatchCollection:
    """
    Get shapes for scatter plot.

    Adapted from `here <https://gist.github.com/syrte/592a062c562cd2a98a83>`_.
    This code is under `The BSD 3-Clause License <http://opensource.org/licenses/BSD-3-Clause>`_.
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
        alpha = ColorConverter().to_rgba_array(c)[..., -1]
        collection.set_facecolor(c)
        collection.set_alpha(alpha)

    return collection


def _make_poly(x: NDArrayA, y: NDArrayA, r: float, n: int, i: int) -> tuple[NDArrayA, NDArrayA]:
    x_i = x + r * np.sin((np.pi / n) * (1 + 2 * i))
    y_i = y + r * np.cos((np.pi / n) * (1 + 2 * i))
    return x_i, y_i


def _plot_edges(
    adata: AnnData,
    coords: NDArrayA,
    connectivity_key: str,
    ax: Axes,
    edges_width: float = 0.1,
    edges_color: str | Sequence[float] | Sequence[str] = "grey",
    **kwargs: Any,
) -> None:
    from networkx import Graph
    from networkx.drawing import draw_networkx_edges

    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"Unable to find `connectivity_key: {connectivity_key}` in `adata.obsp`. Please set `connectivity_key`."
        )

    g = Graph(adata.obsp[connectivity_key])
    if not len(g.edges):
        return None
    edge_collection = draw_networkx_edges(
        g, coords, width=edges_width, edge_color=edges_color, arrows=False, ax=ax, **kwargs
    )
    edge_collection.set_rasterized(sc_settings._vector_friendly)
    ax.add_collection(edge_collection)


def _get_title_axlabels(
    title: _SeqStr | None, axis_label: _SeqStr | None, spatial_key: str, n_plots: int
) -> tuple[_SeqStr | None, Sequence[str]]:
    if title is not None:
        if isinstance(title, (tuple, list)) and len(title) != n_plots:
            raise ValueError(f"Expected `{n_plots}` titles, found `{len(title)}`.")
        elif isinstance(title, str):
            title = [title] * n_plots
    else:
        title = None

    axis_label = spatial_key if axis_label is None else axis_label
    if isinstance(axis_label, list):
        if len(axis_label) != 2:
            raise ValueError(f"Expected axis labels to be of length `2`, found `{len(axis_label)}`.")
        axis_labels = axis_label
    elif isinstance(axis_label, str):
        axis_labels = [axis_label + str(x + 1) for x in range(2)]
    else:
        raise TypeError(f"Expected axis labels to be of type `list` or `str`, found `{type(axis_label).__name__}`.")

    return title, axis_labels


def _get_scalebar(
    scalebar_dx: _SeqFloat | None = None,
    scalebar_units: _SeqStr | None = None,
    len_lib: int | None = None,
) -> tuple[Sequence[float] | None, Sequence[str] | None]:
    if scalebar_dx is not None:
        _scalebar_dx = _get_list(scalebar_dx, _type=float, ref_len=len_lib, name="scalebar_dx")
        scalebar_units = "um" if scalebar_units is None else scalebar_units
        _scalebar_units = _get_list(scalebar_units, _type=str, ref_len=len_lib, name="scalebar_units")
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
    img: NDArrayA | None = None,
    img_cmap: str | None = None,
    img_alpha: float | None = None,
    palette: Palette_t = None,
    alpha: float = 1.0,
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str | None = "right margin",
    legend_fontoutline: int | None = None,
    na_color: str | tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    na_in_legend: bool = True,
    colorbar: bool = True,
    scalebar_dx: Sequence[float] | None = None,
    scalebar_units: Sequence[str] | None = None,
    scalebar_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes:
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(fig_params.ax_labels[0])
    ax.set_ylabel(fig_params.ax_labels[1])
    ax.autoscale_view()  # needed when plotting points but no image

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
            palette = _get_palette(adata, cluster_key=value_to_plot, categories=clusters, palette=palette, alpha=alpha)
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=palette,
                scatter_array=coords,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=[na_color],
                na_in_legend=na_in_legend,
                multi_panel=fig_params.axs is not None,
            )
        elif colorbar:
            # TODO: na_in_legend should have some effect here
            plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30)

    if img is not None:
        ax.imshow(img, cmap=img_cmap, alpha=img_alpha)
    else:
        ax.set_aspect("equal")
        ax.invert_yaxis()

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
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
) -> NDArrayA:
    cell_id = np.array(cell_id)

    if is_categorical_dtype(color_vector):
        if isinstance(na_color, tuple) and len(na_color) == 4 and np.any(color_source_vector.isna()):
            cell_id[color_source_vector.isna()] = 0
        val_im: NDArrayA = map_array(seg, cell_id, color_vector.codes + 1)  # type: ignore[union-attr]
        cols = colors.to_rgba_array(color_vector.categories)  # type: ignore[union-attr]
    else:
        val_im = map_array(seg, cell_id, cell_id)  # replace with same seg id to remove missing segs
        try:
            cols = cmap_params.cmap(cmap_params.norm(color_vector))
        except TypeError:
            assert all(colors.is_color_like(c) for c in color_vector), "Not all values are color-like."
            cols = colors.to_rgba_array(color_vector)

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
    seg_im = np.dstack((seg_im, np.where(val_im > 0, 1, 0)))  # add transparency here
    return seg_im


def _prepare_args_plot(
    adata: AnnData,
    shape: _AvailShapes | None = None,
    color: Sequence[str | None] | str | None = None,
    groups: _SeqStr | None = None,
    img_alpha: float | None = None,
    alpha: float = 1.0,
    use_raw: bool | None = None,
    layer: str | None = None,
    palette: Palette_t = None,
) -> ColorParams:
    img_alpha = 1.0 if img_alpha is None else img_alpha

    # make colors and groups as list
    groups = [groups] if isinstance(groups, str) else groups
    if isinstance(color, list):
        if not len(color):
            color = None
    color = [color] if isinstance(color, str) or color is None else color

    # set palette if missing
    for c in color:
        if c is not None and c in adata.obs and isinstance(adata.obs[c].dtype, CategoricalDtype):
            _maybe_set_colors(source=adata, target=adata, key=c, palette=palette)

    # check raw
    if use_raw is None:
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            f"Cannot use both a layer and the raw representation. Got passed: use_raw={use_raw}, layer={layer}."
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
    norm: _Normalize | None = None,
    library_first: bool = True,
    img_cmap: Colormap | str | None = None,
    frameon: bool | None = None,
    na_color: str | tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0),
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    title: _SeqStr | None = None,
    axis_label: _SeqStr | None = None,
    scalebar_dx: _SeqFloat | None = None,
    scalebar_units: _SeqStr | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    fig: Figure | None = None,
    ax: Axes | Sequence[Axes] | None = None,
    **kwargs: Any,
) -> tuple[FigParams, CmapParams, ScalebarParams, Any]:
    if library_first:
        iter_panels: tuple[range | Sequence[str | None], range | Sequence[str | None]] = (
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
        axs: Sequence[Axes] | None = [plt.subplot(grid[c]) for c in range(num_panels)]
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

    if isinstance(norm, Normalize):
        pass
    elif vcenter is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)

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
) -> tuple[Figure, GridSpec]:
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

    if fig_params.title is None:
        ax.set_title(value_to_plot)
    else:
        ax.set_title(fig_params.title[count])
    return ax


def _set_outline(
    size: float,
    outline: bool = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    **kwargs: Any,
) -> tuple[OutlineParams, Any]:
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
    color_params: ColorParams,
    size: float,
    color_vector: NDArrayA,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),  # TODO(giovp): remove?
    **kwargs: Any,
) -> tuple[Axes, Collection | PatchCollection]:
    if color_params.shape is not None:
        scatter = partial(_shaped_scatter, shape=color_params.shape, alpha=color_params.alpha)
    else:
        scatter = partial(ax.scatter, marker=".", alpha=color_params.alpha, plotnonfinite=True)

    # prevents reusing vmin/vmax when sharing a norm
    norm = copy(cmap_params.norm)
    if outline_params.outline:
        _cax = scatter(
            coords[:, 0],
            coords[:, 1],
            s=outline_params.bg_size,
            c=outline_params.bg_color,
            rasterized=sc_settings._vector_friendly,
            cmap=cmap_params.cmap,
            norm=norm,
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
            norm=norm,
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
        norm=norm,
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
    color_params: ColorParams,
    categorical: bool,
    seg_contourpx: int | None = None,
    seg_outline: bool = False,
    na_color: str | tuple[float, ...] = (0, 0, 0, 0),
    **kwargs: Any,
) -> tuple[Axes, Collection]:
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
    cax = ax.add_image(_cax)

    return ax, cax
