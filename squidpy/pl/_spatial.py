from __future__ import annotations

from cycler import Cycler
from typing import Any, Tuple, Union, Literal, Mapping, Callable, Optional, Sequence
from functools import partial
import itertools

from anndata import AnnData
from scanpy.plotting._tools.scatterplots import _panel_grid, _get_palette

from pandas.core.dtypes.common import is_categorical_dtype
import numpy as np
import pandas as pd

from matplotlib import colors, pyplot as pl, rcParams
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from squidpy._utils import NDArrayA
from squidpy.gr._utils import _assert_spatial_basis
from squidpy.pl._utils import _sanitize_anndata, _assert_value_in_obs
from squidpy._constants._pkg_constants import Key

VBound = Union[str, float, Callable[[Sequence[float]], float]]


def spatial(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    library_id: Optional[Sequence[str] | str | None] = None,
    batch_key: Optional[str | None] = None,
    img: Optional[Sequence[NDArrayA] | NDArrayA | None] = None,
    img_key: str | None = None,
    scale_factor: Optional[Sequence[float] | float | None] = None,
    bw: bool = False,
    color: Optional[Sequence[str] | str | None] = None,
    groups: Optional[Sequence[str] | str | None] = None,
    use_raw: Optional[bool | None] = None,
    layer: Optional[str | None] = None,
    alt_var: Optional[str | None] = None,
    projection: Literal["2d", "3d"] = "2d",
    size: Optional[float | None] = None,
    palette: Optional[Sequence[str] | str | Cycler | None] = None,
    na_color: Optional[str | Tuple[float, ...] | None] = None,
    wspace: Optional[float | None] = None,
    hspace: float = 0.25,
    ncols: int = 4,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    ax: Optional[Axes | None] = None,
    legend_kwargs: Optional[Mapping[str, Sequence[str]] | None] = None,
    label_kwargs: Optional[Mapping[str, Sequence[str]] | None] = None,
) -> Any:
    """Spatial plotting for squidpy."""
    _sanitize_anndata(adata)
    _assert_spatial_basis(adata, spatial_key)

    # get projection
    args_3d = {"projection": "3d"} if projection == "3d" else {}

    # make colors and groups as list
    if groups:
        if isinstance(groups, str):
            groups = [groups]
    if isinstance(color, str):
        color = [color]
    elif color is None:
        color = []

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
    if wspace is None:
        wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02

    # set title
    # set cmap

    # get spatial attributes
    if isinstance(library_id, str):
        library_id = [library_id]
    if isinstance(scale_factor, float):
        scale_factor = [scale_factor]
    if isinstance(img, np.ndarray):
        img = [img]

    library_id, scale_factor, img = _get_spatial_attrs(adata, spatial_key, library_id, img, img_key, scale_factor, bw)

    if batch_key is not None:
        _assert_value_in_obs(adata, key=batch_key, val=library_id)
    if (len(library_id) > 1) and batch_key is None:
        raise ValueError(
            f"Multiple `library_id={library_id}` found but no `batch_key` specified. Please specify `batch_key`."
        )
    coords = _get_coords(adata, library_id, spatial_key, batch_key, scale_factor)

    # initialize axis
    if not isinstance(color, str) and isinstance(color, Sequence) and len(color) > 1:
        if ax is not None:
            raise ValueError("Cannot specify `ax` when plotting multiple panels ")

        # each plot needs to be its own panel
        num_panels = len(color) * len(library_id)
        fig, grid = _panel_grid(hspace, wspace, ncols, num_panels)
    else:
        grid = None
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111, **args_3d)

    vmin, vmax, vcenter, norm = _get_seq_vminmax(vmin, vmax, vcenter, norm)

    # set size
    if size is None:
        size = 1

    # make plots
    for _count, (value_to_plot, _lib) in enumerate(itertools.product(color, library_id)):
        color_source_vector = _get_source_vec(
            adata, value_to_plot, layer=layer, use_raw=use_raw, alt_var=alt_var, groups=groups
        )
        color_vector, categorical = _get_color_vec(
            adata,
            value_to_plot,
            color_source_vector,
            palette=palette,
            na_color=na_color,
        )

    return fig, ax, grid, coords, color_vector, categorical


def _get_spatial_attrs(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    library_id: Optional[Sequence[str] | None] = None,
    img: Optional[Sequence[NDArrayA] | None] = None,
    img_key: str | None = None,
    scale_factor: Optional[Sequence[float] | None] = None,
    bw: bool = False,
) -> Tuple[Sequence[str], Sequence[float], Sequence[NDArrayA]]:
    """Return lists of image attributes saved in adata for plotting."""
    library_id = Key.uns.library_id(adata, spatial_key, library_id, return_all=True)
    if library_id is None:
        raise ValueError(f"Could not fetch `library_id`, check that `spatial_key={spatial_key}` is correct.")

    image_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.image_key, library_id)
    scalefactor_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.scalefactor_key, library_id)

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

    if scale_factor is None:  # get intersection of scale_factor and match to img_key
        scale_factor_key = _get_unique_map(scalefactor_mapping)
        scale_factor_key = [i for i in scale_factor_key if img_key in i][0]
        if len(scale_factor_key) == 0:
            raise ValueError(f"No `scale_factor` found that could match `img_key`: {img_key}.")
        scale_factor = [adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][scale_factor_key] for i in library_id]
    else:
        if len(scale_factor) != len(library_id):
            raise ValueError(
                f"Len of scale_factor list: {len(scale_factor)} is not equal to len of library_id: {len(library_id)}."
            )

    if img is None:
        img = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][img_key] for i in library_id]
    else:
        if len(img) != len(library_id):
            raise ValueError(f"Len of img list: {len(img)} is not equal to len of library_id: {len(library_id)}.")

    if bw:
        img = [np.dot(im[..., :3], [0.2989, 0.5870, 0.1140]) for im in img]

    return library_id, scale_factor, img


def _get_coords(
    adata: AnnData,
    library_id: Sequence[str],
    spatial_key: str = Key.obsm.spatial,
    batch_key: Optional[str | None] = None,
    scale_factor: Optional[Sequence[float] | None] = None,
) -> Sequence[NDArrayA]:

    coords = adata.obsm[spatial_key]

    if (batch_key is not None) and (len(library_id) > 1):
        data_points = []
        for lib in library_id:
            data_points.append(coords[adata.obs[batch_key] == lib, :])
    else:
        data_points = [coords]

    if scale_factor is not None and (len(data_points) == len(scale_factor)):
        for i, sf in enumerate(scale_factor):
            data_points[i] = np.multiply(data_points[i], sf)
    else:
        raise ValueError("Len of `data_points` and `scale_factor` does not match.")

    return data_points


def _get_unique_map(dic: Mapping[str, Any]) -> Any:
    """Get intersection of dict values."""
    return sorted(set.intersection(*map(set, dic.values())))


def _get_seq_vminmax(
    vmin: Any, vmax: Any, vcenter: Any, norm: Any
) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]]:
    if isinstance(vmax, str) or not isinstance(vmax, Sequence):
        vmax = [vmax]
    if isinstance(vmin, str) or not isinstance(vmin, Sequence):
        vmin = [vmin]
    if isinstance(vcenter, str) or not isinstance(vcenter, Sequence):
        vcenter = [vcenter]
    if isinstance(norm, Normalize) or not isinstance(norm, Sequence):
        norm = [norm]

    return vmin, vmax, vcenter, norm


def _get_source_vec(
    adata: AnnData,
    value_to_plot: str | None,
    use_raw: Optional[bool | None] = None,
    alt_var: Optional[str | None] = None,
    layer: Optional[str | None] = None,
    groups: Optional[Sequence[str] | str | None] = None,
) -> NDArrayA | pd.Series:

    if value_to_plot is None:
        return np.broadcast_to(np.nan, adata.n_obs)
    if alt_var is not None and value_to_plot not in adata.obs.columns and value_to_plot not in adata.var_names:
        value_to_plot = adata.var.index[adata.var[alt_var] == value_to_plot][0]
    if use_raw and value_to_plot not in adata.obs.columns:
        values = adata.raw.obs_vector(value_to_plot)
    else:
        values = adata.obs_vector(value_to_plot, layer=layer)
    if groups and is_categorical_dtype(values):
        values = values.replace(values.categories.difference(groups), np.nan)
    return values


def _get_color_vec(
    adata: AnnData,
    value_to_plot: str | None,
    values: NDArrayA | pd.Series,
    palette: Optional[Sequence[str] | str | Cycler | None] = None,
    na_color: Optional[str | Tuple[float, ...] | None] = None,
) -> Tuple[NDArrayA, bool]:
    to_hex = partial(colors.to_hex, keep_alpha=True)
    if value_to_plot is None:
        return np.broadcast_to(to_hex(na_color), adata.n_obs), False
    if not is_categorical_dtype(values):
        return values, False
    elif is_categorical_dtype(values):
        # use scanpy _get_palette to set palette if not present
        color_map = {k: to_hex(v) for k, v in _get_palette(adata, value_to_plot, palette)}
        color_vector = values.map(color_map)  # type: ignore

        # Set color to 'missing color' for all missing values
        if color_vector.isna().any():
            color_vector = color_vector.add_categories([to_hex(na_color)])
            color_vector = color_vector.fillna(to_hex(na_color))
        return color_vector, True
    else:
        raise ValueError(f"Wrong type for value passed `{value_to_plot}`.")
