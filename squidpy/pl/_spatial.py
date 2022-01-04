from __future__ import annotations

from math import cos, sin
from cycler import Cycler
from typing import (
    Any,
    Tuple,
    Union,
    Literal,
    Mapping,
    Optional,
    Sequence,
    MutableMapping,
)
from pathlib import Path
from functools import partial
import itertools

from scanpy import logging as logg
from anndata import AnnData
from scanpy._settings import settings as sc_settings
from scanpy.plotting._utils import VBound, _FontSize, _FontWeight, check_colornorm
from scanpy.plotting._tools.scatterplots import (
    _panel_grid,
    _get_palette,
    _get_vboundnorm,
    _add_categorical_legend,
)

from pandas.core.dtypes.common import is_categorical_dtype
import numpy as np
import pandas as pd

from matplotlib import colors, pyplot as pl, rcParams, patheffects
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection

from squidpy._utils import NDArrayA
from squidpy.gr._utils import _assert_spatial_basis
from squidpy.pl._utils import save_fig, _sanitize_anndata, _assert_value_in_obs
from squidpy._constants._pkg_constants import Key


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
    shape: Literal["circle", "square", "hex"] | None = None,
    use_raw: Optional[bool | None] = None,
    layer: Optional[str | None] = None,
    alt_var: Optional[str | None] = None,
    projection: Literal["2d", "3d"] = "2d",
    size: Optional[float | None] = None,
    palette: Optional[Sequence[str] | str | Cycler | None] = None,
    na_color: Optional[str | Tuple[float, ...] | None] = None,
    frameon: Optional[bool] = None,
    title: Union[str, Sequence[str], None] = None,
    wspace: Optional[float | None] = None,
    hspace: float = 0.25,
    ncols: int = 4,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ("black", "white"),
    ax: Optional[Axes | None] = None,
    save: str | Path | None = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = "bold",
    legend_loc: str = "right margin",
    legend_fontoutline: Optional[int] = None,
    na_in_legend: bool = True,
    scatter_kwargs: Optional[MutableMapping[str, Any] | None] = None,
) -> Any:
    """Spatial plotting for squidpy."""
    _sanitize_anndata(adata)
    _assert_spatial_basis(adata, spatial_key)

    # get projection
    args_3d = {"projection": "3d"} if projection == "3d" else {}

    # set scatter_kwargs
    if scatter_kwargs is None:
        scatter_kwargs = {}

    # set shape
    _avail_shapes = ["circle", "square", "hexagon"]
    if shape is not None:
        if shape not in _avail_shapes:
            raise ValueError(f"Shape: `{shape}` not found. Available shapes are: `{_avail_shapes}`")

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
    axs = []
    library_idx = range(len(library_id))

    vmin, vmax, vcenter, norm = _get_seq_vminmax(vmin, vmax, vcenter, norm)

    # set size
    if size is None:
        size = 1

    # make plots
    for count, (value_to_plot, lib_idx) in enumerate(itertools.product(color, library_idx)):
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

        # order points [skip]
        _coords = coords[lib_idx]  # [order, :]

        # set frame
        if grid:
            ax = pl.subplot(grid[count], **args_3d)
            axs.append(ax)
        if not (sc_settings._frameon if frameon is None else frameon):
            ax.axis("off")

        # set title
        if title is None:
            if value_to_plot is not None:
                ax.set_title(value_to_plot)
            else:
                ax.set_title("")
        elif isinstance(title, list):
            try:
                ax.set_title(title[count])
            except IndexError:
                logg.warning(
                    "The title list is shorter than the number of panels. "
                    "Using 'color' value instead for some plots."
                )
                ax.set_title(value_to_plot)
        else:
            raise ValueError(f"Title: {title} is of wrong type: {type(title)}")

        if not categorical:
            vmin_float, vmax_float, vcenter_float, norm_obj = _get_vboundnorm(
                vmin, vmax, vcenter, norm, count, color_vector
            )
            normalize = check_colornorm(
                vmin_float,
                vmax_float,
                vcenter_float,
                norm_obj,
            )
        else:
            normalize = None

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
                **scatter_kwargs,
            )
        else:

            scatter = (
                partial(ax.scatter, s=size, plotnonfinite=True)
                if shape is None
                else partial(shaped_scatter, s=size, ax=ax, shape=shape)
            )

            if add_outline:
                bg_width, gap_width = outline_width
                point = np.sqrt(size)
                gap_size = (point + (point * gap_width) * 2) ** 2
                bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
                # the default black and white colors can be changes using
                # the contour_config parameter
                bg_color, gap_color = outline_color

                scatter_kwargs["edgecolor"] = "none"  # remove edge from kwargs if present
                if "alpha" in scatter_kwargs.keys():
                    alpha = (
                        scatter_kwargs.pop("alpha") if "alpha" in scatter_kwargs else None
                    )  # remove alpha for outline

                ax.scatter(
                    _coords[:, 0],
                    _coords[:, 1],
                    s=bg_size,
                    marker=".",
                    c=bg_color,
                    rasterized=sc_settings._vector_friendly,
                    norm=normalize,
                    **scatter_kwargs,
                )
                ax.scatter(
                    _coords[:, 0],
                    _coords[:, 1],
                    s=gap_size,
                    marker=".",
                    c=gap_color,
                    rasterized=sc_settings._vector_friendly,
                    norm=normalize,
                    **scatter_kwargs,
                )

                # if user did not set alpha, set alpha to 0.7
                scatter_kwargs["alpha"] = 0.7 if alpha is None else alpha

            cax = scatter(
                _coords[:, 0],
                _coords[:, 1],
                marker=".",
                c=color_vector,
                rasterized=sc_settings._vector_friendly,
                norm=normalize,
                **scatter_kwargs,
            )

        ax.set_yticks([])
        ax.set_xticks([])
        if projection == "3d":
            ax.set_zticks([])

        ax.autoscale_view()

        # plot edges and arrows if needed

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
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=_get_palette(adata, value_to_plot),
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

    if save is not None:
        save_fig(fig, path=save)


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


def shaped_scatter(
    x: NDArrayA,
    y: NDArrayA,
    s: NDArrayA,
    shape: Literal["circle", "square", "hex"],
    ax: Axes,
    c: NDArrayA,
) -> Any:
    """
    Get shapes for scatterplot.

    Adapted from here: https://gist.github.com/syrte/592a062c562cd2a98a83 .
    This code is under [The BSD 3-Clause License](http://opensource.org/licenses/BSD-3-Clause)
    """
    zipped = np.broadcast(x, y, s)
    if shape == "circle":
        patches = [Circle((x_, y_), s_) for x_, y_, s_ in zipped]
    else:
        func = np.vectorize(_make_poly)
        n = 4 if shape == "square" else 6
        r: float = s / (2 * sin(np.pi / n))
        patches = [Polygon(np.stack(func(x_, y_, r, n, range(n)), 1), True) for x_, y_, _ in zipped]
    collection = PatchCollection(patches)
    collection.set_facecolor(c)

    ax.add_collection(collection)

    return collection


def _make_poly(x: float, y: float, r: float, n: int, i: int) -> Tuple[float, float]:
    x_i = x + r * sin((np.pi / n) * (1 + 2 * i))
    y_i = y + r * cos((np.pi / n) * (1 + 2 * i))
    return x_i, y_i
