from __future__ import annotations

from textwrap import dedent
from typing import Any, Callable

from docrep import DocstringProcessor

from squidpy._constants._pkg_constants import Key


def inject_docs(**kwargs: Any) -> Callable[..., Any]:  # noqa: D103
    # taken from scanpy
    def decorator(obj: Any) -> Any:
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj: Any) -> Any:
        obj.__doc__ = dedent(kwargs["__doc__"])
        return obj

    if isinstance(kwargs.get("__doc__", None), str) and len(kwargs) == 1:
        return decorator2

    return decorator


_ConnKey = Key.obsp.spatial_conn()
_adata = """\
adata
    Annotated data object."""
_img_container = """\
img
    High-resolution image."""
_copy = """\
copy
    If ``True``, return the result, otherwise save it to the ``adata`` object."""
_copy_cont = """\
copy
    If ``True``, return the result, otherwise save it to the image container."""
_numba_parallel = """\
numba_parallel
    Whether to use :class:`numba.prange` or not. If `None`, it is determined automatically.
    For small datasets or small number of interactions, it's recommended to set this to `False`."""
_seed = """\
seed
    Random seed for reproducibility."""
_n_perms = """\
n_perms
    Number of permutations for the permutation test."""
_img_layer = """\
layer
    Image layer in ``img`` that should be processed. If `None` and only 1 layer is present, it will be selected."""
_feature_name = """\
feature_name
    Base name of feature in resulting feature values :class:`dict`."""
_feature_ret = """\
    Dictionary of feature values."""
_yx = """\
y
    Coordinate of the crop along the ``height`` dimension in the pixel space.
    If a :class:`float`, it specifies the relative position and must be in `[0, 1]`.
x
    Coordinate of the crop along the ``width`` dimension in the pixel space.
    If a :class:`float`, it specifies the relative position and must be in `[0, 1]`."""
_size = """\
size
    Size of the crop as ``(height, width)``. If a single :class:`int`, the crop will be a square."""
_cluster_key = """\
cluster_key
    Key in :attr:`anndata.AnnData.obs` where clustering is stored."""
_spatial_key = """\
spatial_key
    Key in :attr:`anndata.AnnData.obsm` where spatial coordinates are stored."""
_conn_key = f"""\
connectivity_key
    Key in :attr:`anndata.AnnData.obsp` where spatial connectivities are stored.
    Default is: :attr:`anndata.AnnData.obsp` ``['{_ConnKey}']``."""
_plotting = """\
figsize
    Size of the figure in inches.
dpi
    Dots per inch.
save
    Whether to save the plot."""
_cat_plotting = f"""\
palette
    Categorical colormap for the clusters.
    If `None`, use :attr:`anndata.AnnData.uns` ``['{{cluster_key}}_colors']``, if available.
{_plotting}"""
_heatmap_plotting = f"""\
annotate
    Whether to annotate the cells of the heatmap.
method
    The linkage method to be used for dendrogram/clustering, see :func:`scipy.cluster.hierarchy.linkage`.
title
    The title of the plot.
cmap
    Continuous colormap to use.
cbar_kwargs
    Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
{_cat_plotting}
ax
    Axes, :class:`matplotlib.axes.Axes`."""
_plotting_returns = """\
Nothing, just plots the figure and optionally saves the plot.
"""
_parallelize = """\
n_jobs
    Number of parallel jobs.
backend
    Parallelization backend to use. See :class:`joblib.Parallel` for available options.
show_progress_bar
    Whether to show the progress bar or not."""
_channels = """\
channels
    Channels for this feature is computed. If `None`, use all channels."""
_segment_kwargs = """\
kwargs
    Keyword arguments for the underlying model."""

_ligrec_test_returns = """\
If ``copy = True``, returns a :class:`dict` with following keys:

    - `'means'` - :class:`pandas.DataFrame` containing the mean expression.
    - `'pvalues'` - :class:`pandas.DataFrame` containing the possibly corrected p-values.
    - `'metadata'` - :class:`pandas.DataFrame` containing interaction metadata.

Otherwise, modifies the ``adata`` object with the following key:

    - :attr:`anndata.AnnData.uns` ``['{key_added}']`` - the above mentioned :class:`dict`.

`NaN` p-values mark combinations for which the mean expression of one of the interacting components was 0
or it didn't pass the ``threshold`` percentage of cells being expressed within a given cluster."""
_corr_method = """\
corr_method
    Correction method for multiple testing. See :func:`statsmodels.stats.multitest.multipletests`
    for valid options."""
_custom_fn = """\
Alternatively, any :func:`callable` can be passed as long as it has the following signature:
    :class:`numpy.ndarray` ``(height, width, channels)`` **->** :class:`numpy.ndarray` ``(height, width[, channels])``."""  # noqa: E501
_as_array = """
as_array
    - If `True`, yields a :class:`dict` where keys are layers and values are :class:`numpy.ndarray`.
    - If a :class:`str`, yields one :class:`numpy.ndarray` for the specified layer.
    - If a :class:`typing.Sequence`, yields a :class:`tuple` of :class:`numpy.ndarray` for the specified layers.
    - Otherwise, yields :class:`squidpy.im.ImageContainer`.
"""
_layer_added = """\
layer_added
    Layer of new image layer to add into ``img`` object."""
_chunks_lazy = """\
chunks
    Number of chunks for :mod:`dask`. For automatic chunking, use ``chunks = 'auto'``.
lazy
    Whether to lazily compute the result or not. Only used when ``chunks != None``."""

_ripley_stat_returns = """\
If ``copy = True``, returns a :class:`dict` with following keys:

    - `'{mode}_stat'` - :class:`pandas.DataFrame` containing the statistics of choice for the real observations.
    - `'sims_stat'` - :class:`pandas.DataFrame` containing the statistics of choice for the simulations.
    - `'bins'` - :class:`numpy.ndarray` containing the support.
    - `'pvalues'` - :class:`numpy.ndarray` containing the p-values for the statistics of interest.

Otherwise, modifies the ``adata`` object with the following key:

    - :attr:`anndata.AnnData.uns` ``['{key_added}']`` - the above mentioned :class:`dict`.

Statistics and p-values are computed for each cluster :attr:`anndata.AnnData.obs` ``['{cluster_key}']`` separately."""
_library_id_features = """\
library_id
    Name of the Z-dimension that this function should be applied to."""
_library_id = """\
library_id
    Name of the Z-dimension(s) that this function should be applied to.
    For not specified Z-dimensions, the identity function is applied."""
_img_library_id = """\
library_id
    - If `None`, there should only exist one entry in :attr:`anndata.AnnData.uns` ``['{spatial_key}']``.
    - If a :class:`str`, first search :attr:`anndata.AnnData.obs` ``['{library_id}']`` which contains the mapping
      from observations to library ids, then search :attr:`anndata.AnnData.uns` ``['{spatial_key}']``."""
_library_key = """\
library_key
    Key in :attr:`anndata.AnnData.obs` containing library ids for which to build the spatial graphs separately."""

# static plotting docs
_plotting_kwargs_static = """\
scalebar_kwargs
    Keyword arguments for :meth:`matplotlib_scalebar.ScaleBar`.
edges_kwargs
    Keyword arguments for :func:`networkx.draw_networkx_edges`.
kwargs
    Keyword arguments for :func:`matplotlib.pyplot.scatter` or :func:`matplotlib.pyplot.imshow`.
"""
_plotting_save = f"""\
figsize
    Size of the figure in inches.
dpi
    Dots per inch.
save
    Whether to save the plot.
{_plotting_kwargs_static}"""
_plotting_ax = f"""\
title
    Panel titles.
axis_label
    Panel axis labels.
fig
    Optional :class:`matplotlib.figure.Figure` to use.
ax
    Optional :class:`matplotlib.axes.Axes` to use.
return_ax
    Whether to return :class:`matplotlib.axes.Axes` object(s).
{_plotting_save}"""
_plotting_scalebar = f"""\
scalebar_dx
    Size of one pixel in units specified by ``scalebar_units``.
scalebar_units
    Units of ``scalebar_dx``.
{_plotting_ax}"""
_plotting_legend = f"""\
legend_loc
    Location of the legend, see :class:`matplotlib.legend.Legend`.
legend_fontsize
    Font size of the legend, see :meth:`matplotlib.text.Text.set_fontsize`.
legend_fontweight
    Font weight of the legend, see :meth:`matplotlib.text.Text.set_fontweight`.
legend_fontoutline
    Font outline of the legend, see :class:`matplotlib.patheffects.withStroke`.
legend_na
    Whether to show NA values in the legend.
colorbar
    Whether to show the colorbar, see :func:`matplotlib.pyplot.colorbar`.
{_plotting_scalebar}"""
_plotting_outline = f"""\
outline
    If `True`, a thin border around points/shapes is plotted.
outline_color
    Color of the border.
outline_width
    Width of the border.
{_plotting_legend}"""
_plotting_panels = f"""\
library_first
    If multiple libraries are plotted, set the plotting order with respect to ``color``.
frameon
    If `True`, draw a frame around the panels.
wspace
    Width space between panels.
hspace
    Height space between panels.
ncols
    Number of panels per row.
{_plotting_outline}"""
_plotting_edges = f"""\
connectivity_key
    Key for neighbors graph to plot. Default is: :attr:`anndata.AnnData.obsp` ``['{_ConnKey}']``.
edges_width
    Width of the edges. Only used when ``connectivity_key != None`` .
edges_color
    Color of the edges.
{_plotting_panels}"""
_plotting_sizecoords = f"""\
size
    Size of the scatter point/shape. In case of ``spatial_shape`` it represents to the
    scaling factor for shape (accessed via ``size_key``). In case of ``spatial_point``,
    it represents the ``size`` argument in :func:`matplotlib.pyplot.scatter`.
size_key
    Key of of pixel size of shapes to be plotted, stored in :attr:`anndata.AnnData.uns`.
    Only needed for ``spatial_shape``.
scale_factor
    Scaling factor used to map from coordinate space to pixel space.
    Found by default if ``library_id`` and ``img_key`` can be resolved.
    Otherwise, defaults to `1`.
crop_coord
    Coordinates to use for cropping the image (left, right, top, bottom).
    These coordinates are expected to be in pixel space (same as ``spatial``)
    and will be transformed by ``scale_factor``.
    If not provided, image is automatically cropped to bounds of ``spatial``,
    plus a border.
cmap
    Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`.
palette
    Palette for discrete annotations, see :class:`matplotlib.colors.Colormap`.
alpha
    Alpha value for scatter point/shape.
norm
    Colormap normalization for continuous annotations, see :class:`matplotlib.colors.Normalize`.
na_color
    Color to be used for NAs values, if present.
{_plotting_edges}"""
_plotting_features = f"""\
use_raw
    If True, use :attr:`anndata.AnnData.raw`.
layer
    Key in :attr:`anndata.AnnData.layers` or `None` for :attr:`anndata.AnnData.X`.
alt_var
    Which column to use in :attr:`anndata.AnnData.var` to select alternative ``var_name``.
{_plotting_sizecoords}"""

_cat_plotting = f"""\
palette
    Categorical colormap for the clusters.
    If ``None``, use :attr:`anndata.AnnData.uns` ``['{{cluster_key}}_colors']``, if available.
{_plotting_save}"""

# general static plotting docstrings
_plotting_segment = """\
seg_cell_id
    Column in :attr:`anndata.AnnData.obs` with unique segmentation mask ids. Required to filter
    valid segmentation masks.
seg
    Whether to plot the segmentation mask. One (or more) :class:`numpy.ndarray` can also be
    passed for plotting.
seg_key
    Key of segmentation mask in :attr:`anndata.AnnData.uns`.
seg_contourpx
    Draw contour of specified width for each segment. If `None`, fills
    entire segment, see :func:`skimage.morphology.erosion`.
seg_outline
    Whether to plot boundaries around segmentation masks."""

_plotting_image = """\
img
    Whether to plot the image. One (or more) :class:`numpy.ndarray` can also be
    passed for plotting.
img_res_key
    Key for image resolution, used to get ``img`` and ``scale_factor`` from ``'images'``
    and ``'scalefactors'`` entries for this library.
img_alpha
    Alpha value for the underlying image.
image_cmap
    Colormap for the image, see :class:`matplotlib.colors.Colormap`.
img_channel
    To select which channel to plot (all by default)."""

_shape = """\
shape
    Whether to plot scatter plot of points or regular polygons."""
_color = """\
color
    Key for annotations in :attr:`anndata.AnnData.obs` or variables/genes."""
_groups = """\
groups
    For discrete annotation in ``color``, select which values to plot (other values are set to NAs)."""
_plotting_library_id = """\
library_id
    Select one or some of the unique ``library_id`` that constitute the AnnData to plot."""
_library_key = """\
library_key
    If multiple `library_id`, column in :attr:`anndata.AnnData.obs`
    which stores mapping between ``library_id`` and obs."""

d = DocstringProcessor(
    adata=_adata,
    img_container=_img_container,
    copy=_copy,
    copy_cont=_copy_cont,
    numba_parallel=_numba_parallel,
    seed=_seed,
    n_perms=_n_perms,
    img_layer=_img_layer,
    feature_name=_feature_name,
    yx=_yx,
    feature_ret=_feature_ret,
    size=_size,
    cluster_key=_cluster_key,
    spatial_key=_spatial_key,
    conn_key=_conn_key,
    plotting_save=_plotting_save,
    cat_plotting=_cat_plotting,
    plotting_returns=_plotting_returns,
    parallelize=_parallelize,
    channels=_channels,
    segment_kwargs=_segment_kwargs,
    ligrec_test_returns=_ligrec_test_returns,
    corr_method=_corr_method,
    heatmap_plotting=_heatmap_plotting,
    custom_fn=_custom_fn,
    as_array=_as_array,
    layer_added=_layer_added,
    chunks_lazy=_chunks_lazy,
    ripley_stat_returns=_ripley_stat_returns,
    library_id_features=_library_id_features,
    library_id=_library_id,
    img_library_id=_img_library_id,
    plotting=_plotting,
    plotting_features=_plotting_features,
    plotting_segment=_plotting_segment,
    plotting_image=_plotting_image,
    shape=_shape,
    color=_color,
    groups=_groups,
    plotting_library_id=_plotting_library_id,
    library_key=_library_key,
)
