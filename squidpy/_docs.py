from __future__ import annotations

from docrep import DocstringProcessor
from typing import Any, Callable
from textwrap import dedent


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


_adata = """\
adata
    Annotated data object."""
_img_container = """\
img
    High-resolution image."""
_copy = """\
copy
    If `True`, return the result, otherwise save it to the ``adata`` object."""
_copy_cont = """\
copy
    If `True`, return the result, otherwise save it to the image container."""
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
_conn_key = """\
connectivity_key
    Key in :attr:`anndata.AnnData.obsp` where spatial connectivities are stored."""
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
Nothing, just plots the and optionally saves the plot.
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
    Whether to lazily compute the result or not. Only used when ``chunks != None```."""

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
# plotting docs

_plotting_save = """\
figsize
    Size of the figure in inches.
dpi
    Dots per inch.
save
    Whether to save the plot."""
_plotting_ax = f"""\
title
    Panel titles.
axis_label
    Panel axis labels.
fig
    Optional :class:`matplotlib.figure.Figure` object.
ax
    Optional :class:`matplotlib.axes.Axes` object.
{_plotting_save}"""
_plotting_scalebar = f"""\
scalebar_dx
    Size of one pixel in units specified by `scalebar_units`.
scalebar_units
    Units of `scalebar_dx`.
{_plotting_ax}"""
_plotting_legend = f"""\
legend_loc
    Location of legend, see :class:`matplotlib.legend.Legend`.
legend_fontsize
    Font size of legend, see:meth:`matplotlib.text.Text.set_fontsize`.
legend_fontweight
    Font weight of legend, see :meth:`matplotlib.text.Text.set_fontweight`.
legend_fontoutline
    Font outline of legend, see :class:`matplotlib.patheffects.withStroke`.
legend_na
    If there are missing values, whether they get an entry in the legend.
{_plotting_scalebar}"""
_plotting_outline = f"""\
outline
    If set to True, a thin border around points/shapes is plotted.
outline_color
    Color of the border.
outline_width
    Width of the border.
{_plotting_legend}"""
_plotting_panels = f"""\
library_first
    If multiple libraries are plotted, set the plotting order with respect to `color`.
frameon
    If True, draw a frame around the panel.
wspace
    Width space between panels.
hspace
    Height space between panels.
ncols
    Number of panels per row.
{_plotting_outline}"""
_plotting_edges = f"""\
edges
    If True, draw edges based on graph.
edges_width
    Width of edges.
edges_color
    Color of edges.
connectivity_key
    Key for neighbors graph to plot.
{_plotting_panels}"""
_plotting_sizecoords = f"""\
size
    Size of the scatter point/shape. In case of `spatial_shape` it represents to the
    scaling factor for shape (accessed with `size_key`). In case of `spatial_point`,
    it represents the `size` argument in :func:`matplotlib.pyplot.scatter`.
size_key
    Key of of pixel size of shapes to be plotted, stored in :attr:`anndata.AnnData.uns`.
    Only needed for `spatial_shape`.
scale_factor
    Scaling factor used to map from coordinate space to pixel space.
    Found by default if `library_id` and `img_key` can be resolved.
    Otherwise defaults to `1.`.
crop_coord
    Coordinates to use for cropping the image (left, right, top, bottom).
    These coordinates are expected to be in pixel space (same as `spatial`)
    and will be transformed by `scale_factor`.
    If not provided, image is automatically cropped to bounds of `spatial`,
    plus a border.
cmap
    Colormap for continuous annotations. See :class:`matplotlib.colors.Colormap`.
palette
    Palette for discrete annotations. See :class:`matplotlib.colors.Colormap`.
alpha
    Alpha value for scatter point/shape.
norm
    Colormap normalization for continuous annotations. See :class:`matplotlib.colors.Normalize`.
na_color
    Color to be used for NAs values, if present.
{_plotting_edges}"""
_plotting_features = f"""\
use_raw
    If True, use :attr:`anndata.AnnData.raw`.
layer
    Which layer to use for features.
alt_var
    Which column to use in :attr:`anndata.AnnData.var` to select alternative `var_name`.
{_plotting_sizecoords}"""

_plotting_segment = """\
seg
    Whether to plot the segmentation mask. One (or more) :class:`numpy.ndarray` can also be
    passed for plotting.
seg_key
    Key of segmentation mask in :attr:`anndata.AnnData.uns`.
cell_id_key
    Column in :attr:`anndata.AnnData.obs` with unique segmentation mask ids. Required to filter
    valid segmentation masks.
seg_contourpx
    Draw contour of specified width for each segment. If `None`, fills
    entire segment. See :func:`skimage.morphology.erosion`.
seg_outline
    Whether to plot boundaries around segmentation masks."""

_plotting_image = """\
img
    Whether to plot the image. One (or more) :class:`numpy.ndarray` can also be
    passed for plotting.
img_res_key
    Key for image resolution, used to get `img` and `scale_factor` from `"images"`
    and `"scalefactors"` entries for this library.
img_alpha
    Alpha value for the underlying image.
image_cmap
    Colormap for the image. See :class:`matplotlib.colors.Colormap`.
img_channel
    To select which channel to plot (all by default)."""

_shape = """\
shape
    Whether to plot scatter plot of points or regular polygons."""
_color = """\
color
    Which features to plot from :class:`anndata.AnnData`."""
_groups = """\
groups
    For discrete annotation in `color`, select which values to plot (other values are set to NAs)."""
_plotting_library_id = """\
library_id
    Select one or some of the unique `library_id` that constitute the AnnData to plot."""
_library_key = """\
library_key
    If multiple `library_id`, column in :attr:`anndata.AnnData.obs` which stores mapping between `library_id` and obs"""

_cat_plotting = f"""\
palette
    Categorical colormap for the clusters.
    If `None`, use :attr:`anndata.AnnData.uns` ``['{{cluster_key}}_colors']``, if available.
{_plotting_save}"""

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
{_cat_plotting}"""

_plotting_returns = """\
Nothing, just plots and optionally saves the plot.
"""

_plotting_general_summary = """\
As this function is designed to for imaging data, there are two key assumptions
about how coordinates are handled:

1. The origin (e.g `(0, 0)`) is at the top left - as is common convention
with image data.

2. Coordinates are in the pixel space of the source image, so an equal
aspect ratio is assumed.

If your anndata object has a `"spatial"` entry in `.uns`, the `img_key`, `seg_key`
and `library_id` parameters to find values for `img`, `seg`, `scale_factor`,
and `spot_size` arguments. Alternatively, these values be passed directly.
"""

_plotting_point_summary = """\
The plotted points (dots) do not have a real "size" but only relative to their
coordinate space.
"""

_plotting_shape_summary = """\
The plotted shapes (circles, squares or hexagons) have a real "size" with respect to their
coordinate space, which can be specified via the `size` or `size_key` parameter.

This function allows overlaying data on top of images.

Use the parameter `img_key` to see the image in the background
and the parameter `library_id` to select the image.
By default, `'hires'` key is attempted.
Use `img_alpha`, `img_cmap` or `img_channel` to control how it is displayed.
Use `size` to scale the size of the shapes plotted on top.
"""

_plotting_segment_summary = """\
This function allows overlaying segmentation masks on top of images.

Use the parameter `seg_key` to see the image in the background
and the parameter `library_id` to select the image.

By default, `'segmentation'` `seg_key` is attempted and
`'hires'` image key is attempted.
Use `img_alpha`, `img_cmap` or `img_channel` to control how the image is displayed.
Use `seg_contourpx` or `seg_outline` to control how the segmentation mask is displayed.
"""

_plotting_general_summary = """\
Use the parameter `library_id` to select the image.
If multiple `library_id` are available, use `library_key` to plot subsets of
the :class:`anndata.AnnData`.
Use `crop_coord` to crop the spatial plot based on coordinate boundaries.

As this function is designed to for imaging data, there are two key assumptions
about how coordinates are handled:

1. The origin (e.g `(0, 0)`) is at the top left - as is common convention
with image data.

2. Coordinates are in the pixel space of the source image, so an equal
aspect ratio is assumed.

If your anndata object has a `"spatial"` entry in `.uns`, the `img_key`, `seg_key`
and `library_id` parameters to find values for `img`, `seg` and `scale_factor`.
Alternatively, these values can be passed directly.
"""

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
    plotting_features=_plotting_features,
    plotting_segment=_plotting_segment,
    plotting_image=_plotting_image,
    shape=_shape,
    color=_color,
    groups=_groups,
    plotting_library_id=_plotting_library_id,
    library_key=_library_key,
    plotting_general_summary=_plotting_general_summary,
    plotting_point_summary=_plotting_point_summary,
    plotting_shape_summary=_plotting_shape_summary,
    plotting_segment_summary=_plotting_segment_summary,
)
