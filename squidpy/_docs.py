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
    Whether to use :class:`numba.prange` or not. If `None`, it's determined automatically.
    For small datasets or small number of interactions, it's recommended to set this to `False`."""
_seed = """\
seed
    Random seed for reproducibility."""
_n_perms = """
n_perms
    Number of permutations for the permutation test."""
_img_hr = """\
img
    High-resolution image."""
_img_id = """\
img_id
    Key of image layer in ``img`` that should be processed."""
_feature_name = """\
feature_name
    Base name of feature in resulting feature values dict."""
_feature_ret = """\
    Dictionary of feature values."""
_xy_coord = """\
x
    X coord of crop (in pixel space).
y
    Y coord of crop (in pixel space)."""
_width_height = """\
xs
    Width of the crops in pixels.
ys
    Height of the crops in pixels."""
_cluster_key = """\
cluster_key
    Key in :attr:`anndata.AnnData.obs` where clustering is stored."""
_spatial_key = """\
spatial_key
    Key in :attr:`anndata.AnnData.obsm` where spatial coordinates are stored."""
_conn_key = """\
conn_key
    Key in :attr:`anndata.AnnData.obsp` where spatial connectivities are stored."""
# TODO: https://github.com/Chilipp/docrep/issues/21 fixes this, this is not necessary
_crop_extra = """\
scale
    Resolution of the crop (smaller -> smaller image).
mask_circle
    Mask crop to a circle.
cval
    The value outside image boundaries or the mask.
dtype
    Type to which the output should be (safely) cast. If `None`, don't recast.
    Currently supported dtypes: 'uint8'. TODO: pass actualy types instead of strings."""
_plotting = """\
figsize
    Size of the figure in inches.
dpi
    Dots per inch.
save
    Whether to save the plot."""
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

_ligrec_test_returns = """
If ``copy = True``, returns a :class:`typing.NamedTuple`:

    - `'means'` - :class:`pandas.DataFrame` containing the mean expression.
    - `'pvalues'` - :class:`pandas.DataFrame` containing the possibly corrected p-values.
    - `'metadata'` - :class:`pandas.DataFrame` containing interaction metadata.

Otherwise, it modifies the ``adata`` object with the following key:

    - :attr:`anndata.AnnData.uns` ``['{key_added}']`` - the above mentioned triple.

`NaN` p-values mark combinations for which the mean expression of one of the interacting components was 0
or it didn't pass the ``threshold`` percentage of cells being expressed within a given cluster."""


d = DocstringProcessor(
    adata=_adata,
    img_container=_img_container,
    copy=_copy,
    copy_cont=_copy_cont,
    numba_parallel=_numba_parallel,
    seed=_seed,
    n_perms=_n_perms,
    img_hr=_img_hr,
    img_id=_img_id,
    feature_name=_feature_name,
    xy_coord=_xy_coord,
    feature_ret=_feature_ret,
    width_height=_width_height,
    cluster_key=_cluster_key,
    spatial_key=_spatial_key,
    conn_key=_conn_key,
    crop_extra=_crop_extra,
    plotting=_plotting,
    plotting_returns=_plotting_returns,
    parallelize=_parallelize,
    channels=_channels,
    segment_kwargs=_segment_kwargs,
    ligrec_test_returns=_ligrec_test_returns,
)
