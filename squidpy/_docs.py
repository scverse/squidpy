from textwrap import dedent

from docrep import DocstringProcessor


def inject_docs(**kwargs):  # noqa: D103
    # taken from scanpy
    def decorator(obj):
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj):
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
    If `True`, return the result, otherwise modify the image container."""
_numba_parallel = """\
numba_parallel
    Whether to use :class:`numba.prange` or not. If `None`, it's determined automatically.
    For small datasets or small number of interactions, it's recommended to set this to `False`.
"""
_seed = """\
seed
    Random seed for reproducibility.
"""
_img_hr = """\
img
    High-resolution image.
"""
_img_uint8 = """\
img
    RGB image of type :class:`numpy.uint8` of shape ``(height, width [, channels]).``.
"""
_feature_name = """\
feature_name
    Base name of feature in resulting feature values dict.
"""
_feature_ret = """\
:class:`dict`
    Dictionary of feature values.
"""
_width_height = """\
xs
    Width of the crops in pixels.
ys
    Height of the crops in pixels.
"""
_cluster_key = """\
cluster_key
    Key in :attr:`anndata.AnnData.obs` where clustering is stored.
"""
# TODO: https://github.com/Chilipp/docrep/issues/21 fixed, this is not necessary
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
None
    Nothing, just plots the and optionally saves the plot.
"""


d = DocstringProcessor(
    adata=_adata,
    img_container=_img_container,
    copy=_copy,
    copy_cont=_copy_cont,
    numba_parallel=_numba_parallel,
    seed=_seed,
    img_hr=_img_hr,
    img_uint8=_img_uint8,
    feature_name=_feature_name,
    feature_ret=_feature_ret,
    width_height=_width_height,
    cluster_key=_cluster_key,
    crop_extra=_crop_extra,
    plotting=_plotting,
    plotting_returns=_plotting_returns,
)
