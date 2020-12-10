from time import time
from typing import Any, Set, List, Tuple, Union, Callable, Hashable, Iterable
from functools import wraps

from anndata import AnnData

import numpy as np
import xarray as xr

import tifffile

from squidpy.constants._pkg_constants import Key


# TODO: dead code
def timing(f: Callable[..., Any]) -> Callable[..., Any]:
    """Time a function ``f``."""

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} took: {te-ts} sec")
        return result

    return wrap


def _num_pages(fname: str) -> int:
    """Use tifffile to get the number of pages in the tif."""
    with tifffile.TiffFile(fname) as img:
        num_pages = len(img.pages)
    return num_pages


# helper function for scaling one xarray
def _scale_xarray(arr: xr.DataArray, scale: float) -> xr.DataArray:
    """Scale xarray in x and y dims using skimage.transform.rescale."""
    from skimage.transform import rescale

    dtype = arr.dtype
    dims = arr.dims

    # rescale only in x and y dim
    scales = np.ones(len(dims))
    scales[np.in1d(dims, ["y", "x"])] = scale

    arr = rescale(arr, scales, preserve_range=True, order=1)
    arr = arr.astype(dtype)
    # recreate DataArray
    arr = xr.DataArray(arr, dims=dims)
    return arr


# TODO: dead code
def _round_odd(num: Union[float, int]) -> Union[float, int]:
    """
    Round num to next odd integer value.

    Parameters
    ----------
    num
        Number to round

    Returns
    -------
    int
        rounded number to nearest odd integer value
    """
    res = round(num)
    if res % 2 == 1:
        return res
    if abs(res + 1 - num) < abs(res - 1 - num):
        return res + 1
    return res - 1


# TODO: dead code
def _round_even(num: Union[float, int]) -> Union[float, int]:
    """
    Round num to next even integer value.

    Parameters
    ----------
    num
        Number to round.

    Returns
    -------
    int
        Rounded number to nearest even integer value.
    """
    return 2 * round(num / 2.0)


# TODO: dead code
def _access_img_in_adata(
    adata: AnnData,
    img_key: str,
) -> np.ndarray:
    """
    Return im from anndata instance.

    Attrs:
        adata: Instance to get im from.
        img_key:
    Returns:
        Image (x, y, channels).
    """
    return adata.uns["spatial"][img_key]


# TODO: dead code
def _write_img_in_adata(adata: AnnData, img_key: str, img: np.ndarray):
    """
    Save im in anndata instance.

    Attrs:
        adata: Instance to get im from.
        img_key: Name of im in adata object.
        img:
    Returns:
        Image (x, y, channels).
    """
    adata.uns[Key.uns.spatial][img_key] = img


def _unique_order_preserving(iterable: Iterable[Hashable]) -> Tuple[List[Hashable], Set[Hashable]]:
    """Remove items from an iterable while preserving the order."""
    seen = set()
    return [i for i in iterable if i not in seen and not seen.add(i)], seen
