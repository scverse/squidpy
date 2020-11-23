from time import time
from typing import Any, Union, Callable
from functools import wraps

from anndata import AnnData

import numpy as np

import tifffile


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


def _round_odd(num: Union[float, int]) -> Union[float, int]:
    """
    Round num to next odd integer value.

    Params
    ------
    num: float
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
    else:
        return res - 1


def _round_even(num: Union[float, int]) -> Union[float, int]:
    """
    Round num to next even integer value.

    Params
    ------
    num: float
        Number to round

    Returns
    -------
    int
        rounded number to nearest even integer value
    """
    res = round(num / 2.0)
    return res * 2


def _access_img_in_adata(
    adata: AnnData,
    img_key: str,
) -> np.ndarray:
    """
    Return image from anndata instance.

    Attrs:
        adata: Instance to get image from.
        img_key:
    Returns:
        Image (x, y, channels).
    """
    return adata.uns["spatial"][img_key]


def _write_img_in_adata(adata: AnnData, img_key: str, img: np.ndarray):
    """
    Save image in anndata instance.

    Attrs:
        adata: Instance to get image from.
        img_key: Name of image in adata object.
        img:
    Returns:
        Image (x, y, channels).
    """
    adata.uns["spatial"][img_key] = img
