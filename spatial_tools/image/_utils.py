import anndata
import numpy as np
import tifffile

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} took: {te-ts} sec")
        return result

    return wrap


def _num_pages(fname):
    """
    Use tifffile to get the number of pages in the tif
    """
    with tifffile.TiffFile(fname) as img:
        num_pages = len(img.pages)
    return num_pages


def _access_img_in_adata(
    adata: anndata.AnnData,
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


def _write_img_in_adata(adata: anndata.AnnData, img_key: str, img: np.ndarray):
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
