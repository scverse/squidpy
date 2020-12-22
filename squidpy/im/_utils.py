from typing import Set, List, Tuple, Hashable, Iterable

import numpy as np
import xarray as xr

import tifffile


def _num_pages(fname: str) -> int:
    """Use tifffile to get the number of pages in the tif."""
    with tifffile.TiffFile(fname) as img:
        num_pages = len(img.pages)
    return num_pages


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


def _unique_order_preserving(iterable: Iterable[Hashable]) -> Tuple[List[Hashable], Set[Hashable]]:
    """Remove items from an iterable while preserving the order."""
    seen = set()
    return [i for i in iterable if i not in seen and not seen.add(i)], seen
