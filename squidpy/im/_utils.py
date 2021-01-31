from typing import Any
from dataclasses import dataclass
import warnings

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


def _open_rasterio(path: str, **kwargs: Any) -> xr.DataArray:
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        return xr.open_rasterio(path, **kwargs)


@dataclass(frozen=True)
class CropCoords:
    """Top-left and bottom right-corners of a crop."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x0 > self.x1:
            raise ValueError(f"Expected `x0` <= `x1`, found `{self.x0}` > `{self.x1}`.")
        if self.y0 > self.y1:
            raise ValueError(f"Expected `y0` <= `y1`, found `{self.y0}` > `{self.y1}`.")

    @property
    def dx(self) -> float:
        """Height."""
        return self.x1 - self.x0

    @property
    def dy(self) -> float:
        """Width."""
        return self.x1 - self.x0

    @property
    def center_x(self) -> float:
        """Center of height."""
        return self.x0 + self.dx / 2.0

    @property
    def center_y(self) -> float:
        """Width of height."""
        return self.x0 + self.dy / 2.0
