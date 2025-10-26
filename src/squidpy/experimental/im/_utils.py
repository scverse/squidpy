from __future__ import annotations

from typing import Literal

import spatialdata as sd
import xarray as xr
from spatialdata._logging import logger as logg


def _get_image_data(
    sdata: sd.SpatialData,
    image_key: str,
    scale: str,
) -> xr.DataArray:
    """
    Extract image data from SpatialData object, handling both datatree and direct DataArray images.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    image_key : str
        Key in sdata.images
    scale : str
        Multiscale level, e.g. "scale0", or "auto" for the smallest available scale

    Returns
    -------
    xr.DataArray
        Image data in (c, y, x) format
    """
    img_node = sdata.images[image_key]

    # Check if the image is a datatree (has multiple scales) or a direct DataArray
    if hasattr(img_node, "keys"):
        available_scales = list(img_node.keys())

        if scale == "auto":
            scale = available_scales[-1]
        elif scale not in available_scales:
            print(scale)
            print(available_scales)
            scale = available_scales[-1]
            logg.warning(f"Scale '{scale}' not found, using available scale. Available scales: {available_scales}")

        img_da = img_node[scale].image
    else:
        # It's a direct DataArray (no scales)
        img_da = img_node.image if hasattr(img_node, "image") else img_node

    return _ensure_cyx(img_da)


def _ensure_cyx(img_da: xr.DataArray) -> xr.DataArray:
    """Ensure dims are (c, y, x). Adds a length-1 "c" if missing."""
    dims = list(img_da.dims)
    if "y" not in dims or "x" not in dims:
        raise ValueError(f'Expected dims to include "y" and "x". Found dims={dims}')

    # Handle case where dims are (c, y, x) - keep as is
    if "c" in dims:
        return img_da if dims[0] == "c" else img_da.transpose("c", "y", "x")
    # If no "c" dimension, add one
    return img_da.expand_dims({"c": [0]}).transpose("c", "y", "x")


def _flatten_channels(
    img: xr.DataArray,
    channel_format: Literal["infer", "rgb", "rgba", "multichannel"] = "infer",
) -> xr.DataArray:
    """
    Takes an image of shape (c, y, x) and returns a 2D image of shape (y, x).

    Conversion logic:
    - 1 channel: Returns greyscale (removes channel dimension)
    - 3 channels + "rgb"/"infer": Uses RGB luminance formula
    - 4 channels + "rgba": Uses RGB luminance formula (ignores alpha)
    - 2 channels or 4+ channels + "infer": Automatically treated as multichannel
    - "multichannel": Always uses mean across all channels

    The function is silent unless the channel_format is not "infer".

    Parameters
    ----------
    img : xr.DataArray
        Input image with shape (c, y, x)
    channel_format : Literal["infer", "rgb", "rgba", "multichannel"]
        How to interpret the channels:
        - "infer": Automatically infer format based on number of channels
        - "rgb": Force RGB treatment (requires exactly 3 channels)
        - "rgba": Force RGBA treatment (requires exactly 4 channels)
        - "multichannel": Force multichannel treatment (mean across all channels)

    Returns
    -------
    xr.DataArray
        Greyscale image with shape (y, x)
    """
    n_channels = img.sizes["c"]

    # 1 channel: always return greyscale
    if n_channels == 1:
        return img.squeeze("c")

    # If user explicitly specifies multichannel, always use mean
    if channel_format == "multichannel":
        logg.info(f"Converting {n_channels}-channel image to greyscale using mean across all channels")
        return img.mean(dim="c")

    # Handle explicit RGB specification
    if channel_format == "rgb":
        if n_channels != 3:
            raise ValueError(f"Cannot treat {n_channels}-channel image as RGB (requires exactly 3 channels)")
        logg.info("Converting RGB image to greyscale using luminance formula")
        weights = xr.DataArray([0.299, 0.587, 0.114], dims=["c"], coords={"c": img.coords["c"]})
        return (img * weights).sum(dim="c")

    elif channel_format == "rgba":
        if n_channels != 4:
            raise ValueError(f"Cannot treat {n_channels}-channel image as RGBA (requires exactly 4 channels)")
        logg.info("Converting RGBA image to greyscale using luminance formula (ignoring alpha)")
        weights = xr.DataArray([0.299, 0.587, 0.114, 0.0], dims=["c"], coords={"c": img.coords["c"]})
        return (img * weights).sum(dim="c")

    elif channel_format == "infer":
        if n_channels == 3:
            # 3 channels + infer -> RGB luminance formula
            weights = xr.DataArray([0.299, 0.587, 0.114], dims=["c"], coords={"c": img.coords["c"]})
            return (img * weights).sum(dim="c")
        else:
            # 2 channels or 4+ channels + infer -> multichannel
            return img.mean(dim="c")

    else:
        raise ValueError(
            f"Invalid channel_format: {channel_format}. Must be one of 'infer', 'rgb', 'rgba', 'multichannel'."
        )
