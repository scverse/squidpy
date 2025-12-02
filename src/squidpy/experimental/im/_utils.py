from __future__ import annotations

from typing import Any, Literal

import xarray as xr
from spatialdata._logging import logger


def _get_element_data(
    element_node: Any,
    scale: str | Literal["auto"],
    element_type: str = "element",
    element_key: str = "",
) -> xr.DataArray:
    """
    Extract data array from a spatialdata element (image or label) node.
    Supports multiscale and single-scale elements.

    Parameters
    ----------
    element_node
        The element node from sdata.images[key] or sdata.labels[key]
    scale
        Scale level to use, or "auto" for images (picks coarsest).
    element_type
        Type of element for error messages (e.g., "image", "label").
    element_key
        Key of the element for error messages.

    Returns
    -------
    xr.DataArray of the element data.
    """
    if hasattr(element_node, "keys"):  # multiscale
        available = list(element_node.keys())
        if not available:
            raise ValueError(f"No scales for {element_type} {element_key!r}")

        if scale == "auto":

            def _idx(k: str) -> int:
                num = "".join(ch for ch in k if ch.isdigit())
                return int(num) if num else -1

            chosen = max(available, key=_idx)
        elif scale not in available:
            logger.warning(f"Scale {scale!r} not found. Available: {available}")
            # Try scale0 as fallback, otherwise use first available
            chosen = "scale0" if "scale0" in available else available[0]
            logger.info(f"Using scale {chosen!r}")
        else:
            chosen = scale

        data = element_node[chosen].image
    else:  # single-scale
        data = element_node.image if hasattr(element_node, "image") else element_node

    return data


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
        logger.info(f"Converting {n_channels}-channel image to greyscale using mean across all channels")
        return img.mean(dim="c")

    # Handle explicit RGB specification
    if channel_format == "rgb":
        if n_channels != 3:
            raise ValueError(f"Cannot treat {n_channels}-channel image as RGB (requires exactly 3 channels)")
        logger.info("Converting RGB image to greyscale using luminance formula")
        weights = xr.DataArray([0.299, 0.587, 0.114], dims=["c"], coords={"c": img.coords["c"]})
        return (img * weights).sum(dim="c")

    elif channel_format == "rgba":
        if n_channels != 4:
            raise ValueError(f"Cannot treat {n_channels}-channel image as RGBA (requires exactly 4 channels)")
        logger.info("Converting RGBA image to greyscale using luminance formula (ignoring alpha)")
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
