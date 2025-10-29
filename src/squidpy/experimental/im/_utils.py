from __future__ import annotations

from typing import Any, Literal

import dask.array as da
import numpy as np
import spatialdata as sd
import xarray as xr
from shapely.geometry import Polygon
from spatialdata._logging import logger

from squidpy._utils import _ensure_dim_order


class TileGrid:
    def __init__(
        self,
        H: int,
        W: int,
        tile_size: Literal["auto"] | tuple[int, int] = "auto",
        target_tiles: int = 100,
    ):
        self.H = int(H)
        self.W = int(W)
        if tile_size == "auto":
            size = max(min(self.H // target_tiles, self.W // target_tiles), 100)
            self.ty = int(size)
            self.tx = int(size)
        else:
            self.ty = int(tile_size[0])
            self.tx = int(tile_size[1])
        self.tiles_y = (self.H + self.ty - 1) // self.ty
        self.tiles_x = (self.W + self.tx - 1) // self.tx

    def indices(self) -> np.ndarray:
        return np.array([[iy, ix] for iy in range(self.tiles_y) for ix in range(self.tiles_x)], dtype=int)

    def names(self) -> list[str]:
        return [f"tile_x{ix}_y{iy}" for iy in range(self.tiles_y) for ix in range(self.tiles_x)]

    def bounds(self) -> np.ndarray:
        b: list[list[int]] = []
        for iy in range(self.tiles_y):
            for ix in range(self.tiles_x):
                y0, x0 = iy * self.ty, ix * self.tx
                y1 = (iy + 1) * self.ty if iy < self.tiles_y - 1 else self.H
                x1 = (ix + 1) * self.tx if ix < self.tiles_x - 1 else self.W
                b.append([y0, x0, y1, x1])
        return np.array(b, dtype=int)

    def centroids_and_polygons(self) -> tuple[np.ndarray, list[Polygon]]:
        cents: list[list[float]] = []
        polys: list[Polygon] = []
        for y0, x0, y1, x1 in self.bounds():
            cy = (y0 + y1) / 2
            cx = (x0 + x1) / 2
            cents.append([cy, cx])
            polys.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]))
        return np.array(cents, dtype=float), polys

    def rechunk_and_pad(self, arr_yx: da.Array) -> da.Array:
        if arr_yx.ndim != 2:
            raise ValueError("Expected a 2D array shaped (y, x).")
        pad_y = self.tiles_y * self.ty - int(arr_yx.shape[0])
        pad_x = self.tiles_x * self.tx - int(arr_yx.shape[1])
        a = arr_yx.rechunk((self.ty, self.tx))
        return da.pad(a, ((0, pad_y), (0, pad_x)), mode="edge") if (pad_y > 0 or pad_x > 0) else a

    def coarsen(self, arr_yx: da.Array, reduce: Literal["mean", "sum"] = "mean") -> da.Array:
        reducer = np.mean if reduce == "mean" else np.sum
        return da.coarsen(reducer, arr_yx, {0: self.ty, 1: self.tx}, trim_excess=False)


def _make_tile_grid(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    image_mask_key: str | None = None,
    tile_size: tuple[int, int] = (224, 224),
    center_grid_on_tissue: bool = False,
) -> TileGrid | None:
    pass


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
