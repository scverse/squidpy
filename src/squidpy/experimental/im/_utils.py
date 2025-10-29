from __future__ import annotations

from typing import Any, Literal

import dask.array as da
import geopandas as gpd
import numpy as np
import spatialdata as sd
import xarray as xr
from shapely.geometry import Polygon
from spatialdata._logging import logger
from spatialdata.models import ShapesModel

from squidpy._utils import _ensure_dim_order, _yx_from_shape


class TileGrid:
    def __init__(
        self,
        H: int,
        W: int,
        tile_size: Literal["auto"] | tuple[int, int] = "auto",
        target_tiles: int = 100,
        offset_y: int = 0,
        offset_x: int = 0,
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
        self.offset_y = int(offset_y)
        self.offset_x = int(offset_x)
        # Calculate number of tiles needed to cover entire image, accounting for offset
        # The grid starts at offset_y, offset_x (can be negative)
        # We need tiles from min(0, offset_y) to at least H
        # So total coverage needed is from min(0, offset_y) to H
        grid_start_y = min(0, self.offset_y)
        grid_start_x = min(0, self.offset_x)
        total_h_needed = self.H - grid_start_y
        total_w_needed = self.W - grid_start_x
        self.tiles_y = (total_h_needed + self.ty - 1) // self.ty
        self.tiles_x = (total_w_needed + self.tx - 1) // self.tx

    def indices(self) -> np.ndarray:
        return np.array([[iy, ix] for iy in range(self.tiles_y) for ix in range(self.tiles_x)], dtype=int)

    def names(self) -> list[str]:
        return [f"tile_x{ix}_y{iy}" for iy in range(self.tiles_y) for ix in range(self.tiles_x)]

    def bounds(self) -> np.ndarray:
        b: list[list[int]] = []
        for iy in range(self.tiles_y):
            for ix in range(self.tiles_x):
                y0 = iy * self.ty + self.offset_y
                x0 = ix * self.tx + self.offset_x
                y1 = ((iy + 1) * self.ty + self.offset_y) if iy < self.tiles_y - 1 else self.H
                x1 = ((ix + 1) * self.tx + self.offset_x) if ix < self.tiles_x - 1 else self.W
                # Clamp bounds to image dimensions
                y0 = max(0, min(y0, self.H))
                x0 = max(0, min(x0, self.W))
                y1 = max(0, min(y1, self.H))
                x1 = max(0, min(x1, self.W))
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


def _get_largest_scale_dimensions(
    sdata: sd.SpatialData,
    image_key: str,
) -> tuple[int, int]:
    """
    Get the dimensions (H, W) of the largest/finest scale of an image.

    Parameters
    ----------
    sdata
        SpatialData object containing the image.
    image_key
        Key of the image in ``sdata.images``.

    Returns
    -------
    Tuple of (height, width) in pixels.
    """
    img_node = sdata.images[image_key]

    # Use _get_element_data with "scale0" which is always the largest scale
    # It handles both datatree (multiscale) and single-scale images
    img_da = _get_element_data(img_node, "scale0", "image", image_key)

    # Get spatial dimensions (y, x)
    if "y" in img_da.sizes and "x" in img_da.sizes:
        return int(img_da.sizes["y"]), int(img_da.sizes["x"])
    else:
        # Fallback: assume last two dimensions are spatial
        return int(img_da.shape[-2]), int(img_da.shape[-1])


def _save_tile_grid_to_shapes(
    sdata: sd.SpatialData,
    tg: TileGrid,
    image_key: str,
    shapes_key: str,
) -> None:
    """Save a TileGrid to sdata.shapes as a GeoDataFrame."""
    tile_indices = tg.indices()
    pixel_bounds = tg.bounds()
    _, polys = tg.centroids_and_polygons()

    tile_gdf = gpd.GeoDataFrame(
        {
            "tile_id": tg.names(),
            "tile_y": tile_indices[:, 0],
            "tile_x": tile_indices[:, 1],
            "pixel_y0": pixel_bounds[:, 0],
            "pixel_x0": pixel_bounds[:, 1],
            "pixel_y1": pixel_bounds[:, 2],
            "pixel_x1": pixel_bounds[:, 3],
            "geometry": polys,
        },
        geometry="geometry",
    )

    sdata.shapes[shapes_key] = ShapesModel.parse(tile_gdf)
    logger.info(f"- Saved tile grid as 'sdata.shapes[\"{shapes_key}\"]'")


def _make_tile_grid(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    image_mask_key: str | None = None,
    tile_size: tuple[int, int] = (224, 224),
    center_grid_on_tissue: bool = False,
    scale: str = "auto",
    new_shapes_key: str | None = None,
    inplace: bool = True,
) -> TileGrid | None:
    """
    Create a TileGrid for a spatialdata image.

    Parameters
    ----------
    sdata
        SpatialData object containing the image.
    image_key
        Key of the image in ``sdata.images``.
    image_mask_key
        Optional key of the segmentation mask in ``sdata.labels``.
        Required if ``center_grid_on_tissue`` is ``True``.
    tile_size
        Size of tiles as (height, width) in pixels.
    center_grid_on_tissue
        If ``True`` and ``image_mask_key`` is provided, center the grid
        on the centroid of the mask.
    scale
        Scale level to use for mask processing. The tile grid is always generated
        based on the largest (finest) scale of the image.
    new_shapes_key
        Key to save the tile grid in ``sdata.shapes`` if ``inplace=True``.
        If ``None``, defaults to ``f"{image_key}_tile_grid"``.
    inplace
        If ``True``, save the tile grid to ``sdata.shapes[new_shapes_key]``.
        If ``False``, return the TileGrid object without saving.

    Returns
    -------
    If ``inplace=True``, returns ``None``. Otherwise, returns a TileGrid object.

    Raises
    ------
    KeyError
        If ``image_key`` is not in ``sdata.images``.
    KeyError
        If ``center_grid_on_tissue`` is ``True`` but ``image_mask_key``
        is not provided or not found in ``sdata.labels``.
    """
    # Validate image key
    if image_key not in sdata.images:
        raise KeyError(f"Image key '{image_key}' not found in sdata.images")

    # Get image dimensions from the largest/finest scale
    H, W = _get_largest_scale_dimensions(sdata, image_key)

    ty, tx = tile_size

    # Path 1: Regular grid starting from top-left
    if not center_grid_on_tissue or image_mask_key is None:
        tg = TileGrid(H, W, tile_size=tile_size)
        if inplace:
            shapes_key = new_shapes_key or f"{image_key}_tile_grid"
            _save_tile_grid_to_shapes(sdata, tg, image_key, shapes_key)
            return None
        return tg

    # Path 2: Center grid on tissue mask centroid
    if image_mask_key not in sdata.labels:
        raise KeyError(
            f"Mask key '{image_mask_key}' not found in sdata.labels. Available keys: {list(sdata.labels.keys())}"
        )

    # Get mask and compute centroid
    label_node = sdata.labels[image_mask_key]
    mask_da = _get_element_data(label_node, scale, "label", image_mask_key)

    # Convert to numpy array if needed
    if hasattr(mask_da, "compute"):
        mask = np.asarray(mask_da.compute())
    elif hasattr(mask_da, "values"):
        mask = np.asarray(mask_da.values)
    else:
        mask = np.asarray(mask_da)

    # Ensure 2D (y, x) shape
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask with shape (y, x), got shape {mask.shape}")

    # Ensure mask matches image dimensions
    H_mask, W_mask = mask.shape

    # Compute centroid of mask (where mask > 0)
    mask_bool = mask > 0
    if not mask_bool.any():
        logger.warning("Mask is empty. Using regular grid starting from top-left.")
        tg = TileGrid(H, W, tile_size=tile_size)
        if inplace:
            shapes_key = new_shapes_key or f"{image_key}_tile_grid"
            _save_tile_grid_to_shapes(sdata, tg, image_key, shapes_key)
            return None
        return tg

    # Calculate centroid using center of mass
    y_coords, x_coords = np.where(mask_bool)
    centroid_y_mask = float(np.mean(y_coords))
    centroid_x_mask = float(np.mean(x_coords))

    # Scale centroid coordinates to match image dimensions if mask is at different scale
    if H_mask != H or W_mask != W:
        scale_y = H / H_mask
        scale_x = W / W_mask
        centroid_y = centroid_y_mask * scale_y
        centroid_x = centroid_x_mask * scale_x
        logger.info(
            f"Mask shape {mask.shape} doesn't match image shape ({H}, {W}). "
            f"Scaled centroid coordinates by factors ({scale_y:.2f}, {scale_x:.2f})."
        )
    else:
        centroid_y = centroid_y_mask
        centroid_x = centroid_x_mask

    # Calculate offset to center grid on centroid
    # We want the centroid to be at the center of a tile
    # Find which tile index would contain the centroid if grid started at (0,0)
    tile_idx_y_centroid = int(centroid_y // ty)
    tile_idx_x_centroid = int(centroid_x // tx)

    # Calculate the center position of that tile in a standard grid
    tile_center_y_standard = tile_idx_y_centroid * ty + ty / 2
    tile_center_x_standard = tile_idx_x_centroid * tx + tx / 2

    # Calculate offset needed to move that tile center to the centroid
    offset_y = int(round(centroid_y - tile_center_y_standard))
    offset_x = int(round(centroid_x - tile_center_x_standard))

    tg = TileGrid(H, W, tile_size=tile_size, offset_y=offset_y, offset_x=offset_x)

    if inplace:
        shapes_key = new_shapes_key or f"{image_key}_tile_grid"
        _save_tile_grid_to_shapes(sdata, tg, image_key, shapes_key)
        return None

    return tg


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
