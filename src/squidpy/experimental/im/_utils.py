from __future__ import annotations

from typing import Any, Literal

import dask.array as da
import geopandas as gpd
import numpy as np
import xarray as xr
from shapely import box
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation, set_transformation


class TileGrid:
    """Immutable tile grid definition with cached bounds and centroids."""

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
        self.offset_y = offset_y
        self.offset_x = offset_x
        # Calculate number of tiles needed to cover entire image, accounting for offset
        grid_start_y = min(0, self.offset_y)
        grid_start_x = min(0, self.offset_x)
        total_h_needed = self.H - grid_start_y
        total_w_needed = self.W - grid_start_x
        self.tiles_y = (total_h_needed + self.ty - 1) // self.ty
        self.tiles_x = (total_w_needed + self.tx - 1) // self.tx
        # Cache immutable derived values (vectorized)
        iy = np.repeat(np.arange(self.tiles_y), self.tiles_x)
        ix = np.tile(np.arange(self.tiles_x), self.tiles_y)
        self._indices = np.column_stack([iy, ix])
        self._names = [f"tile_x{x}_y{y}" for y, x in zip(iy, ix, strict=True)]
        self._bounds = self._compute_bounds(iy, ix)
        self._centroids, self._polys = self._compute_centroids_and_polygons()

    def indices(self) -> np.ndarray:
        return self._indices

    def names(self) -> list[str]:
        return self._names

    def bounds(self) -> np.ndarray:
        return self._bounds

    def _compute_bounds(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        y0 = iy * self.ty + self.offset_y
        x0 = ix * self.tx + self.offset_x
        y1 = (iy + 1) * self.ty + self.offset_y
        x1 = (ix + 1) * self.tx + self.offset_x
        # Last row/column extends to image edge
        y1[iy == self.tiles_y - 1] = self.H
        x1[ix == self.tiles_x - 1] = self.W
        # Clamp to image dimensions
        y0 = np.clip(y0, 0, self.H)
        x0 = np.clip(x0, 0, self.W)
        y1 = np.clip(y1, 0, self.H)
        x1 = np.clip(x1, 0, self.W)
        return np.column_stack([y0, x0, y1, x1]).astype(int)

    def centroids_and_polygons(self) -> tuple[np.ndarray, list]:
        return self._centroids, self._polys

    def _compute_centroids_and_polygons(self) -> tuple[np.ndarray, list]:
        y0, x0, y1, x1 = self._bounds[:, 0], self._bounds[:, 1], self._bounds[:, 2], self._bounds[:, 3]
        centroids = np.column_stack([(y0 + y1) / 2.0, (x0 + x1) / 2.0])
        polys = list(box(x0, y0, x1, y1))
        return centroids, polys

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


def _get_mask_dask(sdata: SpatialData, mask_key: str, scale: str) -> da.Array:
    """Extract mask as a lazy dask array from ``sdata.labels``."""
    if mask_key not in sdata.labels:
        raise KeyError(f"Mask key '{mask_key}' not found in sdata.labels")
    label_node = sdata.labels[mask_key]
    mask_xr = _get_element_data(label_node, scale, "label", mask_key)

    arr = mask_xr.data if hasattr(mask_xr, "data") else mask_xr
    if not isinstance(arr, da.Array):
        arr = da.from_array(np.asarray(arr))

    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask with shape (y, x), got shape {arr.shape}")
    return arr


def _get_mask_materialized(sdata: SpatialData, mask_key: str, scale: str) -> np.ndarray:
    """Extract a 2D mask array from ``sdata.labels`` at the requested scale (materialized)."""
    arr = _get_mask_dask(sdata, mask_key, scale)
    return np.asarray(arr.compute())


def _ensure_tissue_mask(
    sdata: SpatialData,
    image_key: str,
    scale: str,
    tissue_mask_key: str | None = None,
) -> str:
    """Return the key of a tissue mask in ``sdata.labels``, creating one if needed.

    If *tissue_mask_key* is given and exists, it is returned as-is.
    Otherwise falls back to ``f"{image_key}_tissue"``, running ``detect_tissue``
    to create it when missing.

    Raises
    ------
    KeyError
        If *tissue_mask_key* is given but not found in ``sdata.labels``.
    Exception
        Any exception raised by ``detect_tissue`` if auto-creation fails.
        Callers needing graceful fallback should wrap this call in try/except.
    """
    if tissue_mask_key is not None:
        if tissue_mask_key not in sdata.labels:
            raise KeyError(f"Tissue mask key '{tissue_mask_key}' not found in sdata.labels")
        return tissue_mask_key

    mask_key = f"{image_key}_tissue"
    if mask_key not in sdata.labels:
        from squidpy.experimental.im._detect_tissue import detect_tissue

        detect_tissue(sdata=sdata, image_key=image_key, scale=scale, inplace=True, new_labels_key=mask_key)
        logger.info(f"Saved tissue mask as 'sdata.labels[\"{mask_key}\"]'")
    return mask_key


def _save_tile_grid_to_shapes(
    sdata: SpatialData,
    tg: TileGrid,
    shapes_key: str,
    *,
    copy_transforms_from_key: str | None = None,
) -> None:
    """Save a TileGrid to ``sdata.shapes`` as a GeoDataFrame.

    Parameters
    ----------
    sdata
        SpatialData object.
    tg
        TileGrid whose bounds/centroids are persisted.
    shapes_key
        Key under which to store the shapes.
    copy_transforms_from_key
        If given, copy the transformations from ``sdata.images[copy_transforms_from_key]``
        onto the new shapes element.
    """
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
    if copy_transforms_from_key is not None:
        transformations = get_transformation(sdata.images[copy_transforms_from_key], get_all=True)
        set_transformation(sdata.shapes[shapes_key], transformations, set_all=True)
    logger.info(f"- Saved tile grid as 'sdata.shapes[\"{shapes_key}\"]'")
