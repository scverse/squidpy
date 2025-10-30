from __future__ import annotations

from typing import Any, Literal

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from shapely.geometry import Polygon
from spatialdata._logging import logger
from spatialdata.models import ShapesModel, TableModel

from squidpy._utils import _ensure_dim_order, _yx_from_shape

__all__ = ["TileGrid", "make_tiles_for_inference"]


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


def make_tiles_for_inference(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    image_mask_key: str | None = None,
    tissue_mask_key: str | None = None,
    sharpness_table_key: str | None = None,
    tile_size: tuple[int, int] = (224, 224),
    center_grid_on_tissue: bool = False,
    scale: str = "auto",
    min_tissue_fraction: float = 1.0,
    min_sharp_fraction: float = 0.95,
    new_shapes_key: str | None = None,
    new_table_key: str | None = None,
    preview: bool = True,
    **kwargs: Any,
) -> None:
    """
    Create and filter tiles suitable for inference.

    This function combines tile grid generation and filtering based on tissue coverage
    and sharpness. It creates a tile grid, filters tiles based on tissue and sharpness
    criteria, and optionally generates a preview plot.

    Parameters
    ----------
    sdata
        SpatialData object containing the image.
    image_key
        Key of the image in ``sdata.images``.
    image_mask_key
        Optional key of the segmentation mask in ``sdata.labels``.
        Required if ``center_grid_on_tissue`` is ``True``.
    tissue_mask_key
        Key of the tissue mask in ``sdata.labels``. If ``None``, uses ``f"{image_key}_tissue"``.
    sharpness_table_key
        Key of the sharpness QC table in ``sdata.tables``. If ``None``, uses
        ``f"qc_img_{image_key}_sharpness"``.
    tile_size
        Size of tiles as (height, width) in pixels.
    center_grid_on_tissue
        If ``True`` and ``image_mask_key`` is provided, center the grid
        on the centroid of the mask.
    scale
        Scale level to use for mask processing. The tile grid is always generated
        based on the largest (finest) scale of the image.
    min_tissue_fraction
        Minimum fraction of tile that must be in tissue. Default is 1.0 (exclusively in tissue).
    min_sharp_fraction
        Minimum fraction of pixels in the tile that must be sharp (non-outlier).
        Tiles will be filtered out if the outlier fraction is >= (1 - min_sharp_fraction).
        For example:
        - To filter tiles where >= 10% of pixels are outliers, use ``min_sharp_fraction=0.9``
          (tiles with 10% or more outliers will be flagged)
        - To filter tiles where >= 5% of pixels are outliers, use ``min_sharp_fraction=0.95``
          (tiles with 5% or more outliers will be flagged)
        Default is 0.95 (tiles with >= 5% outliers will be filtered out).
    new_shapes_key
        Key to save the tile grid in ``sdata.shapes``. If ``None``, defaults to ``f"{image_key}_tile_grid"``.
    new_table_key
        Key to save the tile information table in ``sdata.tables``.
        If ``None``, defaults to ``f"{image_key}_inference_tiles"``.
    preview
        If ``True``, generate a preview plot showing the tile grid colored by classification.
        Default is ``True``.
    **kwargs
        Additional arguments passed to the plotting functions for the preview.

    Returns
    -------
    None
        Results are saved to ``sdata.shapes`` and ``sdata.tables``.

    See Also
    --------
    :func:`_make_tile_grid` : Internal function for tile grid generation.
    :func:`_filter_tiles_for_inference` : Internal function for tile filtering.

    Examples
    --------
    >>> import squidpy as sq
    >>> sdata = sq.datasets.visium_hne_image_crop()
    >>> sq.experimental.im.make_tiles_for_inference(sdata, image_key="image", preview=True)
    """
    # Generate tile grid (without saving first, so we can use it for filtering)
    shapes_key = new_shapes_key or f"{image_key}_tile_grid"
    tg = _make_tile_grid(
        sdata,
        image_key,
        image_mask_key=image_mask_key,
        tile_size=tile_size,
        center_grid_on_tissue=center_grid_on_tissue,
        scale=scale,
        new_shapes_key=shapes_key,
        inplace=False,
    )

    # Now save it
    if tg is not None:
        _save_tile_grid_to_shapes(sdata, tg, image_key, shapes_key)

    # Filter tiles
    table_key = new_table_key or f"{image_key}_inference_tiles"
    if tg is not None:
        _filter_tiles_for_inference(
            sdata,
            tg=tg,
            image_key=image_key,
            tissue_mask_key=tissue_mask_key,
            sharpness_table_key=sharpness_table_key,
            scale="scale0",
            min_tissue_fraction=min_tissue_fraction,
            min_sharp_fraction=min_sharp_fraction,
            new_table_key=table_key,
            shapes_key=shapes_key,
            inplace=True,
        )

        # Update shapes GeoDataFrame with classification for plotting
        if shapes_key in sdata.shapes and table_key in sdata.tables:
            adata_table = sdata.tables[table_key]
            gdf = sdata.shapes[shapes_key]
            # Join classification from table to shapes
            # The table's tile_id corresponds to the GeoDataFrame index
            if "tile_classification" in adata_table.obs.columns:
                classification_map = dict(
                    zip(adata_table.obs["tile_id"], adata_table.obs["tile_classification"], strict=True)
                )
                # Map using GeoDataFrame index to match table's tile_id
                # Convert to categorical to match AnnData format and ensure proper type handling
                classification_values = gdf.index.to_series().map(classification_map)
                gdf["tile_classification"] = pd.Categorical(
                    classification_values,
                    categories=["background", "partial_tissue", "tissue", "sharpness_outlier"],
                )
                # Update the shapes in sdata
                sdata.shapes[shapes_key] = ShapesModel.parse(gdf)

    # Generate preview plot if requested
    if preview and shapes_key in sdata.shapes:
        try:
            (
                sdata.pl.render_images(image_key)
                .pl.render_shapes(shapes_key, color="tile_classification", fill_alpha=0.5, **kwargs)
                .pl.show()
            )
        except (AttributeError, KeyError, ValueError) as e:
            logger.warning(f"Could not generate preview plot: {e}")


def _filter_tiles_for_inference(
    sdata: sd.SpatialData,
    tg: TileGrid,
    image_key: str,
    *,
    tissue_mask_key: str | None = None,
    sharpness_table_key: str | None = None,
    scale: str = "scale0",
    min_tissue_fraction: float = 1.0,
    min_sharp_fraction: float = 0.95,
    new_table_key: str | None = None,
    shapes_key: str | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Filter tiles suitable for inference based on tissue coverage and sharpness.

    Filters out tiles that:
    1. Are not exclusively (100% by default) in tissue
    2. Have less than 50% (by default) of overlapping sharp area flagged as sharp

    Tiles are classified as:
    - "background": No tissue pixels in the tile
    - "partial_tissue": Contains some tissue but below min_tissue_fraction threshold
    - "tissue": Meets tissue requirements and suitable for inference
    - "sharpness_outlier": Meets tissue requirements but fails sharpness filtering

    Uses pixel-precise masking: creates a full-size binary mask marking outlier pixels
    from sharpness analysis, then extracts each inference tile's region and calculates
    the exact fraction of outlier pixels. This accurately handles different tile dimensions.

    Parameters
    ----------
    sdata
        SpatialData object.
    tg
        TileGrid object with tiles to filter.
    image_key
        Key of the image in ``sdata.images``.
    tissue_mask_key
        Key of the tissue mask in ``sdata.labels``. If ``None``, uses ``f"{image_key}_tissue"``.
    sharpness_table_key
        Key of the sharpness QC table in ``sdata.tables``. If ``None``, uses
        ``f"qc_img_{image_key}_sharpness"``.
    scale
        Scale level to use for mask processing.
    min_tissue_fraction
        Minimum fraction of tile that must be in tissue. Default is 1.0 (exclusively in tissue).
    min_sharp_fraction
        Minimum fraction of pixels in the tile that must be sharp (non-outlier).
        Tiles will be filtered out if the outlier fraction is >= (1 - min_sharp_fraction).
        For example:
        - To filter tiles where >= 10% of pixels are outliers, use ``min_sharp_fraction=0.9``
          (tiles with 10% or more outliers will be flagged)
        - To filter tiles where >= 5% of pixels are outliers, use ``min_sharp_fraction=0.95``
          (tiles with 5% or more outliers will be flagged)
        Default is 0.95 (tiles with >= 5% outliers will be filtered out).
    new_table_key
        Key to save the tile information table in ``sdata.tables`` if ``inplace=True``.
        If ``None``, defaults to ``f"{image_key}_tile_grid"``.
    shapes_key
        Key of the tile grid shapes in ``sdata.shapes``. If ``None``, uses ``f"{image_key}_tile_grid"``.
    inplace
        If ``True``, save the tile information as AnnData in ``sdata.tables``.

    Returns
    -------
    If ``inplace=False``, returns a boolean array indicating which tiles are suitable for inference
    (True = suitable). If ``inplace=True``, returns ``None`` (results are saved to ``sdata``).
    """
    tile_bounds = tg.bounds()
    tile_indices = tg.indices()
    n_tiles = len(tile_bounds)
    suitable = np.ones(n_tiles, dtype=bool)

    # Track classification info for each tile
    # Categories: "background", "partial_tissue", "tissue", "sharpness_outlier"
    tile_classification = np.full(n_tiles, "tissue", dtype=object)
    sharpness_outlier_fraction = np.full(n_tiles, np.nan, dtype=np.float32)

    # Get tissue mask
    mask_key = tissue_mask_key or f"{image_key}_tissue"
    if mask_key not in sdata.labels:
        logger.warning(f"Tissue mask '{mask_key}' not found. Skipping tissue filtering.")
    else:
        from ._qc_sharpness import _get_mask_from_labels

        mask = _get_mask_from_labels(sdata, mask_key, scale)
        H_mask, W_mask = mask.shape

        # Check tissue coverage for each tile
        for i, (y0, x0, y1, x1) in enumerate(tile_bounds):
            # Clamp to mask dimensions
            y0_clamped = max(0, min(int(y0), H_mask))
            y1_clamped = max(0, min(int(y1), H_mask))
            x0_clamped = max(0, min(int(x0), W_mask))
            x1_clamped = max(0, min(int(x1), W_mask))

            if y1_clamped > y0_clamped and x1_clamped > x0_clamped:
                tile_region = mask[y0_clamped:y1_clamped, x0_clamped:x1_clamped]
                tissue_fraction = float(np.mean(tile_region > 0))
                if tissue_fraction < min_tissue_fraction:
                    suitable[i] = False
                    if tissue_fraction == 0.0:
                        tile_classification[i] = "background"
                    else:
                        tile_classification[i] = "partial_tissue"
                else:
                    tile_classification[i] = "tissue"
            else:
                # Tile is outside mask bounds
                suitable[i] = False
                tile_classification[i] = "background"

    logger.info(f"After tissue filtering: {suitable.sum()}/{n_tiles} tiles remaining.")

    # Check sharpness if table exists
    sharpness_key = sharpness_table_key or f"qc_img_{image_key}_sharpness"
    has_sharpness = sharpness_key in sdata.tables

    if has_sharpness:
        adata_sharpness: AnnData = sdata.tables[sharpness_key]

        # Check if sharpness outlier column exists
        if "sharpness_outlier" not in adata_sharpness.obs.columns:
            logger.warning("Sharpness outlier labels not found in table. Skipping sharpness filtering.")
            has_sharpness = False
        else:
            # Get sharpness tile grid information
            sharpness_grid_key = f"qc_img_{image_key}_sharpness_grid"
            if sharpness_grid_key not in sdata.shapes:
                logger.warning(f"Sharpness tile grid '{sharpness_grid_key}' not found. Skipping sharpness filtering.")
                has_sharpness = False

    if has_sharpness:
        sharpness_gdf = sdata.shapes[sharpness_grid_key]
        sharpness_outliers = adata_sharpness.obs["sharpness_outlier"].astype(str) == "True"

        # Get sharpness tile bounds
        sharpness_bounds = np.column_stack(
            [
                sharpness_gdf["pixel_y0"].values,
                sharpness_gdf["pixel_x0"].values,
                sharpness_gdf["pixel_y1"].values,
                sharpness_gdf["pixel_x1"].values,
            ]
        )

        # Get image dimensions to create full-size outlier mask
        H, W = _get_largest_scale_dimensions(sdata, image_key)

        # Create pixel-precise mask: 1 where pixels are in outlier tiles, 0 otherwise
        outlier_mask = np.zeros((H, W), dtype=np.uint8)

        for (sy0, sx0, sy1, sx1), is_outlier in zip(sharpness_bounds, sharpness_outliers, strict=True):
            if is_outlier:
                # Mark all pixels in this outlier tile as 1
                sy0_int = max(0, int(sy0))
                sy1_int = min(H, int(sy1))
                sx0_int = max(0, int(sx0))
                sx1_int = min(W, int(sx1))
                if sy1_int > sy0_int and sx1_int > sx0_int:
                    outlier_mask[sy0_int:sy1_int, sx0_int:sx1_int] = 1

        # For each inference tile, extract the mask region and calculate outlier pixel fraction
        # Calculate for ALL tiles, not just those passing tissue filtering
        for i in range(n_tiles):
            y0, x0, y1, x1 = tile_bounds[i]

            # Extract relevant region from outlier mask
            y0_int = max(0, int(y0))
            y1_int = min(H, int(y1))
            x0_int = max(0, int(x0))
            x1_int = min(W, int(x1))

            if y1_int > y0_int and x1_int > x0_int:
                tile_mask_region = outlier_mask[y0_int:y1_int, x0_int:x1_int]

                # Calculate fraction of pixels that are outliers (value == 1)
                outlier_fraction = float(np.mean(tile_mask_region))
                sharpness_outlier_fraction[i] = outlier_fraction

                # Only apply sharpness filtering to tiles that passed tissue filtering
                if suitable[i]:
                    # Filter out tiles where outlier_fraction >= (1 - min_sharp_fraction)
                    # For example: min_sharp_fraction=0.9 means filter if >= 10% outliers
                    max_allowed_outlier_fraction = 1.0 - min_sharp_fraction
                    if outlier_fraction >= max_allowed_outlier_fraction:
                        suitable[i] = False
                        # Update classification - only for tiles that passed tissue filtering
                        if tile_classification[i] == "tissue":
                            tile_classification[i] = "sharpness_outlier"
            else:
                # Tile is outside image bounds
                # Only update suitable if it was still True (to avoid overwriting tissue filtering)
                if suitable[i]:
                    suitable[i] = False
                    if tile_classification[i] == "tissue":
                        tile_classification[i] = "background"

    if has_sharpness:
        logger.info(f"After sharpness filtering: {suitable.sum()}/{n_tiles} tiles suitable for inference.")
    else:
        logger.info(f"No sharpness filtering applied. {suitable.sum()}/{n_tiles} tiles suitable for inference.")

    # Save tile information as AnnData if requested
    if inplace:
        # Use different keys for shapes and table to avoid conflicts
        shapes_key_used = shapes_key or f"{image_key}_tile_grid"
        table_key = new_table_key or f"{image_key}_inference_tiles"

        # Ensure shapes exist
        if shapes_key_used not in sdata.shapes:
            logger.warning(f"Tile grid shapes '{shapes_key_used}' not found. Creating shapes first.")
            _save_tile_grid_to_shapes(sdata, tg, image_key, shapes_key_used)

        # Create AnnData with tile information
        cents, _ = tg.centroids_and_polygons()

        # Create empty AnnData (no features, just observations)
        adata = AnnData(obs=pd.DataFrame(index=tg.names()))
        adata.obs["centroid_y"] = cents[:, 0]
        adata.obs["centroid_x"] = cents[:, 1]
        adata.obs["tile_y"] = tile_indices[:, 0]
        adata.obs["tile_x"] = tile_indices[:, 1]
        adata.obs["pixel_y0"] = tile_bounds[:, 0]
        adata.obs["pixel_x0"] = tile_bounds[:, 1]
        adata.obs["pixel_y1"] = tile_bounds[:, 2]
        adata.obs["pixel_x1"] = tile_bounds[:, 3]
        adata.obs["tile_classification"] = pd.Categorical(
            tile_classification, categories=["background", "partial_tissue", "tissue", "sharpness_outlier"]
        )
        adata.obs["sharpness_outlier_fraction"] = sharpness_outlier_fraction
        adata.obs["suitable_for_inference"] = pd.Categorical(suitable.astype(str), categories=["False", "True"])
        adata.obsm["spatial"] = cents

        # Link to shapes via spatialdata_attrs
        adata.uns["spatialdata_attrs"] = {
            "region": shapes_key_used,
            "region_key": "grid_name",
            "instance_key": "tile_id",
        }
        adata.obs["grid_name"] = pd.Categorical([shapes_key_used] * len(adata))
        adata.obs["tile_id"] = sdata.shapes[shapes_key_used].index

        # Store metadata
        H, W = _get_largest_scale_dimensions(sdata, image_key)
        adata.uns["tile_grid"] = {
            "tile_size_y": tg.ty,
            "tile_size_x": tg.tx,
            "image_height": H,
            "image_width": W,
            "n_tiles_y": tg.tiles_y,
            "n_tiles_x": tg.tiles_x,
            "image_key": image_key,
            "scale": scale,
            "offset_y": tg.offset_y,
            "offset_x": tg.offset_x,
            "min_tissue_fraction": min_tissue_fraction,
            "min_sharp_fraction": min_sharp_fraction,
            "n_tissue_tiles": int((tile_classification == "tissue").sum()),
            "n_background_tiles": int((tile_classification == "background").sum()),
            "n_partial_tissue_tiles": int((tile_classification == "partial_tissue").sum()),
            "n_sharpness_outlier_tiles": int((tile_classification == "sharpness_outlier").sum()),
            "n_suitable_tiles": int(suitable.sum()),
        }

        sdata.tables[table_key] = TableModel.parse(adata)
        logger.info(f"- Saved tile information as 'sdata.tables[\"{table_key}\"]'")
        return None

    return suitable


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
