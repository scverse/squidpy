from __future__ import annotations

import itertools
from typing import Literal

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from dask.base import is_dask_collection
from shapely.geometry import Polygon
from spatialdata._logging import logger
from spatialdata.models import Labels2DModel, ShapesModel
from spatialdata.models._utils import SpatialElement
from spatialdata.transformations import get_transformation, set_transformation

from squidpy._utils import _yx_from_shape

from ._utils import _get_element_data

__all__ = ["make_tiles", "make_tiles_from_spots"]


class _TileGrid:
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
        self.H = H
        self.W = W
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
        # The grid starts at offset_y, offset_x (can be negative)
        # We need tiles from min(0, offset_y) to at least H
        # So total coverage needed is from min(0, offset_y) to H
        grid_start_y = min(0, self.offset_y)
        grid_start_x = min(0, self.offset_x)
        total_h_needed = self.H - grid_start_y
        total_w_needed = self.W - grid_start_x
        self.tiles_y = (total_h_needed + self.ty - 1) // self.ty
        self.tiles_x = (total_w_needed + self.tx - 1) // self.tx
        # Cache immutable derived values
        self._indices = np.array([[iy, ix] for iy in range(self.tiles_y) for ix in range(self.tiles_x)], dtype=int)
        self._names = [f"tile_x{ix}_y{iy}" for iy in range(self.tiles_y) for ix in range(self.tiles_x)]
        self._bounds = self._compute_bounds()
        self._centroids_polys = self._compute_centroids_and_polygons()

    def indices(self) -> np.ndarray:
        return self._indices

    def names(self) -> list[str]:
        return self._names

    def bounds(self) -> np.ndarray:
        return self._bounds

    def _compute_bounds(self) -> np.ndarray:
        b: list[list[int]] = []
        for iy, ix in itertools.product(range(self.tiles_y), range(self.tiles_x)):
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
        return self._centroids_polys

    def _compute_centroids_and_polygons(self) -> tuple[np.ndarray, list[Polygon]]:
        cents: list[list[float]] = []
        polys: list[Polygon] = []
        for y0, x0, y1, x1 in self._bounds:
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


class _SpotTileGrid:
    """Tile container for Visium spots, used with ``_filter_tiles``."""

    def __init__(self, centers: np.ndarray, tile_size: tuple[int, int], spot_ids: np.ndarray | None = None):
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("Expected centers of shape (n, 2) for (x, y) coordinates.")
        self.centers = centers.astype(float)
        self.tx = int(tile_size[1])
        self.ty = int(tile_size[0])
        if self.tx <= 0 or self.ty <= 0:
            raise ValueError("Derived tile size must be positive in both dimensions.")
        self._spot_ids = spot_ids if spot_ids is not None else np.arange(len(centers))
        self._bounds = self._compute_bounds()
        self._indices = np.zeros((len(self.centers), 2), dtype=int)
        self._names = [f"spot_tile_{spot_id}" for spot_id in self._spot_ids]
        self._centroids_polys = self._compute_centroids_and_polygons()

    def bounds(self) -> np.ndarray:
        return self._bounds

    def _compute_bounds(self) -> np.ndarray:
        half_h = self.ty / 2.0
        half_w = self.tx / 2.0
        x = self.centers[:, 0]
        y = self.centers[:, 1]
        y0 = np.floor(y - half_h)
        y1 = np.ceil(y + half_h)
        x0 = np.floor(x - half_w)
        x1 = np.ceil(x + half_w)
        return np.column_stack([y0, x0, y1, x1]).astype(int)

    def indices(self) -> np.ndarray:
        # Dummy indices; not used downstream but kept for API compatibility.
        return self._indices

    def names(self) -> list[str]:
        return self._names

    def centroids_and_polygons(self) -> tuple[np.ndarray, list[Polygon]]:
        return self._centroids_polys

    def _compute_centroids_and_polygons(self) -> tuple[np.ndarray, list[Polygon]]:
        bounds = self._bounds
        polys: list[Polygon] = []
        cents: list[list[float]] = []
        for y0, x0, y1, x1 in bounds:
            cy = (y0 + y1) / 2.0
            cx = (x0 + x1) / 2.0
            cents.append([cy, cx])
            polys.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]))
        return np.asarray(cents, dtype=float), polys


def _get_largest_scale_dimensions(
    sdata: sd.SpatialData,
    image_key: str,
) -> tuple[int, int]:
    """Get the dimensions (H, W) of the largest/finest scale of an image."""
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


def _choose_label_scale_for_image(label_node: Labels2DModel, target_hw: tuple[int, int]) -> str:
    """Pick the label scale closest to the target image height/width."""
    if not hasattr(label_node, "keys"):
        return "scale0"  # single-scale labels default to their only scale
    target_h, target_w = target_hw
    best = None
    best_diff = float("inf")
    for k in label_node.keys():
        y, x = _yx_from_shape(label_node[k].image.shape)
        diff = abs(y - target_h) + abs(x - target_w)
        if diff == 0:
            return k
        if diff < best_diff:
            best_diff = diff
            best = k
    return best or "scale0"


def _save_tiles_to_shapes(
    sdata: sd.SpatialData,
    tg: _TileGrid,
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
    # we know that a) the element exists and b) it has at least an Identity transformation
    transformations = get_transformation(sdata.images[image_key], get_all=True)
    set_transformation(sdata.shapes[shapes_key], transformations, set_all=True)
    logger.info(f"Saved tile grid as 'sdata.shapes[\"{shapes_key}\"]'")


def _save_spot_tiles_to_shapes(
    sdata: sd.SpatialData,
    tg: _SpotTileGrid,
    shapes_key: str,
    spot_ids: np.ndarray,
    source_shapes_key: str,
) -> None:
    """Save spot-centered tiles as polygons, copying transformations from the source shapes."""
    tile_bounds = tg.bounds()
    _, polys = tg.centroids_and_polygons()
    tile_gdf = gpd.GeoDataFrame(
        {
            "tile_id": tg.names(),
            "spot_id": spot_ids,
            "pixel_y0": tile_bounds[:, 0],
            "pixel_x0": tile_bounds[:, 1],
            "pixel_y1": tile_bounds[:, 2],
            "pixel_x1": tile_bounds[:, 3],
            "geometry": polys,
        },
        geometry="geometry",
    )

    sdata.shapes[shapes_key] = ShapesModel.parse(tile_gdf)
    try:
        transformations = get_transformation(sdata.shapes[source_shapes_key], get_all=True)
    except (KeyError, ValueError):
        transformations = None
    if transformations:
        set_transformation(sdata.shapes[shapes_key], transformations, set_all=True)
    logger.info(f"Saved spot-aligned tiles as 'sdata.shapes[\"{shapes_key}\"]'")


def _propagate_spot_classification(sdata: sd.SpatialData, tiles_key: str, spots_key: str) -> None:
    """Copy tile classifications from a tiles table back to the corresponding spots."""
    if tiles_key not in sdata.shapes or spots_key not in sdata.shapes:
        return
    tiles_gdf = sdata.shapes[tiles_key]
    if "spot_id" not in tiles_gdf.columns or "tile_classification" not in tiles_gdf.columns:
        logger.warning("Spot tiles missing required columns for classification propagation.")
        return
    classification_map = dict(zip(tiles_gdf["spot_id"], tiles_gdf["tile_classification"], strict=False))
    spots_gdf = sdata.shapes[spots_key]
    spots_gdf = spots_gdf.copy()
    spots_gdf["tile_classification"] = pd.Categorical(
        spots_gdf.index.to_series().map(classification_map),
        categories=["background", "partial_tissue", "tissue"],
    )
    sdata.shapes[spots_key] = ShapesModel.parse(spots_gdf)


def make_tiles(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    image_mask_key: str | None = None,
    tissue_mask_key: str | None = None,
    tile_size: tuple[int, int] = (224, 224),
    center_grid_on_tissue: bool = False,
    scale: str = "auto",
    min_tissue_fraction: float = 1.0,
    new_shapes_key: str | None = None,
    preview: bool = True,
) -> None:
    """
    Create a regular grid of tiles over an image, classify them by tissue coverage, and optionally render a preview.

    Tiles are generated on the highest-resolution image scale and classified into three categories based on the
    supplied (or automatically derived) tissue mask:

        - ``"background"``: no tissue pixels in the tile.
        - ``"partial_tissue"``: some tissue but below ``min_tissue_fraction``.
        - ``"tissue"``: at least ``min_tissue_fraction`` of the tile is tissue.

    The resulting grid is stored in ``sdata.shapes[new_shapes_key]`` (default ``f"{image_key}_tiles"``) with
    one row per tile and columns such as ``pixel_y0``, ``pixel_x0``, ``pixel_y1``, ``pixel_x1``, and
    ``tile_classification``.

    Parameters
    ----------
    sdata
        SpatialData object containing the image and (optionally) label masks.
    image_key
        Key of the image in ``sdata.images`` to tile.
    image_mask_key
        Optional key of a segmentation or tissue mask in ``sdata.labels`` used purely to position the grid.
        If ``center_grid_on_tissue`` is ``True`` and this is not provided, the function will fall back to
        ``tissue_mask_key`` or ``f"{image_key}_tissue"`` if present, or automatically run
        :func:`~squidpy.experimental.im.detect_tissue` to create one.
    tissue_mask_key
        Key of the tissue mask in ``sdata.labels`` used for classification. If ``None``, the function uses
        ``f"{image_key}_tissue"`` and will automatically run :func:`detect_tissue` to create this mask if it
        does not exist.
    tile_size
        Size of tiles as ``(height, width)`` in pixels on the largest image scale.
    center_grid_on_tissue
        If ``True``, center the tile grid on the centroid of the mask given by ``image_mask_key`` /
        ``tissue_mask_key`` / ``f"{image_key}_tissue"`` (created on the fly if needed). If ``False``, the grid
        starts at the top-left corner of the image.
    scale
        Label scale to use when reading the mask for centering and classification. If ``"auto"``, the function
        chooses the label scale whose shape is closest to the full-resolution image dimensions.
    min_tissue_fraction
        Minimum fraction of a tile that must be tissue for it to be considered suitable for inference. Tiles
        below this threshold are classified as ``"background"`` (0%) or ``"partial_tissue"`` (>0%).
    new_shapes_key
        Key under which to store the tile grid in ``sdata.shapes``. Defaults to ``f"{image_key}_tiles"``.
    preview
        If ``True``, render a preview using ``sdata.pl.render_images(image_key)`` overlaid with the tiles colored
        by ``tile_classification``.

    Returns
    -------
    None
        All results are written in-place to ``sdata.shapes``.

    See Also
    --------
    detect_tissue
        Helper used to automatically derive a tissue mask when none is provided.
    make_tiles_from_spots
        Create tiles centered on Visium spots instead of a regular grid.
    """
    # Derive mask key for centering if needed
    mask_key_for_grid = image_mask_key
    default_mask_key = tissue_mask_key or f"{image_key}_tissue"
    scale_for_grid = scale
    if center_grid_on_tissue and mask_key_for_grid is None:
        if default_mask_key in sdata.labels:
            mask_key_for_grid = default_mask_key
        else:
            try:
                from ._detect_tissue import detect_tissue

                detect_tissue(
                    sdata,
                    image_key=image_key,
                    scale=scale,
                    new_labels_key=default_mask_key,
                    inplace=True,
                )
            except (ImportError, KeyError, ValueError, RuntimeError) as e:  # pragma: no cover - defensive
                logger.warning(
                    "center_grid_on_tissue=True but no mask key provided/found; "
                    "detect_tissue failed (%s). Using default grid origin.",
                    e,
                )
            else:
                mask_key_for_grid = default_mask_key
    if center_grid_on_tissue and mask_key_for_grid is not None and scale == "auto":
        label_node = sdata.labels.get(mask_key_for_grid)
        if label_node is not None:
            target_hw = _get_largest_scale_dimensions(sdata, image_key)
            scale_for_grid = _choose_label_scale_for_image(label_node, target_hw)

    # Build tile grid (keep locally for filtering)
    shapes_key = new_shapes_key or f"{image_key}_tiles"
    tg = _make_tiles(
        sdata,
        image_key=image_key,
        image_mask_key=mask_key_for_grid,
        tile_size=tile_size,
        center_grid_on_tissue=center_grid_on_tissue,
        scale=scale_for_grid,
    )

    _save_tiles_to_shapes(sdata, tg, image_key, shapes_key)

    # Filter tiles
    if tg is not None:
        classification_mask_key = tissue_mask_key or f"{image_key}_tissue"
        if classification_mask_key not in sdata.labels:
            logger.info(
                "No tissue mask provided/found; running detect_tissue to create '%s' for classification.",
                classification_mask_key,
            )
            try:
                from ._detect_tissue import detect_tissue

                detect_tissue(
                    sdata,
                    image_key=image_key,
                    scale=scale,
                    new_labels_key=classification_mask_key,
                    inplace=True,
                )
            except (ImportError, KeyError, ValueError, RuntimeError) as e:  # pragma: no cover - defensive
                logger.warning("detect_tissue failed (%s); tiles will not be classified.", e)
        if classification_mask_key not in sdata.labels:
            raise KeyError(f"Tissue mask '{classification_mask_key}' not found in sdata.labels.")
        # Use a mask scale that aligns with the full-resolution image; avoid coarsest "auto" selection.
        if scale == "auto":
            label_node = sdata.labels.get(classification_mask_key)
            if label_node is not None:
                target_hw = _get_largest_scale_dimensions(sdata, image_key)
                scale_used = _choose_label_scale_for_image(label_node, target_hw)
            else:
                scale_used = "scale0"
        else:
            scale_used = scale
        _filter_tiles(
            sdata,
            tg=tg,
            image_key=image_key,
            tissue_mask_key=classification_mask_key,
            scale=scale_used,
            min_tissue_fraction=min_tissue_fraction,
            shapes_key=shapes_key,
        )

    # Generate preview plot if requested
    if preview and shapes_key in sdata.shapes:
        try:
            (
                sdata.pl.render_images(image_key)
                .pl.render_shapes(shapes_key, color="tile_classification", fill_alpha=0.5)
                .pl.show()
            )
        except (AttributeError, KeyError, ValueError) as e:
            logger.warning(f"Could not generate preview plot: {e}")


def make_tiles_from_spots(
    sdata: sd.SpatialData,
    *,
    spots_key: str,
    image_key: str | None = None,
    tissue_mask_key: str | None = None,
    scale: str = "auto",
    min_tissue_fraction: float = 1.0,
    new_shapes_key: str | None = None,
    preview: bool = True,
) -> None:
    """
    Create tiles centered on Visium spots, classify them by tissue coverage, and optionally render a preview.

    The function reads spot coordinates from ``sdata.shapes[spots_key]``, infers a square tile size from the
    vertical spacing between spots, and constructs one tile per spot. Tiles are then classified using a tissue
    mask in the same way as :func:`make_tiles`.

    Parameters
    ----------
    sdata
        SpatialData object containing Visium spot shapes, images, and (optionally) label masks.
    spots_key
        Key of the spot shapes in ``sdata.shapes``. The geometry must be point-like and in the same coordinate
        system as the image / tissue mask.
    image_key
        Optional key of the image in ``sdata.images``. If provided, the function will:

            - use it to choose an appropriate label scale for the tissue mask,
            - optionally render the preview with the image as a background, and
            - automatically run :func:`detect_tissue` to create ``f"{image_key}_tissue"`` if
              neither ``tissue_mask_key`` nor an existing ``f"{image_key}_tissue"`` is found.

        If ``None``, tiles can still be created and classified using an explicit ``tissue_mask_key``, but the
        preview will render tiles only (no image background).
    tissue_mask_key
        Key of the tissue mask in ``sdata.labels`` used for classification. If ``None`` and ``image_key`` is
        provided, the function falls back to ``f"{image_key}_tissue"`` and will automatically run
        :func:`detect_tissue` to create it if missing. If both ``image_key`` and ``tissue_mask_key`` are
        ``None``, tiles are created but not classified.
    scale
        Label scale to use when reading the tissue mask. If ``"auto"`` and ``image_key`` is provided, the
        function picks the label scale whose shape is closest to the full-resolution image dimensions.
    min_tissue_fraction
        Minimum fraction of a tile that must be tissue for it to be considered suitable for inference. Tiles
        below this threshold are classified as ``"background"`` (0%) or ``"partial_tissue"`` (>0%).
    new_shapes_key
        Key under which to store the spot-centered tiles in ``sdata.shapes``. Defaults to ``f"{spots_key}_tiles"``.
    preview
        If ``True``, render a preview. When ``image_key`` is provided, this uses
        ``sdata.pl.render_images(image_key)`` and overlays the tiles colored by ``tile_classification``.
        Otherwise only the tiles are rendered.

    Returns
    -------
    None
        Tiles and their classifications are written in-place to ``sdata.shapes``; spot classifications are copied
        back to ``sdata.shapes[spots_key]`` via the ``tile_classification`` column.

    See Also
    --------
    make_tiles
        Create a regular grid of tiles over an image instead of spot-centered tiles.
    detect_tissue
        Helper used to derive tissue masks automatically when needed.
    """

    if spots_key not in sdata.shapes:
        raise KeyError(f"Spots key '{spots_key}' not found in sdata.shapes")

    target_cs: str | None = None
    if image_key is not None:
        mask_key = tissue_mask_key or f"{image_key}_tissue"
        if mask_key in sdata.labels:
            target_cs = _get_primary_coordinate_system(sdata.labels[mask_key])
        image_cs = _get_primary_coordinate_system(sdata.images[image_key])
        if target_cs and image_cs and target_cs != image_cs:
            logger.warning(
                "Coordinate system mismatch between mask (%s) and image (%s). Tile coverage may be incorrect.",
                target_cs,
                image_cs,
            )

    coords, spot_ids = _get_spot_coordinates(sdata, spots_key)
    derived_tile = _derive_tile_size_from_spots(coords)
    logger.info(f"Derived tile size {derived_tile} from {len(coords)} Visium spots (key='{spots_key}').")

    tg = _SpotTileGrid(centers=coords, tile_size=derived_tile, spot_ids=spot_ids)
    shapes_key = new_shapes_key or f"{spots_key}_tiles"
    _save_spot_tiles_to_shapes(sdata, tg, shapes_key, spot_ids, source_shapes_key=spots_key)

    classification_mask_key: str | None = None
    if tissue_mask_key is not None:
        classification_mask_key = tissue_mask_key
    elif image_key is not None:
        classification_mask_key = f"{image_key}_tissue"
        if classification_mask_key not in sdata.labels:
            logger.info(
                "No tissue mask provided/found; running detect_tissue to create '%s' for classification.",
                classification_mask_key,
            )
            try:
                from ._detect_tissue import detect_tissue

                detect_tissue(
                    sdata,
                    image_key=image_key,
                    scale=scale,
                    new_labels_key=classification_mask_key,
                    inplace=True,
                )
            except (ImportError, KeyError, ValueError, RuntimeError) as e:  # pragma: no cover - defensive
                logger.warning("detect_tissue failed (%s); tiles will not be classified.", e)

    if classification_mask_key is not None:
        if image_key is not None:
            if image_key not in sdata.images:
                raise KeyError(f"Image key '{image_key}' not found in sdata.images")

            mask_key = classification_mask_key
            if mask_key in sdata.labels:
                target_hw = _get_largest_scale_dimensions(sdata, image_key)
                scale = _choose_label_scale_for_image(sdata.labels[mask_key], target_hw)
            else:
                scale = "scale0"

            _filter_tiles(
                sdata,
                tg=tg,
                image_key=image_key,
                tissue_mask_key=classification_mask_key,
                scale=scale,
                min_tissue_fraction=min_tissue_fraction,
                shapes_key=shapes_key,
            )
        else:
            if classification_mask_key not in sdata.labels:
                raise KeyError(f"Tissue mask '{classification_mask_key}' not found in sdata.labels.")
            # Without an image we cannot infer the best scale; default to finest scale unless user specified.
            scale_used = "scale0" if scale == "auto" else scale
            _filter_tiles(
                sdata,
                tg=tg,
                image_key=None,
                tissue_mask_key=classification_mask_key,
                scale=scale_used,
                min_tissue_fraction=min_tissue_fraction,
                shapes_key=shapes_key,
            )
        _propagate_spot_classification(sdata, shapes_key, spots_key)
    else:
        logger.info("No mask provided or derived; skipping tissue classification.")

    if preview and shapes_key in sdata.shapes:
        try:
            if image_key is not None:
                (
                    sdata.pl.render_images(image_key)
                    .pl.render_shapes(
                        shapes_key,
                        color="tile_classification" if classification_mask_key is not None else None,
                        fill_alpha=0.5,
                    )
                    .pl.show()
                )
            else:
                sdata.pl.render_shapes(
                    shapes_key,
                    color="tile_classification" if classification_mask_key is not None else None,
                    fill_alpha=0.5,
                ).pl.show()
        except (AttributeError, KeyError, ValueError) as e:
            logger.warning(f"Could not generate preview plot: {e}")


def _filter_tiles(
    sdata: sd.SpatialData,
    tg: _TileGrid,
    image_key: str | None,
    *,
    tissue_mask_key: str | None = None,
    scale: str = "scale0",
    min_tissue_fraction: float = 1.0,
    shapes_key: str | None = None,
) -> np.ndarray | None:
    """
    Filter tiles suitable for inference based solely on tissue coverage.

    Filters out tiles that are not exclusively (100% by default) in tissue.

    Tiles are classified as:
    - "background": No tissue pixels in the tile
    - "partial_tissue": Contains some tissue but below ``min_tissue_fraction``
    - "tissue": Meets tissue requirements and is suitable for inference

    Parameters
    ----------
    sdata
        SpatialData object.
    tg
        TileGrid object with tiles to filter.
    image_key
        Key of the image in ``sdata.images``. Optional if ``tissue_mask_key`` is provided.
    tissue_mask_key
        Key of the tissue mask in ``sdata.labels``. If ``None``, uses ``f"{image_key}_tissue"``.
    scale
        Scale level to use for mask processing.
    min_tissue_fraction
        Minimum fraction of tile that must be in tissue. Default is 1.0 (exclusively in tissue).
    shapes_key
        Key of the tile grid shapes in ``sdata.shapes``. If ``None``, uses ``f"{image_key}_tiles"`` when
        ``image_key`` is provided; otherwise must be supplied.
    """
    tile_bounds = tg.bounds()
    n_tiles = len(tile_bounds)
    suitable = np.ones(n_tiles, dtype=bool)

    # Track classification info for each tile
    # Categories: "background", "partial_tissue", "tissue"
    tile_classification = np.full(n_tiles, "tissue", dtype=object)

    # Get tissue mask
    if tissue_mask_key is not None:
        mask_key = tissue_mask_key
    elif image_key is not None:
        mask_key = f"{image_key}_tissue"
    else:
        raise ValueError("tissue_mask_key must be provided when image_key is None.")
    if mask_key not in sdata.labels:
        raise KeyError(f"Tissue mask '{mask_key}' not found in sdata.labels.")
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

    logger.info(
        f"After tissue filtering: {suitable.sum()}/{n_tiles} ({suitable.sum() / n_tiles * 100:.2f}%) tiles are fully within tissue."
    )

    # Always persist classification on the GeoDataFrame so users can inspect it directly
    if shapes_key is None:
        if image_key is None:
            raise ValueError("shapes_key must be provided when image_key is None.")
        shapes_key_used = f"{image_key}_tiles"
    else:
        shapes_key_used = shapes_key
    if shapes_key_used in sdata.shapes:
        gdf = sdata.shapes[shapes_key_used]
        if len(gdf) != len(tile_classification):
            logger.warning(
                "Tile classification length (%d) does not match GeoDataFrame length (%d); skipping write.",
                len(tile_classification),
                len(gdf),
            )
        else:
            gdf = gdf.copy()
            gdf["tile_classification"] = pd.Categorical(
                tile_classification,
                categories=["background", "partial_tissue", "tissue"],
            )
            sdata.shapes[shapes_key_used] = ShapesModel.parse(gdf)

    return suitable


def _make_tiles(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    image_mask_key: str | None = None,
    tile_size: tuple[int, int] = (224, 224),
    center_grid_on_tissue: bool = False,
    scale: str = "auto",
) -> _TileGrid:
    """Construct a tile grid for an image, optionally centered on a tissue mask."""
    # Validate image key
    if image_key not in sdata.images:
        raise KeyError(f"Image key '{image_key}' not found in sdata.images")

    # Get image dimensions from the largest/finest scale
    H, W = _get_largest_scale_dimensions(sdata, image_key)

    ty, tx = tile_size

    # Path 1: Regular grid starting from top-left
    if not center_grid_on_tissue or image_mask_key is None:
        return _TileGrid(H, W, tile_size=tile_size)

    # Path 2: Center grid on tissue mask centroid
    if image_mask_key not in sdata.labels:
        raise KeyError(
            f"Mask key '{image_mask_key}' not found in sdata.labels. Available keys: {list(sdata.labels.keys())}"
        )

    # Get mask and compute centroid
    label_node = sdata.labels[image_mask_key]
    mask_da = _get_element_data(label_node, scale, "label", image_mask_key)

    # Convert to numpy array if needed
    if is_dask_collection(mask_da):
        mask_da = mask_da.compute()

    if isinstance(mask_da, xr.DataArray):
        mask = np.asarray(mask_da.data)
    else:
        mask = np.asarray(mask_da)

    # Ensure 2D (y, x) shape
    if mask.ndim > 3:
        raise ValueError(f"Expected 2D mask with shape `(y, x)`, got shape `{mask.shape}`.")

    if mask.ndim == 3:
        old_shape = mask.shape
        mask = mask.squeeze()
        if mask.ndim == 2:
            logger.warning(f"Mask had shape `{old_shape}`, squeezed to `{mask.shape}`.")
        else:
            raise ValueError(f"Expected 2D mask with shape `(y, x)`, got shape `{mask.shape}`.")

    # If we made it here, the mask is 2D.

    # Ensure mask matches image dimensions
    H_mask, W_mask = mask.shape

    # Compute centroid of mask (where mask > 0)
    mask_bool = mask > 0
    if not mask_bool.any():
        logger.warning("Mask is empty. Using regular grid starting from top-left.")
        return _TileGrid(H, W, tile_size=tile_size)

    # Calculate centroid using center of mass
    y_coords, x_coords = np.where(mask_bool)
    centroid_y = float(np.mean(y_coords))
    centroid_x = float(np.mean(x_coords))

    # Calculate offset to center grid on centroid
    tile_idx_y_centroid = int(centroid_y // ty)
    tile_idx_x_centroid = int(centroid_x // tx)
    tile_center_y_standard = tile_idx_y_centroid * ty + ty / 2
    tile_center_x_standard = tile_idx_x_centroid * tx + tx / 2
    offset_y = int(round(centroid_y - tile_center_y_standard))
    offset_x = int(round(centroid_x - tile_center_x_standard))

    return _TileGrid(H, W, tile_size=tile_size, offset_y=offset_y, offset_x=offset_x)


def _get_spot_coordinates(
    sdata: sd.SpatialData,
    spots_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract spot centers (x, y) and IDs from a shapes table."""
    gdf = sdata.shapes[spots_key]
    if "geometry" not in gdf:
        raise ValueError(f"Shapes '{spots_key}' lack geometry column required for spot coordinates.")
    centers = np.array([[geom.x, geom.y] for geom in gdf.geometry], dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(
            f"Unexpected geometry layout for '{spots_key}'. Expected point geometries with (x, y) coordinates."
        )
    return centers, gdf.index.to_numpy()


def _get_primary_coordinate_system(element: SpatialElement) -> str | None:
    """Return the first available coordinate system, preferring 'global'."""
    try:
        transformations = get_transformation(element, get_all=True)
    except (KeyError, ValueError):
        return None
    if not transformations:
        return None
    # Prefer 'global' if present
    if "global" in transformations:
        return "global"
    return next(iter(transformations.keys()))


def _derive_tile_size_from_spots(coords: np.ndarray) -> tuple[int, int]:
    """Derive a square tile size from Visium spot spacing."""
    if coords.shape[0] < 2:
        raise ValueError("Need at least 2 spots to derive tile size; ensure the spots table has multiple entries.")
    # Spots are arranged in rows with constant vertical spacing; use this to set tile size.
    y_coords = np.unique(np.sort(coords[:, 1]))
    if len(y_coords) < 2:
        raise ValueError(
            "Unable to derive row spacing from spot coordinates; check coordinate system and spot positions."
        )
    diffs = np.diff(y_coords)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("Spot coordinates do not contain distinct rows; verify spot grid spacing.")
    # Use the most frequent spacing (mode) to be robust to outliers.
    values, counts = np.unique(np.round(diffs, decimals=6), return_counts=True)
    row_spacing = float(values[np.argmax(counts)])
    if not np.isfinite(row_spacing) or row_spacing <= 0:
        raise ValueError(
            "Unable to derive a valid spacing from spot coordinates; ensure spots are in consistent units."
        )
    side = max(1, int(np.floor(row_spacing)))
    return side, side


def _get_mask_from_labels(sdata: sd.SpatialData, mask_key: str, scale: str) -> np.ndarray:
    """Extract a 2D mask array from ``sdata.labels`` at the requested scale."""
    if mask_key not in sdata.labels:
        raise KeyError(f"Mask key '{mask_key}' not found in sdata.labels")

    label_node = sdata.labels[mask_key]
    mask_da = _get_element_data(label_node, scale, "label", mask_key)

    if is_dask_collection(mask_da):
        mask_da = mask_da.compute()

    if isinstance(mask_da, xr.DataArray):
        mask = np.asarray(mask_da.data)
    else:
        mask = np.asarray(mask_da)

    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask with shape (y, x), got shape {mask.shape}")

    return mask
