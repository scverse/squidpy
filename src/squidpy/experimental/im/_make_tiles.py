from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import spatialdata as sd
from anndata import AnnData
from dask.base import is_dask_collection
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from spatialdata import get_centroids
from spatialdata._logging import logger
from spatialdata.models import ShapesModel, TableModel
from spatialdata.models._utils import SpatialElement
from spatialdata.transformations import get_transformation, set_transformation

from squidpy._utils import _yx_from_shape
from squidpy._validators import assert_in_range, assert_key_in_sdata, assert_positive
from squidpy.experimental.im._utils import (
    TileGrid,
    _choose_label_scale_for_image,
    get_element_data,
    get_mask_materialized,
)

__all__ = ["make_tiles", "make_tiles_from_spots"]


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
    *,
    image_key: str,
) -> tuple[int, int]:
    """Get the dimensions (H, W) of the largest/finest scale of an image.

    Callers must validate *image_key* before calling this helper.
    """
    img_node = sdata.images[image_key]

    # Use get_element_data with "scale0" which is always the largest scale
    # It handles both datatree (multiscale) and single-scale images
    img_da = get_element_data(img_node, "scale0", "image", image_key)

    # Get spatial dimensions (y, x)
    if "y" in img_da.sizes and "x" in img_da.sizes:
        return int(img_da.sizes["y"]), int(img_da.sizes["x"])
    else:
        # Fallback: assume last two dimensions are spatial
        return int(img_da.shape[-2]), int(img_da.shape[-1])


def _save_spot_tiles_to_shapes(
    sdata: sd.SpatialData,
    *,
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


def _propagate_spot_classification(sdata: sd.SpatialData, *, tiles_key: str, spots_key: str) -> None:
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
    *,
    image_key: str | Sequence[str],
    tile_size: int | tuple[int, int] | None = None,
    tile_size_um: float | None = None,
    target_mpp: float | None = None,
    image_mpp: float | Mapping[str, float] | None = None,
    mpp_from: str | Mapping[str, str] | None = None,
    centered_on: str | None = None,
    table_key: str | None = None,
    tissue_mask_key: str | Mapping[str, str] | None = None,
    min_tissue_fraction: float = 0.25,
    center_grid_on_tissue: bool = False,
    image_mask_key: str | None = None,
    scale: str = "auto",
    key_added: str | None = None,
    new_shapes_key: str | None = None,
    preview: bool = True,
) -> None:
    """
    Create tiles over one image or a cohort of images, measure their tissue coverage, and link them to observations.

    Tiles are either a regular grid over the tissue (``centered_on=None``) or one square crop centred on each
    shape of ``sdata.shapes[centered_on]``, e.g. Visium spots. Every tile is kept: tiles with less tissue than
    ``min_tissue_fraction`` are flagged with ``in_tissue=False``, not removed.

    Tile sizes are in pixels by default. When the pixel size of an image is known, ``tile_size_um`` gives the
    size in micrometres and ``target_mpp`` picks the pyramid level whose pixel size is closest to it. The pixel
    size is never guessed. It comes from ``image_mpp`` (a value, e.g. from
    :func:`~squidpy.experimental.utils.derive_mpp_from_shapes` for Visium spots or Visium HD bins), or from
    ``mpp_from``, an element stored in micrometres such as Xenium or MERSCOPE transcripts. Without either, the
    tiles stay in pixels and ``uns`` records that.

    Parameters
    ----------
    sdata
        SpatialData object containing the images.
    image_key
        Key of an image in ``sdata.images``, or a list of keys for a cohort. Each image gets its own tile shapes
        and all tiles go into one table.
    tile_size
        Tile size as ``(height, width)``, or one int for square tiles, in pixels of the full-resolution image.
        Defaults to ``(224, 224)``. Can't be combined with ``tile_size_um``.
    tile_size_um
        Square tile side in micrometres. Used only when the pixel size is known.
    target_mpp
        Pixel size, in micrometres per pixel, at which the tiles should be read later. The pyramid level whose
        pixel size is closest to it (on a log scale) is chosen, and the tile bounds on that level are stored.
        Requires ``tile_size_um``. No resampling is done.
    image_mpp
        Micrometres per pixel of the full-resolution image, one value or a mapping from image key to value.
    mpp_from
        Key of a points or shapes element whose coordinates are in micrometres (e.g. Xenium ``"transcripts"``),
        or a mapping from image key to such a key. The pixel size follows from the element's transformation
        relative to the image.
    centered_on
        If `None`, tile the tissue with a regular grid. Otherwise the key of a shapes element in
        ``sdata.shapes``: one crop per shape, centred on its centroid.
    table_key
        Key of a table in ``sdata.tables`` whose observations get ``obs["tile_id"]``: the grid tile under each
        observation's centroid, the observation's own crop when ``centered_on`` is the element the table
        annotates, or `NA` outside any tile.
    tissue_mask_key
        Tissue for each image: a key in ``sdata.shapes`` (polygons, preferred) or ``sdata.labels``, or a mapping
        from image key to such a key. Defaults to ``f"{image_key}_tissue"``; if that doesn't exist,
        :func:`~squidpy.experimental.im.detect_tissue` creates it.
    min_tissue_fraction
        Minimum fraction of a tile covered by tissue for ``in_tissue=True``.
    center_grid_on_tissue
        If `True`, shift the grid so that a tile is centred on the tissue centroid (or on the centroid of
        ``image_mask_key``). Grid tiles only.
    image_mask_key
        Optional key of a mask in ``sdata.labels`` used only to position the grid when
        ``center_grid_on_tissue=True``.
    scale
        Label scale used when :func:`~squidpy.experimental.im.detect_tissue` has to create a mask.
    key_added
        Key of the tiles table in ``sdata.tables``. Defaults to ``f"{image_key}_tiles_table"`` for one image and
        ``"tiles_table"`` for a cohort.
    new_shapes_key
        Key of the tile shapes for a single image. Defaults to ``f"{image_key}_tiles"``. For a cohort the shapes
        are always ``f"{image_key}_tiles"``.
    preview
        If `True`, render the first image with its tiles coloured by ``tile_classification``.

    Returns
    -------
    `None`. Writes:

        - ``sdata.shapes[f"{image_key}_tiles"]`` per image: one polygon per tile in the image's pixel frame, with
          the image's transformations, and the columns ``tissue_fraction``, ``in_tissue``, ``specimen_id``,
          ``tile_classification`` and the pixel bounds on the full-resolution image (``pixel_*``) and on the
          chosen level (``level_*``).
        - ``sdata.tables[key_added]``: one row per tile across all images, annotating those shapes, with
          ``obs`` columns ``slide``, ``region``, ``tile_id``, ``tissue_fraction``, ``in_tissue`` and
          ``specimen_id``, tile centres in ``obsm["spatial"]``, and the parameters, pixel size and its source
          per image in ``uns["make_tiles"]``.
        - ``sdata.tables[table_key].obs["tile_id"]`` if ``table_key`` is given.

    See Also
    --------
    detect_tissue
        Creates the tissue polygons used to measure tissue coverage.
    make_tiles_from_spots
        Older spot-centred tiles; ``make_tiles(..., centered_on="spots")`` replaces it.
    """
    keys = [image_key] if isinstance(image_key, str) else list(image_key)
    if not keys:
        raise ValueError("`image_key` must name at least one image.")
    if len(set(keys)) != len(keys):
        raise ValueError(f"`image_key` contains duplicates: {keys}.")
    for key in keys:
        assert_key_in_sdata(sdata, key, attr="images")
    if new_shapes_key is not None and len(keys) > 1:
        raise ValueError("`new_shapes_key` only applies to a single image; a cohort uses `f'{image_key}_tiles'`.")
    if tile_size_um is not None and tile_size is not None:
        raise ValueError("Pass either `tile_size` (pixels) or `tile_size_um` (micrometres), not both.")
    if target_mpp is not None and tile_size_um is None:
        raise ValueError("`target_mpp` requires `tile_size_um`.")
    if tile_size_um is not None:
        assert_positive(tile_size_um, name="tile_size_um")
    if target_mpp is not None:
        assert_positive(target_mpp, name="target_mpp")
    size_px = _as_pair(tile_size if tile_size is not None else (224, 224))
    assert_positive(size_px[0], name="tile_size[0]")
    assert_positive(size_px[1], name="tile_size[1]")
    assert_in_range(min_tissue_fraction, 0, 1, name="min_tissue_fraction")
    if centered_on is not None:
        assert_key_in_sdata(sdata, centered_on, attr="shapes")
    if table_key is not None:
        assert_key_in_sdata(sdata, table_key, attr="tables")

    table_name = key_added or (f"{keys[0]}_tiles_table" if len(keys) == 1 else "tiles_table")
    built: list[_ImageTiles] = []
    for key in keys:
        shapes_key = new_shapes_key or f"{key}_tiles"
        mpp0, mpp_source = _resolve_mpp(sdata, key, image_mpp=image_mpp, mpp_from=mpp_from)
        levels = _image_levels(sdata.images[key])
        level, side_s0, side_level, info = _tile_geometry(
            key, levels, mpp0=mpp0, size_px=size_px, tile_size_um=tile_size_um, target_mpp=target_mpp
        )
        tissue = _resolve_tissue(sdata, key, tissue_mask_key, scale=scale)
        H, W = levels[0][1], levels[0][2]
        if centered_on is None:
            origin = None
            if center_grid_on_tissue:
                origin = _grid_origin_on_centroid(sdata, key, tissue, image_mask_key, side_s0)
            tiles = _grid_tiles(key, tissue.extent(H, W, levels), side_s0, origin=origin)
        else:
            tiles = _centred_tiles(sdata, key, centered_on, side_s0)
        tiles.clamp(H, W)
        fraction, specimen = tissue.fraction(tiles.full)
        downsample = levels[level][3]
        built.append(
            _ImageTiles(
                image_key=key,
                shapes_key=shapes_key,
                tiles=tiles,
                fraction=fraction,
                specimen=specimen,
                in_tissue=fraction >= min_tissue_fraction,
                min_tissue_fraction=min_tissue_fraction,
                downsample=downsample,
                info={
                    **info,
                    "units": "um" if tile_size_um is not None and mpp0 is not None else "px",
                    "mpp_source": mpp_source,
                    "mpp0": mpp0,
                    "level": levels[level][0],
                    "downsample": downsample,
                    "tile_size_px_level": side_level,
                    "tile_size_px_scale0": list(side_s0),
                    "tissue_source": tissue.source,
                    "n_tiles": len(fraction),
                    "n_in_tissue": int((fraction >= min_tissue_fraction).sum()),
                },
            )
        )

    for item in built:
        _write_tile_shapes(sdata, item)
    params = {
        "tile_size": list(size_px),
        "tile_size_um": tile_size_um,
        "target_mpp": target_mpp,
        "centered_on": centered_on,
        "min_tissue_fraction": min_tissue_fraction,
        "center_grid_on_tissue": center_grid_on_tissue,
        "table_key": table_key,
    }
    _write_tile_table(sdata, built, table_name=table_name, params=params)
    if table_key is not None:
        _link_observations(sdata, built, table_key=table_key, tiles_table=table_name, centered_on=centered_on)

    if preview:
        first = built[0]
        try:
            (
                sdata.pl.render_images(first.image_key)
                .pl.render_shapes(first.shapes_key, color="tile_classification", fill_alpha=0.5)
                .pl.show()
            )
        except (AttributeError, KeyError, ValueError) as e:
            logger.warning(f"Could not generate preview plot: {e}")


# ---------------------------------------------------------------------------
# make_tiles helpers
# ---------------------------------------------------------------------------


def _as_pair(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int | np.integer):
        return int(size), int(size)
    return int(size[0]), int(size[1])


def _image_levels(node: Any) -> list[tuple[str, int, int, float]]:
    """Return ``(name, height, width, downsample)`` for every pyramid level, finest first."""
    if hasattr(node, "keys"):
        dims = [(k, *_yx_from_shape(node[k].image.shape)) for k in node.keys()]
    else:
        dims = [("scale0", *_yx_from_shape(node.shape))]
    dims.sort(key=lambda d: -d[1] * d[2])
    H0, W0 = dims[0][1], dims[0][2]
    return [(name, h, w, float(np.mean([H0 / h, W0 / w]))) for name, h, w in dims]


def _affine_to_image_frame(sdata: sd.SpatialData, element_key: str, image_key: str) -> np.ndarray:
    """Return the 3x3 affine that maps an element's native coordinates to the image's pixel frame."""
    element = sdata[element_key]
    el_tfs = get_transformation(element, get_all=True)
    img_tfs = get_transformation(sdata.images[image_key], get_all=True)
    shared = [cs for cs in el_tfs if cs in img_tfs]
    if not shared:
        raise ValueError(
            f"Element '{element_key}' and image '{image_key}' share no coordinate system "
            f"(element: {sorted(el_tfs)}, image: {sorted(img_tfs)})."
        )
    cs = "global" if "global" in shared else shared[0]
    to_cs = np.asarray(el_tfs[cs].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=float)
    img_to_cs = np.asarray(img_tfs[cs].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=float)
    return np.linalg.inv(img_to_cs) @ to_cs


def _resolve_mpp(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    image_mpp: float | Mapping[str, float] | None,
    mpp_from: str | Mapping[str, str] | None,
) -> tuple[float | None, str | None]:
    """Return the full-resolution pixel size of an image and where it came from, or ``(None, None)``."""
    if image_mpp is not None:
        value = image_mpp[image_key] if isinstance(image_mpp, Mapping) else image_mpp
        assert_positive(value, name="image_mpp")
        return float(value), "user"
    if mpp_from is not None:
        element_key = mpp_from[image_key] if isinstance(mpp_from, Mapping) else mpp_from
        A = _affine_to_image_frame(sdata, element_key, image_key)[:2, :2]
        s1, s2 = np.linalg.svd(A, compute_uv=False)
        if abs(s1 - s2) / max(s1, s2) > 1e-3:
            raise ValueError(
                f"The transformation from '{element_key}' to image '{image_key}' is anisotropic "
                f"(scales {s1:.6g}, {s2:.6g}); a single pixel size is not defined."
            )
        return 1.0 / float(s1), f"element:{element_key}"
    return None, None


def _tile_geometry(
    image_key: str,
    levels: list[tuple[str, int, int, float]],
    *,
    mpp0: float | None,
    size_px: tuple[int, int],
    tile_size_um: float | None,
    target_mpp: float | None,
) -> tuple[int, tuple[int, int], list[int], dict[str, Any]]:
    """Choose the pyramid level and the tile side on it and on the full-resolution image."""
    if tile_size_um is None or mpp0 is None:
        if tile_size_um is not None:
            logger.warning(
                f"Pixel size of image '{image_key}' is unknown, so `tile_size_um`/`target_mpp` are ignored and tiles "
                f"are {size_px} pixels. Pass `image_mpp` (e.g. from `derive_mpp_from_shapes`) or `mpp_from`."
            )
        return 0, size_px, list(size_px), {}
    mpps = np.array([mpp0 * d for *_, d in levels])
    level = 0 if target_mpp is None else int(np.argmin(np.abs(np.log(mpps / target_mpp))))
    side_level = max(1, round(tile_size_um / mpps[level]))
    side_s0 = max(1, round(side_level * levels[level][3]))
    info: dict[str, Any] = {"tile_size_um": tile_size_um, "actual_mpp": float(mpps[level])}
    if target_mpp is not None:
        info["target_mpp"] = target_mpp
        if abs(np.log2(mpps[level] / target_mpp)) > 0.5:
            logger.warning(
                f"Image '{image_key}': the closest pyramid level has {mpps[level]:.3g} um/px, far from "
                f"`target_mpp={target_mpp}`. Tiles are not resampled."
            )
    return level, (side_s0, side_s0), [side_level, side_level], info


class _Tiles:
    """Tile bounds ``(y0, x0, y1, x1)`` on the full-resolution image, with names and grid indices."""

    def __init__(self, bounds: np.ndarray, names: list[str], grid: tuple[np.ndarray, np.ndarray, Any] | None = None):
        self.bounds = bounds.astype(np.int64)
        self.full = self.bounds.copy()  # unclamped, for areas and centres
        self.names = names
        self.grid = grid  # (iy, ix, (y_start, x_start, ty, tx, ny, nx)) for grid tiles
        self.centered_on_ids: np.ndarray | None = None

    def clamp(self, H: int, W: int) -> None:
        self.bounds[:, [0, 2]] = np.clip(self.bounds[:, [0, 2]], 0, H)
        self.bounds[:, [1, 3]] = np.clip(self.bounds[:, [1, 3]], 0, W)

    def centres(self) -> np.ndarray:
        """Centres as ``(x, y)`` in pixels of the full-resolution image."""
        return np.column_stack([(self.full[:, 1] + self.full[:, 3]) / 2, (self.full[:, 0] + self.full[:, 2]) / 2])


def _grid_tiles(
    image_key: str, extent: tuple[int, int, int, int], side: tuple[int, int], *, origin: tuple[int, int] | None
) -> _Tiles:
    """Regular grid of ``side`` tiles covering ``extent = (y0, x0, y1, x1)``, anchored at ``origin`` if given."""
    ty, tx = side
    y0, x0, y1, x1 = extent
    if origin is not None:  # shift the grid phase so that a tile starts at `origin`
        y0 = origin[0] - int(np.ceil((origin[0] - y0) / ty)) * ty
        x0 = origin[1] - int(np.ceil((origin[1] - x0) / tx)) * tx
    ny = max(1, int(np.ceil((y1 - y0) / ty)))
    nx = max(1, int(np.ceil((x1 - x0) / tx)))
    iy, ix = np.divmod(np.arange(ny * nx), nx)
    bounds = np.column_stack([y0 + iy * ty, x0 + ix * tx, y0 + (iy + 1) * ty, x0 + (ix + 1) * tx])
    names = [f"{image_key}_x{x}_y{y}" for y, x in zip(iy, ix, strict=True)]
    return _Tiles(bounds, names, grid=(iy, ix, (y0, x0, ty, tx, ny, nx)))


def _centred_tiles(sdata: sd.SpatialData, image_key: str, shapes_key: str, side: tuple[int, int]) -> _Tiles:
    """One tile of ``side`` per shape, centred on the shape's centroid in the image's pixel frame."""
    gdf = sdata.shapes[shapes_key]
    cen = gdf.geometry.centroid
    xy = (
        np.column_stack([cen.x.to_numpy(), cen.y.to_numpy(), np.ones(len(gdf))])
        @ _affine_to_image_frame(sdata, shapes_key, image_key).T
    )
    ty, tx = side
    y0 = np.round(xy[:, 1] - ty / 2).astype(np.int64)
    x0 = np.round(xy[:, 0] - tx / 2).astype(np.int64)
    if len(gdf) > 1_000_000:
        logger.warning(
            f"`centered_on='{shapes_key}'` creates {len(gdf)} overlapping crops; a grid (`centered_on=None`) is "
            "usually better for bins or cells."
        )
    ids = gdf.index.to_numpy()
    tiles = _Tiles(np.column_stack([y0, x0, y0 + ty, x0 + tx]), [f"{image_key}_{i}" for i in ids])
    tiles.centered_on_ids = ids
    return tiles


class _Tissue:
    """Tissue of one image, as polygons in the image's pixel frame or as a labels mask."""

    def __init__(self, source: str, polygons: np.ndarray | None = None, ids: np.ndarray | None = None, label_node=None):
        self.source = source
        self.polygons = polygons
        self.ids = ids
        self.label_node = label_node

    def extent(self, H: int, W: int, levels: list[tuple[str, int, int, float]]) -> tuple[int, int, int, int]:
        """Bounding box ``(y0, x0, y1, x1)`` of the tissue on the full-resolution image."""
        if self.polygons is not None:
            if len(self.polygons) == 0:
                logger.warning(f"Tissue '{self.source}' is empty; the grid covers the whole image.")
                return 0, 0, H, W
            minx, miny, maxx, maxy = shapely.total_bounds(self.polygons)
            box = (int(np.floor(miny)), int(np.floor(minx)), int(np.ceil(maxy)), int(np.ceil(maxx)))
        else:
            mask, d = self._coarsest_mask()
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                logger.warning(f"Tissue '{self.source}' is empty; the grid covers the whole image.")
                return 0, 0, H, W
            box = (
                int(np.floor(ys.min() * d)),
                int(np.floor(xs.min() * d)),
                int(np.ceil((ys.max() + 1) * d)),
                int(np.ceil((xs.max() + 1) * d)),
            )
        return max(0, box[0]), max(0, box[1]), min(H, box[2]), min(W, box[3])

    def centroid(self) -> tuple[float, float] | None:
        """Tissue centroid ``(y, x)`` on the full-resolution image."""
        if self.polygons is not None:
            if len(self.polygons) == 0:
                return None
            c = shapely.centroid(shapely.union_all(self.polygons))
            return float(c.y), float(c.x)
        mask, d = self._coarsest_mask()
        ys, xs = np.nonzero(mask)
        return None if ys.size == 0 else (float((ys.mean() + 0.5) * d), float((xs.mean() + 0.5) * d))

    def _label_levels(self) -> list[tuple[str, int, int]]:
        node = self.label_node
        if hasattr(node, "keys"):
            dims = [(k, *_yx_from_shape(node[k].image.shape)) for k in node.keys()]
        else:
            dims = [("scale0", *_yx_from_shape(node.shape))]
        return sorted(dims, key=lambda d: -d[1] * d[2])

    def _read_level(self, name: str) -> np.ndarray:
        node = self.label_node
        arr = node[name].image if hasattr(node, "keys") else node
        return np.asarray(arr.data.compute() if is_dask_collection(arr.data) else arr.data).squeeze() > 0

    def _coarsest_mask(self) -> tuple[np.ndarray, float]:
        dims = self._label_levels()
        name, h, w = dims[-1]
        return self._read_level(name), dims[0][1] / h * self._label_scale0_factor

    _label_scale0_factor: float = 1.0  # labels level 0 / image level 0, set by _resolve_tissue

    def fraction(self, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tissue fraction and largest-overlap specimen id for every tile."""
        n = len(bounds)
        specimen = np.full(n, None, dtype=object)
        if self.polygons is not None:
            return self._fraction_polygons(bounds, specimen)
        return self._fraction_raster(bounds), specimen

    def _fraction_polygons(self, full: np.ndarray, specimen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(full)
        fraction = np.zeros(n)
        if n == 0 or len(self.polygons) == 0:
            return fraction, specimen
        boxes = shapely.box(full[:, 1], full[:, 0], full[:, 3], full[:, 2])
        tile_area = shapely.area(boxes)
        tree = shapely.STRtree(self.polygons)
        t_in, p_in = tree.query(boxes, predicate="within")
        fraction[t_in] = 1.0
        specimen[t_in] = self.ids[p_in]
        t, p = tree.query(boxes, predicate="intersects")
        boundary = ~np.isin(t, t_in)
        t, p = t[boundary], p[boundary]
        if len(t):
            area = shapely.area(shapely.intersection(boxes[t], self.polygons[p]))
            fraction += np.bincount(t, weights=area, minlength=n) / tile_area
            order = np.lexsort((-area, t))  # per tile, largest overlap first
            first = order[np.r_[True, t[order][1:] != t[order][:-1]]]
            has_area = area[first] > 0
            specimen[t[first][has_area]] = self.ids[p[first][has_area]]
        return np.clip(fraction, 0.0, 1.0), specimen

    def _fraction_raster(self, full: np.ndarray) -> np.ndarray:
        dims = self._label_levels()
        side = float(np.min(full[:, 2:] - full[:, :2])) if len(full) else 1.0
        # coarsest level with at least 8 mask pixels per tile side; the finest level otherwise
        candidates = [d for d in dims if side / (dims[0][1] / d[1] * self._label_scale0_factor) >= 8]
        name, h, _ = candidates[-1] if candidates else dims[0]
        d = dims[0][1] / h * self._label_scale0_factor
        mask = self._read_level(name)
        integral = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
        integral[1:, 1:] = mask.cumsum(0).cumsum(1)
        y0 = np.clip(np.floor(full[:, 0] / d).astype(np.int64), 0, mask.shape[0])
        x0 = np.clip(np.floor(full[:, 1] / d).astype(np.int64), 0, mask.shape[1])
        y1 = np.clip(np.ceil(full[:, 2] / d).astype(np.int64), 0, mask.shape[0])
        x1 = np.clip(np.ceil(full[:, 3] / d).astype(np.int64), 0, mask.shape[1])
        tissue = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        area = (np.ceil(full[:, 2] / d) - np.floor(full[:, 0] / d)) * (
            np.ceil(full[:, 3] / d) - np.floor(full[:, 1] / d)
        )
        self.source = f"{self.source}@{name}"
        return np.clip(tissue / np.maximum(area, 1), 0.0, 1.0)


def _resolve_tissue(
    sdata: sd.SpatialData, image_key: str, tissue_mask_key: str | Mapping[str, str] | None, *, scale: str
) -> _Tissue:
    """Find the tissue for an image: polygons in ``sdata.shapes`` first, then a labels mask, else detect it."""
    given = tissue_mask_key[image_key] if isinstance(tissue_mask_key, Mapping) else tissue_mask_key
    key = given or f"{image_key}_tissue"
    if key not in sdata.shapes and key not in sdata.labels:
        if given is not None:
            raise KeyError(f"Tissue '{key}' not found in `sdata.shapes` or `sdata.labels`.")
        from squidpy.experimental.im._detect_tissue import detect_tissue

        logger.info(f"No tissue found for '{image_key}'; running detect_tissue to create '{key}'.")
        detect_tissue(sdata, image_key=image_key, scale=scale, new_labels_key=key, inplace=True)
    if key in sdata.shapes:
        gdf = sdata.shapes[key]
        A = _affine_to_image_frame(sdata, key, image_key)
        geoms = gdf.geometry.to_numpy()
        if not np.allclose(A, np.eye(3)):
            geoms = shapely.transform(geoms, lambda xy: xy @ A[:2, :2].T + A[:2, 2])
        return _Tissue(f"shapes:{key}", polygons=np.asarray(geoms), ids=gdf.index.astype(str).to_numpy())
    tissue = _Tissue(f"labels:{key}", label_node=sdata.labels[key])
    A = _affine_to_image_frame(sdata, key, image_key)
    tissue._label_scale0_factor = float(np.sqrt(abs(np.linalg.det(A[:2, :2]))))
    return tissue


def _grid_origin_on_centroid(
    sdata: sd.SpatialData,
    image_key: str,
    tissue: _Tissue,
    image_mask_key: str | None,
    side: tuple[int, int],
) -> tuple[int, int] | None:
    """Top-left corner of the tile that is centred on the tissue (or ``image_mask_key``) centroid."""
    source = tissue if image_mask_key is None else _resolve_tissue(sdata, image_key, image_mask_key, scale="auto")
    centroid = source.centroid()
    if centroid is None:
        logger.warning("Mask is empty; the grid is not centred on tissue.")
        return None
    return int(round(centroid[0] - side[0] / 2)), int(round(centroid[1] - side[1] / 2))


@dataclass
class _ImageTiles:
    image_key: str
    shapes_key: str
    tiles: _Tiles
    fraction: np.ndarray
    specimen: np.ndarray
    in_tissue: np.ndarray
    min_tissue_fraction: float
    downsample: float
    info: dict[str, Any]

    def classification(self) -> pd.Categorical:
        labels = np.where(self.in_tissue, "tissue", np.where(self.fraction > 0, "partial_tissue", "background"))
        return pd.Categorical(labels, categories=["background", "partial_tissue", "tissue"])

    def level_bounds(self) -> np.ndarray:
        d = self.downsample
        b = self.tiles.bounds
        return np.column_stack([np.floor(b[:, :2] / d), np.ceil(b[:, 2:] / d)]).astype(np.int64)


def _write_tile_shapes(sdata: sd.SpatialData, item: _ImageTiles) -> None:
    b = item.tiles.bounds
    lb = item.level_bounds()
    columns: dict[str, Any] = {"tile_id": item.tiles.names}
    if item.tiles.grid is not None:
        columns["tile_y"], columns["tile_x"] = item.tiles.grid[0], item.tiles.grid[1]
    else:
        columns["centered_on_id"] = item.tiles.centered_on_ids
    columns.update(
        {
            "pixel_y0": b[:, 0],
            "pixel_x0": b[:, 1],
            "pixel_y1": b[:, 2],
            "pixel_x1": b[:, 3],
            "level_y0": lb[:, 0],
            "level_x0": lb[:, 1],
            "level_y1": lb[:, 2],
            "level_x1": lb[:, 3],
            "tissue_fraction": item.fraction.astype(np.float32),
            "in_tissue": item.in_tissue,
            "specimen_id": pd.array(item.specimen, dtype="string"),
            "tile_classification": item.classification(),
        }
    )
    gdf = gpd.GeoDataFrame(columns, geometry=list(shapely.box(b[:, 1], b[:, 0], b[:, 3], b[:, 2])))
    gdf.index = pd.Index(item.tiles.names)
    sdata.shapes[item.shapes_key] = ShapesModel.parse(gdf)
    set_transformation(
        sdata.shapes[item.shapes_key], get_transformation(sdata.images[item.image_key], get_all=True), set_all=True
    )
    logger.info(f"Saved {len(gdf)} tiles as 'sdata.shapes[\"{item.shapes_key}\"]'")


def _write_tile_table(sdata: sd.SpatialData, built: list[_ImageTiles], *, table_name: str, params: dict) -> None:
    obs = pd.concat(
        [
            pd.DataFrame(
                {
                    "slide": item.image_key,
                    "region": item.shapes_key,
                    "tile_id": item.tiles.names,
                    "tissue_fraction": item.fraction.astype(np.float32),
                    "in_tissue": item.in_tissue,
                    "specimen_id": pd.array(item.specimen, dtype="string"),
                }
            )
            for item in built
        ],
        ignore_index=True,
    )
    obs.index = obs["tile_id"].astype(str).to_numpy()
    obs["slide"] = pd.Categorical(obs["slide"], categories=[i.image_key for i in built])
    obs["region"] = pd.Categorical(obs["region"], categories=[i.shapes_key for i in built])
    adata = AnnData(obs=obs, obsm={"spatial": np.concatenate([i.tiles.centres() for i in built])})
    adata.uns["make_tiles"] = {
        "params": _drop_none(params),
        "per_image": {i.image_key: _drop_none(i.info) for i in built},
    }
    sdata.tables[table_name] = TableModel.parse(
        adata, region=[i.shapes_key for i in built], region_key="region", instance_key="tile_id"
    )
    logger.info(f"Saved the tiles table as 'sdata.tables[\"{table_name}\"]'")


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _link_observations(
    sdata: sd.SpatialData, built: list[_ImageTiles], *, table_key: str, tiles_table: str, centered_on: str | None
) -> None:
    """Write ``obs["tile_id"]`` onto an omics table: the tile under each observation's centroid, else NA."""
    adata = sdata.tables[table_key]
    attrs = adata.uns[TableModel.ATTRS_KEY]
    regions = attrs[TableModel.REGION_KEY]
    regions = [regions] if isinstance(regions, str) else list(regions)
    region_key, instance_key = attrs[TableModel.REGION_KEY_KEY], attrs[TableModel.INSTANCE_KEY]
    tile_id = pd.Series(pd.NA, index=adata.obs_names, dtype="string")
    for region in regions:
        matches = [i for i in built if _shares_coordinate_system(sdata, region, i.image_key)]
        if len(matches) > 1:
            raise ValueError(
                f"Region '{region}' of table '{table_key}' shares a coordinate system with several images "
                f"({[i.image_key for i in matches]}); link the images one at a time."
            )
        if not matches:
            logger.warning(f"Region '{region}' shares no coordinate system with the tiled images; its tile_id is NA.")
            continue
        item = matches[0]
        rows = (adata.obs[region_key] == region).to_numpy()
        instances = adata.obs.loc[rows, instance_key].to_numpy()
        if centered_on == region:
            tile_id[rows] = [f"{item.image_key}_{i}" for i in instances]
            continue
        cs = _shared_cs(sdata, region, item.image_key)
        cen = get_centroids(sdata[region], coordinate_system=cs)
        cen = cen.compute() if hasattr(cen, "compute") else cen
        img_to_cs = get_transformation(sdata.images[item.image_key], to_coordinate_system=cs)
        inv = np.linalg.inv(np.asarray(img_to_cs.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))))
        xy = cen.loc[instances, ["x", "y"]].to_numpy(dtype=float)
        xy = np.column_stack([xy, np.ones(len(xy))]) @ inv.T
        tile_id[rows] = _tile_under(item.tiles, xy[:, 0], xy[:, 1])
    if "tile_id" in adata.obs:
        logger.warning(f"Overwriting `sdata.tables['{table_key}'].obs['tile_id']`.")
    adata.obs["tile_id"] = tile_id.to_numpy()
    adata.uns["make_tiles_link"] = {"tiles_key": tiles_table}


def _tile_under(tiles: _Tiles, x: np.ndarray, y: np.ndarray) -> pd.arrays.StringArray:
    """Name of the tile that contains each point (``(x, y)`` in image pixels), NA outside any tile."""
    names = np.asarray(tiles.names, dtype=object)
    out = np.full(len(x), None, dtype=object)
    if tiles.grid is not None:
        y0, x0, ty, tx, ny, nx = tiles.grid[2]
        iy = np.floor((y - y0) / ty).astype(np.int64)
        ix = np.floor((x - x0) / tx).astype(np.int64)
        ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        idx = iy[ok] * nx + ix[ok]
        b = tiles.bounds[idx]
        inside = (y[ok] >= b[:, 0]) & (y[ok] < b[:, 2]) & (x[ok] >= b[:, 1]) & (x[ok] < b[:, 3])
        hit = np.flatnonzero(ok)[inside]
        out[hit] = names[idx[inside]]
    elif len(names):
        centres = tiles.centres()
        half = (tiles.full[:, 2:] - tiles.full[:, :2]).max(axis=1) / 2
        dist, idx = cKDTree(centres).query(np.column_stack([x, y]), p=np.inf)
        b = tiles.bounds[idx]
        inside = (dist <= half[idx]) & (y >= b[:, 0]) & (y < b[:, 2]) & (x >= b[:, 1]) & (x < b[:, 3])
        out[inside] = names[idx[inside]]
    return pd.array(out, dtype="string")


def _shared_cs(sdata: sd.SpatialData, element_key: str, image_key: str) -> str:
    el = get_transformation(sdata[element_key], get_all=True)
    img = get_transformation(sdata.images[image_key], get_all=True)
    shared = [cs for cs in el if cs in img]
    return "global" if "global" in shared else shared[0]


def _shares_coordinate_system(sdata: sd.SpatialData, element_key: str, image_key: str) -> bool:
    el = get_transformation(sdata[element_key], get_all=True)
    img = get_transformation(sdata.images[image_key], get_all=True)
    return any(cs in img for cs in el)


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

    :func:`make_tiles` with ``centered_on="spots"`` does the same with an explicit tile size, keeps every tile
    in a table and can link the tiles to the observations. This function stays for compatibility.

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

    assert_key_in_sdata(sdata, spots_key, attr="shapes")
    if image_key is not None:
        assert_key_in_sdata(sdata, image_key, attr="images")
    assert_in_range(min_tissue_fraction, 0, 1, name="min_tissue_fraction")

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

    coords, spot_ids = _get_spot_coordinates(sdata, spots_key=spots_key)
    derived_tile = _derive_tile_size_from_spots(coords)
    logger.info(f"Derived tile size {derived_tile} from {len(coords)} Visium spots (key='{spots_key}').")

    tg = _SpotTileGrid(centers=coords, tile_size=derived_tile, spot_ids=spot_ids)
    shapes_key = new_shapes_key or f"{spots_key}_tiles"
    _save_spot_tiles_to_shapes(sdata, tg=tg, shapes_key=shapes_key, spot_ids=spot_ids, source_shapes_key=spots_key)

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
                from squidpy.experimental.im._detect_tissue import detect_tissue

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
            mask_key = classification_mask_key
            if mask_key in sdata.labels:
                target_hw = _get_largest_scale_dimensions(sdata, image_key=image_key)
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
            assert_key_in_sdata(sdata, classification_mask_key, attr="labels")
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
        _propagate_spot_classification(sdata, tiles_key=shapes_key, spots_key=spots_key)
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
    *,
    tg: TileGrid,
    image_key: str | None,
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
    assert_key_in_sdata(sdata, mask_key, attr="labels")
    mask = get_mask_materialized(sdata, mask_key=mask_key, scale=scale)
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


def _get_spot_coordinates(
    sdata: sd.SpatialData,
    *,
    spots_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract spot centers (x, y) and IDs from a shapes table.

    Callers must validate *spots_key* before calling this helper.
    """
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
    transformations = get_transformation(element, get_all=True)
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
