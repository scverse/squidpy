from __future__ import annotations

from typing import Any, Literal

import dask.array as da
import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from shapely import box
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Labels2DModel, ShapesModel
from spatialdata.transformations import Scale, Sequence, get_transformation, set_transformation

from squidpy._utils import _yx_from_shape
from squidpy._validators import normalize_choice


class TileGrid:
    """Immutable tile grid definition with cached bounds and centroids."""

    def __init__(
        self,
        H: int,
        W: int,
        *,
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


def get_element_data(
    element_node: Any,
    scale: str | Literal["auto"],
    element_type: str = "element",
    element_key: str = "",
    *,
    prefer: Literal["coarsest", "finest"] = "coarsest",
) -> xr.DataArray:
    """
    Extract data array from a spatialdata element (image or label) node.
    Supports multiscale and single-scale elements.

    Parameters
    ----------
    element_node
        The element node from sdata.images[key] or sdata.labels[key]
    scale
        Scale level to use, or "auto" to pick a level according to ``prefer``.
    element_type
        Type of element for error messages (e.g., "image", "label").
    element_key
        Key of the element for error messages.
    prefer
        Which level ``scale="auto"`` resolves to on a multiscale element:
        ``"coarsest"`` (smallest, cheapest) or ``"finest"`` (full resolution).
        Ignored for explicit scales and single-scale elements.

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

            chosen = min(available, key=_idx) if prefer == "finest" else max(available, key=_idx)
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


def flatten_channels(
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
    channel_format = normalize_choice(channel_format, ("infer", "rgb", "rgba", "multichannel"), name="channel_format")
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

    elif n_channels == 3:
        # 3 channels + infer -> RGB luminance formula
        weights = xr.DataArray([0.299, 0.587, 0.114], dims=["c"], coords={"c": img.coords["c"]})
        return (img * weights).sum(dim="c")
    else:
        # 2 channels or 4+ channels + infer -> multichannel
        return img.mean(dim="c")


def get_mask_dask(sdata: SpatialData, *, mask_key: str, scale: str) -> da.Array:
    """Extract mask as a lazy dask array from ``sdata.labels``."""
    if mask_key not in sdata.labels:
        raise KeyError(f"Mask key '{mask_key}' not found in sdata.labels")
    label_node = sdata.labels[mask_key]
    mask_xr = get_element_data(label_node, scale, "label", mask_key)

    arr = mask_xr.data if hasattr(mask_xr, "data") else mask_xr
    if not isinstance(arr, da.Array):
        arr = da.from_array(np.asarray(arr))

    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask with shape (y, x), got shape {arr.shape}")
    return arr


def get_mask_materialized(sdata: SpatialData, *, mask_key: str, scale: str) -> np.ndarray:
    """Extract a 2D mask array from ``sdata.labels`` at the requested scale (materialized)."""
    arr = get_mask_dask(sdata, mask_key=mask_key, scale=scale)
    return np.asarray(arr.compute())


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


def resolve_tissue_mask(
    sdata: SpatialData,
    *,
    image_key: str,
    scale: str,
    tissue_mask_key: str | None = None,
    auto_create: bool = True,
) -> str:
    """Return the key of a tissue mask, creating one if needed.

    A tissue mask is either a shapes element of tissue polygons (what :func:`detect_tissue` writes) or,
    for objects made with older versions, a labels element. If *tissue_mask_key* is given and exists, it
    is returned as-is. Otherwise falls back to ``f"{image_key}_tissue"``. When that key is missing,
    the behaviour depends on *auto_create*: if ``True`` ``detect_tissue`` is run to create it; if
    ``False`` a :class:`KeyError` is raised asking the caller to run ``detect_tissue`` first.

    Raises
    ------
    KeyError
        If *tissue_mask_key* is given but not found in ``sdata.shapes`` or ``sdata.labels``, or if
        *auto_create* is ``False`` and no tissue mask exists.
    Exception
        Any exception raised by ``detect_tissue`` if auto-creation fails.
        Callers needing graceful fallback should wrap this call in try/except.
    """

    def _exists(key: str) -> bool:
        return key in sdata.shapes or key in sdata.labels

    if tissue_mask_key is not None:
        if not _exists(tissue_mask_key):
            raise KeyError(f"Tissue mask key '{tissue_mask_key}' not found in sdata.labels or sdata.shapes")
        return tissue_mask_key

    mask_key = f"{image_key}_tissue"
    if not _exists(mask_key):
        if not auto_create:
            raise KeyError(
                f"No tissue mask found (looked for {mask_key!r} in sdata.shapes and sdata.labels). Run "
                f"`squidpy.experimental.im.detect_tissue(sdata, image_key={image_key!r})` first, "
                "or pass an explicit `tissue_mask_key`."
            )
        from squidpy.experimental.im._detect_tissue import detect_tissue

        detect_tissue(sdata=sdata, image_key=image_key, scale=scale, inplace=True, new_shapes_key=mask_key)
    elif mask_key in sdata.labels:
        logger.info(
            f"Using the raster tissue mask 'sdata.labels[{mask_key!r}]'. Re-run `detect_tissue` to get tissue polygons."
        )
    return mask_key


def scaled_image_transformations(sdata: SpatialData, *, image_key: str, hw: tuple[int, int]) -> dict[str, Any]:
    """Transformations of ``sdata.images[image_key]`` for an element laid out in pixels of an ``hw``-sized copy.

    The image's transformations map ``scale0`` pixels, so an element in coarser pixels first gets scaled up
    to ``scale0`` pixels.
    """
    image_node = sdata.images[image_key]
    transformations = get_transformation(image_node, get_all=True)
    full = get_element_data(image_node, "scale0", "image", image_key)
    sy, sx = full.sizes["y"] / hw[0], full.sizes["x"] / hw[1]
    if (sy, sx) == (1.0, 1.0):
        return transformations
    to_scale0 = Scale([sx, sy], axes=("x", "y"))
    return {cs: Sequence([to_scale0, t]) for cs, t in transformations.items()}


def _polygonal(geom: Any) -> Any:
    """Keep only the polygon parts of a geometry (``make_valid`` can return collections with lines)."""
    import shapely

    polys = []
    for part in shapely.get_parts(geom):
        polys.extend(p for p in shapely.get_parts(part) if p.geom_type == "Polygon")
    return polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)


def labels_to_polygons(
    labels: np.ndarray, *, scale_xy: tuple[float, float], tolerance: float = 1.0
) -> gpd.GeoDataFrame:
    """Vectorise integer labels ``1..K`` into one (Multi)Polygon per label, holes kept.

    Each row's runs of a label become unit boxes whose union follows the pixel edges exactly, so the result
    is valid and keeps holes. It is then simplified by ``tolerance`` label pixels and scaled by ``scale_xy``
    (``x``, ``y``) into the target pixel frame.
    """
    import shapely
    from scipy import ndimage as ndi

    ids: list[int] = []
    geoms: list[Any] = []
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        runs = np.diff(np.pad(labels[sl] == lab, ((0, 0), (1, 1))).astype(np.int8), axis=1)
        rows, starts = np.nonzero(runs == 1)
        _, ends = np.nonzero(runs == -1)  # row-major order, so starts and ends pair up per row
        y0, x0 = sl[0].start, sl[1].start
        geom = shapely.union_all(shapely.box(starts + x0, rows + y0, ends + x0, rows + y0 + 1))
        if tolerance > 0:
            geom = shapely.simplify(geom, tolerance, preserve_topology=True)
        geom = _polygonal(shapely.make_valid(geom))
        ids.append(lab)
        geoms.append(shapely.transform(geom, lambda c: c * np.asarray(scale_xy, dtype=float)))
    gdf = gpd.GeoDataFrame({"specimen_id": np.asarray(ids, dtype=np.int64)}, geometry=geoms, index=ids)
    gdf["area_px"] = gdf.area
    return gdf


def rasterize_polygons(
    gdf: gpd.GeoDataFrame, hw: tuple[int, int], *, scale_xy: tuple[float, float], chunks: Any = "auto"
) -> da.Array:
    """Lazily draw polygons given in ``scale0`` pixels onto an ``hw`` grid of coarser pixels; boolean ``(y, x)``.

    One output pixel spans ``scale_xy`` (``x``, ``y``) ``scale0`` pixels; a pixel is ``True`` when its centre lies
    inside a polygon and outside its holes. Each chunk only draws the polygons that touch it.
    """
    import shapely
    from skimage.draw import polygon as draw_polygon

    geoms = list(gdf.geometry)
    tree = shapely.STRtree(geoms)
    sx, sy = scale_xy

    def _block(block_info: dict[Any, Any] | None = None) -> np.ndarray:
        (y0, y1), (x0, x1) = block_info[None]["array-location"]
        out = np.zeros((y1 - y0, x1 - x0), dtype=bool)

        def _fill(target: np.ndarray, ring: Any, value: bool) -> None:
            c = np.asarray(ring.coords)
            rr, cc = draw_polygon(c[:, 1] / sy - y0 - 0.5, c[:, 0] / sx - x0 - 0.5, shape=target.shape)
            target[rr, cc] = value

        for i in tree.query(shapely.box(x0 * sx, y0 * sy, x1 * sx, y1 * sy)):
            for poly in shapely.get_parts(geoms[i]):
                # each part on its own canvas, then OR-ed in, so a part lying in another part's hole survives
                part = np.zeros_like(out)
                _fill(part, poly.exterior, True)
                for hole in poly.interiors:
                    _fill(part, hole, False)
                out |= part
        return out

    norm = da.core.normalize_chunks(chunks, hw, dtype=bool)
    return da.map_blocks(_block, chunks=norm, dtype=bool, meta=np.empty((0, 0), dtype=bool))


def tissue_polygons_in_image_pixels(sdata: SpatialData, *, mask_key: str, image_key: str) -> gpd.GeoDataFrame:
    """Tissue polygons from ``sdata.shapes[mask_key]`` mapped into the ``scale0`` pixel frame of an image.

    The mapping goes through the coordinate systems the two elements share, so polygons stored in another
    frame (e.g. micrometres) line up with the image.
    """
    import shapely
    from spatialdata.transformations import get_transformation_between_coordinate_systems

    tissue = sdata.shapes[mask_key]
    m = get_transformation_between_coordinate_systems(sdata, tissue, sdata.images[image_key]).to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    if np.allclose(m, np.eye(3)):
        return tissue
    return tissue.set_geometry(shapely.transform(tissue.geometry.values, lambda c: c @ m[:2, :2].T + m[:2, 2]))


def tissue_mask_raster(
    sdata: SpatialData, *, mask_key: str, image_key: str, target_hw: tuple[int, int], chunks: Any = "auto"
) -> da.Array:
    """Boolean tissue mask on the pixel grid of an image level of shape ``target_hw``, lazy.

    Works for tissue polygons in ``sdata.shapes`` (mapped into the image's pixels through the coordinate
    systems, then drawn on demand) and for raster masks in ``sdata.labels`` (the label level closest to
    ``target_hw``, resized with nearest neighbours if it doesn't match).
    """
    if mask_key in sdata.shapes:
        tissue = tissue_polygons_in_image_pixels(sdata, mask_key=mask_key, image_key=image_key)
        full = get_element_data(sdata.images[image_key], "scale0", "image", image_key)
        scale_xy = (full.sizes["x"] / target_hw[1], full.sizes["y"] / target_hw[0])
        return rasterize_polygons(tissue, target_hw, scale_xy=scale_xy, chunks=chunks)

    label_node = sdata.labels[mask_key]
    mask = get_mask_dask(sdata, mask_key=mask_key, scale=_choose_label_scale_for_image(label_node, target_hw)) > 0
    if tuple(mask.shape) != tuple(target_hw):
        from skimage.transform import resize

        mask = da.from_array(resize(np.asarray(mask), target_hw, order=0, preserve_range=True) > 0.5)
    return mask.rechunk(chunks)


def save_tile_grid_to_shapes(
    sdata: SpatialData,
    *,
    tg: TileGrid,
    shapes_key: str,
    copy_transforms_from_key: str | None = None,
    scale: str = "scale0",
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
    scale
        Scale level of that image whose pixels ``tg`` is laid out in. The image's
        transformations map ``scale0`` pixels, so a grid on a coarser level first gets
        scaled up to ``scale0`` pixels.
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
        if scale == "scale0":
            transformations = get_transformation(sdata.images[copy_transforms_from_key], get_all=True)
        else:
            transformations = scaled_image_transformations(sdata, image_key=copy_transforms_from_key, hw=(tg.H, tg.W))
        set_transformation(sdata.shapes[shapes_key], transformations, set_all=True)
    logger.info(f"- Saved tile grid as 'sdata.shapes[\"{shapes_key}\"]'")


def _tissue_fraction_polygons(
    bounds: np.ndarray, polygons: np.ndarray, ids: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of each tile covered by tissue polygons, and the specimen with the largest overlap.

    Parameters
    ----------
    bounds
        Tile bounds ``(y0, x0, y1, x1)``, shape ``(n, 4)``, in the same pixel frame as ``polygons``.
    polygons
        Tissue ``Polygon``/``MultiPolygon`` geometries, one per specimen; holes are respected.
    ids
        Specimen id per polygon (e.g. the ``specimen_id`` column of a ``detect_tissue`` result). Defaults
        to the polygon position.

    Returns
    -------
    ``(fraction, specimen_id)``: ``area(tile & tissue) / area(tile)`` clipped to ``[0, 1]``, and the id of
    the specimen with the largest positive overlap (`None` where a tile touches no tissue).
    """
    n = len(bounds)
    fraction = np.zeros(n)
    specimen = np.full(n, None, dtype=object)
    if n == 0 or len(polygons) == 0:
        return fraction, specimen
    ids = np.arange(len(polygons)) if ids is None else np.asarray(ids)
    boxes = shapely.box(bounds[:, 1], bounds[:, 0], bounds[:, 3], bounds[:, 2])
    tile_area = shapely.area(boxes)
    tree = shapely.STRtree(polygons)
    # tiles inside a specimen need no intersection; "within" excludes tiles over holes
    t_in, p_in = tree.query(boxes, predicate="within")
    fraction[t_in] = 1.0
    specimen[t_in] = ids[p_in]
    t, p = tree.query(boxes, predicate="intersects")
    boundary = ~np.isin(t, t_in)
    t, p = t[boundary], p[boundary]
    if len(t):
        area = shapely.area(shapely.intersection(boxes[t], polygons[p]))
        fraction += np.bincount(t, weights=area, minlength=n) / tile_area
        order = np.lexsort((-area, t))  # per tile, largest overlap first
        first = order[np.r_[True, t[order][1:] != t[order][:-1]]]
        has_area = area[first] > 0
        specimen[t[first][has_area]] = ids[p[first][has_area]]
    return np.clip(fraction, 0.0, 1.0), specimen
