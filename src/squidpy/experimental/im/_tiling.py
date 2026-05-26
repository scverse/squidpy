"""Cell-aware tiling for large images.

Splits a label image into overlapping tiles such that every cell is fully
contained in exactly one tile.  Cells are assigned to tiles by centroid:
the tile whose non-overlapping base region contains the centroid owns the
cell.  Non-owned cells are zeroed out in each tile's mask so that
downstream processing never double-counts.

All functions accept pre-computed centroid dicts and image shapes — they
never materialize the full image or label array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
from skimage.measure import regionprops


@dataclass(frozen=True)
class CellInfo:
    """Centroid and bounding box for a single label."""

    label: int
    centroid_y: float
    centroid_x: float
    bbox_h: int  # height of bounding box
    bbox_w: int  # width of bounding box


@dataclass(frozen=True)
class TileSpec:
    """Specification for a single tile.

    Attributes
    ----------
    base
        The non-overlapping region ``(y0, x0, y1, x1)`` used for centroid
        ownership.  Tiles partition the image into a grid of base regions.
    crop
        The extended region ``(y0, x0, y1, x1)`` that includes the overlap
        margin.  This is the actual slice extracted from the image/labels.
    owned_ids
        Label IDs whose centroid falls inside ``base``.  Only these labels
        are kept in the tile's mask; all others are zeroed out.
    """

    base: tuple[int, int, int, int]
    crop: tuple[int, int, int, int]
    owned_ids: frozenset[int]


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------


def compute_cell_info(labels: np.ndarray) -> dict[int, CellInfo]:
    """Compute centroid and bounding-box size for every label from a numpy array.

    Parameters
    ----------
    labels
        2-D integer label image where 0 is background.

    Returns
    -------
    Mapping from label ID to :class:`CellInfo`.
    """
    props = regionprops(labels)
    info: dict[int, CellInfo] = {}
    for p in props:
        min_row, min_col, max_row, max_col = p.bbox
        info[p.label] = CellInfo(
            label=p.label,
            centroid_y=p.centroid[0],
            centroid_x=p.centroid[1],
            bbox_h=max_row - min_row,
            bbox_w=max_col - min_col,
        )
    return info


def compute_cell_info_multiscale(
    labels_node: xr.DataTree,
    target_scale: str = "scale0",
) -> dict[int, CellInfo]:
    """Compute centroids using the coarsest scale of a multiscale label pyramid.

    Reads only the smallest resolution, then scales coordinates to *target_scale*.
    """
    available = list(labels_node.keys())
    if not available:
        return {}

    def _spatial_size(k: str) -> int:
        da = labels_node[k].ds["image"]
        h = int(da.sizes.get("y", da.shape[-2]))
        w = int(da.sizes.get("x", da.shape[-1]))
        return h * w

    coarsest = min(available, key=_spatial_size)
    coarse_da = labels_node[coarsest].ds["image"]
    coarse_labels = np.asarray(coarse_da.values).squeeze()

    if coarse_labels.ndim != 2:
        raise ValueError(f"Expected 2-D labels at scale {coarsest}, got shape {coarse_labels.shape}")

    target_da = labels_node[target_scale].ds["image"]
    target_h, target_w = target_da.sizes.get("y", target_da.shape[-2]), target_da.sizes.get("x", target_da.shape[-1])
    coarse_h, coarse_w = coarse_labels.shape
    scale_y = target_h / coarse_h
    scale_x = target_w / coarse_w

    props = regionprops(coarse_labels)
    return {
        p.label: CellInfo(
            label=p.label,
            centroid_y=p.centroid[0] * scale_y,
            centroid_x=p.centroid[1] * scale_x,
            bbox_h=int(np.ceil((p.bbox[2] - p.bbox[0]) * scale_y)),
            bbox_w=int(np.ceil((p.bbox[3] - p.bbox[1]) * scale_x)),
        )
        for p in props
    }


def compute_cell_info_tiled(
    labels_da: xr.DataArray,
    chunk_size: int = 4096,
) -> dict[int, CellInfo]:
    """Compute centroids by reading label tiles — never materializes the full array.

    For cells spanning multiple chunks, centroids are computed as
    area-weighted means of per-chunk centroids.

    Parameters
    ----------
    labels_da
        2-D (y, x) dask-backed xarray DataArray.
    chunk_size
        Size of chunks to read at a time.
    """
    H = int(labels_da.sizes.get("y", labels_da.shape[-2]))
    W = int(labels_da.sizes.get("x", labels_da.shape[-1]))

    # Per-label accumulators: [sum_y*area, sum_x*area, total_area, min_y, max_y, min_x, max_x]
    stats: dict[int, list[float]] = {}

    for y0 in range(0, H, chunk_size):
        y1 = min(y0 + chunk_size, H)
        for x0 in range(0, W, chunk_size):
            x1 = min(x0 + chunk_size, W)
            chunk = labels_da.isel(y=slice(y0, y1), x=slice(x0, x1)).values
            if chunk.ndim > 2:
                chunk = chunk.squeeze()

            for p in regionprops(chunk):
                lid = p.label
                cy_global = float(p.centroid[0] + y0)
                cx_global = float(p.centroid[1] + x0)
                area = float(p.area)
                min_row = float(p.bbox[0] + y0)
                max_row = float(p.bbox[2] + y0)
                min_col = float(p.bbox[1] + x0)
                max_col = float(p.bbox[3] + x0)

                if lid not in stats:
                    stats[lid] = [cy_global * area, cx_global * area, area, min_row, max_row, min_col, max_col]
                else:
                    s = stats[lid]
                    s[0] += cy_global * area
                    s[1] += cx_global * area
                    s[2] += area
                    s[3] = min(s[3], min_row)
                    s[4] = max(s[4], max_row)
                    s[5] = min(s[5], min_col)
                    s[6] = max(s[6], max_col)

    result: dict[int, CellInfo] = {}
    for lid, s in stats.items():
        if lid == 0:
            continue
        result[lid] = CellInfo(
            label=lid,
            centroid_y=s[0] / s[2],
            centroid_x=s[1] / s[2],
            bbox_h=int(s[4] - s[3]),
            bbox_w=int(s[6] - s[5]),
        )
    return result


# ---------------------------------------------------------------------------
# Tile spec building
# ---------------------------------------------------------------------------


def _auto_margin(cell_info: dict[int, CellInfo]) -> int:
    """Compute the minimum margin that covers the largest cell's half-extent."""
    if not cell_info:
        return 0
    max_extent = max(max(c.bbox_h, c.bbox_w) for c in cell_info.values())
    # Centroid can be at most half a bbox away from the cell's edge.
    # Add 1 pixel for safety (rounding / off-by-one).
    return int(np.ceil(max_extent / 2)) + 1


def build_tile_specs(
    image_shape: tuple[int, int],
    cell_info: dict[int, CellInfo],
    tile_size: int = 2048,
    overlap_margin: int | Literal["auto"] = "auto",
) -> list[TileSpec]:
    """Build tile specifications from pre-computed centroids.

    No pixel data is needed — only the image dimensions and centroid dict.

    Parameters
    ----------
    image_shape
        ``(H, W)`` of the full-resolution image/labels.
    cell_info
        Pre-computed centroids from :func:`compute_cell_info`,
        :func:`compute_cell_info_multiscale`, or :func:`compute_cell_info_tiled`.
    tile_size
        Side length of the non-overlapping base grid cells.
    overlap_margin
        Pixel margin added around each base region.  ``"auto"`` computes the
        minimum margin from the largest cell's bounding box.

    Returns
    -------
    List of :class:`TileSpec`, one per grid cell that owns at least one
    label.  Empty tiles (no cells) are omitted.
    """
    H, W = image_shape
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    if isinstance(overlap_margin, str) and overlap_margin == "auto":
        margin = _auto_margin(cell_info)
    else:
        margin = int(overlap_margin)
    if margin < 0:
        raise ValueError(f"overlap_margin must be non-negative, got {margin}")

    cell_to_tile: dict[int, tuple[int, int]] = {}
    for lid, ci in cell_info.items():
        tile_row = min(int(ci.centroid_y) // tile_size, (H - 1) // tile_size)
        tile_col = min(int(ci.centroid_x) // tile_size, (W - 1) // tile_size)
        cell_to_tile[lid] = (tile_row, tile_col)

    tile_to_cells: dict[tuple[int, int], set[int]] = {}
    for lid, key in cell_to_tile.items():
        tile_to_cells.setdefault(key, set()).add(lid)

    specs: list[TileSpec] = []
    for (row, col), owned in sorted(tile_to_cells.items()):
        by0 = row * tile_size
        bx0 = col * tile_size
        by1 = min(by0 + tile_size, H)
        bx1 = min(bx0 + tile_size, W)

        cy0 = max(by0 - margin, 0)
        cx0 = max(bx0 - margin, 0)
        cy1 = min(by1 + margin, H)
        cx1 = min(bx1 + margin, W)

        specs.append(
            TileSpec(
                base=(by0, bx0, by1, bx1),
                crop=(cy0, cx0, cy1, cx1),
                owned_ids=frozenset(owned),
            )
        )

    return specs


# ---------------------------------------------------------------------------
# Tile extraction
# ---------------------------------------------------------------------------


def extract_tile(
    image: np.ndarray,
    labels: np.ndarray,
    spec: TileSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a tile from numpy arrays, zeroing out non-owned cells.

    Parameters
    ----------
    image
        ``(C, H, W)`` numpy array.
    labels
        ``(H, W)`` numpy label array.
    spec
        Tile specification.

    Returns
    -------
    tile_image, tile_labels
    """
    cy0, cx0, cy1, cx1 = spec.crop
    tile_image = image[:, cy0:cy1, cx0:cx1]
    tile_labels = labels[cy0:cy1, cx0:cx1].copy()
    _zero_non_owned(tile_labels, spec.owned_ids)
    return tile_image, tile_labels


def extract_tile_lazy(
    image_da: xr.DataArray,
    labels_da: xr.DataArray,
    spec: TileSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a tile from dask-backed xarray arrays.

    Materializes only the tile's crop region (~2k×2k), not the full image.

    Parameters
    ----------
    image_da
        ``(c, y, x)`` dask-backed DataArray.
    labels_da
        ``(y, x)`` dask-backed DataArray.
    spec
        Tile specification.

    Returns
    -------
    tile_image
        ``(C, crop_h, crop_w)`` numpy array.
    tile_labels
        ``(crop_h, crop_w)`` numpy array with non-owned cells zeroed.
    """
    cy0, cx0, cy1, cx1 = spec.crop
    tile_image = image_da.isel(y=slice(cy0, cy1), x=slice(cx0, cx1)).values
    tile_labels = labels_da.isel(y=slice(cy0, cy1), x=slice(cx0, cx1)).values.copy()
    if tile_labels.ndim > 2:
        tile_labels = tile_labels.squeeze()
    _zero_non_owned(tile_labels, spec.owned_ids)
    return tile_image, tile_labels


def extract_labels_tile_lazy(
    labels_da: xr.DataArray,
    spec: TileSpec,
) -> np.ndarray:
    """Extract a labels-only tile from a dask-backed DataArray.

    Like :func:`extract_tile_lazy` but skips the image entirely.
    Materializes only the crop region.

    Parameters
    ----------
    labels_da
        ``(y, x)`` dask-backed DataArray.
    spec
        Tile specification.

    Returns
    -------
    ``(crop_h, crop_w)`` numpy array with non-owned cells zeroed.
    """
    cy0, cx0, cy1, cx1 = spec.crop
    tile_labels = labels_da.isel(y=slice(cy0, cy1), x=slice(cx0, cx1)).values.copy()
    if tile_labels.ndim > 2:
        tile_labels = tile_labels.squeeze()
    _zero_non_owned(tile_labels, spec.owned_ids)
    return tile_labels


def _zero_non_owned(tile_labels: np.ndarray, owned_ids: frozenset[int]) -> None:
    """Zero out labels not in *owned_ids* (in-place).

    Uses a boolean lookup table indexed by label ID for O(n) per-pixel
    cost when label IDs are dense.  Falls back to :func:`numpy.isin`
    when the maximum label ID is large relative to the tile size, so
    sparse-but-large ID spaces (e.g. globally-unique segmentation IDs
    from multi-FOV pipelines) don't allocate an oversized LUT.
    """
    if tile_labels.size == 0:
        return

    if not owned_ids:
        tile_labels[:] = 0
        return

    max_id = int(tile_labels.max())
    # LUT is cheaper than np.isin only when max_id fits in roughly one
    # tile's worth of bool entries; above that, the alloc dominates.
    if max_id < tile_labels.size:
        lut = np.zeros(max_id + 1, dtype=bool)
        for lid in owned_ids:
            if lid <= max_id:
                lut[lid] = True
        tile_labels[~lut[tile_labels]] = 0
    else:
        owned_arr = np.fromiter(owned_ids, dtype=tile_labels.dtype, count=len(owned_ids))
        tile_labels[~np.isin(tile_labels, owned_arr)] = 0


# ---------------------------------------------------------------------------
# Coverage verification
# ---------------------------------------------------------------------------


def verify_coverage(
    all_label_ids: set[int],
    specs: list[TileSpec],
) -> None:
    """Assert that tile specs provide full, non-overlapping cell coverage.

    Parameters
    ----------
    all_label_ids
        Set of all nonzero label IDs expected in the image.
    specs
        Tile specifications to verify.

    Raises
    ------
    ValueError
        If any cell is missing or assigned to more than one tile.
    """
    owned_union: set[int] = set()
    for spec in specs:
        overlap = owned_union & spec.owned_ids
        if overlap:
            raise ValueError(f"Cells {overlap} assigned to multiple tiles")
        owned_union |= spec.owned_ids

    missing = all_label_ids - owned_union
    if missing:
        raise ValueError(f"Cells {missing} not assigned to any tile")

    extra = owned_union - all_label_ids
    if extra:
        raise ValueError(f"Tile specs reference non-existent labels {extra}")
