"""Cell-aware tiling for large images.

Splits a label image into overlapping tiles such that every cell is fully
contained in exactly one tile.  Cells are assigned to tiles by centroid:
the tile whose non-overlapping base region contains the centroid owns the
cell.  Non-owned cells are zeroed out in each tile's mask so that
downstream processing never double-counts.

Two parallel APIs are exposed:

* In-memory: ``compute_cell_info(labels) -> dict`` + ``extract_tile``.
* Lazy / xarray-backed: ``compute_cell_info_multiscale``,
  ``compute_cell_info_tiled``, ``extract_tile_lazy``.

``build_tile_specs`` takes only ``(shape, cell_info)``, so it is agnostic
to whether the labels are in memory, dask-backed, or multiscale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr
from skimage.measure import regionprops

__all__ = [
    "CellInfo",
    "TileSpec",
    "build_tile_specs",
    "compute_cell_info",
    "compute_cell_info_multiscale",
    "compute_cell_info_tiled",
    "extract_tile",
    "extract_tile_lazy",
    "verify_coverage",
]


@dataclass(frozen=True)
class CellInfo:
    """Centroid and bounding box for a single label."""

    label: int
    centroid_y: float
    centroid_x: float
    bbox_h: int  # height of bounding box (pixels)
    bbox_w: int  # width of bounding box (pixels)


@dataclass(frozen=True)
class TileSpec:
    """Specification for a single tile.

    Attributes
    ----------
    base
        Non-overlapping region ``(y0, x0, y1, x1)`` used for centroid
        ownership.  Tiles partition the image into a grid of base regions.
    crop
        Extended region ``(y0, x0, y1, x1)`` that includes the overlap
        margin.  This is the actual slice extracted from the image/labels.
    owned_ids
        Label IDs whose centroid falls inside ``base``.  Only these labels
        are kept in the tile's mask; all others are zeroed out.
    """

    base: tuple[int, int, int, int]
    crop: tuple[int, int, int, int]
    owned_ids: frozenset[int] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Cell info — in-memory
# ---------------------------------------------------------------------------


def compute_cell_info(labels: np.ndarray) -> dict[int, CellInfo]:
    """Compute centroid and bounding-box size for every label.

    Parameters
    ----------
    labels
        2-D integer label image where 0 is background.

    Returns
    -------
    Mapping from label ID to :class:`CellInfo`.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected 2-D labels, got shape {labels.shape}")
    props = regionprops(labels)
    info: dict[int, CellInfo] = {}
    for p in props:
        min_row, min_col, max_row, max_col = p.bbox
        info[p.label] = CellInfo(
            label=p.label,
            centroid_y=float(p.centroid[0]),
            centroid_x=float(p.centroid[1]),
            bbox_h=max_row - min_row,
            bbox_w=max_col - min_col,
        )
    return info


# ---------------------------------------------------------------------------
# Cell info — multiscale (read coarse pyramid level, scale back to target)
# ---------------------------------------------------------------------------


def _pick_coarsest_scale(label_tree: xr.DataTree) -> str:
    """Return the coarsest scale key in a multiscale DataTree."""
    scales = sorted(label_tree.keys(), key=lambda s: int(s.replace("scale", "")))
    return scales[-1]


def _scale_dims(node: xr.DataTree | xr.DataArray) -> tuple[int, int]:
    """Return (H, W) of a single scale level."""
    if isinstance(node, xr.DataTree):
        # spatialdata stores the array under .ds["image"]
        da = node.ds["image"]
    else:
        da = node
    return int(da.sizes["y"]), int(da.sizes["x"])


def compute_cell_info_multiscale(
    label_tree: xr.DataTree,
    target_scale: str,
) -> dict[int, CellInfo]:
    """Compute cell info from the coarsest scale, rescaled to target scale.

    Reading the coarsest scale avoids materializing the full-res labels
    just to find centroids.

    Parameters
    ----------
    label_tree
        Multi-scale labels (e.g. ``sdata.labels[key]``).
    target_scale
        Scale level whose pixel grid the returned centroids/bbox refer to.

    Returns
    -------
    Cell info dict, in ``target_scale`` pixel coordinates.
    """
    if target_scale not in label_tree:
        raise ValueError(f"target_scale '{target_scale}' not found in DataTree. Available: {list(label_tree.keys())}")

    coarsest = _pick_coarsest_scale(label_tree)
    if coarsest == target_scale:
        labels_arr = label_tree[coarsest].ds["image"].values
        if labels_arr.ndim > 2:
            labels_arr = labels_arr.squeeze()
        return compute_cell_info(labels_arr)

    coarse_h, coarse_w = _scale_dims(label_tree[coarsest])
    target_h, target_w = _scale_dims(label_tree[target_scale])

    sy = target_h / coarse_h
    sx = target_w / coarse_w

    labels_arr = label_tree[coarsest].ds["image"].values
    if labels_arr.ndim > 2:
        labels_arr = labels_arr.squeeze()
    coarse_info = compute_cell_info(labels_arr)

    rescaled: dict[int, CellInfo] = {}
    for lid, ci in coarse_info.items():
        rescaled[lid] = CellInfo(
            label=ci.label,
            centroid_y=ci.centroid_y * sy,
            centroid_x=ci.centroid_x * sx,
            bbox_h=int(np.ceil(ci.bbox_h * sy)),
            bbox_w=int(np.ceil(ci.bbox_w * sx)),
        )
    return rescaled


# ---------------------------------------------------------------------------
# Cell info — tiled (single-scale large labels, no full materialization)
# ---------------------------------------------------------------------------


def compute_cell_info_tiled(
    labels_da: xr.DataArray,
    chunk: int = 4096,
) -> dict[int, CellInfo]:
    """Compute cell info by tile-streaming the labels array.

    Accumulates pixel sums + bbox per label across non-overlapping tiles.
    Cells that span tile boundaries are merged correctly because the per-
    label statistics are additive.

    Parameters
    ----------
    labels_da
        Lazy/eager 2-D xarray DataArray of integer labels.
    chunk
        Tile side length for streaming reads.

    Returns
    -------
    Cell info dict in ``labels_da``'s native pixel grid.
    """
    if labels_da.ndim > 2:
        labels_da = labels_da.squeeze()
    if labels_da.ndim != 2:
        raise ValueError(f"Expected 2-D labels, got shape {labels_da.shape}")

    H, W = int(labels_da.sizes["y"]), int(labels_da.sizes["x"])

    # Per-label accumulators
    area: dict[int, int] = {}
    sum_y: dict[int, float] = {}
    sum_x: dict[int, float] = {}
    min_y: dict[int, int] = {}
    min_x: dict[int, int] = {}
    max_y: dict[int, int] = {}
    max_x: dict[int, int] = {}

    for y0 in range(0, H, chunk):
        y1 = min(y0 + chunk, H)
        for x0 in range(0, W, chunk):
            x1 = min(x0 + chunk, W)
            tile = labels_da.isel(y=slice(y0, y1), x=slice(x0, x1)).values
            if tile.ndim > 2:
                tile = tile.squeeze()
            ids = np.unique(tile)
            ids = ids[ids != 0]
            if ids.size == 0:
                continue
            for lid in ids:
                lid_int = int(lid)
                ys, xs = np.where(tile == lid)
                # Convert tile-local coords to global
                ys_g = ys + y0
                xs_g = xs + x0
                area[lid_int] = area.get(lid_int, 0) + ys.size
                sum_y[lid_int] = sum_y.get(lid_int, 0.0) + float(ys_g.sum())
                sum_x[lid_int] = sum_x.get(lid_int, 0.0) + float(xs_g.sum())
                ymin, ymax = int(ys_g.min()), int(ys_g.max())
                xmin, xmax = int(xs_g.min()), int(xs_g.max())
                min_y[lid_int] = min(min_y.get(lid_int, ymin), ymin)
                min_x[lid_int] = min(min_x.get(lid_int, xmin), xmin)
                max_y[lid_int] = max(max_y.get(lid_int, ymax), ymax)
                max_x[lid_int] = max(max_x.get(lid_int, xmax), xmax)

    info: dict[int, CellInfo] = {}
    for lid_int, a in area.items():
        info[lid_int] = CellInfo(
            label=lid_int,
            centroid_y=sum_y[lid_int] / a,
            centroid_x=sum_x[lid_int] / a,
            bbox_h=max_y[lid_int] - min_y[lid_int] + 1,
            bbox_w=max_x[lid_int] - min_x[lid_int] + 1,
        )
    return info


# ---------------------------------------------------------------------------
# Tile specification
# ---------------------------------------------------------------------------


def _auto_margin(cell_info: dict[int, CellInfo]) -> int:
    """Compute the minimum margin that covers the largest cell's half-extent."""
    if not cell_info:
        return 0
    max_extent = max(max(c.bbox_h, c.bbox_w) for c in cell_info.values())
    return int(np.ceil(max_extent / 2)) + 1


def build_tile_specs(
    shape: tuple[int, int],
    cell_info: dict[int, CellInfo],
    tile_size: int = 2048,
    overlap_margin: int | Literal["auto"] = "auto",
) -> list[TileSpec]:
    """Build tile specifications from precomputed cell info.

    The new ``(shape, cell_info)`` signature makes this agnostic to label
    materialization — caller supplies dims and centroids, this function
    just partitions.

    Parameters
    ----------
    shape
        ``(H, W)`` of the labels array.
    cell_info
        Output of :func:`compute_cell_info` (or one of its variants).
    tile_size
        Side length of the non-overlapping base grid cells.
    overlap_margin
        Pixel margin added around each base region.  ``"auto"`` computes
        the minimum margin from the largest cell's bounding box.

    Returns
    -------
    List of :class:`TileSpec`, one per grid cell that owns at least one
    label.  Empty tiles are omitted.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    H, W = shape

    if isinstance(overlap_margin, str) and overlap_margin == "auto":
        margin = _auto_margin(cell_info)
    else:
        margin = int(overlap_margin)
    if margin < 0:
        raise ValueError(f"overlap_margin must be non-negative, got {margin}")

    # Assign each cell to a base-grid cell by its centroid
    cell_to_tile: dict[int, tuple[int, int]] = {}
    for lid, ci in cell_info.items():
        tile_row = min(int(ci.centroid_y) // tile_size, max((H - 1) // tile_size, 0))
        tile_col = min(int(ci.centroid_x) // tile_size, max((W - 1) // tile_size, 0))
        cell_to_tile[lid] = (tile_row, tile_col)

    tile_to_cells: dict[tuple[int, int], set[int]] = {}
    for lid, key in cell_to_tile.items():
        tile_to_cells.setdefault(key, set()).add(lid)

    n_rows = max((H + tile_size - 1) // tile_size, 1)
    n_cols = max((W + tile_size - 1) // tile_size, 1)

    specs: list[TileSpec] = []
    for row in range(n_rows):
        for col in range(n_cols):
            owned = tile_to_cells.get((row, col), set())
            if not owned:
                continue
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


def _zero_non_owned(tile_labels: np.ndarray, owned: frozenset[int]) -> np.ndarray:
    """Return a copy of ``tile_labels`` with non-owned labels set to 0."""
    out = tile_labels.copy()
    unique_in_crop = np.unique(out)
    for lid in unique_in_crop:
        if lid != 0 and int(lid) not in owned:
            out[out == lid] = 0
    return out


def extract_tile(
    image: np.ndarray,
    labels: np.ndarray,
    spec: TileSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a tile's image and mask from in-memory arrays.

    Parameters
    ----------
    image
        3-D array of shape ``(C, H, W)``.
    labels
        2-D integer label image of shape ``(H, W)``.
    spec
        Tile specification.

    Returns
    -------
    tile_image
        Cropped image of shape ``(C, crop_h, crop_w)``.
    tile_labels
        Cropped label image with non-owned cells zeroed out.
    """
    cy0, cx0, cy1, cx1 = spec.crop
    tile_image = image[:, cy0:cy1, cx0:cx1]
    tile_labels = _zero_non_owned(labels[cy0:cy1, cx0:cx1], spec.owned_ids)
    return tile_image, tile_labels


def extract_tile_lazy(
    image_da: xr.DataArray,
    labels_da: xr.DataArray,
    spec: TileSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a tile by lazy slicing then materializing only the crop.

    Parameters
    ----------
    image_da
        Lazy DataArray of shape ``(C, H, W)``.
    labels_da
        Lazy 2-D DataArray of labels.
    spec
        Tile specification.

    Returns
    -------
    tile_image
        Numpy ``(C, crop_h, crop_w)``.
    tile_labels
        Numpy ``(crop_h, crop_w)`` with non-owned cells zeroed out.
    """
    cy0, cx0, cy1, cx1 = spec.crop
    tile_image = image_da.isel(y=slice(cy0, cy1), x=slice(cx0, cx1)).values
    tile_labels_raw = labels_da.isel(y=slice(cy0, cy1), x=slice(cx0, cx1)).values
    if tile_labels_raw.ndim > 2:
        tile_labels_raw = tile_labels_raw.squeeze()
    tile_labels = _zero_non_owned(tile_labels_raw, spec.owned_ids)
    return tile_image, tile_labels


# ---------------------------------------------------------------------------
# Coverage verification
# ---------------------------------------------------------------------------


def verify_coverage(label_ids: set[int], specs: list[TileSpec]) -> None:
    """Assert that tile specs provide full, non-overlapping cell coverage.

    Parameters
    ----------
    label_ids
        Set of all expected nonzero label IDs.
    specs
        Tile specifications.

    Raises
    ------
    AssertionError
        If any cell is missing, duplicated, or unknown.
    """
    owned_union: set[int] = set()
    for spec in specs:
        overlap = owned_union & spec.owned_ids
        assert not overlap, f"Cells {overlap} assigned to multiple tiles"
        owned_union |= spec.owned_ids

    missing = label_ids - owned_union
    assert not missing, f"Cells {missing} not assigned to any tile"

    extra = owned_union - label_ids
    assert not extra, f"Tile specs reference non-existent labels {extra}"
