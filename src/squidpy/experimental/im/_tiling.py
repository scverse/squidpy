"""Cell-aware tiling for large images.

Splits a label image into overlapping tiles such that every cell is fully
contained in exactly one tile.  Cells are assigned to tiles by centroid:
the tile whose non-overlapping base region contains the centroid owns the
cell.  Non-owned cells are zeroed out in each tile's mask so that
downstream processing never double-counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
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
    owned_ids: frozenset[int] = field(default_factory=frozenset)


def _compute_cell_info(labels: np.ndarray) -> dict[int, CellInfo]:
    """Compute centroid and bounding-box size for every label.

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


def _auto_margin(cell_info: dict[int, CellInfo]) -> int:
    """Compute the minimum margin that covers the largest cell's half-extent."""
    if not cell_info:
        return 0
    max_half = max(max(c.bbox_h, c.bbox_w) for c in cell_info.values())
    # Full bbox extent: a cell's centroid can be at most half a bbox away
    # from its edge, so margin = ceil(max_extent / 2) guarantees coverage.
    # Add 1 pixel for safety (rounding / off-by-one).
    return int(np.ceil(max_half / 2)) + 1


def build_tile_specs(
    labels: np.ndarray,
    tile_size: int = 2048,
    overlap_margin: int | Literal["auto"] = "auto",
) -> list[TileSpec]:
    """Build tile specifications for a label image.

    Each tile gets a non-overlapping *base* region (for centroid ownership)
    and an extended *crop* region (base + margin on each side).  Every
    nonzero label is assigned to exactly one tile.

    Parameters
    ----------
    labels
        2-D integer label image (0 = background).
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
    if labels.ndim != 2:
        raise ValueError(f"Expected 2-D labels, got shape {labels.shape}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    H, W = labels.shape
    cell_info = _compute_cell_info(labels)

    if isinstance(overlap_margin, str) and overlap_margin == "auto":
        margin = _auto_margin(cell_info)
    else:
        margin = int(overlap_margin)
    if margin < 0:
        raise ValueError(f"overlap_margin must be non-negative, got {margin}")

    # Assign each cell to a base-grid cell by its centroid
    cell_to_tile: dict[int, tuple[int, int]] = {}
    for lid, ci in cell_info.items():
        tile_row = min(int(ci.centroid_y) // tile_size, (H - 1) // tile_size)
        tile_col = min(int(ci.centroid_x) // tile_size, (W - 1) // tile_size)
        cell_to_tile[lid] = (tile_row, tile_col)

    # Group cells by tile
    tile_to_cells: dict[tuple[int, int], set[int]] = {}
    for lid, key in cell_to_tile.items():
        tile_to_cells.setdefault(key, set()).add(lid)

    # Build specs (skip empty tiles)
    n_rows = (H + tile_size - 1) // tile_size
    n_cols = (W + tile_size - 1) // tile_size

    specs: list[TileSpec] = []
    for row in range(n_rows):
        for col in range(n_cols):
            owned = tile_to_cells.get((row, col), set())
            if not owned:
                continue

            # Base region (non-overlapping)
            by0 = row * tile_size
            bx0 = col * tile_size
            by1 = min(by0 + tile_size, H)
            bx1 = min(bx0 + tile_size, W)

            # Crop region (with margin, clamped)
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


def extract_tile(
    image: np.ndarray,
    labels: np.ndarray,
    spec: TileSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a tile's image and mask, zeroing out non-owned cells.

    Parameters
    ----------
    image
        3-D array of shape ``(C, H, W)``.
    labels
        2-D integer label image of shape ``(H, W)``.
    spec
        Tile specification from :func:`build_tile_specs`.

    Returns
    -------
    tile_image
        Cropped image of shape ``(C, crop_h, crop_w)``.
    tile_labels
        Cropped label image with non-owned cells zeroed out.
    """
    cy0, cx0, cy1, cx1 = spec.crop
    tile_image = image[:, cy0:cy1, cx0:cx1]
    tile_labels = labels[cy0:cy1, cx0:cx1].copy()

    # Zero out labels not owned by this tile
    unique_in_crop = np.unique(tile_labels)
    for lid in unique_in_crop:
        if lid != 0 and lid not in spec.owned_ids:
            tile_labels[tile_labels == lid] = 0

    return tile_image, tile_labels


def verify_coverage(
    labels: np.ndarray,
    specs: list[TileSpec],
) -> None:
    """Assert that tile specs provide full, non-overlapping cell coverage.

    Raises
    ------
    AssertionError
        If any cell is missing or assigned to more than one tile.
    """
    all_label_ids = set(np.unique(labels))
    all_label_ids.discard(0)

    owned_union: set[int] = set()
    for spec in specs:
        overlap = owned_union & spec.owned_ids
        assert not overlap, f"Cells {overlap} assigned to multiple tiles"
        owned_union |= spec.owned_ids

    missing = all_label_ids - owned_union
    assert not missing, f"Cells {missing} not assigned to any tile"

    extra = owned_union - all_label_ids
    assert not extra, f"Tile specs reference non-existent labels {extra}"
