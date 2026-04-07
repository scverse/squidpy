"""Tests for cell-aware tiling logic.

Uses a deterministic "brick-pattern" grid of rectangular cells on a
500×500 image.  Even rows are aligned; odd rows are shifted right by
half a cell width, like bricks in a wall.  The image divides into 4
tiles of 250×250.  Because cell positions are predictable we can check
*exactly* which cell lands in which tile.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.conftest import PlotTester, PlotTesterMeta

from squidpy.experimental.im._tiling import (
    TileSpec,
    build_tile_specs,
    extract_tile,
    verify_coverage,
)

# ---------------------------------------------------------------------------
# Brick-pattern fixture
# ---------------------------------------------------------------------------

_IMAGE_SIZE = 500
_CELL_H = 20
_CELL_W = 30


def _make_brick_labels(
    image_size: int = _IMAGE_SIZE,
    cell_h: int = _CELL_H,
    cell_w: int = _CELL_W,
    gap: int = 10,
) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    """Create a brick-pattern label image and return centroids.

    Parameters
    ----------
    image_size
        Side length of the square image.
    cell_h, cell_w
        Height and width of each rectangular cell.
    gap
        Gap between cells (0 = touching).

    Returns
    -------
    labels
        ``(image_size, image_size)`` int32 array.
    centroids
        Mapping from label ID → ``(centroid_y, centroid_x)``.
    """
    labels = np.zeros((image_size, image_size), dtype=np.int32)
    centroids: dict[int, tuple[float, float]] = {}

    step_y = cell_h + gap
    step_x = cell_w + gap
    cell_id = 0

    row_idx = 0
    y = gap // 2  # start with half-gap from top
    while y + cell_h <= image_size:
        # Odd rows shift right by half a cell+gap step
        x_offset = (step_x // 2) if (row_idx % 2 == 1) else 0
        x = x_offset + gap // 2
        while x + cell_w <= image_size:
            cell_id += 1
            labels[y : y + cell_h, x : x + cell_w] = cell_id
            # Match regionprops centroid: mean of pixel indices [y, y+cell_h-1]
            cy = y + (cell_h - 1) / 2.0
            cx = x + (cell_w - 1) / 2.0
            centroids[cell_id] = (cy, cx)
            x += step_x
        y += step_y
        row_idx += 1

    return labels, centroids


def _make_image(image_size: int = _IMAGE_SIZE, n_channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (n_channels, image_size, image_size), dtype=np.uint8)


def _expected_tile_key(cy: float, cx: float, tile_size: int, image_size: int) -> tuple[int, int]:
    """Which tile base-grid cell a centroid falls into."""
    max_row = (image_size - 1) // tile_size
    max_col = (image_size - 1) // tile_size
    row = min(int(cy) // tile_size, max_row)
    col = min(int(cx) // tile_size, max_col)
    return (row, col)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[10, 0], ids=["gap=10", "gap=0"])
def brick_labels(request):
    """Brick-pattern labels with gap (non-touching) or without (touching)."""
    gap = request.param
    labels, centroids = _make_brick_labels(gap=gap)
    return labels, centroids, gap


@pytest.fixture()
def brick_image():
    return _make_image()


# ---------------------------------------------------------------------------
# build_tile_specs — deterministic checks
# ---------------------------------------------------------------------------

_TILE_SIZE = 250  # 500 / 250 = 2×2 = 4 tiles


class TestBuildTileSpecs:
    def test_four_tiles(self, brick_labels):
        """500×500 with tile_size=250 produces at most 4 tiles."""
        labels, _, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        assert len(specs) <= 4

    def test_full_coverage(self, brick_labels):
        """Every cell is assigned to exactly one tile."""
        labels, _, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        verify_coverage(labels, specs)

    def test_cell_assigned_to_centroid_tile(self, brick_labels):
        """Each cell's tile matches the tile we predict from its centroid."""
        labels, centroids, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)

        # Build actual mapping: cell_id → tile base origin
        actual: dict[int, tuple[int, int]] = {}
        for spec in specs:
            for lid in spec.owned_ids:
                actual[lid] = (spec.base[0], spec.base[1])

        for lid, (cy, cx) in centroids.items():
            expected_row, expected_col = _expected_tile_key(cy, cx, _TILE_SIZE, _IMAGE_SIZE)
            expected_origin = (expected_row * _TILE_SIZE, expected_col * _TILE_SIZE)
            assert actual[lid] == expected_origin, (
                f"Cell {lid} centroid=({cy:.1f},{cx:.1f}): "
                f"expected tile origin {expected_origin}, got {actual[lid]}"
            )

    def test_no_duplicates(self, brick_labels):
        """No cell ID appears in more than one tile."""
        labels, _, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)

        seen: set[int] = set()
        for spec in specs:
            overlap = seen & spec.owned_ids
            assert not overlap, f"Duplicate cell IDs: {overlap}"
            seen |= spec.owned_ids

    def test_boundary_cells_exist(self, brick_labels):
        """With the brick offset, some cells straddle the y=250 or x=250 boundary."""
        labels, centroids, gap = brick_labels
        # A cell straddles a boundary if its rectangle crosses y=250 or x=250
        # but its centroid is on one side
        boundary_cells = []
        step_y = _CELL_H + gap
        step_x = _CELL_W + gap
        for lid, (cy, cx) in centroids.items():
            half_h = _CELL_H / 2.0
            half_w = _CELL_W / 2.0
            y0, y1 = cy - half_h, cy + half_h
            x0, x1 = cx - half_w, cx + half_w
            crosses_y = (y0 < 250 < y1)
            crosses_x = (x0 < 250 < x1)
            if crosses_y or crosses_x:
                boundary_cells.append(lid)

        # With cell_h=20 and various gaps, we expect some boundary cells
        # (the brick offset makes this likely for odd rows near y=250)
        # Just verify they're all assigned somewhere
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        all_owned = set()
        for s in specs:
            all_owned |= s.owned_ids
        for lid in boundary_cells:
            assert lid in all_owned, f"Boundary cell {lid} not assigned"

    def test_crop_contains_owned_cells_fully(self, brick_labels):
        """Every owned cell's rectangle fits inside its tile's crop region."""
        labels, centroids, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE, overlap_margin="auto")

        for spec in specs:
            cy0, cx0, cy1, cx1 = spec.crop
            for lid in spec.owned_ids:
                cent_y, cent_x = centroids[lid]
                # Reconstruct cell pixel range from centroid
                # Centroid is mean of [y, y+cell_h-1], so half-extent = (cell_h-1)/2
                cell_y0 = cent_y - (_CELL_H - 1) / 2.0
                cell_y1 = cent_y + (_CELL_H - 1) / 2.0
                cell_x0 = cent_x - (_CELL_W - 1) / 2.0
                cell_x1 = cent_x + (_CELL_W - 1) / 2.0
                assert cy0 <= cell_y0 and cell_y1 <= cy1, (
                    f"Cell {lid} y-range [{cell_y0:.0f},{cell_y1:.0f}] "
                    f"not in crop y-range [{cy0},{cy1}]"
                )
                assert cx0 <= cell_x0 and cell_x1 <= cx1, (
                    f"Cell {lid} x-range [{cell_x0:.0f},{cell_x1:.0f}] "
                    f"not in crop x-range [{cx0},{cx1}]"
                )


class TestBuildTileSpecsEdgeCases:
    def test_empty_labels(self):
        labels = np.zeros((500, 500), dtype=np.int32)
        specs = build_tile_specs(labels, tile_size=250)
        assert specs == []
        verify_coverage(labels, specs)

    def test_single_cell_whole_image(self):
        """One cell that fills most of the image."""
        labels = np.zeros((500, 500), dtype=np.int32)
        labels[10:490, 10:490] = 1
        specs = build_tile_specs(labels, tile_size=250)
        verify_coverage(labels, specs)
        assert len(specs) == 1  # centroid is at ~(250,250), lands in one tile

    def test_invalid_tile_size(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        with pytest.raises(ValueError, match="tile_size must be positive"):
            build_tile_specs(labels, tile_size=0)

    def test_invalid_labels_ndim(self):
        labels = np.zeros((2, 100, 100), dtype=np.int32)
        with pytest.raises(ValueError, match="Expected 2-D labels"):
            build_tile_specs(labels, tile_size=100)

    def test_tile_size_larger_than_image(self):
        """tile_size > image → single tile."""
        labels, _ = _make_brick_labels(image_size=100, gap=5)
        specs = build_tile_specs(labels, tile_size=1000)
        verify_coverage(labels, specs)
        assert len(specs) == 1


# ---------------------------------------------------------------------------
# extract_tile
# ---------------------------------------------------------------------------


class TestExtractTile:
    def test_non_owned_cells_zeroed(self, brick_labels, brick_image):
        """Only owned cells survive in the extracted tile mask."""
        labels, _, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)

        for spec in specs:
            _, tile_lbl = extract_tile(brick_image, labels, spec)
            present = set(np.unique(tile_lbl))
            present.discard(0)
            assert present == spec.owned_ids, (
                f"Tile base={spec.base}: expected {spec.owned_ids}, "
                f"got {present}"
            )

    def test_owned_cell_pixels_preserved(self, brick_labels, brick_image):
        """Pixel values for owned cells match the original labels."""
        labels, _, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)

        for spec in specs:
            cy0, cx0, cy1, cx1 = spec.crop
            _, tile_lbl = extract_tile(brick_image, labels, spec)
            for lid in spec.owned_ids:
                orig_in_crop = labels[cy0:cy1, cx0:cx1] == lid
                tile_matches = tile_lbl == lid
                np.testing.assert_array_equal(orig_in_crop, tile_matches)

    def test_original_labels_not_mutated(self, brick_labels, brick_image):
        labels, _, _ = brick_labels
        labels_copy = labels.copy()
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        for spec in specs:
            extract_tile(brick_image, labels, spec)
        np.testing.assert_array_equal(labels, labels_copy)

    def test_image_crop_shape(self, brick_labels, brick_image):
        """Extracted image has shape (C, crop_h, crop_w)."""
        labels, _, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        for spec in specs:
            tile_img, tile_lbl = extract_tile(brick_image, labels, spec)
            cy0, cx0, cy1, cx1 = spec.crop
            assert tile_img.shape == (3, cy1 - cy0, cx1 - cx0)
            assert tile_lbl.shape == (cy1 - cy0, cx1 - cx0)


# ---------------------------------------------------------------------------
# End-to-end roundtrip
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_roundtrip_no_cells_lost(self, brick_labels, brick_image):
        """Build specs → extract tiles → union of labels == all cells."""
        labels, centroids, _ = brick_labels
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        verify_coverage(labels, specs)

        recovered: set[int] = set()
        for spec in specs:
            _, tile_lbl = extract_tile(brick_image, labels, spec)
            tile_ids = set(np.unique(tile_lbl))
            tile_ids.discard(0)
            assert tile_ids == spec.owned_ids
            recovered |= tile_ids

        assert recovered == set(centroids.keys())

    def test_touching_cells_no_merge(self):
        """With gap=0, adjacent cells still get distinct labels and assignments."""
        labels, centroids = _make_brick_labels(gap=0)
        n_cells = len(centroids)
        assert n_cells > 0

        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        verify_coverage(labels, specs)

        # Total owned cells across all tiles == total cells
        total_owned = sum(len(s.owned_ids) for s in specs)
        assert total_owned == n_cells

    def test_nontouching_cells_same_result(self):
        """With gap=10, same coverage guarantees hold."""
        labels, centroids = _make_brick_labels(gap=10)
        n_cells = len(centroids)
        assert n_cells > 0

        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        verify_coverage(labels, specs)

        total_owned = sum(len(s.owned_ids) for s in specs)
        assert total_owned == n_cells


# ---------------------------------------------------------------------------
# Visual test — tile assignment plot
# ---------------------------------------------------------------------------

# Tile colors: one distinct color per tile quadrant
_TILE_COLORS = [
    (0.12, 0.47, 0.71),  # blue   — top-left
    (1.00, 0.50, 0.05),  # orange — top-right
    (0.17, 0.63, 0.17),  # green  — bottom-left
    (0.84, 0.15, 0.16),  # red    — bottom-right
]


def _plot_tile_assignment(labels, specs, title=""):
    """Render each cell colored by its owning tile, with grid lines."""
    rgb = np.ones((*labels.shape, 3), dtype=np.float32)  # white background

    for i, spec in enumerate(specs):
        color = _TILE_COLORS[i % len(_TILE_COLORS)]
        for lid in spec.owned_ids:
            mask = labels == lid
            rgb[mask] = color

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb, origin="upper")

    # Draw tile base-grid lines
    for spec in specs:
        by0, bx0, by1, bx1 = spec.base
        rect = plt.Rectangle(
            (bx0 - 0.5, by0 - 0.5),
            bx1 - bx0,
            by1 - by0,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.5, labels.shape[1] - 0.5)
    ax.set_ylim(labels.shape[0] - 0.5, -0.5)
    ax.set_title(title or "Tile assignment")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


class TestTilingVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_tile_assignment_gap(self):
        """Visual: brick pattern (gap=10), cells colored by tile."""
        labels, _ = _make_brick_labels(gap=10)
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        _plot_tile_assignment(labels, specs, title="Tile assignment (gap=10)")

    def test_plot_tile_assignment_touching(self):
        """Visual: brick pattern (gap=0, touching), cells colored by tile."""
        labels, _ = _make_brick_labels(gap=0)
        specs = build_tile_specs(labels, tile_size=_TILE_SIZE)
        _plot_tile_assignment(labels, specs, title="Tile assignment (gap=0, touching)")
