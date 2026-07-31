"""Tests for cell-aware tiling logic.

Uses a deterministic "brick-pattern" grid of rectangular cells on a
500x500 image.  Even rows are aligned; odd rows are shifted right by
half a cell width, like bricks in a wall.  The image divides into 4
tiles of 250x250.  Because cell positions are predictable we can check
*exactly* which cell lands in which tile.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from squidpy.experimental.im._tiling import (
    CellInfo,
    _zero_non_owned,
    build_tile_specs,
    compute_cell_info,
    compute_cell_info_multiscale,
    compute_cell_info_tiled,
    extract_tile_lazy,
)
from tests.conftest import PlotTester, PlotTesterMeta


# Test-only helpers: an eager tile-extraction reference (the production path is
# extract_tile_lazy) and a coverage-invariant checker for build_tile_specs.
# Neither is used in production, so they live with the tests rather than ship.
def extract_tile(image, labels, spec):
    """Eager numpy tile extraction; reference for extract_tile_lazy."""
    cy0, cx0, cy1, cx1 = spec.crop
    tile_image = image[:, cy0:cy1, cx0:cx1]
    tile_labels = labels[cy0:cy1, cx0:cx1].copy()
    _zero_non_owned(tile_labels, spec.owned_ids)
    return tile_image, tile_labels


def verify_coverage(all_label_ids, specs):
    """Assert tile specs give full, non-overlapping cell coverage."""
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


# Brick-pattern fixture

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
        Mapping from label ID -> ``(centroid_y, centroid_x)``.
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


_TILE_SIZE = 250  # 500 / 250 = 2x2 = 4 tiles


def _specs_from_labels(labels, tile_size=_TILE_SIZE, overlap_margin="auto"):
    """Convenience: compute cell info + build tile specs from a numpy label array."""
    cell_info = compute_cell_info(labels)
    return build_tile_specs(labels.shape, cell_info, tile_size=tile_size, overlap_margin=overlap_margin)


def _label_ids(labels):
    """All nonzero label IDs as a set."""
    ids = set(np.unique(labels).tolist())
    ids.discard(0)
    return ids


def _make_ci(label: int, cy: float, cx: float, h: int = 4, w: int = 4) -> CellInfo:
    """Build a CellInfo for tests that need a minimal hand-constructed cell."""
    return CellInfo(label=label, centroid_y=cy, centroid_x=cx, bbox_h=h, bbox_w=w)


# Fixtures


@pytest.fixture(params=[10, 0], ids=["gap=10", "gap=0"])
def brick_labels(request):
    """Brick-pattern labels with gap (non-touching) or without (touching)."""
    return _make_brick_labels(gap=request.param)


@pytest.fixture()
def brick_image():
    return _make_image()


# build_tile_specs - deterministic checks


class TestBuildTileSpecs:
    def test_four_tiles(self, brick_labels):
        """500x500 with tile_size=250 produces at most 4 tiles."""
        labels, _ = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        assert len(specs) <= 4

    def test_full_coverage(self, brick_labels):
        """Every cell is assigned to exactly one tile."""
        labels, _ = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        verify_coverage(_label_ids(labels), specs)

    def test_cell_assigned_to_centroid_tile(self, brick_labels):
        """Each cell's tile matches the tile we predict from its centroid."""
        labels, centroids = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)

        # Build actual mapping: cell_id -> tile base origin
        actual: dict[int, tuple[int, int]] = {}
        for spec in specs:
            for lid in spec.owned_ids:
                actual[lid] = (spec.base[0], spec.base[1])

        for lid, (cy, cx) in centroids.items():
            expected_row, expected_col = _expected_tile_key(cy, cx, _TILE_SIZE, _IMAGE_SIZE)
            expected_origin = (expected_row * _TILE_SIZE, expected_col * _TILE_SIZE)
            assert actual[lid] == expected_origin, (
                f"Cell {lid} centroid=({cy:.1f},{cx:.1f}): expected tile origin {expected_origin}, got {actual[lid]}"
            )

    def test_no_duplicates(self, brick_labels):
        """No cell ID appears in more than one tile."""
        labels, _ = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)

        seen: set[int] = set()
        for spec in specs:
            overlap = seen & spec.owned_ids
            assert not overlap, f"Duplicate cell IDs: {overlap}"
            seen |= spec.owned_ids

    def test_boundary_cells_exist(self, brick_labels):
        """With the brick offset, some cells straddle the y=250 or x=250 boundary."""
        labels, centroids = brick_labels
        # A cell straddles a boundary if its rectangle crosses y=250 or x=250
        # but its centroid is on one side.
        boundary_cells = []
        for lid, (cy, cx) in centroids.items():
            half_h = _CELL_H / 2.0
            half_w = _CELL_W / 2.0
            y0, y1 = cy - half_h, cy + half_h
            x0, x1 = cx - half_w, cx + half_w
            crosses_y = y0 < 250 < y1
            crosses_x = x0 < 250 < x1
            if crosses_y or crosses_x:
                boundary_cells.append(lid)

        # Fail loudly if the fixture stops producing boundary cells: this
        # test is otherwise a no-op and silently misses regressions.
        assert boundary_cells, "Fixture produced no tile-boundary cells; test would pass vacuously."

        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        all_owned = set()
        for s in specs:
            all_owned |= s.owned_ids
        for lid in boundary_cells:
            assert lid in all_owned, f"Boundary cell {lid} not assigned"

    def test_crop_contains_owned_cells_fully(self, brick_labels):
        """Every owned cell's rectangle fits inside its tile's crop region."""
        labels, centroids = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE, overlap_margin="auto")

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
                    f"Cell {lid} y-range [{cell_y0:.0f},{cell_y1:.0f}] not in crop y-range [{cy0},{cy1}]"
                )
                assert cx0 <= cell_x0 and cell_x1 <= cx1, (
                    f"Cell {lid} x-range [{cell_x0:.0f},{cell_x1:.0f}] not in crop x-range [{cx0},{cx1}]"
                )


class TestBuildTileSpecsEdgeCases:
    def test_empty_labels(self):
        labels = np.zeros((500, 500), dtype=np.int32)
        specs = _specs_from_labels(labels, tile_size=250)
        assert specs == []
        verify_coverage(_label_ids(labels), specs)

    def test_single_cell_whole_image(self):
        """One cell that fills most of the image."""
        labels = np.zeros((500, 500), dtype=np.int32)
        labels[10:490, 10:490] = 1
        specs = _specs_from_labels(labels, tile_size=250)
        verify_coverage(_label_ids(labels), specs)
        assert len(specs) == 1  # centroid is at ~(250,250), lands in one tile

    def test_invalid_tile_size(self):
        # Pass a non-empty cell_info so the test exercises the tile_size guard
        # rather than an empty-dict short-circuit if validation order ever shifts.
        ci = {1: CellInfo(label=1, centroid_y=50, centroid_x=50, bbox_h=4, bbox_w=4)}
        with pytest.raises(ValueError, match="tile_size must be positive"):
            build_tile_specs((100, 100), ci, tile_size=0)

    def test_tile_size_larger_than_image(self):
        """tile_size > image -> single tile."""
        labels, _ = _make_brick_labels(image_size=100, gap=5)
        specs = _specs_from_labels(labels, tile_size=1000)
        verify_coverage(_label_ids(labels), specs)
        assert len(specs) == 1


# extract_tile


class TestExtractTile:
    def test_non_owned_cells_zeroed(self, brick_labels, brick_image):
        """Only owned cells survive in the extracted tile mask."""
        labels, _ = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)

        for spec in specs:
            _, tile_lbl = extract_tile(brick_image, labels, spec)
            present = set(np.unique(tile_lbl))
            present.discard(0)
            assert present == spec.owned_ids, f"Tile base={spec.base}: expected {spec.owned_ids}, got {present}"

    def test_owned_cell_pixels_preserved(self, brick_labels, brick_image):
        """Pixel values for owned cells match the original labels."""
        labels, _ = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)

        for spec in specs:
            cy0, cx0, cy1, cx1 = spec.crop
            _, tile_lbl = extract_tile(brick_image, labels, spec)
            for lid in spec.owned_ids:
                orig_in_crop = labels[cy0:cy1, cx0:cx1] == lid
                tile_matches = tile_lbl == lid
                np.testing.assert_array_equal(orig_in_crop, tile_matches)

    def test_original_labels_not_mutated(self, brick_labels, brick_image):
        labels, _ = brick_labels
        labels_copy = labels.copy()
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        for spec in specs:
            extract_tile(brick_image, labels, spec)
        np.testing.assert_array_equal(labels, labels_copy)

    def test_image_crop_shape(self, brick_labels, brick_image):
        """Extracted image has shape (C, crop_h, crop_w)."""
        labels, _ = brick_labels
        n_channels = brick_image.shape[0]
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        for spec in specs:
            tile_img, tile_lbl = extract_tile(brick_image, labels, spec)
            cy0, cx0, cy1, cx1 = spec.crop
            assert tile_img.shape == (n_channels, cy1 - cy0, cx1 - cx0)
            assert tile_lbl.shape == (cy1 - cy0, cx1 - cx0)


# End-to-end roundtrip


class TestEndToEnd:
    def test_roundtrip_no_cells_lost(self, brick_labels, brick_image):
        """Build specs -> extract tiles -> union of labels == all cells."""
        labels, centroids = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        verify_coverage(_label_ids(labels), specs)

        recovered: set[int] = set()
        for spec in specs:
            _, tile_lbl = extract_tile(brick_image, labels, spec)
            tile_ids = set(np.unique(tile_lbl))
            tile_ids.discard(0)
            assert tile_ids == spec.owned_ids
            recovered |= tile_ids

        assert recovered == set(centroids.keys())


# Note: gap=0 (touching) and gap=10 (non-touching) are both covered by
# test_roundtrip_no_cells_lost via the brick_labels fixture's parametrisation.


# Visual test - tile assignment plot

# Tile colors: one distinct color per tile quadrant
_TILE_COLORS = [
    (0.12, 0.47, 0.71),  # blue   - top-left
    (1.00, 0.50, 0.05),  # orange - top-right
    (0.17, 0.63, 0.17),  # green  - bottom-left
    (0.84, 0.15, 0.16),  # red    - bottom-right
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


# Lazy / multiscale helpers


def _make_multiscale_tree(labels: np.ndarray, n_scales: int = 3) -> xr.DataTree:
    """Build a tiny multiscale DataTree by integer-downsampling."""
    scales: dict[str, xr.DataTree] = {}
    for i in range(n_scales):
        step = 2**i
        sub = labels[::step, ::step]
        ds = xr.Dataset({"image": xr.DataArray(sub, dims=("y", "x"))})
        scales[f"scale{i}"] = xr.DataTree(ds)
    return xr.DataTree.from_dict(scales)


class TestComputeCellInfoMultiscale:
    def test_target_is_coarsest_matches_eager(self):
        labels, _ = _make_brick_labels(gap=10)
        tree = _make_multiscale_tree(labels, n_scales=3)
        # scale2 is coarsest. Target it -> use that scale directly.
        info_ms = compute_cell_info_multiscale(tree, target_scale="scale2")
        info_eager = compute_cell_info(tree["scale2"].ds["image"].values)
        assert set(info_ms.keys()) == set(info_eager.keys())
        for lid in info_ms:
            assert info_ms[lid].centroid_y == pytest.approx(info_eager[lid].centroid_y, abs=0.5)
            assert info_ms[lid].centroid_x == pytest.approx(info_eager[lid].centroid_x, abs=0.5)

    def test_rescale_to_finer(self):
        labels, _ = _make_brick_labels(gap=10)
        tree = _make_multiscale_tree(labels, n_scales=3)
        info_ms = compute_cell_info_multiscale(tree, target_scale="scale0")
        info_eager = compute_cell_info(labels)
        # Centroids should be close (within ~1 px due to coarse-scale quantization)
        assert set(info_ms.keys()) == set(info_eager.keys())
        for lid in info_ms:
            assert info_ms[lid].centroid_y == pytest.approx(info_eager[lid].centroid_y, abs=4.0)
            assert info_ms[lid].centroid_x == pytest.approx(info_eager[lid].centroid_x, abs=4.0)


class TestComputeCellInfoTiled:
    def test_matches_eager_no_cell_spans_tiles(self):
        labels, _ = _make_brick_labels(gap=10)  # cells are 20x30, well below chunk
        labels_da = xr.DataArray(labels, dims=("y", "x"))
        info_tiled = compute_cell_info_tiled(labels_da, chunk_size=128)
        info_eager = compute_cell_info(labels)
        assert set(info_tiled.keys()) == set(info_eager.keys())
        for lid in info_eager:
            assert info_tiled[lid].centroid_y == pytest.approx(info_eager[lid].centroid_y, abs=1e-6)
            assert info_tiled[lid].centroid_x == pytest.approx(info_eager[lid].centroid_x, abs=1e-6)
            assert info_tiled[lid].bbox_h == info_eager[lid].bbox_h
            assert info_tiled[lid].bbox_w == info_eager[lid].bbox_w

    def test_matches_eager_cells_span_tile_boundary(self):
        # A 100x100 cell crossing chunk boundary at 50.
        labels = np.zeros((200, 200), dtype=np.int32)
        labels[30:130, 30:130] = 1
        labels_da = xr.DataArray(labels, dims=("y", "x"))
        info_tiled = compute_cell_info_tiled(labels_da, chunk_size=50)
        info_eager = compute_cell_info(labels)
        assert set(info_tiled.keys()) == set(info_eager.keys())
        for lid in info_eager:
            assert info_tiled[lid].centroid_y == pytest.approx(info_eager[lid].centroid_y, abs=1e-6)
            assert info_tiled[lid].centroid_x == pytest.approx(info_eager[lid].centroid_x, abs=1e-6)
            assert info_tiled[lid].bbox_h == info_eager[lid].bbox_h
            assert info_tiled[lid].bbox_w == info_eager[lid].bbox_w

    def test_empty_labels(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels_da = xr.DataArray(labels, dims=("y", "x"))
        assert compute_cell_info_tiled(labels_da, chunk_size=32) == {}


class TestExtractTileLazy:
    def test_matches_eager(self, brick_labels, brick_image):
        labels, _ = brick_labels
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        labels_da = xr.DataArray(labels, dims=("y", "x"))
        image_da = xr.DataArray(brick_image, dims=("c", "y", "x"))
        for spec in specs:
            img_e, lbl_e = extract_tile(brick_image, labels, spec)
            img_l, lbl_l = extract_tile_lazy(image_da, labels_da, spec)
            np.testing.assert_array_equal(img_e, img_l)
            np.testing.assert_array_equal(lbl_e, lbl_l)


class TestVerifyCoverage:
    def test_detects_duplicate(self):
        spec_a = build_tile_specs((100, 100), {1: _make_ci(1, 25, 25)}, tile_size=50)
        spec_b = build_tile_specs((100, 100), {1: _make_ci(1, 25, 25)}, tile_size=50)
        with pytest.raises(ValueError, match="multiple tiles"):
            verify_coverage({1}, spec_a + spec_b)

    def test_detects_missing(self):
        specs = build_tile_specs((100, 100), {}, tile_size=50)
        with pytest.raises(ValueError, match="not assigned"):
            verify_coverage({42}, specs)

    def test_detects_extra(self):
        specs = build_tile_specs((100, 100), {1: _make_ci(1, 25, 25)}, tile_size=50, overlap_margin=0)
        with pytest.raises(ValueError, match="non-existent"):
            verify_coverage(set(), specs)


class TestTilingVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_tile_assignment_gap(self):
        """Visual: brick pattern (gap=10), cells colored by tile."""
        labels, _ = _make_brick_labels(gap=10)
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        _plot_tile_assignment(labels, specs, title="Tile assignment (gap=10)")

    def test_plot_tile_assignment_touching(self):
        """Visual: brick pattern (gap=0, touching), cells colored by tile."""
        labels, _ = _make_brick_labels(gap=0)
        specs = _specs_from_labels(labels, tile_size=_TILE_SIZE)
        _plot_tile_assignment(labels, specs, title="Tile assignment (gap=0, touching)")
