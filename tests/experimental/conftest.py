"""Shared fixtures for experimental tests.

Provides synthetic SpatialData objects for testing segmentation QC metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy import ndimage
from skimage.draw import ellipse
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

# ---------------------------------------------------------------------------
# Tile-boundary QC fixture
# ---------------------------------------------------------------------------

_IMAGE_SIZE = 400
_TILE_BORDERS = (133, 267)  # 3×3 grid on 400 px → borders at 133, 267
_BORDER_GAP = 2  # pixels zeroed at each tile border
_CELL_GAP = 2  # minimum gap between any two cells
_N_CELLS_TARGET = 40
_SEMI_AXIS_RANGE = (8, 20)  # semi-axis lengths in pixels


@dataclass
class TileBoundaryGroundTruth:
    """Ground-truth metadata for the tile-boundary fixture."""

    cut_cell_ids: frozenset[int] = field(default_factory=frozenset)
    intact_cell_ids: frozenset[int] = field(default_factory=frozenset)
    original_n_cells: int = 0
    tile_borders_y: tuple[int, ...] = _TILE_BORDERS
    tile_borders_x: tuple[int, ...] = _TILE_BORDERS


def _place_ellipsoids(
    shape: tuple[int, int],
    n_target: int,
    semi_range: tuple[int, int],
    cell_gap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place non-overlapping ellipsoids via rejection sampling.

    Returns an ``(H, W)`` int32 label array with IDs 1..N.
    """
    H, W = shape
    labels = np.zeros(shape, dtype=np.int32)
    cell_id = 0
    max_attempts = n_target * 20  # allow generous retries

    for _ in range(max_attempts):
        if cell_id >= n_target:
            break

        # Random ellipse parameters
        cy = rng.integers(semi_range[1] + cell_gap, H - semi_range[1] - cell_gap)
        cx = rng.integers(semi_range[1] + cell_gap, W - semi_range[1] - cell_gap)
        r_radius = rng.integers(semi_range[0], semi_range[1] + 1)
        c_radius = rng.integers(semi_range[0], semi_range[1] + 1)
        angle = rng.uniform(0, np.pi)

        # Rasterise candidate ellipse
        rr, cc = ellipse(cy, cx, r_radius, c_radius, shape=shape, rotation=angle)
        if len(rr) == 0:
            continue

        # Check for overlap (including gap buffer)
        # Dilate existing labels by cell_gap and check candidate pixels
        if cell_gap > 0:
            occupied = ndimage.binary_dilation(
                labels > 0,
                iterations=cell_gap,
            )
        else:
            occupied = labels > 0

        if occupied[rr, cc].any():
            continue

        cell_id += 1
        labels[rr, cc] = cell_id

    return labels


def _apply_tile_cuts(
    labels: np.ndarray,
    borders_y: tuple[int, ...],
    borders_x: tuple[int, ...],
    gap: int,
) -> np.ndarray:
    """Zero out pixels along tile borders to simulate segmentation seams.

    For each border coordinate, a stripe of width ``gap`` centred on the
    border is erased.
    """
    out = labels.copy()
    half = gap // 2

    for by in borders_y:
        out[by - half : by - half + gap, :] = 0
    for bx in borders_x:
        out[:, bx - half : bx - half + gap] = 0

    return out


def _relabel_and_track(
    original: np.ndarray,
    cut: np.ndarray,
) -> tuple[np.ndarray, frozenset[int], frozenset[int]]:
    """Relabel connected components after cutting and classify fragments.

    Returns
    -------
    relabelled
        New label array with unique IDs for each fragment.
    cut_ids
        Fragment IDs that came from an original cell that was split.
    intact_ids
        Fragment IDs from cells that remained whole.
    """
    # Relabel connected components (each fragment gets a new ID)
    relabelled, n_fragments = ndimage.label(cut > 0)

    # Map each fragment back to its original cell ID
    # For each new fragment, find which original cell(s) it overlaps with
    cut_ids: set[int] = set()
    intact_ids: set[int] = set()

    # Build reverse mapping: original_id → set of fragment_ids
    orig_to_fragments: dict[int, set[int]] = {}
    for frag_id in range(1, n_fragments + 1):
        frag_mask = relabelled == frag_id
        orig_ids_in_frag = set(np.unique(original[frag_mask])) - {0}

        for oid in orig_ids_in_frag:
            orig_to_fragments.setdefault(oid, set()).add(frag_id)

    # Classify: if an original cell maps to >1 fragment, all its fragments are "cut"
    for _orig_id, frag_set in orig_to_fragments.items():
        if len(frag_set) > 1:
            cut_ids.update(frag_set)
        else:
            intact_ids.update(frag_set)

    return relabelled, frozenset(cut_ids), frozenset(intact_ids)


def make_tile_boundary_sdata() -> tuple[SpatialData, TileBoundaryGroundTruth]:
    """Build a 400x400 SpatialData with ellipsoid cells cut by a 3x3 tile grid.

    Returns a tuple of ``(sdata, ground_truth)`` where ``ground_truth``
    contains the sets of cut and intact cell IDs for test assertions.

    The labels are dask-backed to exercise lazy codepaths.
    """
    rng = np.random.default_rng(42)

    # 1. Place ellipsoids
    original_labels = _place_ellipsoids(
        shape=(_IMAGE_SIZE, _IMAGE_SIZE),
        n_target=_N_CELLS_TARGET,
        semi_range=_SEMI_AXIS_RANGE,
        cell_gap=_CELL_GAP,
        rng=rng,
    )
    n_original = len(np.unique(original_labels)) - 1  # exclude background

    # 2. Apply tile cuts (zero out 2px stripes at borders)
    cut_labels = _apply_tile_cuts(
        original_labels,
        borders_y=_TILE_BORDERS,
        borders_x=_TILE_BORDERS,
        gap=_BORDER_GAP,
    )

    # 3. Relabel fragments and track ground truth
    relabelled, cut_ids, intact_ids = _relabel_and_track(original_labels, cut_labels)

    # 4. Wrap as dask array (chunks matching ~tile size for realistic access)
    dask_labels = da.from_array(relabelled, chunks=(200, 200))
    labels_xr = xr.DataArray(dask_labels, dims=["y", "x"])

    # 5. Dummy image for API compatibility
    image_data = rng.integers(0, 255, (3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint8)
    image_xr = xr.DataArray(image_data, dims=["c", "y", "x"], coords={"c": ["R", "G", "B"]})

    sdata = SpatialData(
        images={"image": Image2DModel.parse(image_xr)},
        labels={"labels": Labels2DModel.parse(labels_xr)},
    )

    ground_truth = TileBoundaryGroundTruth(
        cut_cell_ids=cut_ids,
        intact_cell_ids=intact_ids,
        original_n_cells=n_original,
    )

    return sdata, ground_truth


@pytest.fixture()
def sdata_tile_boundary() -> tuple[SpatialData, TileBoundaryGroundTruth]:
    """Fixture wrapper around :func:`make_tile_boundary_sdata`."""
    return make_tile_boundary_sdata()
