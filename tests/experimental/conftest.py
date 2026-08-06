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

# Tile-boundary QC fixture

_IMAGE_SIZE = 600
_TILE_BORDERS = (200, 400)  # 3x3 grid on 600 px - borders at 200, 400
_BORDER_GAP = 2  # pixels zeroed at each tile border
_SEMI_AXIS_RANGE = (5, 10)  # semi-axis lengths in pixels
_GRID_STEP = 24  # spacing between cell centers on the grid


@dataclass
class TileBoundaryGroundTruth:
    """Ground-truth metadata for the tile-boundary fixture."""

    cut_cell_ids: frozenset[int] = field(default_factory=frozenset)
    intact_cell_ids: frozenset[int] = field(default_factory=frozenset)
    original_n_cells: int = 0
    tile_borders_y: tuple[int, ...] = _TILE_BORDERS
    tile_borders_x: tuple[int, ...] = _TILE_BORDERS


def _place_ellipsoids_grid(
    shape: tuple[int, int],
    semi_range: tuple[int, int],
    grid_step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place non-overlapping ellipsoids on a jittered grid.

    Cell centers are placed on a regular grid with spacing ``grid_step``,
    then jittered by a small random offset.  Each cell gets random
    semi-axes and rotation.  The grid guarantees no overlaps as long as
    ``grid_step >= 2 * semi_range[1] + margin``, so no collision
    checking is needed.

    Returns an ``(H, W)`` int32 label array with IDs 1..N.
    """
    H, W = shape
    labels = np.zeros(shape, dtype=np.int32)
    margin = semi_range[1] + 1
    max_jitter = (grid_step - 2 * semi_range[1]) // 2

    # Build grid centers
    ys = np.arange(margin, H - margin, grid_step)
    xs = np.arange(margin, W - margin, grid_step)

    cell_id = 0
    for y0 in ys:
        for x0 in xs:
            cy = y0 + rng.integers(-max_jitter, max_jitter + 1)
            cx = x0 + rng.integers(-max_jitter, max_jitter + 1)
            r_radius = rng.integers(semi_range[0], semi_range[1] + 1)
            c_radius = rng.integers(semi_range[0], semi_range[1] + 1)
            angle = rng.uniform(0, np.pi)

            rr, cc = ellipse(cy, cx, r_radius, c_radius, shape=shape, rotation=angle)
            if len(rr) == 0:
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
    relabelled, n_fragments = ndimage.label(cut > 0)

    cut_ids: set[int] = set()
    intact_ids: set[int] = set()

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

    original_labels = _place_ellipsoids_grid(
        shape=(_IMAGE_SIZE, _IMAGE_SIZE),
        semi_range=_SEMI_AXIS_RANGE,
        grid_step=_GRID_STEP,
        rng=rng,
    )
    n_original = len(np.unique(original_labels)) - 1

    cut_labels = _apply_tile_cuts(
        original_labels,
        borders_y=_TILE_BORDERS,
        borders_x=_TILE_BORDERS,
        gap=_BORDER_GAP,
    )

    relabelled, cut_ids, intact_ids = _relabel_and_track(original_labels, cut_labels)

    dask_labels = da.from_array(relabelled, chunks=(200, 200))
    labels_xr = xr.DataArray(dask_labels, dims=["y", "x"])

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


def make_clean_sdata() -> SpatialData:
    """Build a SpatialData with natural ellipsoid cells and NO tile cuts.

    This is the negative control: no tiling artifacts exist, so the
    spatial post-processing should flag zero outliers.
    """
    rng = np.random.default_rng(123)
    labels = _place_ellipsoids_grid(
        shape=(_IMAGE_SIZE, _IMAGE_SIZE),
        semi_range=_SEMI_AXIS_RANGE,
        grid_step=_GRID_STEP,
        rng=rng,
    )
    dask_labels = da.from_array(labels, chunks=(200, 200))
    labels_xr = xr.DataArray(dask_labels, dims=["y", "x"])

    image_data = rng.integers(0, 255, (3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint8)
    image_xr = xr.DataArray(image_data, dims=["c", "y", "x"], coords={"c": ["R", "G", "B"]})

    return SpatialData(
        images={"image": Image2DModel.parse(image_xr)},
        labels={"labels": Labels2DModel.parse(labels_xr)},
    )


@pytest.fixture()
def sdata_clean() -> SpatialData:
    """Fixture wrapper around :func:`make_clean_sdata`."""
    return make_clean_sdata()


# Dense-tissue + wide-gap + single-sided seam fixture (stresses emergent-seam detection)

_DENSE_SIZE = 420
_DENSE_BORDERS = (140, 280)  # two internal seams per axis (3x3 tile grid)
_DENSE_GAP = 8  # wide inter-FOV gap (>> the current stitcher's 3px max_gap)


@dataclass
class DenseSeamGroundTruth:
    """Ground truth for the dense-tissue seam fixture."""

    cut_cell_ids: frozenset[int] = field(default_factory=frozenset)
    seam_coords: tuple[int, ...] = _DENSE_BORDERS
    gap: int = _DENSE_GAP


def make_dense_seam_sdata() -> tuple[SpatialData, DenseSeamGroundTruth]:
    """Dense **touching** Voronoi cells cut by a wide seam, with single-sided drop-out.

    Mimics CosMx-style per-FOV segmentation on dense tissue: convex Voronoi cells with
    thin membranes, a wide zeroed seam band, and small remnants dropped so many cuts leave
    only one segmentable half.  Convex cells never self-fragment, so the ground truth (a
    cell is cut iff its pre-cut mask overlaps the seam band) is exact.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    size, borders, gap = _DENSE_SIZE, _DENSE_BORDERS, _DENSE_GAP
    pts = [(y + rng.integers(-4, 5), x + rng.integers(-4, 5)) for y in range(15, size, 15) for x in range(15, size, 15)]
    tree = cKDTree(np.array(pts))
    yy, xx = np.mgrid[0:size, 0:size]
    d, idx = tree.query(np.column_stack([yy.ravel(), xx.ravel()]), k=2)
    cell = (idx[:, 0] + 1).reshape(size, size).astype(np.int32)
    cell[((d[:, 1] - d[:, 0]) < 1.5).reshape(size, size)] = 0  # thin membranes
    orig = cell.copy()

    half = gap // 2
    seam = np.zeros((size, size), bool)
    for b in borders:
        seam[b - half : b - half + gap, :] = True
        seam[:, b - half : b - half + gap] = True
    cut_orig = set(np.unique(orig[seam & (orig > 0)])) - {0}
    orig_area = {i: int((orig == i).sum()) for i in np.unique(orig) if i}

    seg = orig.copy()
    seg[seam] = 0
    frag, nfrag = ndimage.label(seg > 0)
    parent = np.zeros(nfrag + 1, np.int64)
    frags_of: dict[int, list[int]] = {}
    for f in range(1, nfrag + 1):
        vals = orig[frag == f]
        vals = vals[vals > 0]
        parent[f] = np.bincount(vals).argmax() if vals.size else 0
        frags_of.setdefault(int(parent[f]), []).append(f)

    keep = np.ones(nfrag + 1, bool)
    keep[0] = False
    for o in cut_orig:
        for f in frags_of.get(o, []):
            if (frag == f).sum() / max(orig_area.get(o, 1), 1) < 0.45 and rng.random() < 0.75:
                keep[f] = False
    frag2, nf2 = ndimage.label(np.where(keep[frag], frag, 0) > 0)

    cut_ids: set[int] = set()
    for f in range(1, nf2 + 1):
        vals = orig[frag2 == f]
        vals = vals[vals > 0]
        if vals.size and int(np.bincount(vals).argmax()) in cut_orig:
            cut_ids.add(f)

    labels_xr = xr.DataArray(da.from_array(frag2.astype(np.int32), chunks=(128, 128)), dims=["y", "x"])
    image_xr = xr.DataArray(
        rng.integers(0, 255, (3, size, size), dtype=np.uint8), dims=["c", "y", "x"], coords={"c": ["R", "G", "B"]}
    )
    sdata = SpatialData(
        images={"image": Image2DModel.parse(image_xr)},
        labels={"labels": Labels2DModel.parse(labels_xr)},
    )
    return sdata, DenseSeamGroundTruth(cut_cell_ids=frozenset(cut_ids))


@pytest.fixture()
def sdata_dense_seam() -> tuple[SpatialData, DenseSeamGroundTruth]:
    """Fixture wrapper around :func:`make_dense_seam_sdata`."""
    return make_dense_seam_sdata()
