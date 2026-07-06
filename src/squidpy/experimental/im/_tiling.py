"""Cell-aware tiling for large images.

Splits a label image into overlapping tiles such that every cell is fully
contained in exactly one tile.  Cells are assigned to tiles by centroid:
the tile whose non-overlapping base region contains the centroid owns the
cell.  Non-owned cells are zeroed out in each tile's mask so that
downstream processing never double-counts.

All functions accept pre-computed centroid dicts and image shapes; they
never materialize the full image or label array.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr
from skimage.measure import regionprops
from spatialdata._logging import logger as logg

from squidpy._utils import _get_n_cores


def yx_size(da: xr.DataArray) -> tuple[int, int]:
    """``(height, width)`` of a DataArray, falling back to its last two axes."""
    return int(da.sizes.get("y", da.shape[-2])), int(da.sizes.get("x", da.shape[-1]))


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Drop singleton leading dims so a labels array is 2-D."""
    return arr.squeeze() if arr.ndim > 2 else arr


def _materialize(da_slice: xr.DataArray) -> np.ndarray:
    """Materialize a crop to numpy with the synchronous scheduler.

    A bare ``.values`` inside a distributed worker would route ``.compute()`` back
    to the scheduler and deadlock; forcing synchronous keeps the read worker-local.
    No-op for numpy-backed arrays.
    """
    return da_slice.compute(scheduler="synchronous").values


@dataclass(frozen=True)
class CellInfo:
    """Centroid and bounding box for a single label."""

    label: int
    centroid_y: float
    centroid_x: float
    bbox_h: int  # height of bounding box
    bbox_w: int  # width of bounding box
    bbox_y0: int = 0  # top edge (row) of bounding box
    bbox_x0: int = 0  # left edge (col) of bounding box


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


# Centroid computation


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
            bbox_y0=min_row,
            bbox_x0=min_col,
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
        h, w = yx_size(labels_node[k].ds["image"])
        return h * w

    coarsest = min(available, key=_spatial_size)
    coarse_labels = np.asarray(labels_node[coarsest].ds["image"].values).squeeze()

    if coarse_labels.ndim != 2:
        raise ValueError(f"Expected 2-D labels at scale {coarsest}, got shape {coarse_labels.shape}")

    target_h, target_w = yx_size(labels_node[target_scale].ds["image"])
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
            bbox_y0=int(np.floor(p.bbox[0] * scale_y)),
            bbox_x0=int(np.floor(p.bbox[1] * scale_x)),
        )
        for p in props
    }


@dataclass
class _Accum:
    """Per-label running totals while streaming chunks (a cell may span chunks)."""

    sum_y: float = 0.0  # centroid_y * area, summed across chunks (for area-weighted centroid)
    sum_x: float = 0.0
    area: float = 0.0
    min_y: float = np.inf
    max_y: float = -np.inf
    min_x: float = np.inf
    max_x: float = -np.inf


def compute_cell_info_tiled(
    labels_da: xr.DataArray,
    chunk_size: int = 4096,
) -> dict[int, CellInfo]:
    """Compute per-label centroids and bounding boxes without materializing the full array.

    The labels are read in ``chunk_size`` blocks. This chunking is internal to the
    scan and is independent of the featurization tiles from :func:`build_tile_specs`.
    A label spanning a block boundary is partitioned across blocks; its centroid is
    recovered as the area-weighted mean of the per-block centroids and its bounding
    box as the union of the per-block boxes.

    Parameters
    ----------
    labels_da
        2-D (y, x) dask-backed xarray DataArray.
    chunk_size
        Side length in pixels of each read block.
    """
    height, width = yx_size(labels_da)
    accums: dict[int, _Accum] = {}

    for y0 in range(0, height, chunk_size):
        for x0 in range(0, width, chunk_size):
            chunk = _as_2d(labels_da.isel(y=slice(y0, y0 + chunk_size), x=slice(x0, x0 + chunk_size)).values)
            for prop in regionprops(chunk):
                a = accums.setdefault(prop.label, _Accum())
                area = float(prop.area)
                a.area += area
                a.sum_y += (prop.centroid[0] + y0) * area
                a.sum_x += (prop.centroid[1] + x0) * area
                a.min_y, a.max_y = min(a.min_y, prop.bbox[0] + y0), max(a.max_y, prop.bbox[2] + y0)
                a.min_x, a.max_x = min(a.min_x, prop.bbox[1] + x0), max(a.max_x, prop.bbox[3] + x0)

    return {
        lid: CellInfo(
            label=lid,
            centroid_y=a.sum_y / a.area,
            centroid_x=a.sum_x / a.area,
            bbox_h=int(a.max_y - a.min_y),
            bbox_w=int(a.max_x - a.min_x),
            bbox_y0=int(a.min_y),
            bbox_x0=int(a.min_x),
        )
        for lid, a in accums.items()
        if lid != 0
    }


# Tile spec building


def _auto_margin(cell_info: dict[int, CellInfo]) -> int:
    """Compute the minimum margin that covers the largest cell's half-extent."""
    if not cell_info:
        return 0
    max_extent = max(max(c.bbox_h, c.bbox_w) for c in cell_info.values())
    # Centroid can be at most half a bbox away from the cell's edge.
    # Add 1 pixel for safety (rounding / off-by-one).
    return int(np.ceil(max_extent / 2)) + 1


def build_tile_specs(
    grid_shape: tuple[int, int],
    cell_info: dict[int, CellInfo],
    tile_size: int = 2048,
    overlap_margin: int | Literal["auto"] = "auto",
) -> list[TileSpec]:
    """Build tile specifications from pre-computed centroids.

    No pixel data is needed, only the grid dimensions and centroid dict.

    Parameters
    ----------
    grid_shape
        ``(height, width)`` of the full-resolution labels grid.
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
    height, width = grid_shape
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    margin = _auto_margin(cell_info) if overlap_margin == "auto" else int(overlap_margin)
    if margin < 0:
        raise ValueError(f"overlap_margin must be non-negative, got {margin}")

    cell_to_tile: dict[int, tuple[int, int]] = {}
    for lid, cell in cell_info.items():
        tile_row = min(int(cell.centroid_y) // tile_size, (height - 1) // tile_size)
        tile_col = min(int(cell.centroid_x) // tile_size, (width - 1) // tile_size)
        cell_to_tile[lid] = (tile_row, tile_col)

    tile_to_cells: dict[tuple[int, int], set[int]] = {}
    for lid, key in cell_to_tile.items():
        tile_to_cells.setdefault(key, set()).add(lid)

    specs: list[TileSpec] = []
    for (row, col), owned in sorted(tile_to_cells.items()):
        by0 = row * tile_size
        bx0 = col * tile_size
        by1 = min(by0 + tile_size, height)
        bx1 = min(bx0 + tile_size, width)

        cy0 = max(by0 - margin, 0)
        cx0 = max(bx0 - margin, 0)
        cy1 = min(by1 + margin, height)
        cx1 = min(bx1 + margin, width)

        specs.append(
            TileSpec(
                base=(by0, bx0, by1, bx1),
                crop=(cy0, cx0, cy1, cx1),
                owned_ids=frozenset(owned),
            )
        )

    return specs


# Tile extraction


def extract_tile_lazy(
    image_da: xr.DataArray,
    labels_da: xr.DataArray,
    spec: TileSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a tile from dask-backed xarray arrays.

    Materializes only the tile's crop region (~2k x 2k), not the full image.

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
    tile_image = _materialize(image_da.isel(y=slice(cy0, cy1), x=slice(cx0, cx1)))
    return tile_image, extract_labels_tile_lazy(labels_da, spec)


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
    tile_labels = _as_2d(_materialize(labels_da.isel(y=slice(cy0, cy1), x=slice(cx0, cx1))).copy())
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


# Tiled execution engine
# ----------------------
#
# Shared by the tiled featurizers (calculate_image_features, calculate_tiling_qc):
# run a per-tile function over `build_tile_specs` output across an appropriate
# scheduler.  The two callers differ only in `kind`: tiling QC's per-tile work is
# numba `nogil` (threads scale), while image featurization is GIL-bound Python
# (needs processes).  Everything else - client detection, worker-count
# resolution, streaming collection, progress - is common.


def _has_distributed_client() -> bool:
    """Return ``True`` iff a ``dask.distributed.Client`` is active in this process.

    Mirrors the public dask idiom: if a Client is in scope, work submitted to it
    runs there automatically.  ``ImportError`` guards a dask install without the
    distributed extra; ``ValueError`` is what ``get_client`` raises when no
    Client is active.
    """
    try:
        from dask.distributed import get_client

        get_client()
    except (ImportError, ValueError):
        return False
    return True


def _log_progress(done: int, total: int, desc: str) -> None:
    """Emit a ~10% heartbeat so long (Xenium-scale) runs show liveness."""
    if total <= 0:
        return
    step = max(1, total // 10)
    if done == 1 or done == total or done % step == 0:
        logg.info(f"Processed {done}/{total} {desc}.")


def _run_on_client(
    client: Any,
    specs: Sequence[Any],
    process_fn: Callable[..., Any],
    scatter: Sequence[Any],
    desc: str,
) -> list[Any]:
    """Submit one task per spec to ``client`` and collect results in spec order.

    ``scatter`` objects are sent to the workers once (broadcast) so their backing
    graph is not re-embedded in every task. Each result is gathered and its future
    released as it completes (``as_completed``), so worker memory is reclaimed
    incrementally rather than pinning every per-tile result until the end.
    """
    from dask.distributed import as_completed

    scattered = client.scatter(list(scatter), broadcast=True) if scatter else []
    futures = [client.submit(process_fn, spec, *scattered, pure=False) for spec in specs]
    order = {fut.key: i for i, fut in enumerate(futures)}
    results: list[Any] = [None] * len(futures)
    for done, (fut, res) in enumerate(as_completed(futures, with_results=True), start=1):
        results[order[fut.key]] = res
        fut.release()  # the driver now holds the result; free the worker-side copy
        _log_progress(done, len(futures), desc)
    return results


def _run_tiled(
    specs: Sequence[Any],
    process_fn: Callable[..., Any],
    *,
    n_jobs: int = 1,
    kind: Literal["threads", "processes"] = "processes",
    scatter: Sequence[Any] = (),
    desc: str = "tiles",
) -> list[Any]:
    """Run ``process_fn(spec, *scatter)`` over tile ``specs``; return results in spec order.

    Engine selection: an active ``distributed.Client`` wins; else ``n_jobs`` (repo
    ``_get_n_cores`` convention, ``1``/``0``/``None`` serial) picks workers and
    ``kind`` picks the scheduler -- ``"threads"`` for GIL-releasing work (numba
    ``nogil``), ``"processes"`` for GIL-bound work (a ``LocalCluster``, since the
    local multiprocessing scheduler does not fork). ``scatter`` holds large objects
    passed after ``spec`` (broadcast once on the distributed path); ``process_fn``
    must be picklable for ``kind="processes"``.
    """
    n = len(specs)

    # An active Client always wins, regardless of `kind` or `n_jobs`.
    if _has_distributed_client():
        from dask.distributed import get_client

        # Warn only when an explicit worker count (not a default) is overridden.
        if n_jobs not in (None, 1, -1):
            logg.warning("`n_jobs` is ignored when an active dask.distributed Client is in scope.")
        return _run_on_client(get_client(), specs, process_fn, scatter, desc)

    workers = 1 if n_jobs in (None, 0) else _get_n_cores(n_jobs)

    # GIL-bound work with >1 worker needs real worker processes: the local
    # multiprocessing scheduler does not fork for this graph, so use a
    # distributed LocalCluster. (dask's ProgressBar cannot observe a distributed
    # cluster; _run_on_client emits its own heartbeat.)
    if kind == "processes" and workers > 1 and n > 1:
        from dask.distributed import Client, LocalCluster

        with (
            LocalCluster(n_workers=workers, threads_per_worker=1, processes=True, dashboard_address=None) as cluster,
            Client(cluster) as client,
        ):
            return _run_on_client(client, specs, process_fn, scatter, desc)

    # Local dask scheduler with a ProgressBar: synchronous for serial / single
    # tile, threads for GIL-releasing work. Bind scatter into the closure rather
    # than passing the dask-backed arrays as delayed args -- dask materializes a
    # collection given to delayed in full before the call, reading the whole
    # image per tile instead of each tile's crop.
    import dask
    from dask.diagnostics import ProgressBar

    scheduler = "synchronous" if (workers == 1 or n <= 1) else "threads"

    def _task(spec):
        return process_fn(spec, *scatter)

    tasks = [dask.delayed(_task)(spec) for spec in specs]
    with ProgressBar():
        return list(dask.compute(*tasks, scheduler=scheduler, num_workers=workers))
