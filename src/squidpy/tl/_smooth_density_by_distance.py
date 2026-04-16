from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scanpy import logging as logg
from shapely import STRtree, points
from spatialdata import SpatialData

from squidpy._docs import d
from squidpy._validators import assert_key_in_sdata, assert_non_empty_sequence, assert_non_negative, assert_positive
from squidpy.gr._utils import _save_data
from squidpy.tl._ring_density import (
    _apply_affine_to_xy,
    _as_list,
    _get_affine_matrix,
    _prepare_contours,
    _resolve_feature_key,
    _resolve_metadata_columns,
    _signed_distance,
)

__all__ = ["smooth_density_by_distance"]

_GAUSSIAN_KERNEL = "gaussian"
_KERNEL_TRUNCATION = 4.0


@d.dedent
def smooth_density_by_distance(
    sdata: SpatialData,
    contour_key: str,
    bandwidth: float,
    table_key: str | None = None,
    points_key: str = "transcripts",
    feature_key: str | None = None,
    feature_values: str | Sequence[str] | None = None,
    grid_step: float | None = None,
    inward: float = 0.0,
    outward: float = 50.0,
    coordinate_system: str = "global",
    contour_query: str | None = None,
    metadata_keys: str | Sequence[str] | None = None,
    key_added: str = "smooth_density_by_distance",
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    Compute a smooth signed-distance density profile around contour annotations stored in ``SpatialData``.

    Parameters
    ----------
    sdata
        SpatialData object containing the contour annotations and transcript points.
    contour_key
        Key in ``sdata.shapes`` with the contour polygons used as density anchors.
    bandwidth
        Gaussian kernel bandwidth in the target coordinate system.
    table_key
        Key in ``sdata.tables`` used to store results in ``.uns`` when ``copy = False``.
    points_key
        Key in ``sdata.points`` with the transcript points.
    feature_key
        Column in the transcript points table used for feature filtering. If ``None`` and ``feature_values`` is
        provided, the key is inferred from SpatialData metadata.
    feature_values
        Optional transcript feature value or values to keep before counting.
    grid_step
        Distance between consecutive signed-distance evaluation points. Defaults to ``bandwidth / 4``.
    inward
        Maximum inward distance from the contour boundary.
    outward
        Maximum outward distance from the contour boundary.
    coordinate_system
        Coordinate system in which distances should be computed.
    contour_query
        Optional pandas query string used to subset the contours before computation.
    metadata_keys
        Optional contour metadata columns to copy into the output table. By default, a small useful subset is copied
        when present.
    key_added
        Key used when storing the resulting dataframe in ``sdata.tables[table_key].uns``.
    copy
        If ``True``, return a dataframe. Otherwise, store the result in ``.uns``.

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with one row per contour and signed-distance grid point.
    Otherwise, stores the dataframe in ``sdata.tables[table_key].uns[key_added]``.
    """
    start = logg.info(f"Computing {key_added}")

    assert_positive(bandwidth, name="bandwidth")
    assert_non_negative(inward, name="inward")
    assert_non_negative(outward, name="outward")
    if inward == 0 and outward == 0:
        raise ValueError("At least one of `inward` or `outward` must be greater than 0.")

    if grid_step is None:
        grid_step = bandwidth / 4.0
    assert_positive(grid_step, name="grid_step")

    assert_key_in_sdata(sdata, contour_key, attr="shapes")
    assert_key_in_sdata(sdata, points_key, attr="points")
    feature_values = (
        assert_non_empty_sequence(feature_values, name="feature_values")
        if feature_values is not None
        else None
    )

    contours = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        coordinate_system=coordinate_system,
        contour_query=contour_query,
    )
    if contours.empty:
        raise ValueError("No contours remain after applying the requested filters.")

    signed_distance_grid = _build_signed_distance_grid(inward=inward, outward=outward, grid_step=grid_step)
    contour_geometries = np.asarray(contours.geometry.values, dtype=object)
    contour_boundaries = np.asarray([geom.boundary for geom in contour_geometries], dtype=object)
    support_geometries = np.asarray(
        [
            _build_support_geometry(
                geom,
                inward=inward,
                outward=outward,
                cutoff_radius=_KERNEL_TRUNCATION * bandwidth,
            )
            for geom in contour_geometries
        ],
        dtype=object,
    )

    count_density = _count_smoothed_transcripts(
        sdata=sdata,
        points_key=points_key,
        coordinate_system=coordinate_system,
        feature_key=feature_key,
        feature_values=feature_values,
        contour_geometries=contour_geometries,
        contour_boundaries=contour_boundaries,
        support_geometries=support_geometries,
        signed_distance_grid=signed_distance_grid,
        inward=inward,
        outward=outward,
        bandwidth=bandwidth,
    )
    geometry_measure = _compute_geometry_measure(
        contour_geometries=contour_geometries,
        signed_distance_grid=signed_distance_grid,
        inward=inward,
        outward=outward,
        grid_step=grid_step,
    )

    result = _assemble_smooth_result(
        contours=contours,
        contour_key=contour_key,
        points_key=points_key,
        signed_distance_grid=signed_distance_grid,
        bandwidth=bandwidth,
        grid_step=grid_step,
        count_density=count_density,
        geometry_measure=geometry_measure,
        feature_values=feature_values,
        metadata_keys=metadata_keys,
    )

    if copy:
        logg.info("Finish", time=start)
        return result

    if table_key is None:
        raise ValueError("`table_key` must be specified when `copy=False` to determine where to store the result.")
    assert_key_in_sdata(sdata, table_key, attr="tables")
    _save_data(sdata.tables[table_key], attr="uns", key=key_added, data=result, time=start)
    return None


def _build_signed_distance_grid(inward: float, outward: float, grid_step: float) -> np.ndarray:
    grid = np.arange(-float(inward), float(outward) + grid_step, grid_step, dtype=float)
    if grid.size == 0:
        return np.asarray([-float(inward), float(outward)], dtype=float)
    if np.isclose(grid[-1], outward):
        grid[-1] = float(outward)
    elif grid[-1] < outward:
        grid = np.append(grid, float(outward))
    else:
        grid = np.append(grid[grid < outward], float(outward))
    return np.unique(grid)


def _build_support_geometry(geometry: Any, inward: float, outward: float, cutoff_radius: float) -> Any:
    outer = geometry.buffer(outward + cutoff_radius)
    inner = geometry.buffer(-(inward + cutoff_radius))
    if inner.is_empty:
        return outer
    return outer.difference(inner)


def _count_smoothed_transcripts(
    sdata: SpatialData,
    points_key: str,
    coordinate_system: str,
    feature_key: str | None,
    feature_values: Sequence[str] | None,
    contour_geometries: np.ndarray,
    contour_boundaries: np.ndarray,
    support_geometries: np.ndarray,
    signed_distance_grid: np.ndarray,
    inward: float,
    outward: float,
    bandwidth: float,
) -> np.ndarray:
    points_ddf = sdata.points[points_key]
    required_columns = ["x", "y"]
    if feature_values is not None:
        feature_key = _resolve_feature_key(points_ddf=points_ddf, feature_key=feature_key)
        required_columns.append(feature_key)
        feature_values = set(feature_values)
    points_ddf = points_ddf[required_columns]

    matrix = _get_affine_matrix(points_ddf, coordinate_system=coordinate_system)
    tree = STRtree(support_geometries)
    counts = np.zeros((len(contour_geometries), len(signed_distance_grid)), dtype=float)
    lower = -float(inward)
    upper = float(outward)

    for delayed_partition in points_ddf.to_delayed():
        partition = delayed_partition.compute()
        if feature_values is not None:
            partition = partition.loc[partition[feature_key].isin(feature_values)]
        if partition.empty:
            continue

        xy = partition[["x", "y"]].to_numpy(dtype=float)
        transformed_xy = _apply_affine_to_xy(xy, matrix)
        point_geometries = np.asarray(points(transformed_xy[:, 0], transformed_xy[:, 1]), dtype=object)

        pair_idx = tree.query(point_geometries)
        if pair_idx.size == 0:
            continue

        point_idx = pair_idx[0]
        contour_idx = pair_idx[1]
        signed = _signed_distance(
            point_geometries=point_geometries.take(point_idx),
            contour_geometries=contour_geometries.take(contour_idx),
            contour_boundaries=contour_boundaries.take(contour_idx),
        )
        valid = (signed >= lower) & (signed <= upper)
        if not np.any(valid):
            continue

        valid_contour_idx = contour_idx[valid]
        valid_signed = signed[valid]
        for current_contour_idx in np.unique(valid_contour_idx):
            signed_for_contour = valid_signed[valid_contour_idx == current_contour_idx]
            _accumulate_gaussian_with_reflection(
                counts[current_contour_idx],
                signed_distance_grid=signed_distance_grid,
                signed_distances=signed_for_contour,
                bandwidth=bandwidth,
                lower=lower,
                upper=upper,
            )

    return counts


def _accumulate_gaussian_with_reflection(
    accumulator: np.ndarray,
    signed_distance_grid: np.ndarray,
    signed_distances: np.ndarray,
    bandwidth: float,
    lower: float,
    upper: float,
    chunk_size: int = 2048,
) -> None:
    if signed_distances.size == 0:
        return

    reflected_left = 2.0 * lower - signed_distances
    reflected_right = 2.0 * upper - signed_distances
    cutoff = _KERNEL_TRUNCATION * bandwidth

    for start in range(0, signed_distances.size, chunk_size):
        slc = slice(start, start + chunk_size)
        accumulator += _gaussian_kernel_sum(
            signed_distance_grid=signed_distance_grid,
            sample_locations=signed_distances[slc],
            bandwidth=bandwidth,
            cutoff=cutoff,
        )
        accumulator += _gaussian_kernel_sum(
            signed_distance_grid=signed_distance_grid,
            sample_locations=reflected_left[slc],
            bandwidth=bandwidth,
            cutoff=cutoff,
        )
        accumulator += _gaussian_kernel_sum(
            signed_distance_grid=signed_distance_grid,
            sample_locations=reflected_right[slc],
            bandwidth=bandwidth,
            cutoff=cutoff,
        )


def _gaussian_kernel_sum(
    signed_distance_grid: np.ndarray,
    sample_locations: np.ndarray,
    bandwidth: float,
    cutoff: float,
) -> np.ndarray:
    if sample_locations.size == 0:
        return np.zeros(len(signed_distance_grid), dtype=float)

    diffs = signed_distance_grid[:, None] - sample_locations[None, :]
    mask = np.abs(diffs) <= cutoff
    scaled = diffs / bandwidth
    weights = np.exp(-0.5 * scaled**2) / (np.sqrt(2.0 * np.pi) * bandwidth)
    weights[~mask] = 0.0
    return np.asarray(weights.sum(axis=1), dtype=float)


def _compute_geometry_measure(
    contour_geometries: np.ndarray,
    signed_distance_grid: np.ndarray,
    inward: float,
    outward: float,
    grid_step: float,
) -> np.ndarray:
    geometry_measure = np.zeros((len(contour_geometries), len(signed_distance_grid)), dtype=float)
    lower = -float(inward)
    upper = float(outward)
    half_step = grid_step / 2.0

    for contour_idx, geometry in enumerate(contour_geometries):
        for grid_idx, signed_distance in enumerate(signed_distance_grid):
            interval_start = max(lower, signed_distance - half_step)
            interval_end = min(upper, signed_distance + half_step)
            interval_width = interval_end - interval_start
            if interval_width <= 0:
                geometry_measure[contour_idx, grid_idx] = 0.0
                continue

            local_area = _shell_area(geometry, interval_start, interval_end)
            geometry_measure[contour_idx, grid_idx] = local_area / interval_width if local_area > 0 else 0.0

    return geometry_measure


def _shell_area(geometry: Any, distance_start: float, distance_end: float) -> float:
    if distance_end <= distance_start:
        return 0.0

    if distance_end <= 0:
        outer = geometry.buffer(-abs(distance_end))
        inner = geometry.buffer(-abs(distance_start))
        return max(float(outer.area) - float(inner.area), 0.0)

    if distance_start >= 0:
        return max(float(geometry.buffer(distance_end).area) - float(geometry.buffer(distance_start).area), 0.0)

    inner_core = geometry.buffer(-abs(distance_start))
    return max(float(geometry.buffer(distance_end).area) - float(inner_core.area), 0.0)


def _assemble_smooth_result(
    contours: gpd.GeoDataFrame,
    contour_key: str,
    points_key: str,
    signed_distance_grid: np.ndarray,
    bandwidth: float,
    grid_step: float,
    count_density: np.ndarray,
    geometry_measure: np.ndarray,
    feature_values: Sequence[str] | None,
    metadata_keys: str | Sequence[str] | None,
) -> pd.DataFrame:
    metadata_columns = _resolve_metadata_columns(contours=contours, metadata_keys=metadata_keys)
    records: list[dict[str, Any]] = []
    feature_values_list = _as_list(feature_values) if feature_values is not None else None

    for contour_idx, contour_row in contours.iterrows():
        for grid_idx, signed_distance in enumerate(signed_distance_grid):
            local_geometry_measure = float(geometry_measure[contour_idx, grid_idx])
            local_count_density = float(count_density[contour_idx, grid_idx])
            record: dict[str, Any] = {
                "contour_key": contour_key,
                "contour_id": contour_row["_squidpy_contour_id"],
                "target": "transcripts",
                "target_key": points_key,
                "signed_distance": float(signed_distance),
                "bandwidth": float(bandwidth),
                "grid_step": float(grid_step),
                "kernel": _GAUSSIAN_KERNEL,
                "count_density": local_count_density,
                "geometry_measure": local_geometry_measure,
                "density": np.nan if local_geometry_measure <= 0 else local_count_density / local_geometry_measure,
            }
            if feature_values_list is not None:
                record["feature_values"] = tuple(feature_values_list)
            for column in metadata_columns:
                record[column] = contour_row[column]
            records.append(record)

    return pd.DataFrame.from_records(records)
