from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scanpy import logging as logg
from shapely import STRtree, affinity, boundary, contains, distance, points
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from squidpy._docs import d
from squidpy._validators import (
    assert_key_in_sdata,
    assert_non_empty_sequence,
    assert_non_negative,
    assert_one_of,
    assert_positive,
)
from squidpy.gr._utils import _save_data

__all__ = ["ring_density"]


@d.dedent
def ring_density(
    sdata: SpatialData,
    contour_key: str,
    target: Literal["cells", "transcripts"] = "cells",
    table_key: str | None = None,
    points_key: str = "transcripts",
    shape_key: str | None = None,
    ring_width: float = 50.0,
    inward: float = 0.0,
    outward: float = 50.0,
    coordinate_system: str = "global",
    contour_query: str | None = None,
    feature_key: str | None = None,
    feature_values: str | Sequence[str] | None = None,
    metadata_keys: str | Sequence[str] | None = None,
    key_added: str = "ring_density",
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    Compute inward/outward ring density around contour annotations stored in ``SpatialData``.

    Parameters
    ----------
    sdata
        SpatialData object containing the contour annotations and the target elements.
    contour_key
        Key in ``sdata.shapes`` with the contour polygons used as density anchors.
    target
        Whether to count cell centroids or transcript points.
    table_key
        Key in ``sdata.tables`` used to infer the cell-associated shape layer and to store results in ``.uns`` when
        ``copy = False``.
    points_key
        Key in ``sdata.points`` used when ``target = 'transcripts'``.
    shape_key
        Optional key in ``sdata.shapes`` to use when ``target = 'cells'``. If omitted, the region linked to
        ``sdata.tables[table_key]`` is used.
    ring_width
        Width of each ring in the target coordinate system.
    inward
        Maximum inward distance from the contour boundary.
    outward
        Maximum outward distance from the contour boundary.
    coordinate_system
        Coordinate system in which distances should be computed.
    contour_query
        Optional pandas query string used to subset the contours before computation.
    feature_key
        Column in the transcript points table used for feature filtering. If ``None`` and ``feature_values`` is
        provided, the key is inferred from SpatialData metadata.
    feature_values
        Optional transcript feature value or values to keep before counting.
    metadata_keys
        Optional contour metadata columns to copy into the output table. By default, a small useful subset is copied
        when present.
    key_added
        Key used when storing the resulting dataframe in ``sdata.tables[table_key].uns``.
    copy
        If ``True``, return a dataframe. Otherwise, store the result in ``.uns``.

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with one row per contour and ring.
    Otherwise, stores the dataframe in ``sdata.tables[table_key].uns[key_added]``.
    """
    start = logg.info(f"Computing {key_added}")

    assert_positive(ring_width, name="ring_width")
    assert_non_negative(inward, name="inward")
    assert_non_negative(outward, name="outward")
    if inward == 0 and outward == 0:
        raise ValueError("At least one of `inward` or `outward` must be greater than 0.")
    assert_key_in_sdata(sdata, contour_key, attr="shapes")
    assert_one_of(target, ["cells", "transcripts"], name="target")

    feature_values = (
        assert_non_empty_sequence(feature_values, name="feature_values") if feature_values is not None else None
    )

    contours = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        coordinate_system=coordinate_system,
        contour_query=contour_query,
    )
    if contours.empty:
        raise ValueError("No contours remain after applying the requested filters.")

    intervals = _build_ring_intervals(ring_width=ring_width, inward=inward, outward=outward)
    interval_edges = np.asarray([intervals[0][0], *[end for _, end in intervals]], dtype=float)

    contour_geometries = np.asarray(contours.geometry.values, dtype=object)
    contour_boundaries = np.asarray([boundary(geom) for geom in contour_geometries], dtype=object)
    max_radius = max(inward, outward)
    expanded_geometries = np.asarray(
        [geom.buffer(max_radius) if max_radius > 0 else geom for geom in contour_geometries],
        dtype=object,
    )

    if target == "cells":
        resolved_shape_key = _resolve_shape_key(sdata=sdata, table_key=table_key, shape_key=shape_key)
        target_geometries = _prepare_cell_targets(
            sdata=sdata,
            table_key=table_key,
            shape_key=resolved_shape_key,
            coordinate_system=coordinate_system,
        )
        counts = _count_geometry_targets(
            target_geometries=target_geometries,
            expanded_geometries=expanded_geometries,
            contour_geometries=contour_geometries,
            contour_boundaries=contour_boundaries,
            interval_edges=interval_edges,
        )
        target_key = resolved_shape_key
    else:
        counts = _count_transcript_targets(
            sdata=sdata,
            points_key=points_key,
            coordinate_system=coordinate_system,
            expanded_geometries=expanded_geometries,
            contour_geometries=contour_geometries,
            contour_boundaries=contour_boundaries,
            interval_edges=interval_edges,
            feature_key=feature_key,
            feature_values=feature_values,
        )
        target_key = points_key

    areas = np.asarray(
        [
            [_ring_area(geom, ring_start=ring_start, ring_end=ring_end) for ring_start, ring_end in intervals]
            for geom in contour_geometries
        ],
        dtype=float,
    )

    result = _assemble_result(
        contours=contours,
        intervals=intervals,
        counts=counts,
        areas=areas,
        contour_key=contour_key,
        target=target,
        target_key=target_key,
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


def _prepare_contours(
    sdata: SpatialData,
    contour_key: str,
    coordinate_system: str,
    contour_query: str | None,
) -> gpd.GeoDataFrame:
    contours = sdata.shapes[contour_key].copy()
    matrix = _get_affine_matrix(sdata.shapes[contour_key], coordinate_system=coordinate_system)
    contours.geometry = contours.geometry.apply(lambda geom: _apply_affine_to_geometry(geom, matrix))
    if contour_query is not None:
        contours = contours.query(contour_query)
    return contours.reset_index(names="_squidpy_contour_id")


def _prepare_cell_targets(
    sdata: SpatialData,
    table_key: str | None,
    shape_key: str,
    coordinate_system: str,
) -> np.ndarray:
    shapes = sdata.shapes[shape_key]
    table_index: pd.Index | None = None
    if table_key is not None:
        assert_key_in_sdata(sdata, table_key, attr="tables")
        table_index = sdata.tables[table_key].obs_names

    if table_index is not None:
        keep_mask = shapes.index.isin(table_index)
        shapes = shapes.loc[keep_mask]

    geometry = shapes.geometry
    if (geometry.geom_type == "Point").all():
        x = geometry.x.to_numpy(dtype=float)
        y = geometry.y.to_numpy(dtype=float)
    else:
        centroids = geometry.centroid
        x = centroids.x.to_numpy(dtype=float)
        y = centroids.y.to_numpy(dtype=float)

    matrix = _get_affine_matrix(sdata.shapes[shape_key], coordinate_system=coordinate_system)
    transformed_xy = _apply_affine_to_xy(np.column_stack([x, y]), matrix)
    return np.asarray(points(transformed_xy[:, 0], transformed_xy[:, 1]), dtype=object)


def _count_geometry_targets(
    target_geometries: np.ndarray,
    expanded_geometries: np.ndarray,
    contour_geometries: np.ndarray,
    contour_boundaries: np.ndarray,
    interval_edges: np.ndarray,
) -> np.ndarray:
    counts = np.zeros((len(contour_geometries), len(interval_edges) - 1), dtype=np.int64)
    if len(target_geometries) == 0:
        return counts

    tree = STRtree(target_geometries)
    for contour_idx, expanded_geom in enumerate(expanded_geometries):
        target_idx = tree.query(expanded_geom)
        if len(target_idx) == 0:
            continue

        candidate_points = target_geometries.take(target_idx)
        signed = _signed_distance(
            point_geometries=candidate_points,
            contour_geometries=np.repeat(contour_geometries[contour_idx], len(candidate_points)),
            contour_boundaries=np.repeat(contour_boundaries[contour_idx], len(candidate_points)),
        )
        ring_idx = _assign_intervals(signed_distances=signed, interval_edges=interval_edges)
        valid = ring_idx >= 0
        if np.any(valid):
            np.add.at(counts[contour_idx], ring_idx[valid], 1)
    return counts


def _count_transcript_targets(
    sdata: SpatialData,
    points_key: str,
    coordinate_system: str,
    expanded_geometries: np.ndarray,
    contour_geometries: np.ndarray,
    contour_boundaries: np.ndarray,
    interval_edges: np.ndarray,
    feature_key: str | None,
    feature_values: Sequence[str] | None,
) -> np.ndarray:
    assert_key_in_sdata(sdata, points_key, attr="points")

    points_ddf = sdata.points[points_key]
    required_columns = ["x", "y"]
    if feature_values is not None:
        feature_key = _resolve_feature_key(points_ddf=points_ddf, feature_key=feature_key)
        feature_values = set(feature_values)
        required_columns.append(feature_key)
    points_ddf = points_ddf[required_columns]

    matrix = _get_affine_matrix(points_ddf, coordinate_system=coordinate_system)
    tree = STRtree(expanded_geometries)
    counts = np.zeros((len(contour_geometries), len(interval_edges) - 1), dtype=np.int64)

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
        ring_idx = _assign_intervals(signed_distances=signed, interval_edges=interval_edges)
        valid = ring_idx >= 0
        if np.any(valid):
            np.add.at(counts, (contour_idx[valid], ring_idx[valid]), 1)

    return counts


def _assemble_result(
    contours: gpd.GeoDataFrame,
    intervals: list[tuple[float, float]],
    counts: np.ndarray,
    areas: np.ndarray,
    contour_key: str,
    target: str,
    target_key: str | None,
    feature_values: Sequence[str] | None,
    metadata_keys: str | Sequence[str] | None,
) -> pd.DataFrame:
    metadata_columns = _resolve_metadata_columns(contours=contours, metadata_keys=metadata_keys)
    records: list[dict[str, Any]] = []
    feature_values_list = _as_list(feature_values) if feature_values is not None else None

    for contour_idx, contour_row in contours.iterrows():
        for ring_idx, (ring_start, ring_end) in enumerate(intervals):
            area = float(areas[contour_idx, ring_idx])
            count = int(counts[contour_idx, ring_idx])
            record: dict[str, Any] = {
                "contour_key": contour_key,
                "contour_id": contour_row["_squidpy_contour_id"],
                "target": target,
                "target_key": target_key,
                "ring_start": ring_start,
                "ring_end": ring_end,
                "ring_mid": 0.5 * (ring_start + ring_end),
                "count": count,
                "area": area,
                "density": np.nan if area <= 0 else count / area,
            }
            if feature_values_list is not None:
                record["feature_values"] = tuple(feature_values_list)
            for column in metadata_columns:
                record[column] = contour_row[column]
            records.append(record)
    return pd.DataFrame.from_records(records)


def _resolve_shape_key(sdata: SpatialData, table_key: str | None, shape_key: str | None) -> str:
    if shape_key is not None:
        assert_key_in_sdata(sdata, shape_key, attr="shapes")
        return shape_key

    if table_key is None:
        raise ValueError("Specify either `shape_key` or `table_key` when `target='cells'`.")
    assert_key_in_sdata(sdata, table_key, attr="tables")

    attrs = sdata.tables[table_key].uns.get("spatialdata_attrs", {})
    region = attrs.get("region")
    if region is None:
        raise KeyError(
            f"Unable to infer a shape key from `sdata.tables[{table_key!r}].uns['spatialdata_attrs']['region']`."
        )

    if isinstance(region, str):
        resolved_region = region
    else:
        regions = assert_non_empty_sequence(region, name="regions")
        if len(regions) != 1:
            raise ValueError(
                f"Unable to infer a unique shape key from `sdata.tables[{table_key!r}].uns['spatialdata_attrs']['region']`. "
                "Please specify `shape_key` explicitly."
            )
        resolved_region = str(regions[0])

    assert_key_in_sdata(sdata, resolved_region, attr="shapes")
    return resolved_region


def _resolve_feature_key(points_ddf: Any, feature_key: str | None) -> str:
    if feature_key is not None:
        if feature_key not in points_ddf.columns:
            raise KeyError(f"Feature key `{feature_key}` not found in the transcript points dataframe.")
        return feature_key

    attrs = getattr(points_ddf, "attrs", {})
    inferred = attrs.get("spatialdata_attrs", {}).get("feature_key")
    if inferred is None:
        raise KeyError("Unable to infer `feature_key` from SpatialData metadata. Please specify it explicitly.")
    if inferred not in points_ddf.columns:
        raise KeyError(f"Inferred feature key `{inferred}` not found in the transcript points dataframe.")
    return str(inferred)


def _resolve_metadata_columns(contours: gpd.GeoDataFrame, metadata_keys: str | Sequence[str] | None) -> list[str]:
    if metadata_keys is None:
        defaults = ["classification_name", "assigned_structure", "annotation_source"]
        return [column for column in defaults if column in contours.columns]
    return [column for column in _as_list(metadata_keys) if column in contours.columns]


def _build_ring_intervals(ring_width: float, inward: float, outward: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []

    current = -float(inward)
    while current < 0:
        next_edge = min(current + ring_width, 0.0)
        intervals.append((current, next_edge))
        current = next_edge

    current = 0.0
    while current < outward:
        next_edge = min(current + ring_width, float(outward))
        intervals.append((current, next_edge))
        current = next_edge

    return intervals


def _ring_area(geometry: Any, ring_start: float, ring_end: float) -> float:
    if ring_end <= 0:
        outer = geometry.buffer(-abs(ring_end))
        inner = geometry.buffer(-abs(ring_start))
        return max(outer.area - inner.area, 0.0)
    return max(geometry.buffer(ring_end).area - geometry.buffer(ring_start).area, 0.0)


def _signed_distance(
    point_geometries: np.ndarray,
    contour_geometries: np.ndarray,
    contour_boundaries: np.ndarray,
) -> np.ndarray:
    dist = np.asarray(distance(point_geometries, contour_boundaries), dtype=float)
    inside = np.asarray(contains(contour_geometries, point_geometries), dtype=bool)
    signed = np.where(dist == 0, 0.0, np.where(inside, -dist, dist))
    return np.asarray(signed, dtype=float)


def _assign_intervals(signed_distances: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    ring_idx = np.searchsorted(interval_edges, signed_distances, side="right") - 1
    on_last_edge = np.isclose(signed_distances, interval_edges[-1])
    ring_idx[on_last_edge] = len(interval_edges) - 2
    valid = (ring_idx >= 0) & (ring_idx < len(interval_edges) - 1)
    return np.where(valid, ring_idx, -1)


def _get_affine_matrix(element: Any, coordinate_system: str) -> np.ndarray:
    transformation = get_transformation(element, to_coordinate_system=coordinate_system)
    matrix = transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine matrix, found shape {matrix.shape}.")
    return np.asarray(matrix, dtype=float)


def _apply_affine_to_geometry(geometry: Any, matrix: np.ndarray) -> Any:
    return affinity.affine_transform(
        geometry,
        [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]],
    )


def _apply_affine_to_xy(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = (matrix[:2, :2] @ xy.T).T + matrix[:2, 2]
    return np.asarray(transformed, dtype=float)


def _as_list(values: str | Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, Iterable):
        return [str(value) for value in values]
    return [str(values)]
