from __future__ import annotations

import geopandas as gpd
import numpy as np
import scanpy as sc
import spatialdata as sd
from dask.dataframe import DataFrame as DaskDataFrame
from spatialdata import SpatialData, subset_sdata_by_table_mask
from spatialdata._logging import logger as logg
from spatialdata.models import (
    get_table_keys,
    points_dask_dataframe_to_geopandas,
    points_geopandas_to_dask_dataframe,
)




def filter_cells(
    data: ad.AnnData | sd.SpatialData,
    tables: list[str] | str | None = None,
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_counts: int | None = None,
    max_genes: int | None = None,
    inplace: bool = True,
    filter_labels: bool = True,
) -> sd.SpatialData | None:
    """\
    Squidpy's implementation of :func:`scanpy.pp.filter_cells` for :class:`anndata.AnnData` and :class:`spatialdata.SpatialData` objects.
    For :class:`spatialdata.SpatialData` objects, this function filters the following elements:


    - labels: filtered based on the values of the images which are assumed to be the instance_id.
    - points: filtered based on the index which is assumed to be the instance_id.
    - shapes: filtered based on the instance_id column.


    See :func:`scanpy.pp.filter_cells` for more details regarding the filtering
    behavior.

    Parameters
    ----------
    data
        :class:`spatialdata.SpatialData` object.
    tables
        If :class:`spatialdata.SpatialData` object, the tables to filter. If `None`, all tables are filtered.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_genes
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_genes
        Maximum number of genes expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.
    filter_labels
        Whether to filter labels. If `True`, then labels are filtered based on the instance_id column.

    Returns
    -------
    If `inplace` then returns `None`, otherwise returns the filtered :class:`spatialdata.SpatialData` object.
    """
    if not isinstance(data, sd.SpatialData):
        raise ValueError(
            f"Expected `SpatialData`, found `{type(data)}` instead. Perhaps you want to use `scanpy.pp.filter_cells` instead."
        )

    return _filter_cells_spatialdata(data, tables, min_counts, min_genes, max_counts, max_genes, inplace, filter_labels)


def _get_only_annotated_shape(sdata: sd.SpatialData, table_name: str) -> str | None:
    table = sdata.tables[table_name]

    # only one shape needs to be annotated to filter points within it
    # other annotations can't be points

    regions, _, _ = get_table_keys(table)
    if len(regions) == 0:
        return None

    if isinstance(regions, str):
        regions = [regions]

    res = None
    for r in regions:
        if r in sdata.points:
            return None
        if r in sdata.shapes:
            if res is not None:
                return None
            res = r

    return res


def _annotated_points_by_shape_membership(
    sdata: SpatialData,
    point_key: str,
    shape_key: str,
) -> DaskDataFrame:
    """Annotate points by shape membership.

    Parameters
    ----------
    sdata
        The SpatialData object to annotate.
    point_key
        The key of the points to annotate.
    shape_key
        The key of the shapes to annotate.

    Returns
    -------
    The annotated points.
    """
    points = sdata.points[point_key]
    shapes = sdata.shapes[shape_key]
    points_gdf = points_dask_dataframe_to_geopandas(points)
    res = points_gdf.sjoin(shapes, how="left", predicate="within")
    return points_geopandas_to_dask_dataframe(res)


def _filter_cells_spatialdata(
    data: sd.SpatialData,
    tables: list[str] | str | None = None,
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_counts: int | None = None,
    max_genes: int | None = None,
    inplace: bool = True,
    filter_labels: bool = True,
) -> sd.SpatialData | None:
    if isinstance(tables, str):
        tables = [tables]
    elif tables is None:
        tables = list(data.tables.keys())

    if len(tables) == 0:
        raise ValueError("Expected at least one table to be filtered, found `0`")

    if not all(t in data.tables for t in tables):
        raise ValueError(f"Expected all tables to be in `{data.tables.keys()}`.")

    for t in tables:
        if "spatialdata_attrs" not in data.tables[t].uns:
            raise ValueError(f"Table `{t}` does not have 'spatialdata_attrs' to indicate what it annotates.")

    if not inplace:
        logg.warning(
            "Creating a deepcopy of the SpatialData object, depending on the size of the object this can take a while."
        )
        data_out = sd.deepcopy(data)
    else:
        data_out = data

    for t in tables:
        table_old = data_out.tables[t]
        mask_filtered, _ = sc.pp.filter_cells(
            table_old,
            min_counts=min_counts,
            min_genes=min_genes,
            max_counts=max_counts,
            max_genes=max_genes,
            inplace=False,
        )
        if mask_filtered.sum() == 0:
            raise ValueError(f"Filter results in empty table when filtering table `{t}`.")
        sdata_filtered = subset_sdata_by_table_mask(sdata=data_out, table_name=t, mask=mask_filtered)
        data_out.tables[t] = sdata_filtered.tables[t]
        for k in list(sdata_filtered.points.keys()):
            data_out.points[k] = sdata_filtered.points[k]
        for k in list(sdata_filtered.shapes.keys()):
            data_out.shapes[k] = sdata_filtered.shapes[k]
        if filter_labels:
            for k in list(sdata_filtered.labels.keys()):
                data_out.labels[k] = sdata_filtered.labels[k]
        shape_name = _get_only_annotated_shape(data_out, t)
        if shape_name is not None:
            for p in data_out.points:
                _, _, instance_key = get_table_keys(table_old)
                shape_index_name = data_out.shapes[shape_name].index.name
                new_points = _annotated_points_by_shape_membership(
                    sdata=data_out,
                    shape_key=shape_name,
                    point_key=p,
                )
                shape_index_name += "_right"
                removed_instance_ids = list(np.unique(table_old.obs[instance_key][~mask_filtered]))
                # drop points that are not in any shape
                new_points = new_points.dropna()
                # drop points that are in the removed_instance_ids
                new_points = new_points[~new_points[shape_index_name].isin(removed_instance_ids)]
                data_out.points[p] = new_points

    if inplace:
        return None
    return data_out
