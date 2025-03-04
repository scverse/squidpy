from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd


def filter_cells(
    data: ad.AnnData | sd.SpatialData,
    table: str | None = None,
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_counts: int | None = None,
    max_genes: int | None = None,
    inplace: bool = True,
) -> ad.AnnData | sd.SpatialData | None:
    if not isinstance(data, ad.AnnData | sd.SpatialData):
        raise ValueError(f"Expected `AnnData` or `SpatialData`, found `{type(data)}`")

    if isinstance(data, ad.AnnData) and table is not None:
        raise ValueError("When filtering `AnnData`, `table` is not used.")

    tables_to_use: list[str] = []

    if isinstance(data, sd.SpatialData) and table is not None:
        if isinstance(table, str):
            tables_to_use = [table]

    if isinstance(data, sd.SpatialData) and table is None:
        if isinstance(table, str):
            tables_to_use = list(data.tables.keys())

    if tables_to_use is not None and len(tables_to_use) == 0:
        raise ValueError("Expected at least one table to be filtered, found `0`")

    if any(t not in data.tables for t in tables_to_use):
        raise ValueError(f"Expected all tables to be in `{data.tables.keys()}`.`")

    # mimic scanpy's behavior in only allowing one filtering parameter per call
    n_given_options = sum(option is not None for option in [min_genes, min_counts, max_genes, max_counts])
    if n_given_options > 1:
        raise ValueError("Only one filtering parameter can be provided per call (scanpy behavior).")

    for param_name, param_value in [
        ("min_counts", min_counts),
        ("min_genes", min_genes),
        ("max_counts", max_counts),
        ("max_genes", max_genes),
    ]:
        if param_value is not None and not isinstance(param_value, int):
            raise ValueError(f"Expected `{param_name}` to be an integer, found `{type(param_value)}`")

    if not isinstance(inplace, bool):
        raise ValueError(f"Expected `inplace` to be a boolean, found `{type(inplace)}`")

    def _apply_anndata_filters(
        data: ad.AnnData,
        min_counts: int | None,
        min_genes: int | None,
        max_counts: int | None,
        max_genes: int | None,
        inplace: bool = True,
    ) -> ad.AnnData | None:
        result = data if inplace else data.copy()

        # robust way to feed in whichever filtering parameters is not None
        filter_params = {
            "min_counts": min_counts,
            "min_genes": min_genes,
            "max_counts": max_counts,
            "max_genes": max_genes,
        }

        for param_name, param_value in filter_params.items():
            if param_value is not None:
                # Always modify result in place since we're using our own copy
                sc.pp.filter_cells(result, **{param_name: param_value}, inplace=inplace)

        # Return the filtered data if not in place
        return None if inplace else result

    if isinstance(data, ad.AnnData):
        data_out = data if inplace else data.copy()

        _apply_anndata_filters(data_out, min_counts, min_genes, max_counts, max_genes, inplace=inplace)

        return None if inplace else data_out

    # if it's SpatialData, we need to filter other elements in the object
    elif isinstance(data, sd.SpatialData):
        if not inplace:
            data_out = sd.SpatialData(
                images=data.images if data.images is not None else None,
                labels=data.labels if data.labels is not None else None,
                points=data.points if data.points is not None else None,
                shapes=data.shapes if data.shapes is not None else None,
                tables=data.tables if data.tables is not None else None,
            )
        else:
            data_out = data

        for t in tables_to_use:
            if "spatialdata_attrs" in data.tables[t].uns:
                instance_key = data.tables[t].uns["spatialdata_attrs"]["instance_key"]
                region_key = data.tables[t].uns["spatialdata_attrs"]["region_key"]
                region = data.tables[t].uns["spatialdata_attrs"]["region"]

            filter_params = {
                "min_counts": min_counts,
                "min_genes": min_genes,
                "max_counts": max_counts,
                "max_genes": max_genes,
            }

            # remove the rows from the table
            table_old = data.tables[t].copy()
            for param_name, param_value in filter_params.items():
                if param_value is not None:
                    # Always modify result in place since we're using our own copy
                    mask, _ = sc.pp.filter_cells(table_old, **{param_name: param_value}, inplace=inplace)

            data_out.tables[t] = table_old[~mask]

            # remove the rows from the shapes
            removed_obs = table_old.obs[mask][[instance_key, region_key]]

            assert removed_obs[region_key].unique() == region

            idx_to_remove = removed_obs[instance_key].values.tolist()
            ele_to_modify = data.shapes[region].copy()
            filtered_gdf = ele_to_modify[~ele_to_modify.index.isin(idx_to_remove)]

            data_out.shapes[region] = filtered_gdf

        return data_out
