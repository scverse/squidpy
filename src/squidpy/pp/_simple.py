from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import xarray as xr
from spatialdata._logging import logger as logg
from spatialdata.models import Labels2DModel, PointsModel, ShapesModel, get_model
from spatialdata.transformations import get_transformation
from xarray import DataTree


def filter_cells(
    data: ad.AnnData | sd.SpatialData,
    tables: list[str] | str | None = None,
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_counts: int | None = None,
    max_genes: int | None = None,
    inplace: bool = True,
    filter_labels: bool = True,
) -> ad.AnnData | sd.SpatialData | None:
    if not isinstance(data, ad.AnnData | sd.SpatialData):
        raise ValueError(f"Expected `AnnData` or `SpatialData`, found `{type(data)}`")

    if isinstance(data, ad.AnnData):
        if tables is not None:
            raise ValueError("When filtering `AnnData`, `tables` is not used.")
        return sc.pp.filter_cells(
            data,
            min_counts=min_counts,
            min_genes=min_genes,
            max_counts=max_counts,
            max_genes=max_genes,
            inplace=inplace,
        )

    return _filter_cells_spatialdata(data, tables, min_counts, min_genes, max_counts, max_genes, inplace, filter_labels)


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

        table_filtered = table_old[mask_filtered]
        if table_filtered.n_obs == 0 or table_filtered.n_vars == 0:
            raise ValueError(f"Filter results in empty table when filtering table `{t}`.")
        data_out.tables[t] = table_filtered

        instance_key = data.tables[t].uns["spatialdata_attrs"]["instance_key"]
        region_key = data.tables[t].uns["spatialdata_attrs"]["region_key"]

        # region can annotate one (dtype str) or multiple (dtype list[str])
        region = data.tables[t].uns["spatialdata_attrs"]["region"]
        if isinstance(region, str):
            region = [region]

        removed_obs = table_old.obs[~mask_filtered][[instance_key, region_key]]

        # iterate over all elements that the table annotates (region var)
        for r in region:
            element_model = get_model(data_out[r])

            ids_to_remove = removed_obs.query(f"{region_key} == '{r}'")[instance_key].tolist()
            if element_model is ShapesModel:
                data_out.shapes[r] = _filter_shapesmodel_by_instance_ids(
                    element=data_out.shapes[r], ids_to_remove=ids_to_remove
                )

            if filter_labels:
                logg.warning("Filtering labels, this can be slow depending on the resolution.")
                if element_model is Labels2DModel:
                    new_label = _filter_labels2dmodel_by_instance_ids(
                        element=data_out.labels[r], ids_to_remove=ids_to_remove
                    )

                    del data_out.labels[r]

                    data_out.labels[r] = new_label

    if inplace:
        return None
    return data_out


def _filter_shapesmodel_by_instance_ids(element: ShapesModel, ids_to_remove: list[str]) -> ShapesModel:
    return element[~element.index.isin(ids_to_remove)]


def _filter_labels2dmodel_by_instance_ids(element: Labels2DModel, ids_to_remove: list[int]) -> Labels2DModel:
    def set_ids_in_label_to_zero(image: xr.DataArray, ids_to_remove: list[int]) -> xr.DataArray:
        # Use apply_ufunc for efficient processing
        def _mask_block(block: xr.DataArray) -> xr.DataArray:
            # Create a copy to avoid modifying read-only array
            result = block.copy()
            result[np.isin(result, ids_to_remove)] = 0
            return result

        processed = xr.apply_ufunc(
            _mask_block,
            image,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[image.dtype],
            dataset_fill_value=0,
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

        # Force computation to ensure the changes are materialized
        computed_result = processed.compute()

        # Create a new DataArray to ensure persistence
        result = xr.DataArray(
            data=computed_result.data,
            coords=image.coords,
            dims=image.dims,
            attrs=image.attrs.copy(),  # Preserve all attributes
        )

        return result

    if isinstance(element, xr.DataArray):
        return Labels2DModel.parse(set_ids_in_label_to_zero(element, ids_to_remove))

    if isinstance(element, DataTree):
        # we extract the info to just reconstruct the DataTree after filtering the max scale
        max_scale = list(element.keys())[0]
        scale_factors = _get_scale_factors(element)
        scale_factors = [int(sf[0]) for sf in scale_factors]

        return Labels2DModel.parse(
            data=set_ids_in_label_to_zero(element[max_scale].image, ids_to_remove),
            scale_factors=scale_factors,
        )


def _get_scale_factors(labels_element: Labels2DModel) -> list[tuple[float, float]]:
    scales = list(labels_element.keys())

    # Calculate relative scale factors between consecutive scales
    scale_factors = []
    for i in range(len(scales) - 1):
        y_size_current = labels_element[scales[i]].image.shape[0]
        x_size_current = labels_element[scales[i]].image.shape[1]
        y_size_next = labels_element[scales[i + 1]].image.shape[0]
        x_size_next = labels_element[scales[i + 1]].image.shape[1]
        y_factor = y_size_current / y_size_next
        x_factor = x_size_current / x_size_next

        scale_factors.append((y_factor, x_factor))

    return scale_factors
