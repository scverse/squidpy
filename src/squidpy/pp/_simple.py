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
    table: str | None = None,
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_counts: int | None = None,
    max_genes: int | None = None,
    inplace: bool = True,
    filter_labels: bool = True,
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
        tables_to_use = list(data.tables.keys())

    if not isinstance(data, ad.AnnData) and tables_to_use is not None and not tables_to_use:
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

    # if it's an AnnData object, we add a pseudo tablename
    if isinstance(data, ad.AnnData):
        tables_to_use = ["adata"]

    # we need to filter the adata object either way
    for t in tables_to_use:
        filter_params = {
            "min_counts": min_counts,
            "min_genes": min_genes,
            "max_counts": max_counts,
            "max_genes": max_genes,
        }

        if isinstance(data, ad.AnnData):
            table_old = data
        else:
            table_old = data.tables[t] if inplace else data.tables[t].copy()

        for param_name, param_value in filter_params.items():
            if param_value is not None:
                if inplace and isinstance(data, ad.AnnData):
                    sc.pp.filter_cells(table_old, **{param_name: param_value}, inplace=True)
                elif not inplace:
                    # inplace=False gives us boolean vector of which rows to remove
                    mask_to_remove, _ = sc.pp.filter_cells(table_old, **{param_name: param_value}, inplace=False)

                    if isinstance(data, ad.AnnData):
                        return table_old[~mask_to_remove]

                    # we're SpatialData now
                    assert isinstance(data, sd.SpatialData)

                    if not inplace:
                        logg.warning(
                            "Creating a deepcopy of the SpatialData object, depending on the size of the object this can take a while."
                        )
                        data_out = sd.deepcopy(data)

                        # elements_dict = {}
                        # for _, element_name, element in data.gen_elements():
                        #     elements_dict[element_name] = sd.deepcopy(element)
                        # deepcopied_attrs = data.attrs
                        # data_out = sd.SpatialData.from_elements_dict(elements_dict, attrs=deepcopied_attrs)

                    else:
                        data_out = data

                    table_filtered = table_old[~mask_to_remove]
                    if table_filtered.n_obs == 0 or table_filtered.n_vars == 0:
                        raise ValueError(f"Filter results in empty table when filtering table `{t}`.")
                    data_out.tables[t] = table_filtered

                    # if this doesn't exist, the table doesn't annotate anything
                    if "spatialdata_attrs" not in data.tables[t].uns:
                        raise ValueError(
                            f"Table `{t}` does not have 'spatialdata_attrs' to indicate what it annotates."
                        )

                    instance_key = data.tables[t].uns["spatialdata_attrs"]["instance_key"]
                    region_key = data.tables[t].uns["spatialdata_attrs"]["region_key"]

                    # region can annotate one (dtype str) or multiple (dtype list[str])
                    region = data.tables[t].uns["spatialdata_attrs"]["region"]
                    if isinstance(region, str):
                        region = [region]

                    removed_obs = table_old.obs[mask_to_remove][[instance_key, region_key]]

                    # iterate over all elements that the table annotates (region var)
                    for r in region:
                        element_model = get_model(data_out[r])

                        ids_to_remove = removed_obs.query(f"{region_key} == '{r}'")[instance_key].tolist()
                        if element_model == ShapesModel:
                            data_out.shapes[r] = _filter_ShapesModel_by_instance_ids(
                                element=data_out.shapes[r], ids_to_remove=ids_to_remove
                            )

                        if filter_labels:
                            logg.warning("Filtering labels, this can be slow depending on the resolution.")
                            if element_model == Labels2DModel:
                                new_label = _filter_Labels2DModel_by_instance_ids(
                                    element=data_out.labels[r], ids_to_remove=ids_to_remove
                                )

                                del data_out.labels[r]

                                data_out.labels[r] = new_label

    if not inplace:
        return data_out


def _filter_ShapesModel_by_instance_ids(element: ShapesModel, ids_to_remove: list[str]) -> ShapesModel:
    return element[~element.index.isin(ids_to_remove)]


def _filter_Labels2DModel_by_instance_ids(element: Labels2DModel, ids_to_remove: list[str]) -> Labels2DModel:
    def set_ids_in_label_to_zero(image: xr.DataArray, ids_to_remove: list[int]) -> xr.DataArray:
        # Use apply_ufunc for efficient processing
        def _mask_block(block):
            # Create a copy to avoid modifying read-only array
            result = block.copy()
            result[np.isin(result, masks)] = 0
            return result

        processed = xr.apply_ufunc(
            _mask_block,
            image,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[image.dtype],
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
