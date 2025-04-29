"""Experimental feature extraction module."""

from __future__ import annotations

import itertools
import os
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from cp_measure.bulk import get_core_measurements, get_correlation_measurements
from numba import njit, prange
from scipy import ndimage
from skimage import measure
from skimage.measure import label
from spatialdata import SpatialData, rasterize
from spatialdata._logging import logger as logg
from spatialdata.models import TableModel

from squidpy._constants._constants import ImageFeature
from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, _get_n_cores, parallelize

__all__ = ["calculate_image_features"]

# Define constant property sets
_MASK_PROPS = {
    "area",
    "area_filled",
    "area_convex",
    "num_pixels",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "equivalent_diameter",
    "extent",
    "feret_diameter_max",
    "solidity",
    "euler_number",
    "centroid",
    "centroid_local",
    "perimeter",
    "perimeter_crofton",
    "inertia_tensor",
    "inertia_tensor_eigvals",
}
_INTENSITY_PROPS = {
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "intensity_std",
}

# Define array types using modern syntax
NDArray = npt.NDArray[Any]  # Generic array
FloatArray = npt.NDArray[np.float32]  # Float32 array
IntArray = npt.NDArray[np.int_]  # Integer array
BoolArray = npt.NDArray[np.bool_]  # Boolean array

# Define property sets at module level for better performance
_SCALAR_PROPS = frozenset(
    {
        "area",
        "area_filled",
        "area_convex",
        "num_pixels",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "equivalent_diameter",
        "extent",
        "feret_diameter_max",
        "solidity",
        "euler_number",
        "perimeter",
        "perimeter_crofton",
    }
)

_ARRAY_1D_PROPS = frozenset({"centroid", "centroid_local"})
_ARRAY_2D_PROPS = frozenset({"inertia_tensor"})
_SPECIAL_PROPS = frozenset({"inertia_tensor_eigvals"})


@d.dedent
@inject_docs(f=ImageFeature)
def calculate_image_features(
    sdata: SpatialData,
    image_key: str,
    labels_key: str | None = None,
    shapes_key: str | None = None,
    scale: str | None = None,
    measurements: list[str] | str | None = None,
    adata_key_added: str = "morphology",
    invalid_as_zero: bool = True,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = False,  # slower, needs to be optimised
    verbose: bool = False,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Calculate features from segmentation masks using CellProfiler measurements.

    This function uses the `cp_measure` package to extract features from
    segmentation masks. It supports both basic shape features and
    intensity-based features if an intensity image is provided.

    Parameters
    ----------
    sdata
        The spatial data object containing the segmentation masks.
    labels_key
        Key in :attr:`spatialdata.SpatialData.labels` containing the
        segmentation masks.
    shapes_key
        Key in :attr:`spatialdata.SpatialData.shapes` containing the
        shape features.
    image_key
        Key in :attr:`spatialdata.SpatialData.images` containing the
        intensity image.
    adata_key_added
        Key to store the AnnData object in the SpatialData object.
    %(parallelize)s

    Returns
    -------
    A :class:`pandas.DataFrame` with the calculated features. If the image has
    multiple channels, features are calculated for each channel separately and
    channel names are appended to the feature names.

    Notes
    -----
    This is an experimental feature that requires the `cp_measure` package
    to be installed.
    """

    if image_key not in sdata.images.keys():
        raise ValueError(f"Image key '{image_key}' not found, valid keys: {list(sdata.images.keys())}")

    if labels_key is not None and shapes_key is not None:
        raise ValueError("Use either `labels_key` or `shapes_key`, not both.")

    if labels_key is not None and labels_key not in sdata.labels.keys():
        raise ValueError(f"Labels key '{labels_key}' not found, valid keys: {list(sdata.labels.keys())}")

    if shapes_key is not None and shapes_key not in sdata.shapes.keys():
        raise ValueError(f"Shapes key '{shapes_key}' not found, valid keys: {list(sdata.shapes.keys())}")

    if (
        isinstance(sdata.images[image_key], xr.DataTree) or isinstance(sdata.labels[labels_key], xr.DataTree)
    ) and scale is None:
        raise ValueError("When using multi-scale data, please specify the scale.")

    if scale is not None and not isinstance(scale, str):
        raise ValueError("Scale must be a string.")

    image = _get_array_from_DataTree_or_DataArray(sdata.images[image_key], scale)
    labels = _get_array_from_DataTree_or_DataArray(sdata.labels[labels_key], scale) if labels_key is not None else None

    if labels is not None and image.shape[1:] != labels.shape:
        raise ValueError(
            f"Image dimensions {image.shape[1:]} do not match labels dimensions {labels.shape} at scale '{scale}'"
        )

    if shapes_key is not None:
        scale_str = f" (using scale '{scale}')" if scale is not None else ""
        logg.info(f"Converting shapes to labels{scale_str}.")
        _, max_y, max_x = image.shape
        labels = np.asarray(
            rasterize(
                sdata.shapes[shapes_key],
                ["x", "y"],
                min_coordinate=[0, 0],
                max_coordinate=[max_x, max_y],
                target_coordinate_system="global",
                target_unit_to_pixels=1.0,
                return_regions_as_labels=True,
            )
        )
    else:
        labels = _get_array_from_DataTree_or_DataArray(sdata.labels[labels_key], scale)

    available_measurements = [
        "skimage:label",
        "skimage:label+image",
        "cpmeasure:core",
        "cpmeasure:correlation",
    ]

    if measurements is None:
        measurements = available_measurements

    if isinstance(measurements, str):
        measurements = [measurements]

    if isinstance(measurements, list):
        invalid_measurements = [m for m in measurements if m not in available_measurements]
        if invalid_measurements:
            raise ValueError(
                f"Invalid measurement(s): {invalid_measurements}, available measurements: {available_measurements}"
            )

    if labels.size == 0:
        raise ValueError("Labels array is empty")

    max_label = int(labels.max())
    if max_label == 0:
        raise ValueError("No cells found in labels (max label is 0)")

    channel_names = None
    if hasattr(sdata.images[image_key], "coords") and "c" in sdata.images[image_key].coords:
        channel_names = sdata.images[image_key].coords["c"].values

    if image.ndim == 2:
        image = image[None, :, :]
    elif image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

    if image.shape[1:] != labels.shape:
        raise ValueError(f"Image and labels have mismatched dimensions: image {image.shape[1:]}, labels {labels.shape}")

    if "cpmeasure:correlation" in measurements:
        measurements_corr = get_correlation_measurements()

    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids != 0]
    # Sort cell_ids to ensure consistent order
    cell_ids = np.sort(cell_ids)
    cell_ids_list = cell_ids.tolist()  # Convert to list for parallelize

    all_features = []
    n_channels = image.shape[0]
    n_jobs = _get_n_cores(n_jobs)

    logg.info(f"Using '{n_jobs}' core(s).")

    if "skimage:label" in measurements:
        logg.info("Calculating 'skimage' label features.")
        res = parallelize(
            _get_regionprops_features,
            collection=cell_ids_list,
            extractor=pd.concat,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
            verbose=verbose,
        )(labels=labels, intensity_image=None)
        all_features.append(res)

    if "skimage:label+image" in measurements:
        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx] if channel_names is not None else f"{ch_idx}"
            ch_image = image[ch_idx]
            logg.info(f"Calculating 'skimage' image features for channel '{ch_idx}'.")
            res = parallelize(
                _get_regionprops_features,
                collection=cell_ids_list,
                extractor=pd.concat,
                n_jobs=n_jobs,
                backend=backend,
                show_progress_bar=show_progress_bar,
                verbose=verbose,
            )(labels=labels, intensity_image=ch_image)
            # Append channel names to each feature column
            res = res.rename(columns=lambda col, ch_name=ch_name: f"{col}_{ch_name}")
            all_features.append(res)

    if "cpmeasure:core" in measurements:
        measurements_core = get_core_measurements()
        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx] if channel_names is not None else f"{ch_idx}"
            ch_image = image[ch_idx]
            logg.info(f"Calculating 'cpmeasure' core features for channel '{ch_idx}'.")
            res = parallelize(
                _calculate_features_helper,
                collection=cell_ids_list,
                extractor=pd.concat,
                n_jobs=n_jobs,
                backend=backend,
                show_progress_bar=show_progress_bar,
                verbose=verbose,
            )(labels, ch_image, None, measurements_core, ch_name)
            all_features.append(res)

    if "cpmeasure:correlation" in measurements:
        for ch1_idx in range(n_channels):
            for ch2_idx in range(ch1_idx + 1, n_channels):
                ch1_name = channel_names[ch1_idx] if channel_names is not None else f"{ch1_idx}"
                ch2_name = channel_names[ch2_idx] if channel_names is not None else f"{ch2_idx}"
                logg.info(
                    f"Calculating 'cpmeasure' correlation features between channels '{ch1_name}' and '{ch2_name}'."
                )
                ch1_image = image[ch1_idx]
                ch2_image = image[ch2_idx]
                res = parallelize(
                    _calculate_features_helper,
                    collection=cell_ids_list,
                    extractor=pd.concat,
                    n_jobs=n_jobs,
                    backend=backend,
                    show_progress_bar=show_progress_bar,
                    verbose=verbose,
                )(labels, ch1_image, ch2_image, measurements_corr, ch1_name, ch2_name)
                all_features.append(res)

    combined_features = pd.concat(all_features, axis=1)

    if invalid_as_zero:
        combined_features = combined_features.replace([np.inf, -np.inf], 0)
        combined_features = combined_features.fillna(0)

    # Ensure cell IDs are preserved in the correct order
    combined_features = combined_features.loc[cell_ids]

    adata = ad.AnnData(X=combined_features)
    adata.obs_names = [f"cell_{i}" for i in cell_ids]
    adata.var_names = combined_features.columns

    adata.uns["spatialdata_attrs"] = {
        "region": labels_key if labels_key is not None else shapes_key,
        "region_key": "region",
        "instance_key": "label_id",
    }
    adata.obs["region"] = pd.Categorical([labels_key if labels_key is not None else shapes_key] * len(adata))
    # here we either use the cell_ids or the index of the shapes. Needed
    # because when converting the shapes to labels, a potential index 0
    # in the shapes is set to 1 in the labels and therefore we'd otherwise
    # be off-by-one in the label_id.
    adata.obs["label_id"] = sdata.shapes[shapes_key].index.values if shapes_key is not None else cell_ids

    if inplace:
        sdata.tables[adata_key_added] = TableModel.parse(adata)
    else:
        return combined_features


def _extract_features_from_regionprops(
    region_obj: Any,
    props: set[str],
    cell_id: int,
    skip_callable: bool = False,
) -> dict[str, float]:
    """Extract features from a regionprops object given a list of properties."""
    cell_features = {}

    for prop in props:
        try:
            value = getattr(region_obj, prop)
            if skip_callable and callable(value):
                continue

            if prop in _SCALAR_PROPS:
                cell_features[prop] = float(value)
            elif prop in _ARRAY_1D_PROPS:
                # Convert to array only once
                value = np.asarray(value)
                for i, v in enumerate(value):
                    cell_features[f"{prop}_{i}"] = float(v)
            elif prop in _ARRAY_2D_PROPS:
                # Convert to array only once
                value = np.asarray(value)
                for i in range(value.shape[0]):
                    for j in range(value.shape[1]):
                        cell_features[f"{prop}_{i}x{j}"] = float(value[i, j])
            elif prop in _SPECIAL_PROPS:
                # Convert to array only once
                value = np.asarray(value)
                for i, v in enumerate(value):
                    cell_features[f"{prop}_{i}"] = float(v)
            else:
                # Fallback for any other properties
                if isinstance(value, (np.ndarray, list, tuple)):
                    value = np.asarray(value)
                    if value.ndim == 1:
                        for i, v in enumerate(value):
                            cell_features[f"{prop}_{i}"] = float(v)
                    elif value.ndim == 2:
                        for i in range(value.shape[0]):
                            for j in range(value.shape[1]):
                                cell_features[f"{prop}_{i}x{j}"] = float(value[i, j])
                    else:
                        cell_features[prop] = float(value.flatten()[0])
                else:
                    cell_features[prop] = float(value)

        except (ValueError, TypeError, AttributeError) as e:
            logg.warning(f"Error calculating {prop} for cell {cell_id}: {str(e)}")
            continue

    return cell_features


def _calculate_regionprops_from_crop(
    cell_mask_cropped: NDArray,
    intensity_image_cropped: NDArray | None,
    cell_id: int,
) -> dict[str, float]:
    """
    Calculate regionprops features from pre-cropped arrays.
    Uses intensity-based properties if an intensity image is provided.
    """
    if intensity_image_cropped is None:
        region_props = measure.regionprops(label_image=label(cell_mask_cropped))
        if not region_props:
            return {}
        return _extract_features_from_regionprops(region_props[0], _MASK_PROPS, cell_id)
    else:
        region_props = measure.regionprops(
            label_image=label(cell_mask_cropped),
            intensity_image=intensity_image_cropped,
        )
        if not region_props:
            return {}
        return _extract_features_from_regionprops(region_props[0], _INTENSITY_PROPS, cell_id, skip_callable=True)


def _append_channel_names(
    features: dict[str, Any],
    channel1: str | None,
    channel2: str | None = None,
) -> dict[str, Any]:
    """Append channel name(s) to all keys in the feature dictionary."""
    if channel2 is None:
        return {f"{k}_{channel1}": v for k, v in features.items()}
    else:
        return {f"{k}_{channel1}_{channel2}": v for k, v in features.items()}


def _prepare_images_for_measurement(
    name: str,
    cell_mask: NDArray,
    img1: NDArray,
    img2: NDArray | None,
    conv_params: dict[str, Any],
) -> tuple[NDArray, NDArray | None, NDArray | None]:
    """
    Convert inputs to the appropriate dtype based on the measurement type.
    """
    if name in conv_params.get("uint8_features", []):
        mask = cell_mask.astype(np.uint8)
        image1_prepared = img1.astype(np.uint8)
        image2_prepared = None if img2 is None else img2.astype(np.uint8)
    elif name == "texture":
        mask = cell_mask.astype(np.uint8)
        image1_prepared = (img1.astype(np.float32) - conv_params["img1_min"]) / conv_params["img1_range"]
        image2_prepared = (
            None if img2 is None else (img2.astype(np.float32) - conv_params["img2_min"]) / conv_params["img2_range"]
        )
    elif name in conv_params.get("float_features", []):
        mask = cell_mask.astype(np.float32)
        image1_prepared = img1.astype(np.float32)
        image2_prepared = None if img2 is None else img2.astype(np.float32)
    else:
        mask = cell_mask.astype(np.float32)
        image1_prepared = img1.astype(np.float32)
        image2_prepared = None if img2 is None else img2.astype(np.float32)
    return mask, image1_prepared, image2_prepared


@njit(fastmath=True)
def _get_cell_crops_numba(
    cell_id: int,
    labels: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    pad: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated version of _get_cell_crops.

    Note: image1 and image2 should be passed as empty arrays (np.zeros((0,0))) if not used.
    """
    # Find cell boundaries using vectorized operations
    cell_mask = labels == cell_id
    if not np.any(cell_mask):
        return (
            np.zeros((0, 0), dtype=np.bool_),
            np.zeros((0, 0), dtype=image1.dtype),
            np.zeros((0, 0), dtype=image2.dtype),
        )

    # Get non-zero indices efficiently
    y_indices, x_indices = np.nonzero(cell_mask)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Get image dimensions
    height, width = labels.shape

    # Calculate padding with boundary checks in one step
    y_pad_min = min(pad, y_min)
    y_pad_max = min(pad, height - y_max - 1)
    x_pad_min = min(pad, x_min)
    x_pad_max = min(pad, width - x_max - 1)

    # Calculate crop dimensions with padding
    y_start = y_min - y_pad_min
    y_end = y_max + y_pad_max + 1
    x_start = x_min - x_pad_min
    x_end = x_max + x_pad_max + 1

    # Create output arrays with exact size
    y_size = y_end - y_start
    x_size = x_end - x_start

    # Create cell mask crop
    cell_mask_cropped = np.zeros((y_size, x_size), dtype=np.bool_)
    for i in range(y_size):
        for j in range(x_size):
            cell_mask_cropped[i, j] = cell_mask[y_start + i, x_start + j]

    # Handle image crops efficiently
    if image1.size > 0:
        image1_cropped = np.zeros((y_size, x_size), dtype=image1.dtype)
        for i in range(y_size):
            for j in range(x_size):
                image1_cropped[i, j] = image1[y_start + i, x_start + j]
    else:
        image1_cropped = np.zeros((0, 0), dtype=image1.dtype)

    if image2.size > 0:
        image2_cropped = np.zeros((y_size, x_size), dtype=image2.dtype)
        for i in range(y_size):
            for j in range(x_size):
                image2_cropped[i, j] = image2[y_start + i, x_start + j]
    else:
        image2_cropped = np.zeros((0, 0), dtype=image2.dtype)

    return cell_mask_cropped, image1_cropped, image2_cropped


def _get_cell_crops(
    cell_id: int,
    labels: NDArray,
    image1: NDArray | None = None,
    image2: NDArray | None = None,
    pad: int = 1,
    verbose: bool = False,
) -> tuple[NDArray, NDArray | None, NDArray | None] | None:
    """Generator function to get cropped arrays for a cell."""
    # Create empty arrays for unused images
    empty_image = np.zeros((0, 0), dtype=np.float32)
    image1_np = image1 if image1 is not None else empty_image
    image2_np = image2 if image2 is not None else empty_image

    # Use Numba-accelerated version
    cell_mask_cropped, image1_cropped, image2_cropped = _get_cell_crops_numba(
        cell_id, labels, image1_np, image2_np, pad
    )

    # Return None if no cell found
    if cell_mask_cropped.size == 0:
        return None

    # Convert back to None for unused images
    image1_cropped = image1_cropped if image1 is not None else None
    image2_cropped = image2_cropped if image2 is not None else None

    return cell_mask_cropped, image1_cropped, image2_cropped


def _get_regionprops_features(
    cell_ids: Sequence[int],
    labels: NDArray,
    intensity_image: NDArray | None = None,
    queue: Any | None = None,
) -> pd.DataFrame:
    """Calculate regionprops features for each cell from the full label image."""
    # Initialize features dictionary with None values to preserve order
    features = dict.fromkeys(cell_ids, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Process cells in order to preserve order
        for cell_id in cell_ids:
            crop = _get_cell_crops(cell_id, labels, image1=intensity_image)
            if crop is None:
                continue
            cell_mask_cropped, intensity_image_cropped, _ = crop
            cell_features = _calculate_regionprops_from_crop(cell_mask_cropped, intensity_image_cropped, cell_id)
            features[cell_id] = cell_features
            if queue is not None:
                queue.put(Signal.UPDATE)
        if queue is not None:
            queue.put(Signal.FINISH)

    # Convert to DataFrame while preserving order
    df = pd.DataFrame.from_dict(features, orient="index")
    # Ensure the index matches the input cell_ids order
    df = df.reindex(cell_ids)
    return df


def _measurement_wrapper(
    func: Callable[..., dict[str, Any]],
    mask: NDArray,
    image1: NDArray | None,
    image2: NDArray | None = None,
) -> dict[str, Any]:
    """Wrapper function to handle both core and correlation measurements.

    Parameters
    ----------
    func
        The measurement function to call
    mask
        The cell mask
    image1
        First image (or only image for core measurements)
    image2
        Second image for correlation measurements. If None, this is a core
        measurement.

    Returns
    -------
    Dictionary of feature values
    """
    if image1 is None:
        return {}  # Return empty dict if no image data

    try:
        if image2 is None:
            return func(mask, image1)
        else:
            # Check if we have valid data for correlation
            if not np.any(mask) or not np.any(image1) or not np.any(image2):
                # Get feature names from a successful call to maintain structure
                dummy_mask = np.ones((2, 2), dtype=bool)
                dummy_img = np.ones((2, 2), dtype=image1.dtype)
                feature_names = func(dummy_img, dummy_img, dummy_mask).keys()
                # Return dictionary with NaN values for all features
                return {name: np.nan for name in feature_names}
            return func(image1, image2, mask)
    except (IndexError, ValueError) as e:
        # Handle cases where correlation calculation fails
        if "index 0 is out of bounds" in str(e) or "size 0" in str(e):
            # Get feature names from a successful call to maintain structure
            dummy_mask = np.ones((2, 2), dtype=bool)
            dummy_img = np.ones((2, 2), dtype=image1.dtype)
            feature_names = func(dummy_img, dummy_img, dummy_mask).keys()
            # Return dictionary with NaN values for all features
            return {name: np.nan for name in feature_names}
        raise  # Re-raise other errors


def _calculate_features_helper(
    cell_ids: Sequence[int],
    labels: NDArray,
    image1: NDArray,
    image2: NDArray | None,
    measurements: dict[str, Any],
    channel1_name: str | None = None,
    channel2_name: str | None = None,
    queue: Any | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Helper function to calculate features for a subset of cells."""
    # Initialize features dictionary with None values to preserve order
    features_dict = dict.fromkeys(cell_ids, None)

    # Pre-allocate lists for type conversion
    uint8_features = [
        "radial_distribution",
        "radial_zernikes",
        "intensity",
        "sizeshape",
        "zernike",
        "ferret",
    ]
    float_features = ["manders_fold", "rwc"]

    # Pre-compute normalization if needed
    conv_params: dict[str, Any] = {
        "uint8_features": uint8_features,
        "float_features": float_features,
    }
    if "texture" in measurements:
        img1_min = image1.min()
        img1_max = image1.max()
        conv_params["img1_min"] = img1_min
        conv_params["img1_range"] = img1_max - img1_min + 1e-10
        if image2 is not None:
            img2_min = image2.min()
            img2_max = image2.max()
            conv_params["img2_min"] = img2_min
            conv_params["img2_range"] = img2_max - img2_min + 1e-10

    # Process cells in order to preserve order
    for cell_id in cell_ids:
        crop = _get_cell_crops(cell_id, labels, image1, image2, verbose=verbose)
        if crop is None:
            continue
        cell_mask_cropped, image1_cropped, image2_cropped = crop
        cell_features = {}

        # Calculate regionprops features using cached crop
        try:
            region_features = _calculate_regionprops_from_crop(
                cell_mask_cropped,
                image1_cropped if image2 is None else None,
                cell_id,
            )
            if image2 is None:
                region_features = _append_channel_names(region_features, channel1_name)
            else:
                region_features = _append_channel_names(region_features, channel1_name, channel2_name)
            cell_features.update(region_features)
        except (ValueError, TypeError, AttributeError) as e:
            if verbose:
                logg.warning(f"Failed to calculate regionprops features for cell {cell_id}: {str(e)}")

        # Calculate cp-measure features for each measurement
        for name, func in measurements.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if image1_cropped is None:
                        continue
                    mask_conv, img1_conv, img2_conv = _prepare_images_for_measurement(
                        name,
                        cell_mask_cropped,
                        image1_cropped,
                        image2_cropped,
                        conv_params,
                    )
                    feature_dict = _measurement_wrapper(func, mask_conv, img1_conv, img2_conv)
                    # Ensure each feature returns a single value
                    for k, v in feature_dict.items():
                        if len(v) > 1:
                            raise ValueError(f"Feature {k} has more than one value.")
                        else:
                            feature_dict[k] = float(v[0])
                    if image2 is None:
                        feature_dict = _append_channel_names(feature_dict, channel1_name)
                    else:
                        feature_dict = _append_channel_names(feature_dict, channel1_name, channel2_name)
                    cell_features.update(feature_dict)
            except (ValueError, TypeError, AttributeError) as e:
                if verbose:
                    logg.warning(f"Failed to calculate '{name}' features for cell {cell_id}: {str(e)}")

        features_dict[cell_id] = cell_features

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    # Convert to DataFrame while preserving order
    df = pd.DataFrame.from_dict(features_dict, orient="index")
    # Ensure the index matches the input cell_ids order
    df = df.reindex(cell_ids)
    return df


def _get_array_from_DataTree_or_DataArray(
    data: xr.DataTree | xr.DataArray,
    scale: str | None = None,
) -> NDArray:
    """
    Returns a NumPy array for the given data and scale.
    If data is an xr.DataTree, it checks for the scale key and computes the image.
    If data is an xr.DataArray, it computes the array (ignoring scale).

    Parameters
    ----------
    data
        The xarray data to convert to a NumPy array
    scale
        Optional scale key for DataTree data

    Returns
    -------
    np.ndarray
        The computed NumPy array
    """
    if not isinstance(data, xr.DataTree):
        return np.asarray(data.compute())
    if scale is None:
        raise ValueError("Scale must be provided for DataTree data")
    if scale not in data:
        raise ValueError(f"Scale '{scale}' not found. Available scales: {list(data.keys())}")
    return np.asarray(data[scale].image.compute())
