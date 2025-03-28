"""Experimental feature extraction module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable
import warnings
import anndata as ad
import numpy as np
import pandas as pd
from cp_measure.bulk import get_core_measurements, get_correlation_measurements
from spatialdata._logging import logger as logg
from spatialdata import SpatialData
import xarray as xr

from squidpy._constants._constants import ImageFeature
from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, _get_n_cores, parallelize
from spatialdata.models import TableModel

__all__ = ["calculate_image_features"]


def _measurement_wrapper(
    func: Callable,
    mask: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray | None = None,
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
    return func(mask, image1) if image2 is None else func(image1, image2, mask)


def _calculate_features_helper(
    cell_ids: Sequence[int],
    labels: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray | None,
    measurements: dict[str, Any],
    channel1_name: str | None = None,
    channel2_name: str | None = None,
    queue: Any | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Helper function to calculate features for a subset of cells."""
    features_dict = {}

    # Pre-allocate arrays for type conversion
    uint8_features = [
        "radial_distribution",
        "radial_zernikes",
        "intensity",
        "sizeshape",
        "zernike",
        "ferret",
    ]
    float_features = ["manders_fold", "rwc"]

    # Pre-compute image normalization if needed
    if "texture" in measurements:
        img1_min = image1.min()
        img1_max = image1.max()
        img1_range = img1_max - img1_min + 1e-10
        if image2 is not None:
            img2_min = image2.min()
            img2_max = image2.max()
            img2_range = img2_max - img2_min + 1e-10

    for cell_id in cell_ids:
        # Get cell mask and find bounding box in one step
        cell_mask = labels == cell_id
        y_indices, x_indices = np.where(cell_mask)
        if len(y_indices) == 0:  # Skip empty cells
            continue

        # Get bounding box with padding
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        pad = 5
        y_min = max(0, y_min - pad)
        y_max = min(labels.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(labels.shape[1], x_max + pad)

        # Crop all arrays at once
        cell_mask_cropped = cell_mask[y_min:y_max, x_min:x_max]
        image1_cropped = image1[y_min:y_max, x_min:x_max]
        image2_cropped = None if image2 is None else image2[y_min:y_max, x_min:x_max]

        # Quick shape check
        if cell_mask_cropped.shape != image1_cropped.shape or (
            image2_cropped is not None
            and cell_mask_cropped.shape != image2_cropped.shape
        ):
            if verbose:
                logg.warning(
                    f"Shape mismatch for cell {cell_id}: "
                    f"mask {cell_mask_cropped.shape}, "
                    f"image1 {image1_cropped.shape}, "
                    f"image2 {image2_cropped.shape if image2_cropped is not None else 'None'}"
                )
            continue

        cell_features = {}
        # Calculate all available features
        for name, func in measurements.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Pre-convert inputs based on feature type
                    if name in uint8_features:
                        mask = cell_mask_cropped.astype(np.uint8)
                        img1 = image1_cropped.astype(np.uint8)
                        img2 = (
                            None
                            if image2_cropped is None
                            else image2_cropped.astype(np.uint8)
                        )
                    elif name == "texture":
                        mask = cell_mask_cropped.astype(np.uint8)
                        img1 = (
                            image1_cropped.astype(np.float32) - img1_min
                        ) / img1_range
                        img2 = (
                            None
                            if image2_cropped is None
                            else (image2_cropped.astype(np.float32) - img2_min)
                            / img2_range
                        )
                    elif name in float_features:
                        mask = cell_mask_cropped.astype(np.float32)
                        img1 = image1_cropped.astype(np.float32)
                        img2 = (
                            None
                            if image2_cropped is None
                            else image2_cropped.astype(np.float32)
                        )
                    else:
                        mask = cell_mask_cropped.astype(np.float32)
                        img1 = image1_cropped.astype(np.float32)
                        img2 = (
                            None
                            if image2_cropped is None
                            else image2_cropped.astype(np.float32)
                        )

                    feature_dict = _measurement_wrapper(func, mask, img1, img2)

                    for k, v in feature_dict.items():
                        if len(v) > 1:
                            raise ValueError(f"Feature {k} has more than one value.")
                        else:
                            feature_dict[k] = float(v[0])

                    # Append channel names efficiently
                    if image2 is None:
                        feature_dict = {
                            f"{k}_ch{channel1_name}": v for k, v in feature_dict.items()
                        }
                    else:
                        feature_dict = {
                            f"{k}_ch{channel1_name}_ch{channel2_name}": v
                            for k, v in feature_dict.items()
                        }

                    cell_features.update(feature_dict)
            except Exception as e:
                if verbose:
                    logg.warning(
                        f"Failed to calculate '{name}' features for cell {cell_id}: {str(e)}"
                    )

        features_dict[cell_id] = cell_features

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame.from_dict(features_dict, orient="index")


@d.dedent
@inject_docs(f=ImageFeature)
def calculate_image_features(
    sdata: SpatialData,
    labels_key: str,
    image_key: str,
    scale: str | None = None,
    adata_key_added: str = "morphology",
    invalid_as_zero: bool = True,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    verbose: bool = False,
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

    if (
        (isinstance(sdata.images[image_key], xr.DataTree)
        or isinstance(sdata.labels[labels_key], xr.DataTree))
        and scale is None
    ):
        raise ValueError("When using multi-scale data, please specify the scale.")

    if scale is not None and not isinstance(scale, str):
        raise ValueError("Scale must be a string.")

    if scale is not None:
        image = np.asarray(sdata.images[image_key][scale].image.compute())
        labels = np.asarray(sdata.labels[labels_key][scale].image.compute())
    else:
        image = np.asarray(sdata.images[image_key].compute())
        labels = np.asarray(sdata.labels[labels_key].compute())

    # Check if labels are empty
    if labels.size == 0:
        raise ValueError("Labels array is empty")

    max_label = int(labels.max())
    if max_label == 0:
        raise ValueError("No cells found in labels (max label is 0)")

    # Get channel names if available
    channel_names = None
    if (
        hasattr(sdata.images[image_key], "coords")
        and "c" in sdata.images[image_key].coords
    ):
        channel_names = sdata.images[image_key].coords["c"].values

    # Handle image dimensions
    if image.ndim == 2:
        image = image[None, :, :]  # Add channel dimension
    elif image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

    # Check if image and labels have matching dimensions
    if image.shape[1:] != labels.shape:
        raise ValueError(
            f"Image and labels have mismatched dimensions: "
            f"image {image.shape[1:]}, labels {labels.shape}"
        )

    # Get core measurements from cp_measure
    measurements_core = get_core_measurements()
    measurements_corr = get_correlation_measurements()

    # Get unique cell IDs from labels, excluding background (0)
    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids != 0]

    # Process each channel
    all_features = []
    n_channels = image.shape[0]
    n_jobs = _get_n_cores(n_jobs)

    logg.info(f"Using '{n_jobs}' core(s).")

    # First process core measurements for each channel
    for ch_idx in range(n_channels):
        logg.info(f"Calculating core features for channel '{ch_idx}'.")

        ch_name = channel_names[ch_idx] if channel_names is not None else f"ch{ch_idx}"
        ch_image = image[ch_idx]

        res = parallelize(
            _calculate_features_helper,
            collection=cell_ids,
            extractor=pd.concat,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
            verbose=verbose,
        )(labels, ch_image, None, measurements_core, ch_name)

        all_features.append(res)

    # Then process correlation measurements between channels
    for ch1_idx in range(n_channels):
        for ch2_idx in range(ch1_idx + 1, n_channels):
            ch1_name = (
                channel_names[ch1_idx] if channel_names is not None else f"ch{ch1_idx}"
            )
            ch2_name = (
                channel_names[ch2_idx] if channel_names is not None else f"ch{ch2_idx}"
            )

            logg.info(
                f"Calculating correlation features between channels "
                f"'{ch1_name}' and '{ch2_name}'."
            )

            ch1_image = image[ch1_idx]
            ch2_image = image[ch2_idx]

            # Parallelize feature calculation
            res = parallelize(
                _calculate_features_helper,
                collection=cell_ids,
                extractor=pd.concat,
                n_jobs=n_jobs,
                backend=backend,
                show_progress_bar=show_progress_bar,
                verbose=verbose,
            )(labels, ch1_image, ch2_image, measurements_corr, ch1_name, ch2_name)

            all_features.append(res)

    # Create AnnData object from results
    combined_features = pd.concat(all_features, axis=1)
    if invalid_as_zero:
        combined_features = combined_features.replace([np.inf, -np.inf], 0)
        combined_features = combined_features.fillna(0)

    # Ensure cell IDs are preserved in the correct order
    cell_ids = sorted(combined_features.index)
    combined_features = combined_features.loc[cell_ids]

    adata = ad.AnnData(X=combined_features)
    adata.obs_names = [f"cell_{i}" for i in cell_ids]
    adata.var_names = combined_features.columns

    adata.uns["spatialdata_attrs"] = {
        "region": labels_key,
        "region_key": "region",
        "instance_key": "label_id",
    }
    adata.obs["region"] = pd.Categorical([labels_key] * len(adata))
    adata.obs["label_id"] = cell_ids

    # Add the AnnData object to the SpatialData object
    sdata.tables[adata_key_added] = TableModel.parse(adata)
