"""Experimental feature extraction module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings
import anndata as ad
import numpy as np
import pandas as pd
from cp_measure.bulk import get_core_measurements
from spatialdata._logging import logger as logg
from spatialdata import SpatialData

from squidpy._constants._constants import ImageFeature
from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, _get_n_cores, parallelize
from spatialdata.models import TableModel
__all__ = ["calculate_image_features"]


@d.dedent
@inject_docs(f=ImageFeature)
def calculate_image_features(
    sdata: SpatialData,
    labels_key: str,
    image_key: str,
    adata_key_added: str = "morphology",
    invalid_as_zero: bool = True,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
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
    # Get the image and labels
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
    measurements = get_core_measurements()

    # Process each channel
    all_features = []
    n_channels = image.shape[0]
    n_jobs = _get_n_cores(n_jobs)

    for ch_idx in range(n_channels):
        ch_name = channel_names[ch_idx] if channel_names is not None else f"ch{ch_idx}"

        logg.info(
            f"Calculating features for channel '{ch_idx}' " f"using '{n_jobs}' core(s)"
        )

        ch_image = image[ch_idx]

        # Get cell IDs
        cell_ids = range(1, max_label + 1)

        # Parallelize feature calculation
        res = parallelize(
            _calculate_image_features_helper,
            collection=cell_ids,
            extractor=pd.concat,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )(labels, ch_image, measurements, ch_name)

        all_features.append(res)

    # Create AnnData object from results
    combined_features = pd.concat(all_features, axis=1)
    if invalid_as_zero:
        combined_features = combined_features.replace([np.inf, -np.inf], 0)
        combined_features = combined_features.fillna(0)
    adata = ad.AnnData(X=combined_features)
    adata.obs_names = [f"cell_{i}" for i in range(1, max_label + 1)]
    adata.var_names = combined_features.columns

    adata.uns["spatialdata_attrs"] = {
        "region": labels_key,
        "region_key": "region", 
        "instance_key": "label_id", 
    }
    adata.obs["region"] = pd.Categorical([labels_key] * len(adata))
    adata.obs["label_id"] = range(1, max_label + 1)
    # adata.obs[["region", "spot_id"]]

    # Add the AnnData object to the SpatialData object
    sdata.tables[adata_key_added] = TableModel.parse(adata)

    # Combine features from all channels
    # return pd.concat(all_features, axis=1)


def _calculate_image_features_helper(
    cell_ids: Sequence[int],
    labels: np.ndarray,
    image: np.ndarray,
    measurements: dict[str, Any],
    channel_name: str | None = None,
    queue: Any | None = None,
) -> pd.DataFrame:
    """Helper function to calculate features for a subset of cells."""
    features_list = []
    for cell_id in cell_ids:
        # Get cell mask
        cell_mask = (labels == cell_id).astype(np.uint8)

        # Find bounding box of the cell
        y_indices, x_indices = np.where(cell_mask)
        if len(y_indices) == 0:  # Skip empty cells
            continue

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Add padding to ensure we capture the full cell
        pad = 5
        y_min = max(0, y_min - pad)
        y_max = min(labels.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(labels.shape[1], x_max + pad)

        # Crop both mask and image to the bounding box
        cell_mask_cropped = cell_mask[y_min:y_max, x_min:x_max]
        image_cropped = image[y_min:y_max, x_min:x_max]

        cell_features = {}
        # Calculate all available features
        for name, func in measurements.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    feature_dict = func(cell_mask_cropped, image_cropped)
                    # Convert numpy arrays to scalars
                    feature_dict = {
                        k: (
                            float(v[0])
                            if isinstance(v, np.ndarray) and v.size == 1
                            else v
                        )
                        for k, v in feature_dict.items()
                    }
                    # Append channel name to feature names
                    feature_dict = {
                        f"{k}_ch{channel_name}": v for k, v in feature_dict.items()
                    }
                    cell_features.update(feature_dict)
            except Exception as e:
                logg.warning(
                    f"Failed to calculate {name} features for cell {cell_id}: "
                    f"{str(e)}"
                )

        features_list.append(cell_features)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame(features_list, index=cell_ids)
