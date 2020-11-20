"""Spatial tools general utility functions."""
import os
import glob
from typing import Optional

import scanpy as sc
import anndata as ad
import scanpy.logging as logg

import pandas as pd

from .image.object import ImageContainer


def read_seqfish(base_path: str, dataset: str):
    """
    Read seqfish dataset.

    Parameters
    ----------
    base_path
        Path to a directory where the dataset is stores.
    dataset
        Type of the dataset. Can be one of `'ob'` or `'svz'`.

    Returns
    -------
    :class:`anndata.AnnData`
        Annotated data object.
    """
    if dataset == "ob":
        counts_path = os.path.join(base_path, "sourcedata", "ob_counts.csv")
        clusters_path = os.path.join(base_path, "OB_cell_type_annotations.csv")
        centroids_path = os.path.join(base_path, "sourcedata", "ob_cellcentroids.csv")
    elif dataset == "svz":
        counts_path = os.path.join(base_path, "sourcedata", "cortex_svz_counts.csv")
        clusters_path = os.path.join(base_path, "cortex_svz_cell_type_annotations.csv")
        centroids_path = os.path.join(base_path, "sourcedata", "cortex_svz_cellcentroids.csv")
    else:
        print("Dataset not available")

    counts = pd.read_csv(counts_path)
    clusters = pd.read_csv(clusters_path)
    centroids = pd.read_csv(centroids_path)

    adata = ad.AnnData(counts, obs=pd.concat([clusters, centroids], axis=1))
    adata.obsm["spatial"] = adata.obs[["X", "Y"]].to_numpy()

    adata.obs["louvain"] = pd.Categorical(adata.obs["louvain"])

    return adata


def read_visium_data(dataset_folder: str, count_file: Optional[str] = None, image_file: Optional[str] = None):
    """
    Read adata and tif image from visium dataset.

    Args
    ----
    count_file: Optional[str]
        Name of the h5 file in dataset_folder. If not specified, will use *filtered_feature_bc_matrix.h5
    image_file: Optional[str]
        Name of the .tif file in dataset_folder. If not specified, will use *image.tif

    Returns
    -------
    AnnData:
        Count matrix
    ImageContainer:
        High resolution tif image
    """
    if count_file is None:
        files = sorted(glob.glob(os.path.join(dataset_folder, "*filtered_feature_bc_matrix.h5")))
        assert len(files) > 0, f"did not find a count file in {dataset_folder}"
        count_file = files[0]
        logg.warning(f"read_visium_data: setting count_file to {count_file}")
    if image_file is None:
        files = sorted(glob.glob(os.path.join(dataset_folder, "*image.tif")))
        assert len(files) > 0, f"did not find a image file in {dataset_folder}"
        image_file = files[0]
        logg.warning(f"read_visium_data: setting image_file to {image_file}")
    # read adata
    adata = sc.read_visium(dataset_folder, count_file=count_file)
    # read image
    img = ImageContainer(os.path.join(dataset_folder, image_file))
    return adata, img
