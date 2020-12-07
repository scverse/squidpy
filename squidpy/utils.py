"""Spatial tools general utility functions."""
import os
import glob
from typing import Tuple, Union, Optional
from pathlib import Path

import scanpy as sc
import scanpy.logging as logg
from anndata import AnnData

from squidpy._docs import inject_docs  # noqa: F401
from squidpy.constants._constants import Dataset  # noqa: F401
from squidpy.constants._pkg_constants import Key  # noqa: F401
from .im.object import ImageContainer


def read_visium_data(
    dataset_folder: Union[str, Path], count_file: Optional[str] = None, image_file: Optional[str] = None
) -> Tuple[AnnData, ImageContainer]:
    """
    Read the count matrix and tiff im from [Visium]_ dataset.

    Parameters
    ----------
    dataset_folder
        TODO
    count_file
        Name of the .h5 file in the ``dataset_folder``. If not specified, will use `*filtered_feature_bc_matrix.h5`.
    image_file
        Name of the .tiff file in the ``dataset_folder``. If not specified, will use `*im.tif`.
    Returns
    -------
    :class:`anndata.AnnData`
        Annotated data object.
    :class:`squidpy.image.ImageContainer`
        The high resolution tiff im.
    """
    # TODO: refactor asserts
    dataset_folder = Path(dataset_folder)

    if count_file is None:
        files = sorted(glob.glob(os.path.join(dataset_folder, "*filtered_feature_bc_matrix.h5")))
        assert len(files) > 0, f"did not find a count file in {dataset_folder}"
        count_file = files[0]
        logg.warning(f"read_visium_data: setting count_file to {count_file}")
    if image_file is None:
        files = sorted(glob.glob(os.path.join(dataset_folder, "*im.tif")))
        assert len(files) > 0, f"did not find a im file in {dataset_folder}"
        image_file = files[0]
        logg.warning(f"read_visium_data: setting image_file to {image_file}")
    # read adata
    adata = sc.read_visium(dataset_folder, count_file=count_file)
    # read im
    img = ImageContainer(os.path.join(dataset_folder, image_file))
    return adata, img
