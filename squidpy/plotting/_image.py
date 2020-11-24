from typing import Mapping, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from squidpy.image import ImageContainer


def extract(adata, obsm_cols: Optional[Mapping] = None):
    """
    Create a temporary adata object for plotting.

    Move columns defined with `obsm_cols` from obsm to obs to enable the use of
    functions from `scanpy.plotting`.
    Columns are moved to "<obsm-key>_<column-name>".
    If `adata.obs["<obsm-key>_<column-name>"]` already exists, it is overwritten.

    For obsm entries that are np.ndarray, specify integer column names in `obsm_cols`.
    For obsm entries that are pd.DataFrame, specify string column names in `obsm_cols`.

    Params
    ------
    obsm_cols:
        Dict with `adata.obsm` as keys and `adata.obsm[key]` columns names as values.
        If None, defaults to all columns from all entries in `adata.obsm`.

    Returns
    -------
    :class:`AnnData`
        Temporary annotated data object with desired entries in `.obs`.
    """
    # fill obsm_cols if emptpy
    if obsm_cols is None:
        obsm_cols = {}
        for key, obsm in adata.obsm.items():
            if isinstance(obsm, pd.DataFrame):
                obsm_cols[key] = list(obsm.columns)
            else:
                obsm_cols[key] = range(obsm.shape)

    # check obsm_cols
    for key, vals in obsm_cols.items():
        if key not in adata.obsm.keys():
            raise IndexError(f"{key} does not exists in adata.obsm")
        if isinstance(adata.obsm[key], np.ndarray):
            if not np.all([isinstance(v, int) for v in vals]):
                raise ValueError(f"obsm entry {key} is a numpy array, but not all values in obsm_cols['{key}'] are int")

    # create tmp_adata and copy obsm columns
    tmp_adata = adata.copy()
    for key, vals in obsm_cols.items():
        obsm = adata.obsm[key]
        for val in vals:
            if isinstance(adata.obsm[key], pd.DataFrame):
                tmp_adata.obs[f"{key}_{val}"] = obsm.loc[:, val]
            else:
                tmp_adata.obs[f"{key}_{val}"] = obsm[:, val]

    return tmp_adata


def plot_segmentation(img: ImageContainer, key: str) -> None:
    """
    Plot segmentation on entire image.

    Parameters
    ----------
    img
        High-resolution image.
    key
        Name of layer that contains segmentation in img.

    Returns
    -------
    None
        TODO.
    """
    arr = img[key]
    plt.imshow(arr)
    # todo add other channels in background
