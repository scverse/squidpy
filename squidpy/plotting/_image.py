from typing import List, Union, Optional

import anndata as ad

import pandas as pd

import matplotlib.pyplot as plt

from squidpy.image import ImageContainer


def extract(adata: ad.AnnData, obsm_key: Union[List["str"], "str"], prefix: Optional[Union[List["str"], "str"]] = None):
    """
    Create a temporary adata object for plotting.

    Move columns from entry `obsm` in `adata.obsm` to `adata.obs` to enable the use of
    functions from `scanpy.plotting`.
    If `prefix` is specified, columns are moved to `<prefix>_<column-name>`.
    Otherwise, column name is kept. If `adata.obs["column-name"]` already exists, it it overwritten.


    Params
    ------
    obsm_key:
        entry in adata.obsm that should be moved to adata.obs. Can be a list of keys.
    prefix:
        prefix to prepend to each column name. Should be a list if obsm_key is a list.

    Returns
    -------
    :class:`AnnData`
        Temporary annotated data object with desired entries in `.obs`.

    Raises
    ------
    ValueError
        if number of prefixes does not fit to number of obsm_keys.
    """
    # make obsm list
    if isinstance(obsm_key, str):
        obsm_key = [obsm_key]

    if prefix is not None:
        # make prefix list of correct length
        if isinstance(prefix, str):
            prefix = [prefix]
        if len(obsm_key) != len(prefix):
            # repeat prefix if only one was specified
            if len(prefix) == 1:
                prefix = [prefix[0] for _ in obsm_key]
            else:
                raise ValueError(f"length of prefix {len(prefix)} does not fit to length of obsm_key {len(obsm_key)}")
        # append _ to prefix
        prefix = [f"{p}_" for p in prefix]
    else:
        # no prefix
        # TODO default could also be obsm_key
        prefix = ["" for _ in obsm_key]

    # create tmp_adata and copy obsm columns
    tmp_adata = adata.copy()
    for i, cur_obsm_key in enumerate(obsm_key):
        obsm = adata.obsm[cur_obsm_key]
        if isinstance(obsm, pd.DataFrame):
            # names will be column_names
            for col in obsm.columns:
                obs_key = f"{prefix[i]}{col}"
                tmp_adata.obs[obs_key] = obsm.loc[:, col]
        else:
            # names will be integer indices
            for j in range(obsm.shape[1]):
                obs_key = f"{prefix[i]}{j}"
                tmp_adata.obs[obs_key] = obsm[:, j]

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
