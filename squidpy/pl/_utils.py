import os
from typing import List, Union, Optional
from pathlib import Path

import anndata as ad
from scanpy import logging as logg
from scanpy import settings

import pandas as pd

from matplotlib.figure import Figure

from squidpy._docs import d


@d.dedent
def save_fig(fig: Figure, path: Union[str, os.PathLike], make_dir: bool = True, ext: str = "png", **kwargs) -> None:
    """
    Save a figure.

    Parameters
    ----------
    fig
        Figure to save.
    path
        Path where to save the figure. If path is relative, save it under :attr:`scanpy.settings.figdir`.
    make_dir
        Whether to try making the directory if it does not exist.
    ext
        Extension to use if none is provided.
    kwargs
        Keyword arguments for :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    None
        Just saves the plot.
    """
    path = Path(path)

    if os.path.splitext(path)[1] == "":
        path = f"{path}.{ext}"

    if not path.is_absolute():
        path = Path(settings.figdir) / path

    if make_dir:
        try:
            os.makedirs(Path.parent, exist_ok=True)
        except OSError as e:
            logg.debug(f"Unable to create directory `{Path.parent}`. Reason: `{e}`.")

    logg.debug(f"Saving figure to `{path!r}`")

    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("transparent", True)

    fig.savefig(path, **kwargs)


@d.dedent
def extract(
    adata: ad.AnnData,
    obsm_key: Union[List["str"], "str"] = "img_features",
    prefix: Optional[Union[List["str"], "str"]] = None,
) -> ad.AnnData:
    """
    Create a temporary adata object for plotting.

    Move columns from entry `obsm` in `adata.obsm` to `adata.obs` to enable the use of
    `scanpy.plotting` functions.
    If `prefix` is specified, columns are moved to `<prefix>_<column-name>`.
    Otherwise, column name is kept.
    Warning: If `adata.obs["<column-name>"]` already exists, it is overwritten.


    Params
    ------
    %(adata)s
    obsm_key:
        entry in adata.obsm that should be moved to adata.obs. Can be a list of keys.
    prefix:
        prefix to prepend to each column name. Should be a list if obsm_key is a list.

    Returns
    -------
        Temporary annotated data object with desired entries in `.obs`.

    Raises
    ------
    ValueError
        if number of prefixes does not fit to number of obsm_keys.
    """

    def _warn_if_exists_obs(adata, obs_key):
        if obs_key in adata.obs.columns:
            logg.warning(f"{obs_key} in adata.obs will be overwritten by extract.")

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
                raise ValueError(f"length of prefix {len(prefix)} does not fit length of obsm_key {len(obsm_key)}")
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
                _warn_if_exists_obs(tmp_adata, obs_key)
                tmp_adata.obs[obs_key] = obsm.loc[:, col]
        else:
            # names will be integer indices
            for j in range(obsm.shape[1]):
                obs_key = f"{prefix[i]}{j}"
                _warn_if_exists_obs(tmp_adata, obs_key)
                tmp_adata.obs[obs_key] = obsm[:, j]

    return tmp_adata
