import os
from typing import List, Union, Hashable, Iterable, Optional
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
    obsm_key: Union[List[str], str] = "img_features",
    prefix: Optional[Union[List[str], str]] = None,
) -> ad.AnnData:
    """
    Create a temporary adata object for plotting.

    Move columns from entry `obsm` in `adata.obsm` to `adata.obs` to enable the use of `scanpy.plotting` functions.
    If `prefix` is specified, columns are moved to `<prefix>_<column-name>`. Otherwise, column name is kept.

    Warning: If `adata.obs["<column-name>"]` already exists, it is overwritten.

    Parameters
    ----------
    %(adata)s
    obsm_key
        entry in adata.obsm that should be moved to :attr:`anndata.AnnData.obs`. Can be a list of keys.
    prefix
        prefix to prepend to each column name. Should be a list if ``obsm_key`` is a list.

    Returns
    -------
    Temporary :class:`anndata.AnnData` object with desired entries in :attr:`anndata.AnnData.obs`.

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


def _contrasting_color(r: int, g: int, b: int) -> str:
    for val in [r, g, b]:
        assert 0 <= val <= 255

    return "#000000" if r * 0.299 + g * 0.587 + b * 0.114 > 186 else "#ffffff"


def _get_black_or_white(value: float, cmap) -> str:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Value must be in range `[0, 1]`, found `{value}`.")

    r, g, b, *_ = [int(c * 255) for c in cmap(value)]
    return _contrasting_color(r, g, b)


def _unique_order_preserving(iterable: Iterable[Hashable]) -> List[Hashable]:
    """Remove items from an iterable while preserving the order."""
    seen = set()
    return [i for i in iterable if i not in seen and not seen.add(i)]
