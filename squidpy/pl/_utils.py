from typing import Any, List, Tuple, Union, Hashable, Iterable, Optional, Sequence
from pathlib import Path
from functools import wraps
import os

from scanpy import logging as logg, settings
from anndata import AnnData

from numba import njit, prange
from scipy.sparse import issparse, spmatrix
from pandas._libs.lib import infer_dtype
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_categorical_dtype,
)
import numpy as np
import pandas as pd

from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from squidpy._docs import d


@d.dedent
def save_fig(fig: Figure, path: Union[str, Path], make_dir: bool = True, ext: str = "png", **kwargs: Any) -> None:
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
    if os.path.splitext(path)[1] == "":
        path = f"{path}.{ext}"

    path = Path(path)

    if not path.is_absolute():
        path = Path(settings.figdir) / path

    if make_dir:
        try:
            os.makedirs(str(Path.parent), exist_ok=True)
        except OSError as e:
            logg.debug(f"Unable to create directory `{Path.parent}`. Reason: `{e}`.")

    logg.debug(f"Saving figure to `{path!r}`")

    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("transparent", True)

    fig.savefig(path, **kwargs)


@d.dedent
def extract(
    adata: AnnData,
    obsm_key: Union[List[str], str] = "img_features",
    prefix: Optional[Union[List[str], str]] = None,
) -> AnnData:
    """
    Create a temporary :class:`anndata.AnnData` object for plotting.

    Move columns from entry `obsm` in `adata.obsm` to `adata.obs` to enable the use of `scanpy.plotting` functions.
    If `prefix` is specified, columns are moved to `<prefix>_<column-name>`. Otherwise, column name is kept.

    Warning: If `adata.obs["<column-name>"]` already exists, it will be overwritten.

    Parameters
    ----------
    %(adata)s
    obsm_key
        Entries in :attr:`anndata.AnnData.obsm` that should be moved to :attr:`anndata.AnnData.obs`.
    prefix
        Prefix to prepend to each column name. Should be a :class;`list` if ``obsm_key`` is a :class:`list`.

    Returns
    -------
    Temporary :class:`anndata.AnnData` object with desired entries in :attr:`anndata.AnnData.obs`.

    Raises
    ------
    ValueError
        If number of ``prefixes`` does not fit to number of ``obsm_keys``.
    """
    # TODO: move to utils?
    def _warn_if_exists_obs(adata: AnnData, obs_key: str) -> None:
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


def _get_black_or_white(value: float, cmap: Colormap) -> str:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Value must be in range `[0, 1]`, found `{value}`.")

    r, g, b, *_ = [int(c * 255) for c in cmap(value)]
    return _contrasting_color(r, g, b)


def _unique_order_preserving(iterable: Iterable[Hashable]) -> List[Hashable]:
    """Remove items from an iterable while preserving the order."""
    seen = set()
    return [i for i in iterable if i not in seen and not seen.add(i)]


@njit(cache=True, fastmath=True)
def _point_inside_triangles(triangles: np.ndarray) -> bool:
    # modified from napari
    AB = triangles[:, 1, :] - triangles[:, 0, :]
    AC = triangles[:, 2, :] - triangles[:, 0, :]
    BC = triangles[:, 2, :] - triangles[:, 1, :]

    s_AB = -AB[:, 0] * triangles[:, 0, 1] + AB[:, 1] * triangles[:, 0, 0] >= 0
    s_AC = -AC[:, 0] * triangles[:, 0, 1] + AC[:, 1] * triangles[:, 0, 0] >= 0
    s_BC = -BC[:, 0] * triangles[:, 1, 1] + BC[:, 1] * triangles[:, 1, 0] >= 0

    return np.any((s_AB != s_AC) & (s_AB == s_BC))


@njit(parallel=True)
def _points_inside_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    out = np.empty(
        len(
            points,
        ),
        dtype=np.bool_,
    )
    for i in prange(len(out)):
        out[i] = _point_inside_triangles(triangles - points[i])

    return out


def _min_max_norm(vec: Union[spmatrix, np.ndarray]) -> np.ndarray:
    if issparse(vec):
        vec = vec.A.squeeze()
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"Expected `1` dimension, found `{vec.ndim}`.")

    maxx, minn = np.nanmax(vec), np.nanmin(vec)

    return np.ones_like(vec) if np.isclose(minn, maxx) else ((vec - minn) / (maxx - minn))


# TODO: type + comment
def _ensure_dense_vector(fn):
    @wraps(fn)
    def decorator(self: "ALayer", *args, **kwargs) -> Tuple[Optional[Union[pd.Series, np.ndarray]], Optional[str]]:
        normalize = kwargs.pop("normalize", False)
        res, fmt = fn(self, *args, **kwargs)
        if res is None:
            return None, None

        if isinstance(res, pd.Series):
            if is_categorical_dtype(res):
                return res, fmt
            if is_string_dtype(res) or is_object_dtype(res) or is_bool_dtype(res):
                return res.astype("category"), fmt
            if is_integer_dtype(res):
                unique = res.unique()
                n_uniq = len(unique)
                if n_uniq <= 2 and (set(unique) & {0, 1}):
                    return res.astype(bool).astype("category"), fmt
                if len(unique) <= len(res) // 100:
                    return res.astype("category"), fmt
            elif not is_numeric_dtype(res):
                raise TypeError(f"Unable to process `pandas.Series` of type `{infer_dtype(res)}`.")
            res = res.to_numpy()
        elif issparse(res):
            res = res.toarray()
        elif not isinstance(res, (np.ndarray, Sequence)):
            raise TypeError(f"Unable to process result of type `{type(res).__name__}`.")

        res = np.asarray(np.squeeze(res))
        if res.ndim != 1:
            raise ValueError(f"Expected 1-dimensional array, found `{res.ndim}`.")

        return (_min_max_norm(res) if normalize else res), fmt

    return decorator


def _only_not_raw(fn):
    @wraps(fn)
    def decorator(self, *args, **kwargs):
        return None if self.raw else fn(self, *args, **kwargs)

    return decorator


# TODO: clean-up
class ALayer:
    """
    Class which helps with :class:`anndata.AnnData` layer logic.

    Parameters
    ----------
    %(adata)s
    is_raw
        Whether we want to access :attr:`anndata.AnnData.raw`.
    palette
        Color palette for categorical variables which don't have colors in :attr:`anndata.AnnData.uns`.
    """

    VALID_ATTRIBUTES = ("obs", "var", "obsm")

    # TODO: properly type palette
    def __init__(self, adata: AnnData, is_raw: bool = False, palette: Optional[str] = None):
        if is_raw and adata.raw is None:
            raise AttributeError("Attribute `.raw` is `None`.")

        self._adata = adata
        self._layer = None
        self._previous_layer = None
        self._raw = is_raw
        self._palette = palette

    @property
    def adata(self) -> AnnData:
        """The underlying annotated data object."""  # noqa: D401
        return self._adata

    @property
    def layer(self) -> Optional[str]:
        """Layer in :attr:`anndata.AnnData.layers`."""
        return self._layer

    @layer.setter
    def layer(self, layer: Optional[str] = None) -> None:
        if layer not in (None,) + tuple(self.adata.layers.keys()):
            raise KeyError(f"Invalid layer `{layer}`. Valid options are: `{[None] + list(self.adata.layers.keys())}`.")
        self._previous_layer = layer
        # handle in raw setter
        self.raw = False

    @property
    def raw(self) -> bool:
        """Whether to access :attr:`anndata.AnnData.raw`."""
        return self._raw

    @raw.setter
    def raw(self, is_raw: bool) -> None:
        if is_raw:
            if self.adata.raw is None:
                raise AttributeError("Attribute `.raw` is `None`.")
            self._previous_layer = self.layer
            self._layer = None
        else:
            self._layer = self._previous_layer
        self._raw = is_raw

    @_ensure_dense_vector
    def get_obs(self, name: str, **_) -> Tuple[Optional[Union[pd.Series, np.ndarray]], str]:
        """
        Return an observation.

        Parameters
        ----------
        name
            Key in :attr:`anndata.AnnData.obs` to access.

        Returns
        -------
        The values and the formatted ``name``.
        """
        if name not in self.adata.obs.columns:
            raise KeyError(f"Key `{name}` not found in `adata.obs`.")
        return self.adata.obs[name], self._format_key(name, layer_modifier=False)

    @_ensure_dense_vector
    def get_var(self, name: Union[str, int], **_) -> Tuple[Optional[np.ndarray], str]:
        """
        Return a gene.

        Parameters
        ----------
        name
            Gene name in :attr:`anndata.AnnData.var_names` or :attr:`anndata.AnnData.raw.var_names`,
            based on :paramref:`raw`.

        Returns
        -------
        The values and the formatted ``name``.
        """
        adata = self.adata.raw if self.raw else self.adata
        try:
            ix = adata._normalize_indices((slice(None), name))
        except KeyError:
            raise KeyError(f"Key `{name}` not found in `adata.{'raw.' if self.raw else ''}var_names`.") from None

        return self.adata._get_X(use_raw=self.raw, layer=self.layer)[ix], self._format_key(name, layer_modifier=True)

    def get_items(self, attr: str) -> tuple:
        """
        Return valid keys for an attribute.

        Parameters
        ----------
        attr
            Attribute of :mod:`anndata.AnnData` to access.

        Returns
        -------
        The available items.
        """
        adata = self.adata.raw if self.raw and attr in ("var",) else self.adata
        if attr in ("obs", "obsm"):
            return tuple(map(str, getattr(adata, attr).keys()))
        return tuple(map(str, getattr(adata, attr).index))

    @_ensure_dense_vector
    def get_obsm(self, name: str, index: int = 0) -> Tuple[Optional[np.ndarray], str]:
        """
        Return a vector from :attr:`anndata.AnnData.obsm`.

        Parameters
        ----------
        name
            Key in :attr:`anndata.AnnData.obsm`.
        index
            Index of the vector.

        Returns
        -------
        The values and the formatted ``name``.
        """
        if name not in self.adata.obsm:
            raise KeyError(name)
        if not isinstance(index, int):
            raise ValueError(index)
        res = self.adata.obsm[name]

        return (res if res.ndim == 1 else res[:, index]), self._format_key(name, layer_modifier=False, index=index)

    def _format_key(self, key: str, layer_modifier: bool = False, index: Optional[int] = None) -> str:
        if not layer_modifier:
            return str(key) + (f":{index}" if index is not None else "")

        return str(key) + (":raw" if self.raw else f":{self.layer}" if self.layer is not None else "")

    def __repr__(self):
        return f"{self.__class__.__name__}<raw={self.raw}, layer={self.layer}>"

    def __str__(self):
        return repr(self)
