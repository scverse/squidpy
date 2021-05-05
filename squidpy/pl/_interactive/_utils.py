from typing import Dict, Union, Optional

from scanpy import logging as logg
from anndata import AnnData
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation

from numba import njit
from scipy.spatial import KDTree
from pandas._libs.lib import infer_dtype
from pandas.core.dtypes.common import is_categorical_dtype
import numpy as np
import pandas as pd
import dask.array as da

from matplotlib.colors import to_rgb

from squidpy._constants._pkg_constants import Key


def _get_categorical(
    adata: AnnData,
    key: str,
    palette: Optional[str] = None,
    vec: Optional[pd.Series] = None,
) -> np.ndarray:
    if vec is not None:
        if not is_categorical_dtype(vec):
            raise TypeError(f"Expected a `categorical` type, found `{infer_dtype(vec)}`.")
        if key in adata.obs:
            logg.debug(f"Overwriting `adata.obs[{key!r}]`")

        adata.obs[key] = vec.values

    add_colors_for_categorical_sample_annotation(
        adata, key=key, force_update_colors=palette is not None, palette=palette
    )
    col_dict = dict(zip(adata.obs[key].cat.categories, [to_rgb(i) for i in adata.uns[Key.uns.colors(key)]]))

    return np.array([col_dict[v] for v in adata.obs[key]])


def _position_cluster_labels(coords: np.ndarray, clusters: pd.Series, colors: np.ndarray) -> Dict[str, np.ndarray]:
    if not is_categorical_dtype(clusters):
        raise TypeError(f"Expected `clusters` to be `categorical`, found `{infer_dtype(clusters)}`.")

    df = pd.DataFrame(coords)
    df["clusters"] = clusters.values
    df = df.groupby("clusters")[[0, 1]].apply(lambda g: list(np.median(g.values, axis=0)))
    df = pd.DataFrame(list(df), index=df.index)

    kdtree = KDTree(coords)
    clusters = np.full(len(coords), fill_value="", dtype=object)
    # index consists of the categories that need not be string
    clusters[kdtree.query(df.values)[1]] = df.index.astype(str)
    colors = np.array([col if cl != "" else (0, 0, 0) for cl, col in zip(clusters, colors)])

    return {"clusters": clusters, "colors": colors}


def _not_in_01(arr: Union[np.ndarray, da.Array]) -> bool:
    @njit
    def _helper_arr(arr: np.ndarray) -> bool:
        for val in arr.flat:
            if not (0 <= val <= 1):
                return True

        return False

    if isinstance(arr, da.Array):
        return bool(np.min(arr) < 0) or bool(np.max(arr) > 1)

    return bool(_helper_arr(np.asarray(arr)))


def _is_luminance(arr: Union[np.ndarray, da.Array]) -> bool:
    # TODO: maybe do only for images with multiple Z-dim?
    n_channels: int = arr.shape[-1]
    if n_channels not in (3, 4):
        return n_channels != 1
    if np.issubdtype(arr.dtype, np.uint8):
        return False  # assume RGB(A)
    if not np.issubdtype(arr.dtype, np.floating):
        return True

    return _not_in_01(arr)
