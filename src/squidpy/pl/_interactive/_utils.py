from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.colors import to_hex, to_rgb
from numba import njit
from pandas import CategoricalDtype
from pandas._libs.lib import infer_dtype
from pandas.core.dtypes.common import is_categorical_dtype
from scanpy import logging as logg
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation
from scipy.spatial import KDTree

from squidpy._constants._pkg_constants import Key
from squidpy._utils import NDArrayA


def _get_categorical(
    adata: AnnData,
    key: str,
    palette: str | None = None,
    vec: pd.Series | None = None,
) -> NDArrayA:
    if vec is not None:
        if not isinstance(vec.dtype, CategoricalDtype):
            raise TypeError(f"Expected a `categorical` type, found `{infer_dtype(vec)}`.")
        if key in adata.obs:
            logg.debug(f"Overwriting `adata.obs[{key!r}]`")

        adata.obs[key] = vec.values

    add_colors_for_categorical_sample_annotation(
        adata, key=key, force_update_colors=palette is not None, palette=palette
    )
    col_dict = dict(zip(adata.obs[key].cat.categories, [to_rgb(i) for i in adata.uns[Key.uns.colors(key)]]))

    return np.array([col_dict[v] for v in adata.obs[key]])


def _position_cluster_labels(coords: NDArrayA, clusters: pd.Series, colors: NDArrayA) -> dict[str, NDArrayA]:
    if not isinstance(clusters.dtype, CategoricalDtype):
        raise TypeError(f"Expected `clusters` to be `categorical`, found `{infer_dtype(clusters)}`.")

    coords = coords[:, 1:]  # TODO(michalk8): account for current Z-dim?
    df = pd.DataFrame(coords)
    df["clusters"] = clusters.values
    df = df.groupby("clusters")[[0, 1]].apply(lambda g: list(np.median(g.values, axis=0)))
    df = pd.DataFrame(list(df), index=df.index)

    kdtree = KDTree(coords)
    clusters = np.full(len(coords), fill_value="", dtype=object)
    # index consists of the categories that need not be string
    clusters[kdtree.query(df.values)[1]] = df.index.astype(str)
    # napari v0.4.9 - properties must be 1-D in napari/layers/points/points.py:581
    colors = np.array([to_hex(col if cl != "" else (0, 0, 0)) for cl, col in zip(clusters, colors)])

    return {"clusters": clusters, "colors": colors}


def _not_in_01(arr: NDArrayA | da.Array) -> bool:
    @njit
    def _helper_arr(arr: NDArrayA) -> bool:
        for val in arr.flat:
            if not (0 <= val <= 1):
                return True

        return False

    if isinstance(arr, da.Array):
        return bool(np.min(arr) < 0) or bool(np.max(arr) > 1)

    return bool(_helper_arr(np.asarray(arr)))


def _display_channelwise(arr: NDArrayA | da.Array) -> bool:
    n_channels: int = arr.shape[-1]
    if n_channels not in (3, 4):
        return n_channels != 1
    if np.issubdtype(arr.dtype, np.uint8):
        return False  # assume RGB(A)
    if not np.issubdtype(arr.dtype, np.floating):
        return True

    return _not_in_01(arr)
