"""Graph utilities."""
from typing import Any, Tuple, Union, Iterable, Optional, Sequence

from scanpy import logging as logg
from anndata import AnnData

from scipy.sparse import issparse, spmatrix, csc_matrix
from pandas.api.types import infer_dtype, is_categorical_dtype
import numpy as np
import pandas as pd


def _check_tuple_needles(
    needles: Sequence[Tuple[Any, Any]],
    haystack: Sequence[Any],
    msg: str,
    reraise: bool = True,
) -> Sequence[Tuple[Any, Any]]:
    filtered = []

    for needle in needles:
        if not isinstance(needle, Sequence):
            raise TypeError(f"Expected a `Sequence`, found `{type(needle).__name__}`.")
        if len(needle) != 2:
            raise ValueError(f"Expected a `tuple` of length `2`, found `{len(needle)}`.")
        a, b = needle

        if a not in haystack:
            if reraise:
                raise ValueError(msg.format(a))
            else:
                continue
        if b not in haystack:
            if reraise:
                raise ValueError(msg.format(a))
            else:
                continue

        filtered.append((a, b))

    return filtered


# modified from pandas' source code
def _create_sparse_df(
    data: Union[np.ndarray, spmatrix],
    index: Optional[pd.Index] = None,
    columns: Optional[Sequence[Any]] = None,
    fill_value: float = 0,
) -> pd.DataFrame:
    """
    Create a new DataFrame from a scipy sparse matrix or numpy array.

    This is the original :mod:`pandas` implementation with 2 differences:

        - allow creation also from :class:`numpy.ndarray`
        - expose ``fill_values``

    Parameters
    ----------
    data
        Must be convertible to CSC format.
    index
        Row labels to use.
    columns
        Column labels to use.

    Returns
    -------
    Each column of the DataFrame is stored as a :class:`arrays.SparseArray`.
    """
    from pandas import DataFrame
    from pandas._libs.sparse import IntIndex
    from pandas.core.arrays.sparse.accessor import (
        SparseArray,
        SparseDtype,
        SparseFrameAccessor,
    )

    if not issparse(data):
        data = csc_matrix(data)
        sort_indices = False
    else:
        data = data.tocsc()
        sort_indices = True

    data = data.tocsc()
    index, columns = SparseFrameAccessor._prep_index(data, index, columns)
    n_rows, n_columns = data.shape
    # We need to make sure indices are sorted, as we create
    # IntIndex with no input validation (i.e. check_integrity=False ).
    # Indices may already be sorted in scipy in which case this adds
    # a small overhead.
    if sort_indices:
        data.sort_indices()

    indices = data.indices
    indptr = data.indptr
    array_data = data.data
    dtype = SparseDtype(array_data.dtype, fill_value=fill_value)
    arrays = []

    for i in range(n_columns):
        sl = slice(indptr[i], indptr[i + 1])
        idx = IntIndex(n_rows, indices[sl], check_integrity=False)
        arr = SparseArray._simple_new(array_data[sl], idx, dtype)
        arrays.append(arr)

    return DataFrame._from_arrays(arrays, columns=columns, index=index, verify_integrity=False)


def _assert_categorical_obs(adata: AnnData, key: str) -> None:
    if key not in adata.obs:
        raise KeyError()

    if not is_categorical_dtype(adata.obs[key]):
        raise TypeError(f"Expected `adata.obs[{key}]` to be `categorical`, found `{infer_dtype(adata.obs[key])}`.")


def _assert_connectivity_key(adata: AnnData, key: str) -> None:
    if key not in adata.obsp:
        # TODO: nicer message
        raise KeyError(
            f"{key} not present in `adata.obs`"
            "Choose a different connectivity_key or run first "
            "gr.spatial_neighbors on the AnnData object."
        )


def _assert_spatial_basis(adata: AnnData, key: str) -> None:
    if key not in adata.obsm:
        raise KeyError("TODO")


def _subset_by_clusters(
    adata: AnnData, key: str, clusters: Optional[Union[Any, Sequence[Any]]], copy: bool = False
) -> AnnData:
    _assert_categorical_obs(adata, key)

    if clusters is None:
        return adata

    if isinstance(clusters, str) or not isinstance(clusters, Iterable):
        clusters = (clusters,)

    clusters = set(clusters)
    viable_clusters = set(adata.obs[key].cat.categories)
    if not clusters & viable_clusters:
        raise ValueError()

    mask = np.isin(adata.obs[key], tuple(clusters))
    adata = adata[mask, :]

    if not adata.n_obs:
        raise ValueError()

    return adata.copy() if copy else adata


def _assert_positive(n_perms: int, *, name: str) -> None:
    if n_perms <= 0:
        raise ValueError(f"Expected `{name}` to be non-negative, found `{n_perms}`.")


def _save_data(
    adata: AnnData, *, attr: str, key: str, data: Any, prefix: bool = True, time: Optional[Any] = None
) -> None:
    obj = getattr(adata, attr)
    obj[key] = data

    if prefix:
        logg.info(f"Adding `adata.{attr}[{key!r}]`")
    else:
        logg.info(f"       `adata.{attr}[{key!r}]`")
    if time is not None:
        logg.info("Finish", time=time)
