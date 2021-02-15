"""Graph utilities."""
from typing import (
    Any,
    List,
    Tuple,
    Union,
    Hashable,
    Iterable,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

from scanpy import logging as logg
from anndata import AnnData

from scipy.sparse import issparse, spmatrix, csc_matrix
from pandas.api.types import infer_dtype, is_categorical_dtype
import numpy as np
import pandas as pd

from squidpy._utils import _unique_order_preserving


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
        if TYPE_CHECKING:
            assert isinstance(data, spmatrix)
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

    return pd.DataFrame._from_arrays(arrays, columns=columns, index=index, verify_integrity=False)


def _assert_categorical_obs(adata: AnnData, key: str) -> None:
    if key not in adata.obs:
        raise KeyError(f"Cluster key `{key}` not found in `adata.obs`.")

    if not is_categorical_dtype(adata.obs[key]):
        raise TypeError(f"Expected `adata.obs[{key!r}]` to be `categorical`, found `{infer_dtype(adata.obs[key])}`.")


def _assert_connectivity_key(adata: AnnData, key: str) -> None:
    if key not in adata.obsp:
        key_added = key.replace("_connectivities", "")
        raise KeyError(
            f"Spatial connectivity key `{key}` not found in `adata.obsp`. "
            f"Please run `squidpy.gr.spatial_neighbors(..., key_added={key_added!r})` first."
        )


def _assert_spatial_basis(adata: AnnData, key: str) -> None:
    if key not in adata.obsm:
        raise KeyError(f"Spatial basis `{key}` not found in `adata.obsm`.")


def _assert_non_empty_sequence(
    seq: Union[Hashable, Iterable[Hashable]], *, name: str, convert_scalar: bool = True
) -> List[Any]:
    if isinstance(seq, str) or not isinstance(seq, Iterable):
        if not convert_scalar:
            raise TypeError(f"Expected a sequence, found `{type(seq)}`.")
        seq = (seq,)

    res, _ = _unique_order_preserving(seq)
    if not len(res):
        raise ValueError(f"No {name} have been selected.")

    return res


def _get_valid_values(needle: Sequence[Any], haystack: Sequence[Any]) -> Sequence[Any]:
    res = [n for n in needle if n in haystack]
    if not len(res):
        raise ValueError(f"No valid values were found. Valid values are `{sorted(set(haystack))}`.")
    return res


def _assert_positive(value: float, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"Expected `{name}` to be positive, found `{value}`.")


def _assert_non_negative(value: float, *, name: str) -> None:
    if value < 0:
        raise ValueError(f"Expected `{name}` to be non-negative, found `{value}`.")


def _assert_in_range(value: float, minn: float, maxx: float, *, name: str) -> None:
    if not (minn <= value <= maxx):
        raise ValueError(f"Expected `{name}` to be in interval `[{minn}, {maxx}]`, found `{value}`.")


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
