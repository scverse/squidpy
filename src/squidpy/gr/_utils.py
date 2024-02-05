"""Graph utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Iterable,
    Sequence,
    Union,  # noqa: F401
)

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._core.views import ArrayView, SparseCSCView, SparseCSRView
from anndata.utils import make_index_unique
from pandas import CategoricalDtype
from pandas.api.types import infer_dtype, is_categorical_dtype
from scanpy import logging as logg
from scipy.sparse import csc_matrix, csr_matrix, issparse, spmatrix

from squidpy._docs import d
from squidpy._utils import NDArrayA, _unique_order_preserving


def _check_tuple_needles(
    needles: Sequence[tuple[Any, Any]],
    haystack: Sequence[Any],
    msg: str,
    reraise: bool = True,
) -> Sequence[tuple[Any, Any]]:
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
                raise ValueError(msg.format(b))
            else:
                continue

        filtered.append((a, b))

    return filtered


# modified from pandas' source code
def _create_sparse_df(
    data: NDArrayA | spmatrix,
    index: pd.Index | None = None,
    columns: Sequence[Any] | None = None,
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
        pred = (lambda col: ~np.isnan(col)) if fill_value is np.nan else (lambda col: ~np.isclose(col, fill_value))
        dtype = SparseDtype(data.dtype, fill_value=fill_value)
        n_rows, n_cols = data.shape
        arrays = []

        for i in range(n_cols):
            mask = pred(data[:, i])
            idx = IntIndex(n_rows, np.where(mask)[0], check_integrity=False)
            arr = SparseArray._simple_new(data[mask, i], idx, dtype)
            arrays.append(arr)

        return pd.DataFrame._from_arrays(arrays, columns=columns, index=index, verify_integrity=False)

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

    if not isinstance(adata.obs[key].dtype, CategoricalDtype):
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
    seq: Hashable | Iterable[Hashable], *, name: str, convert_scalar: bool = True
) -> list[Any]:
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


def _save_data(adata: AnnData, *, attr: str, key: str, data: Any, prefix: bool = True, time: Any | None = None) -> None:
    obj = getattr(adata, attr)
    obj[key] = data

    if prefix:
        logg.info(f"Adding `adata.{attr}[{key!r}]`")
    else:
        logg.info(f"       `adata.{attr}[{key!r}]`")
    if time is not None:
        logg.info("Finish", time=time)


def _extract_expression(
    adata: AnnData, genes: Sequence[str] | None = None, use_raw: bool = False, layer: str | None = None
) -> tuple[NDArrayA | spmatrix, Sequence[str]]:
    if use_raw and adata.raw is None:
        logg.warning("AnnData object has no attribute `raw`. Setting `use_raw=False`")
        use_raw = False

    if genes is None and "highly_variable" in adata.var:
        # should we use `adata.raw.var["highly_variable"]` if `use_raw=True`?
        genes = adata.var_names.values[adata.var["highly_variable"].values]

    if use_raw:
        genes = list(set(adata.raw.var_names) & set(genes))  # type: ignore[arg-type]
        genes = _assert_non_empty_sequence(genes, name="genes")
        res = adata.raw[:, genes].X
    else:
        genes = _assert_non_empty_sequence(genes, name="genes")

        if layer is None:
            res = adata[:, genes].X
        elif layer not in adata.layers:
            raise KeyError(f"Layer `{layer}` not found in `adata.layers`.")
        else:
            res = adata[:, genes].layers[layer]
            if isinstance(res, AnnData):
                res = res.X
            elif not isinstance(res, (np.ndarray, spmatrix)):
                raise TypeError(f"Invalid expression type `{type(res).__name__}`.")

    # handle views
    if isinstance(res, ArrayView):
        return np.asarray(res), genes
    if isinstance(res, (SparseCSRView, SparseCSCView)):
        mro = type(res).mro()
        if csr_matrix in mro:
            return csr_matrix(res), genes
        if csc_matrix in mro:
            return csc_matrix(res), genes
        raise TypeError(f"Unable to handle type `{type(res)}`.")

    return res, genes


@contextmanager
@d.dedent
def _genesymbols(
    adata: AnnData,
    *,
    key: str | None = None,
    use_raw: bool = False,
    make_unique: bool = False,
) -> AnnData:
    """
    Set gene names from a column in :attr:`anndata.AnnData.var`.

    Parameters
    ----------
    %(adata)s
    key
        Key in :attr:`anndata.AnnData.var` where the gene symbols are stored. If `None`, this operation is a no-op.
    use_raw
        Whether to change the gene names in :attr:`anndata.AnnData.raw`.
    make_unique
        Whether to make the newly assigned gene names unique.
    Yields
    ------
    The same ``adata`` with modified :attr:`anndata.AnnData.var_names`, depending on ``use_raw``.
    """

    def key_present() -> bool:
        if use_raw:
            if adata.raw is None:
                raise AttributeError("No `.raw` attribute found. Try specifying `use_raw=False`.")
            return key in adata.raw.var
        return key in adata.var

    if key is None:
        yield adata
    elif not key_present():
        raise KeyError(f"Unable to find gene symbols in `adata.{'raw.' if use_raw else ''}var[{key!r}]`.")
    else:
        adata_orig = adata
        if use_raw:
            adata = adata.raw

        var_names = adata.var_names.copy()
        try:
            # TODO(michalk8): doesn't update varm (niche)
            adata.var.index = make_index_unique(adata.var[key]) if make_unique else adata.var[key]
            yield adata_orig
        finally:
            # in principle we assume the callee doesn't change the index
            # otherwise, would need to check whether it has been changed and add an option to determine what to do
            adata.var.index = var_names


def _shuffle_group(
    cluster_annotation: NDArrayA,
    libraries: pd.Series[CategoricalDtype],
    rs: np.random.RandomState,
) -> NDArrayA:
    """
    Shuffle values in ``arr`` for each category in ``categories``.

    Useful when the shuffling of categories is used in permutation tests where the order of values in ``arr`` matters
    (e.g. you only want to shuffle cluster annotations for the same slide/library_key, and not across slides)

    Parameters
    ----------
    cluster_annotation
        Array to shuffle.
    libraries
        Categories (e.g. libraries) to subset for shuffling.

    Returns
    -------
    Shuffled annotations.
    """
    cluster_annotation_output = np.empty(libraries.shape, dtype=cluster_annotation.dtype)
    for c in libraries.cat.categories:
        idx = np.where(libraries == c)[0]
        arr_group = cluster_annotation[idx].copy()
        rs.shuffle(arr_group)  # it's done in place hence copy before
        cluster_annotation_output[idx] = arr_group
    return cluster_annotation_output
