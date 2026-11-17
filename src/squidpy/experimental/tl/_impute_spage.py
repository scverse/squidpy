from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from spatialdata import SpatialData
from threadpoolctl import threadpool_limits

from squidpy._docs import d
from squidpy._utils import NDArrayA, get_n_numba_threads
from squidpy._validators import assert_in_range, assert_non_empty_sequence
from squidpy.experimental.tl._utils import query_of, resolve_pair, resolve_table
from squidpy.gr._utils import _extract_expression, _save_data

__all__ = ["impute_spage"]


@d.dedent
def impute_spage(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    genes: str | Sequence[str],
    table_key: str | tuple[str | None, str | None] | None = None,
    layer: str | tuple[str | None, str | None] | None = None,
    n_pv: int = 30,
    n_neighbors: int = 50,
    cosine_threshold: float = 0.3,
    key_added: str = "spage",
    n_jobs: int | None = None,
    inplace: bool = True,
) -> pd.DataFrame | None:
    r"""
    Impute genes in spatial data from a scRNA-seq reference with SpaGE :cite:`abdelaal2020`.

    Both datasets are z-scored on their shared genes, ignoring genes that are constant in either of them,
    and reduced to ``n_pv`` principal components each. Both are then projected onto the reference-side
    principal vectors of the two PCA subspaces whose cosine similarity exceeds ``cosine_threshold``, where
    every query cell is assigned its ``n_neighbors`` nearest reference cells by cosine distance. Neighbors at
    cosine distance :math:`d_i < 1` are kept; with :math:`k` of them, each gets the weight
    :math:`(1 - d_i / \sum_j d_j) / (k - 1)`, and the imputed expression is the weighted sum of their
    reference expression. Cells without a kept neighbor are imputed as 0.

    Unlike the `reference implementation <https://github.com/tabdelaal/SpaGE>`_, which returns `NaN` when the
    formula is :math:`0 / 0`, a single kept neighbor gets weight 1 and kept neighbors that are all at distance 0
    get equal weights. At least two principal vectors must pass ``cosine_threshold``, since along a single one
    every cosine distance is 0 or 2.

    Parameters
    ----------
    data_ref, data_query
        The scRNA-seq reference and the spatial query, each an :class:`~anndata.AnnData` or a
        :class:`~spatialdata.SpatialData`. Leave ``data_query=None`` with ``data_ref`` a SpatialData
        holding both samples, distinguished by a ``table_key`` pair. Both should be normalized the same way.
    genes
        Genes of the reference to impute. They may also be measured in the query.
        The result is dense, with ``n_obs * len(genes)`` values for the query's ``n_obs``.
    table_key
        For SpatialData input, which table holds the data. A single key applies to both sides;
        a ``(ref, query)`` pair addresses each side separately (use `None` for an AnnData side).
    layer
        Layer holding the expression, or :attr:`anndata.AnnData.X` if `None`. A single value applies to
        both sides; a ``(ref, query)`` pair addresses each side separately.
    n_pv
        Number of principal components computed per dataset, and hence the maximum number of principal vectors.
        Must be at least 2, at most the number of shared genes that vary in both datasets, and smaller than
        the number of cells in either dataset.
    n_neighbors
        Number of nearest reference cells per query cell, at most the number of reference cells.
    cosine_threshold
        Principal vectors with a cosine similarity above this value, in :math:`[0, 1)`, are used.
    key_added
        Key in the query's :attr:`anndata.AnnData.obsm` where the result is stored.
    %(n_jobs_threads)s
        A positive value also caps the BLAS and OpenMP threads (via :mod:`threadpoolctl`), which `None` and
        ``-1`` leave as configured.
    inplace
        If `True`, store the result in the query, otherwise return it.

    Returns
    -------
    If ``inplace = False``, returns a :class:`pandas.DataFrame` with the query's cells as rows and ``genes``
    as columns.

    Otherwise, stores it in the query's ``obsm[key_added]`` and returns `None`.
    """
    ref_table, query_table = resolve_pair(table_key, name="table_key")
    ref_layer, query_layer = resolve_pair(layer, name="layer")
    data_query = query_of(
        data_ref, data_query, ref_address=(ref_table,), query_address=(query_table,), key_name="table_key"
    )
    ref = resolve_table(data_ref, ref_table, side="reference")
    query = resolve_table(data_query, query_table, side="query")
    for side, adata in (("reference", ref), ("query", query)):
        if not adata.var_names.is_unique:
            raise ValueError(f"The {side} has duplicate gene names; make them unique with `var_names_make_unique`.")

    n_threads = get_n_numba_threads(n_jobs)
    assert_in_range(n_neighbors, 1, ref.n_obs, name="n_neighbors")
    if not 0 <= cosine_threshold < 1:
        raise ValueError(f"Expected `cosine_threshold` to be in interval `[0, 1)`, found `{cosine_threshold}`.")

    genes = assert_non_empty_sequence(genes, name="genes")
    if missing := [g for g in genes if g not in ref.var_names]:
        raise KeyError(f"Genes not found in the reference: {missing}.")
    shared = query.var_names.intersection(ref.var_names, sort=False).tolist()
    if not shared:
        raise ValueError("The reference and the query have no genes in common.")

    start = logg.info(f"Imputing `{len(genes)}` genes with SpaGE using `{n_threads}` thread(s)")
    with threadpool_limits(None if n_jobs in (None, -1) else n_threads):
        ref_x, ref_offset, ref_varies = _scale(_extract_expression(ref, genes=shared, layer=ref_layer)[0])
        query_x, query_offset, query_varies = _scale(_extract_expression(query, genes=shared, layer=query_layer)[0])
        # a constant gene has no z-score (SpaGE's `zscore` gives NaN) and carries nothing to align on
        if not (varies := ref_varies & query_varies).all():
            logg.warning(f"Ignoring `{np.sum(~varies)}` shared genes that are constant in either dataset")
            ref_x, ref_offset = ref_x[:, varies], ref_offset[varies]
            query_x, query_offset = query_x[:, varies], query_offset[varies]
        assert_in_range(n_pv, 2, min(int(np.sum(varies)), query.n_obs - 1, ref.n_obs - 1), name="n_pv")

        # the covariance solver is deterministic and centers sparse data implicitly; it forms
        # `x.T @ x - n * mean**2`, which loses too much precision in float32
        ref_pcs, query_pcs = (
            sc.pp.pca(x.astype(np.float64, copy=False), n_comps=n_pv, svd_solver="covariance_eigh", return_info=True)[1]
            for x in (ref_x, query_x)
        )
        # principal vectors are the canonical directions between the two PCA subspaces; the singular values
        # are their cosine similarities, sorted in descending order
        u, cosine, _ = np.linalg.svd(ref_pcs @ query_pcs.T)
        n_effective = int(np.sum(cosine > cosine_threshold))
        if n_effective < 2:
            raise ValueError(
                f"Expected at least 2 principal vectors with a cosine similarity above "
                f"`cosine_threshold={cosine_threshold}`, found `{n_effective}`. The largest are `{cosine[:3].round(3)}`."
            )
        pvs = ref_pcs.T @ u[:, :n_effective]

        # z-scored data is `x - offset` (see `_scale`); casting `pvs` keeps float32 data from being copied to
        # float64, and float16 data from being projected in half precision
        ref_proj, query_proj = (
            x @ pvs.astype(np.result_type(x.dtype, np.float32)) - offset @ pvs
            for x, offset in ((ref_x, ref_offset), (query_x, query_offset))
        )
        del ref_x, query_x
        distances, indices = _cosine_neighbors(ref_proj, query_proj, n_neighbors=n_neighbors, n_jobs=n_threads)

        y = _extract_expression(ref, genes=genes, layer=ref_layer)[0]
        y = y.astype(np.result_type(y.dtype, np.float32), copy=False)
        weights = csr_matrix(
            (
                _spage_weights(distances).astype(y.dtype).ravel(),
                indices.ravel(),
                np.arange(0, query.n_obs * n_neighbors + 1, n_neighbors),
            ),
            shape=(query.n_obs, ref.n_obs),
        )
        # TODO: use `fast_array_utils.linalg.matmul` once released, which runs this product in parallel
        imputed = to_dense(weights @ y)

    res = pd.DataFrame(imputed, index=query.obs_names, columns=genes, copy=False)
    if not inplace:
        logg.info("Finish", time=start)
        return res
    _save_data(query, attr="obsm", key=key_added, data=res, time=start)
    return None


def _scale(x: NDArrayA | CSBase) -> tuple[NDArrayA | CSBase, NDArrayA, NDArrayA]:
    """Divide each gene by its standard deviation, keeping sparse input sparse.

    Return the scaled data, the scaled means ``offset`` such that ``x - offset`` is the z-scored data,
    and a mask of the non-constant genes. Constant genes are left unscaled.
    """
    scaler = StandardScaler(with_mean=False).fit(x)
    return scaler.transform(x), scaler.mean_ / scaler.scale_, scaler.var_ > 0


def _cosine_neighbors(ref: NDArrayA, query: NDArrayA, *, n_neighbors: int, n_jobs: int) -> tuple[NDArrayA, NDArrayA]:
    """Cosine distances and indices of each query row's nearest reference rows, as sklearn's `"cosine"` metric gives.

    For unit vectors the cosine distance is half the squared euclidean one, which unlike `"cosine"` takes sklearn's
    parallel search. A (near-)zero row, which :func:`~sklearn.preprocessing.normalize` leaves as it is, has cosine
    distance 1 to everything: as a reference row it becomes a unit vector orthogonal to every query row, and the
    distances of a query row are set to 1.
    """
    ref, query = normalize(ref), normalize(query)
    ref_zero = np.einsum("ij,ij->i", ref, ref) < 0.5
    query_zero = np.einsum("ij,ij->i", query, query) < 0.5
    if ref_zero.any():
        ref = np.hstack([np.where(ref_zero[:, None], 0.0, ref), ref_zero[:, None]])
        query = np.hstack([query, np.zeros((len(query), 1))])
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="sqeuclidean", n_jobs=n_jobs)
    distances, indices = knn.fit(ref).kneighbors(query)
    distances /= 2
    distances[query_zero] = 1
    return distances, indices


def _spage_weights(distances: NDArrayA) -> NDArrayA:
    """SpaGE's neighbor weights, row-wise; see `impute_spage` for the formula and its ``0 / 0`` cases."""
    keep = distances < 1
    d = np.where(keep, distances, 0.0)
    k = keep.sum(axis=1, keepdims=True)
    total = d.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where((k > 1) & (total > 0), (1 - d / total) / (k - 1), 1 / k)
    return np.where(keep, weights, 0.0)
