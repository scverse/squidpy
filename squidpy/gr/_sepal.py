from __future__ import annotations

from typing import (
    Callable,
    Literal,  # < 3.8
    Sequence,
    Union,  # noqa: F401
)

import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit
from scanpy import logging as logg
from scipy.sparse import csr_matrix, issparse, isspmatrix_csr, spmatrix
from sklearn.metrics import pairwise_distances
from spatialdata import SpatialData

from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, Signal, SigQueue, _get_n_cores, parallelize
from squidpy.gr._utils import (
    _assert_connectivity_key,
    _assert_non_empty_sequence,
    _assert_spatial_basis,
    _extract_expression,
    _save_data,
)

__all__ = ["sepal"]


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn())
def sepal(
    adata: AnnData | SpatialData,
    max_neighs: Literal[4, 6],
    genes: str | Sequence[str] | None = None,
    n_iter: int | None = 30000,
    dt: float = 0.001,
    thresh: float = 1e-8,
    connectivity_key: str = Key.obsp.spatial_conn(),
    spatial_key: str = Key.obsm.spatial,
    layer: str | None = None,
    use_raw: bool = False,
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> pd.DataFrame | None:
    """
    Identify spatially variable genes with *Sepal*.

    *Sepal* is a method that simulates a diffusion process to quantify spatial structure in tissue.
    See :cite:`andersson2021` for reference.

    Parameters
    ----------
    %(adata)s
    max_neighs
        Maximum number of neighbors of a node in the graph. Valid options are:

            - `4` - for a square-grid (ST, Dbit-seq).
            - `6` - for a hexagonal-grid (Visium).
    genes
        List of gene names, as stored in :attr:`anndata.AnnData.var_names`, used to compute sepal score.

        If `None`, it's computed :attr:`anndata.AnnData.var` ``['highly_variable']``, if present.
        Otherwise, it's computed for all genes.
    n_iter
        Maximum number of iterations for the diffusion simulation.
        If ``n_iter`` iterations are reached, the simulation will terminate
        even though convergence has not been achieved.
    dt
        Time step in diffusion simulation.
    thresh
        Entropy threshold for convergence of diffusion simulation.
    %(conn_key)s
    %(spatial_key)s
    layer
        Layer in :attr:`anndata.AnnData.layers` to use. If `None`, use :attr:`anndata.AnnData.X`.
    use_raw
        Whether to access :attr:`anndata.AnnData.raw`.
    %(copy)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the sepal scores.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['sepal_score']`` - the sepal scores.

    Notes
    -----
    If some genes in :attr:`anndata.AnnData.uns` ``['sepal_score']`` are `NaN`,
    consider re-running the function with increased ``n_iter``.
    """
    if isinstance(adata, SpatialData):
        adata = adata.table
    _assert_connectivity_key(adata, connectivity_key)
    _assert_spatial_basis(adata, key=spatial_key)
    if max_neighs not in (4, 6):
        raise ValueError(f"Expected `max_neighs` to be either `4` or `6`, found `{max_neighs}`.")

    spatial = adata.obsm[spatial_key].astype(np.float_)

    if genes is None:
        genes = adata.var_names.values
        if "highly_variable" in adata.var.columns:
            genes = genes[adata.var["highly_variable"].values]
    genes = _assert_non_empty_sequence(genes, name="genes")

    n_jobs = _get_n_cores(n_jobs)

    g = adata.obsp[connectivity_key]
    if not isspmatrix_csr(g):
        g = csr_matrix(g)
    g.eliminate_zeros()

    max_n = np.diff(g.indptr).max()
    if max_n != max_neighs:
        raise ValueError(f"Expected `max_neighs={max_neighs}`, found node with `{max_n}` neighbors.")

    # get saturated/unsaturated nodes
    sat, sat_idx, unsat, unsat_idx = _compute_idxs(g, spatial, max_neighs, "l1")

    # get counts
    vals, genes = _extract_expression(adata, genes=genes, use_raw=use_raw, layer=layer)
    start = logg.info(f"Calculating sepal score for `{len(genes)}` genes using `{n_jobs}` core(s)")

    score = parallelize(
        _score_helper,
        collection=np.arange(len(genes)).tolist(),
        extractor=np.hstack,
        use_ixs=False,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(
        vals=vals,
        max_neighs=max_neighs,
        n_iter=n_iter,
        sat=sat,
        sat_idx=sat_idx,
        unsat=unsat,
        unsat_idx=unsat_idx,
        dt=dt,
        thresh=thresh,
    )

    key_added = "sepal_score"
    sepal_score = pd.DataFrame(score, index=genes, columns=[key_added])

    if sepal_score[key_added].isna().any():
        logg.warning("Found `NaN` in sepal scores, consider increasing `n_iter` to a higher value")
    sepal_score.sort_values(by=key_added, ascending=False, inplace=True)

    if copy:
        logg.info("Finish", time=start)
        return sepal_score

    _save_data(adata, attr="uns", key=key_added, data=sepal_score, time=start)


def _score_helper(
    ixs: Sequence[int],
    vals: spmatrix | NDArrayA,
    max_neighs: int,
    n_iter: int,
    sat: NDArrayA,
    sat_idx: NDArrayA,
    unsat: NDArrayA,
    unsat_idx: NDArrayA,
    dt: np.float_,
    thresh: np.float_,
    queue: SigQueue | None = None,
) -> NDArrayA:
    if max_neighs == 4:
        fun = _laplacian_rect
    elif max_neighs == 6:
        fun = _laplacian_hex
    else:
        raise NotImplementedError(f"Laplacian for `{max_neighs}` neighbors is not yet implemented.")

    score, sparse = [], issparse(vals)
    for i in ixs:
        conc = vals[:, i].A.flatten() if sparse else vals[:, i].copy()  # type: ignore[union-attr]
        conc = vals[:, i].A.flatten() if sparse else vals[:, i].copy()  # type: ignore[union-attr]
        time_iter = _diffusion(conc, fun, n_iter, sat, sat_idx, unsat, unsat_idx, dt=dt, thresh=thresh)
        score.append(dt * time_iter)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return np.array(score)


@njit(fastmath=True)
def _diffusion(
    conc: NDArrayA,
    laplacian: Callable[[NDArrayA, NDArrayA, NDArrayA], np.float_],
    n_iter: int,
    sat: NDArrayA,
    sat_idx: NDArrayA,
    unsat: NDArrayA,
    unsat_idx: NDArrayA,
    dt: float = 0.001,
    D: float = 1.0,
    thresh: float = 1e-8,
) -> float:
    """Simulate diffusion process on a regular graph."""
    sat_shape, conc_shape = sat.shape[0], conc.shape[0]
    entropy_arr = np.zeros(n_iter)
    prev_ent = 1.0
    nhood = np.zeros(sat_shape)
    weights = np.ones(sat_shape)

    for i in range(n_iter):
        for j in range(sat_shape):
            nhood[j] = np.sum(conc[sat_idx[j]])
        d2 = laplacian(conc[sat], nhood, weights)

        dcdt = np.zeros(conc_shape)
        dcdt[sat] = D * d2
        conc[sat] += dcdt[sat] * dt
        conc[unsat] += dcdt[unsat_idx] * dt
        # set values below zero to 0
        conc[conc < 0] = 0
        # compute entropy
        ent = _entropy(conc[sat]) / sat_shape
        entropy_arr[i] = np.abs(ent - prev_ent)  # estimate entropy difference
        prev_ent = ent
        if entropy_arr[i] <= thresh:
            break

    tmp = np.nonzero(entropy_arr <= thresh)[0]
    return float(tmp[0] if len(tmp) else np.nan)


# taken from https://github.com/almaan/sepal/blob/master/sepal/models.py
@njit(parallel=False, fastmath=True)
def _laplacian_rect(
    centers: NDArrayA,
    nbrs: NDArrayA,
    h: float,
) -> NDArrayA:
    """
    Five point stencil approximation on rectilinear grid.

    See `Wikipedia <https://en.wikipedia.org/wiki/Five-point_stencil>`_ for more information.
    """
    d2f: NDArrayA = nbrs - 4 * centers
    d2f = d2f / h**2

    return d2f


# taken from https://github.com/almaan/sepal/blob/master/sepal/models.py
@njit(fastmath=True)
def _laplacian_hex(
    centers: NDArrayA,
    nbrs: NDArrayA,
    h: float,
) -> NDArrayA:
    """
    Seven point stencil approximation on hexagonal grid.

    References
    ----------
    Approximate Methods of Higher Analysis,
    Curtis D. Benster, L.V. Kantorovich, V.I. Krylov,
    ISBN-13: 978-0486821603.
    """
    d2f: NDArrayA = nbrs - 6 * centers
    d2f = d2f / h**2
    d2f = (d2f * 2) / 3

    return d2f


# taken from https://github.com/almaan/sepal/blob/master/sepal/models.py
@njit(fastmath=True)
def _entropy(
    xx: NDArrayA,
) -> float:
    """Get entropy of an array."""
    xnz = xx[xx > 0]
    xs = np.sum(xnz)
    xn = xnz / xs
    xl = np.log(xn)
    return float((-xl * xn).sum())


def _compute_idxs(
    g: spmatrix, spatial: NDArrayA, sat_thresh: int, metric: str = "l1"
) -> tuple[NDArrayA, NDArrayA, NDArrayA, NDArrayA]:
    """Get saturated and unsaturated nodes and neighborhood indices."""
    sat, unsat = _get_sat_unsat_idx(g.indptr, g.shape[0], sat_thresh)

    sat_idx, nearest_sat, un_unsat = _get_nhood_idx(sat, unsat, g.indptr, g.indices, sat_thresh)

    # compute dist btwn remaining unsat and all sat
    dist = pairwise_distances(spatial[un_unsat], spatial[sat], metric=metric)
    # assign closest sat to remaining nearest_sat
    nearest_sat[np.isnan(nearest_sat)] = sat[np.argmin(dist, axis=1)]

    return sat, sat_idx, unsat, nearest_sat.astype(np.int32)


@njit
def _get_sat_unsat_idx(g_indptr: NDArrayA, g_shape: int, sat_thresh: int) -> tuple[NDArrayA, NDArrayA]:
    """Get saturated and unsaturated nodes based on thresh."""
    n_indices = np.diff(g_indptr)
    unsat = np.arange(g_shape)[n_indices < sat_thresh]
    sat = np.arange(g_shape)[n_indices == sat_thresh]

    return sat, unsat


@njit
def _get_nhood_idx(
    sat: NDArrayA, unsat: NDArrayA, g_indptr: NDArrayA, g_indices: NDArrayA, sat_thresh: int
) -> tuple[NDArrayA, NDArrayA, NDArrayA]:
    """Get saturated and unsaturated neighborhood indices."""
    # get saturated nhood indices
    sat_idx = np.zeros((sat.shape[0], sat_thresh))
    for idx in range(sat.shape[0]):
        i = sat[idx]
        sat_idx[idx] = g_indices[g_indptr[i] : g_indptr[i + 1]]

    # get closest saturated of unsaturated
    nearest_sat = np.full_like(unsat, fill_value=np.nan, dtype=np.float64)
    for idx in range(unsat.shape[0]):
        i = unsat[idx]
        unsat_neigh = g_indices[g_indptr[i] : g_indptr[i + 1]]
        for u in unsat_neigh:
            if u in sat:  # take the first saturated nhood
                nearest_sat[idx] = u
                break

    # some unsat still don't have a sat nhood
    # return them and compute distances in outer func
    un_unsat = unsat[np.isnan(nearest_sat)]

    return sat_idx.astype(np.int32), nearest_sat, un_unsat
