from typing import Tuple, Union, Callable, Optional, Sequence
from functools import singledispatch
from typing_extensions import Literal  # < 3.8

from scanpy import logging as logg
from anndata import AnnData
from scanpy.get import _get_obs_rep

from numba import njit
from scipy.sparse import spmatrix
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, parallelize, _get_n_cores
from squidpy.gr._utils import (
    _save_data,
    _assert_spatial_basis,
    _assert_connectivity_key,
    _assert_non_empty_sequence,
)
from squidpy._constants._pkg_constants import Key

__all__ = ["sepal"]


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn())
def sepal(
    adata: AnnData,
    genes: Optional[Union[str, Sequence[str]]] = None,
    max_nbrs: Literal[4, 6] = 6,
    n_iter: Optional[int] = 10000,
    dt: float = 0.001,
    thres: float = 1e-8,
    connectivity_key: str = Key.obsp.spatial_conn(),
    spatial_key: str = Key.obsm.spatial,
    layer: Optional[str] = None,
    seed: Optional[int] = None,
    use_raw: bool = False,
    copy: bool = False,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Identify spatially variable genes with Sepal.

    Sepal is a method that simulate a diffusion process to quantify spatial structure in tissue.
    See  :cite:`andersson2021` for reference.

    Parameters
    ----------
    %(adata)s
    genes
        List of gene names, as stored in :attr:`anndata.AnnData.var_names`, used to compute global
        spatial autocorrelation statistic.

        If `None`, it's computed :attr:`anndata.AnnData.var` ``['highly_variable']``, if present. Otherwise,
        it's computed for all genes.
    max_nbrs
        Maximum number of neighbors of a node in the graph, either 4 for a square-grid or 6 for a hexagonal-grid.
    n_iter
        Maximum number of iterations for the diffusion simulation.
    dt
        Time step added in diffusion simulation.
    thres
        Entropy threshold for convergence of diffusion simulation.
    %(conn_key)s
    %(spatial_key)s
    layer
        Layer in :attr:`anndata.AnnData.layers` to use. If `None`, use :attr:`anndata.AnnData.X`.
    %(seed)s
    %(copy)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns a :class:`numpy.ndarray` with the sepal scores.

    Otherwise, adds to ``adata.obs`` the following key:

        - :attr:`anndata.AnnData.obs` ``['sepal']`` - sepal score.
    """
    _assert_connectivity_key(adata, connectivity_key)
    _assert_spatial_basis(adata, key=spatial_key)

    spatial = adata.obsm[spatial_key].astype(np.float_)

    if genes is None:
        if "highly_variable" in adata.var.columns:
            genes = adata[:, adata.var.highly_variable.values].var_names.values
        else:
            genes = adata.var_names.values
    genes = _assert_non_empty_sequence(genes, name="genes")

    n_jobs = _get_n_cores(n_jobs)

    vals = _get_obs_rep(adata[:, genes], use_raw=use_raw, layer=layer)
    vals = _resolve_vals(vals)

    g = adata.obsp[connectivity_key].copy()

    max_n = np.diff(g.indptr).max()
    if max_n != max_nbrs:
        raise ValueError(f"Found node with # neighbors == {max_n}, bu expected `max_nbrs` == {max_nbrs}.")

    # get saturated/unsaturated nodes
    sat, sat_idx, unsat, unsat_idx = _compute_idxs(g, spatial, max_nbrs, "l1")

    start = logg.info(f"Calculating sepal score for `{len(genes)}` genes using `{n_jobs}` core(s)")

    score = parallelize(
        _score_helper,
        collection=np.arange(len(genes)),
        extractor=np.hstack,
        use_ixs=False,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(
        vals=vals,
        max_n=max_n,
        n_iter=n_iter,
        sat=sat,
        sat_idx=sat_idx,
        unsat=unsat,
        unsat_idx=unsat_idx,
        dt=dt,
        thres=thres,
    )

    sepal_score = pd.DataFrame(score, index=genes, columns=["sepal_score"])
    if sepal_score["sepal_score"].isna().any():
        logg.info("Found NaN in sepal scores, consider increase `n_iter` to a higher value.")
    sepal_score.sort_values(by="sepal_score", ascending=False, inplace=True)

    if copy:
        logg.info("Finish", time=start)
        return sepal_score

    _save_data(adata, attr="uns", key="sepal_score", data=sepal_score, time=start)


def _score_helper(
    ixs: Sequence[int],
    vals: Union[spmatrix, np.ndarray],
    max_n: int,
    n_iter: int,
    sat: np.ndarray,
    sat_idx: np.ndarray,
    unsat: np.ndarray,
    unsat_idx: np.ndarray,
    dt: np.float_,
    thres: np.float_,
    queue: Optional[SigQueue] = None,
) -> np.ndarray:

    score = []
    if max_n == 4:
        fun = _laplacian_rect
    elif max_n == 6:
        fun = _laplacian_hex
    else:
        NotImplementedError("Laplacian for `max_nbrs`= {max_n} is not yet implemented.")
    for i in ixs:
        conc = vals[:, i].copy()
        time_iter = _diffusion(
            conc, fun, n_iter, sat, sat_idx, sat.shape[0], unsat, unsat_idx, conc.shape[0], dt=dt, thres=thres
        )
        if time_iter.shape[0] != 0:
            score.append(0.001 * time_iter[0])
        else:
            score.append(np.nan)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return np.array(score)


@njit(parallel=False, fastmath=True)
def _diffusion(
    conc: np.ndarray,
    laplacian: Callable[..., np.float_],
    n_iter: int,
    sat: np.ndarray,
    sat_idx: np.ndarray,
    sat_shape: int,
    unsat: np.ndarray,
    unsat_idx: np.ndarray,
    conc_shape: int,
    dt: float = 0.001,
    D: float = 1.0,
    thres: float = 1e-8,
) -> np.ndarray:
    """Simulate diffusion process on a regular graph."""
    entropy_arr = np.zeros(n_iter)
    ent = np.ones(n_iter + 1)

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
        ent[i + 1] = _entropy(conc[sat]) / sat_shape
        entropy_arr[i] = np.abs(ent[i + 1] - ent[i])  # estimate entropy difference
        if entropy_arr[i] <= thres:
            break

    return np.nonzero(entropy_arr <= thres)[0]


# taken from https://github.com/almaan/sepal/blob/master/sepal/models.py
@njit(parallel=False, fastmath=True)
def _laplacian_rect(
    centers: np.ndarray,
    nbrs: np.ndarray,
    h: np.ndarray,
) -> np.float_:
    """Laplacian approx rectilinear grid."""
    d2f = nbrs - 4 * centers
    d2f = d2f / h ** 2

    return d2f


# taken from https://github.com/almaan/sepal/blob/master/sepal/models.py
@njit(parallel=False, fastmath=True)
def _laplacian_hex(
    centers: np.ndarray,
    nbrs: np.ndarray,
    h: np.ndarray,
) -> np.float_:
    """Laplacian approx hexagonal grid."""
    d2f = nbrs - 6 * centers
    d2f = d2f / h ** 2
    d2f = (d2f * 2) / 3

    return d2f


# taken from https://github.com/almaan/sepal/blob/master/sepal/models.py
@njit(parallel=False, fastmath=True)
def _entropy(
    xx: np.ndarray,
) -> np.float_:
    """Entropy of array."""
    xnz = xx[xx > 0]
    xs = np.sum(xnz)
    xn = xnz / xs
    xl = np.log(xn)
    return (-xl * xn).sum()


def _compute_idxs(
    g: spmatrix, spatial: np.ndarray, sat_thres: int, metric: str = "l1"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get saturated and unsaturated nodes and nhood indexes."""
    sat, unsat = _get_sat_unsat_idx(g.indptr, g.shape[0], sat_thres)

    sat_idx, nearest_sat, un_unsat = _get_nhood_idx(sat, unsat, g.indptr, g.indices, sat_thres)

    # compute dist btwn remaining unsat and all sat
    dist = pairwise_distances(spatial[un_unsat], spatial[sat], metric=metric)
    # assign closest sat to remaining nearest_sat
    nearest_sat[np.isnan(nearest_sat)] = sat[np.argmin(dist, axis=1)]

    return sat, sat_idx, unsat, nearest_sat.astype(np.int32)


@njit(parallel=False)
def _get_sat_unsat_idx(g_indptr: np.ndarray, g_shape: int, sat_thres: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get saturated and unsaturated nodes based on thres."""
    n_indices = np.diff(g_indptr)
    unsat = np.arange(g_shape)[n_indices < sat_thres]
    sat = np.arange(g_shape)[n_indices == sat_thres]

    return sat, unsat


@njit(parallel=False)
def _get_nhood_idx(
    sat: np.ndarray, unsat: np.ndarray, g_indptr: np.ndarray, g_indices: np.ndarray, sat_thres: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get saturated and unsaturated nhood indexes."""
    # get saturated nhood indices
    sat_idx = np.zeros((sat.shape[0], sat_thres))
    for idx in np.arange(sat.shape[0]):
        i = sat[idx]
        s = slice(g_indptr[i], g_indptr[i + 1])
        sat_idx[idx] = g_indices[s]

    # get closest saturated of unsaturated
    nearest_sat = np.zeros(unsat.shape) * np.nan
    for idx in np.arange(unsat.shape[0]):
        i = unsat[idx]
        s = slice(g_indptr[i], g_indptr[i + 1])
        unsat_neigh = g_indices[s]
        for u in unsat_neigh:
            if u in sat:  # take the first saturated nhood
                nearest_sat[idx] = u
                break

    # some unsat still don't have a sat nhood
    # return them and compute distances in outer func
    un_unsat = unsat[np.isnan(nearest_sat)]

    return sat_idx.astype(np.int32), nearest_sat, un_unsat


# interface, taken from: https://github.com/theislab/scanpy/blob/master/scanpy/metrics/_gearys_c.py
@singledispatch
def _resolve_vals(val: np.ndarray) -> np.ndarray:
    return val


@_resolve_vals.register(spmatrix)
def _(val: spmatrix) -> np.ndarray:
    return val.toarray()  # type: ignore[no-any-return]
