from typing import Tuple

from numba import njit
from scipy.sparse import spmatrix
from sklearn.metrics import pairwise_distances
import numpy as np


@njit(parallel=False)
def _diffusion(
    conc: np.ndarray,
    n_iter: int,
    sat: np.ndarray,
    sat_idx: np.ndarray,
    sat_shape: int,
    unsat: np.ndarray,
    unsat_idx: np.ndarray,
    conc_shape: int,
    dt: float = 0.001,
    D: float = 1.0,
    thrs: float = 1e-8,
) -> np.ndarray:
    """Simulate diffusion process on a regular graph."""
    entropy_arr = np.zeros(n_iter)
    ent = np.ones(n_iter + 1)

    nhood = np.zeros(sat_shape)
    weights = np.ones(sat_shape)

    for i in range(n_iter):
        for j in range(sat_shape):
            nhood[j] = np.sum(conc[sat_idx[j]])
        d2 = _laplacian_rect(conc[sat], nhood, weights)

        dcdt = np.zeros(conc_shape)
        dcdt[sat] = D * d2
        conc[sat] += dcdt[sat] * dt
        conc[unsat] += dcdt[unsat_idx] * dt
        # set values below zero to 0
        conc[conc < 0] = 0
        # compute entropy
        ent[i + 1] = _entropy(conc[sat]) / sat_shape
        entropy_arr[i] = np.abs(ent[i + 1] - ent[i])  # estimate entropy difference

    return np.argwhere(entropy_arr <= thrs).flatten()


# taken from https://github.com/almaan/sepal/blob/master/sepal/models.py
@njit(parallel=False)
def _laplacian_rect(
    centers: np.ndarray,
    nbrs: np.ndarray,
    h: np.float_,
) -> np.float_:
    """Laplacian approx rectilinear grid."""
    d2f = nbrs - 4 * centers
    d2f = d2f / h ** 2

    return d2f


# taken from https://github.com/almaan/sepal/blob/master/sepal/models.py
@njit(parallel=False)
def laplacian_hex(
    centers: np.ndarray,
    nbrs: np.ndarray,
    h: np.float_,
) -> np.float_:
    """Laplacian approx hexagonal grid."""
    d2f = nbrs - 6 * centers
    d2f = d2f / h ** 2 * 2 / 3

    return d2f


# taken from https://github.com/almaan/sepal
@njit(parallel=False)
def _entropy(
    xx: np.ndarray,
) -> np.float_:
    """Entropy of array."""
    xnz = xx[xx > 0]
    xs = np.sum(xnz)
    xn = xnz / xs
    xl = np.log(xn)
    return (-xl * xn).sum()


def compute_idxs(
    g: spmatrix, spatial: np.ndarray, sat_thres: int, metric: str = "l1"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get saturated and unsaturated nodes and nhood indexes."""
    sat, unsat = _get_sat_unsat_idx(g.indptr, g.indices, g.shape[0], sat_thres)

    sat_idx, nearest_sat, un_unsat = _get_nhood_idx(
        sat,
        unsat,
        g.indptr,
        g.indices,
    )

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
    sat: np.ndarray, unsat: np.ndarray, g_indptr: np.ndarray, g_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get saturated and unsaturated nhood indexes."""
    # get saturated nhood indices
    sat_idx = np.zeros((sat.shape[0], 4))
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
