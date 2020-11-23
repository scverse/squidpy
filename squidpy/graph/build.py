"""Functions for building graph from spatial coordinates."""
import warnings
from typing import Tuple, Union, Optional

from anndata import AnnData

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from squidpy.constants._pkg_constants import SPATIAL_M


def spatial_connectivity(
    adata: AnnData,
    obsm: str = SPATIAL_M,
    n_rings: int = 1,
    n_neigh: int = 6,
    radius: Optional[float] = None,
    coord_type: Optional[str] = "visium",
    transform: Optional[str] = None,
    key_added: Optional[str] = None,
) -> None:
    """
    Create a graph from spatial coordinates.

    Params
    ------
    adata
        The AnnData object.
    obsm
        Key to spatial coordinates.
    key_added
        Key added to connectivity and distance matrices in :attr:`anndata.AnnData.obsp`.
    n_rings
        Number of rings of neighbors for Visium data.
    n_neigh
        Number of neighborhoods to consider for non-Visium data.
    radius
        Radius of neighbors for non-Visium data.
    coord_type
        Type of coordinate system (Visium vs. general coordinates).
    transform
        Type of adjacency matrix transform: either `spectral` or `cosine`.

    Returns
    -------
    None
        TODO.
    """
    coords = adata.obsm[obsm]

    if coord_type == "visium":
        if n_rings > 1:
            Adj = _build_connectivity(coords, 6, neigh_correct=True, set_diag=True)
            Res = Adj
            Walk = Adj
            # TODO: can't this ben done in log(n_rings - 1) with recursion?
            for i in range(n_rings - 1):
                Walk = Walk @ Adj
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SparseEfficiencyWarning)
                    Walk[Res.nonzero()] = 0.0
                Walk.eliminate_zeros()
                Walk.data[:] = float(i + 2)
                Res = Res + Walk
            Adj = Res
            Adj.setdiag(0.0)
            Adj.eliminate_zeros()

            Dst = Adj.copy()
            Adj.data[:] = 1.0
        else:
            Adj = _build_connectivity(coords, 6, neigh_correct=True)
            Dst = None

    else:
        Adj, Dst = _build_connectivity(coords, n_neigh, radius, return_distance=True)

    # check transform
    if transform == "spectral":
        Adj = _transform_a_spectral(Adj)
    elif transform == "cosine":
        Adj = _transform_a_cosine(Adj)

    if key_added is None:
        key_added = "spatial_neighbors"
        conns_key = "spatial_connectivities"
        dists_key = "spatial_distances"
    else:
        conns_key = key_added + "_connectivities"
        dists_key = key_added + "_distances"

    # add keys
    adata.uns[key_added] = {}

    neighbors_dict = adata.uns[key_added]

    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = dists_key

    neighbors_dict["params"] = {"n_neighbors": n_neigh, "coord_type": coord_type}
    neighbors_dict["params"]["radius"] = radius

    adata.obsp[conns_key] = Adj
    if Dst is not None:
        adata.obsp[dists_key] = Dst


def _build_connectivity(
    coords: np.ndarray,
    n_neigh: int,
    radius: Optional[float] = None,
    neigh_correct: bool = False,
    set_diag: bool = False,
    return_distance: bool = False,
) -> Union[Tuple[csr_matrix, csr_matrix], np.ndarray]:
    """Build connectivity matrix from spatial coordinates."""
    from sklearn.neighbors import NearestNeighbors

    N = coords.shape[0]

    dists_m = None

    tree = NearestNeighbors(n_neighbors=n_neigh or 6, radius=radius or 1, metric="euclidean")
    tree.fit(coords)

    if radius is not None:
        results = tree.radius_neighbors()
        dists = np.concatenate(results[0])
        row_indices = np.concatenate(results[1])
        lengths = [len(x) for x in results[1]]
        col_indices = np.repeat(np.arange(N), lengths)
    else:
        results = tree.kneighbors()
        dists, row_indices = (result.reshape(-1) for result in results)
        col_indices = np.repeat(np.arange(N), n_neigh or 6)
        if neigh_correct:
            dist_cutoff = np.median(dists) * 1.3  # There's a small amount of sway
            mask = dists < dist_cutoff
            row_indices, col_indices = row_indices[mask], col_indices[mask]
            dists = dists[mask]

    if return_distance:
        dists_m = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

    if set_diag:
        row_indices = np.concatenate((row_indices, np.arange(N)))
        col_indices = np.concatenate((col_indices, np.arange(N)))

    conns_m = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(N, N))

    return (conns_m, dists_m) if return_distance else conns_m


def _transform_a_spectral(a: np.ndarray) -> np.ndarray:
    degrees = np.squeeze(np.array(np.sqrt(1 / a.sum(axis=0))))
    a_out = a.multiply(np.outer(degrees, degrees))
    return a_out


def _transform_a_cosine(a: np.ndarray) -> np.ndarray:
    a_out = cosine_similarity(a, dense_output=False)
    return a_out
