"""Functions for building graph from spatial coordinates."""
import warnings
from typing import Optional

from typing import Optional, Union
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def spatial_connectivity(
    adata: "AnnData",  # noqa
    obsm: str = "spatial",
    n_rings: int = 1,
    n_neigh: int = 6,
    radius: Optional[float] = None,
    coord_type: Union[str, None] = "visium",
    weighted_graph: bool = False,
    transform: str = None,
    key_added: str = None,
):
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
        Key added to connectivity matrix in obsp.
    n_rings
        Number of rings of neighbors for Visium data
    n_neigh
        Number of neighborhoods to consider for non-Visium data
    radius
        Radius of neighbors for non-Visium data
    coord_type
        Type of coordinate system (Visium vs. general coordinates)
    weighted_graph
        Output weighted connectivities
    transform
        Type of adjacency matrix transform: either `spectral` or `cosine`

    Returns
    -------
    None
    """
    coords = adata.obsm[obsm]

    if coord_type == "visium":
        if n_rings > 1:
            Adj = _build_connectivity(coords, 6, neigh_correct=True, set_diag=True)
            # get up to n_rings order connections
            if weighted_graph:
                Res = Adj
                Walk = Adj
                for i in range(n_rings - 1):
                    Walk = Walk @ Adj
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                        Walk[Res.nonzero()] = 0.0
                    Walk.eliminate_zeros()
                    Walk.data[:] = float(i + 2)
                    Res = Res + Walk
                Adj = Res
                Adj.setdiag(0.0)
                Adj.eliminate_zeros()
            else:
                Adj += Adj ** n_rings
                Adj.setdiag(0.0)
                Adj.eliminate_zeros()
                Adj.data[:] = 1.0
        else:
            Adj = _build_connectivity(coords, 6, neigh_correct=True)

    else:
        Adj = _build_connectivity(coords, n_neigh, radius)

    # check transform
    if transform == "spectral":
        Adj = _transform_a_spectral(Adj)
    elif transform == "cosine":
        Adj = _transform_a_cosine(Adj)

    if key_added is None:
        key_added = "spatial_neighbors"
        conns_key = "spatial_connectivities"
        dists_key = "distances"
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
    # distances not yet added
    # adata.obsp[dists_key] = None


def _build_connectivity(
    coords: np.ndarray,
    n_neigh: int,
    radius: Optional[float] = None,
    neigh_correct: bool = False,
    set_diag: bool = False,
):
    """Build connectivity matrix from spatial coordinates."""
    from sklearn.neighbors import NearestNeighbors

    N = coords.shape[0]

    tree = NearestNeighbors(n_neighbors=n_neigh or 6, radius=radius or 1, metric="euclidean")
    tree.fit(coords)

    if radius is not None:
        results = tree.radius_neighbors()
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

    if set_diag:
        row_indices = np.concatenate((row_indices, np.arange(N)))
        col_indices = np.concatenate((col_indices, np.arange(N)))

    return sparse.csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(N, N))


def _transform_a_spectral(a):
    degrees = np.squeeze(np.array(np.sqrt(1 / a.sum(axis=0))))
    a_out = a.multiply(np.outer(degrees, degrees))
    return a_out


def _transform_a_cosine(a):
    a_out = cosine_similarity(a, dense_output=False)
    return a_out
