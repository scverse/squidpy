"""Functions for building gr from spatial coordinates."""
import warnings
from typing import Tuple, Union, Optional

from anndata import AnnData

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from squidpy._docs import d, inject_docs
from squidpy.constants._constants import CoordType, Transform
from squidpy.constants._pkg_constants import Key


@d.dedent
@inject_docs(t=Transform, c=CoordType)
def spatial_connectivity(
    adata: AnnData,
    obsm: str = Key.obsm.spatial,
    n_rings: int = 1,
    n_neigh: int = 6,
    radius: Optional[float] = None,
    coord_type: Optional[str] = CoordType.VISIUM.s,
    transform: Optional[str] = None,
    key_added: Optional[str] = None,
) -> None:
    """
    Create a gr from spatial coordinates.

    Parameters
    ----------
    %(adata)s
    obsm
        Key in :attr:`anndata.AnnData.obsm` to spatial coordinates.
    key_added
        Key added to connectivity and distance matrices in :attr:`anndata.AnnData.obsp`.
    n_rings
        Number of rings of neighbors for [Visium]_ data.
    n_neigh
        Number of neighborhoods to consider for non-Visium data.
    radius
        Radius of neighbors for non-Visium data.
    coord_type
        Type of coordinate system. Can be one of the following:

            - `{c.VISIUM.s!r}`: [Visium]_ coordinates.
            - `{c.NONE.v}`: generic coordinates.
    transform
        Type of adjacency matrix transform. Can be one of the following:

            - `{t.SPECTRAL.s!r}`: TODO.
            - `{t.COSINE.s!r}`: TODO.
            - `{t.NONE.v}`: TODO.

    Returns
    -------
    TODO
    """
    transform = Transform(transform)
    coord_type = CoordType(coord_type)

    coords = adata.obsm[obsm]
    if coord_type == CoordType.VISIUM:
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

    elif coord_type == CoordType.NONE:
        Adj, Dst = _build_connectivity(coords, n_neigh, radius, return_distance=True)
    else:
        raise NotImplementedError(coord_type)

    # check transform
    if transform == Transform.SPECTRAL:
        Adj = _transform_a_spectral(Adj)
    elif transform == Transform.COSINE:
        Adj = _transform_a_cosine(Adj)
    elif transform == Transform.NONE:
        pass
    else:
        raise NotImplementedError(transform)

    key_added = Key.uns.spatial_neighs(key_added)
    conns_key = Key.obsp.spatial_conn(key_added)
    dists_key = Key.obsp.spatial_dist(key_added)

    # add keys
    neighbors_dict = adata.uns[key_added] = {}

    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = dists_key

    neighbors_dict["params"] = {"n_neighbors": n_neigh, "coord_type": coord_type.v}
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
    return a.multiply(np.outer(degrees, degrees))


def _transform_a_cosine(a: np.ndarray) -> np.ndarray:
    return cosine_similarity(a, dense_output=False)
