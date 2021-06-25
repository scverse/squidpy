"""Functions for building gr from spatial coordinates."""
from typing import Tuple, Union, Optional
from itertools import chain
import warnings

from scanpy import logging as logg
from anndata import AnnData

from numba import njit
from scipy.sparse import spmatrix, csr_matrix, isspmatrix_csr, SparseEfficiencyWarning
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

from squidpy._docs import d, inject_docs
from squidpy.gr._utils import _save_data, _assert_positive, _assert_spatial_basis
from squidpy._constants._constants import CoordType, Transform
from squidpy._constants._pkg_constants import Key

__all__ = ["spatial_neighbors"]


@d.dedent
@inject_docs(t=Transform, c=CoordType)
def spatial_neighbors(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    coord_type: Optional[Union[str, CoordType]] = None,
    n_rings: int = 1,
    neigh_grid: int = 6,
    n_neigh: int = 6,
    delaunay: bool = False,
    radius: Optional[float] = None,
    transform: Optional[Union[str, Transform]] = None,
    set_diag: bool = False,
    key_added: Optional[str] = None,
) -> None:
    """
    Create a graph from spatial coordinates.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    coord_type
        Type of coordinate system. Valid options are:

            - `{c.GRID!r}` - grid coordinates.
            - `{c.GENERIC!r}` - generic coordinates.

        If `None`, use `{c.GRID!r}` if ``spatial_key`` is present in :attr:`anndata.AnnData.uns`,
        with `neigh_grid=6` (Visium), otherwise use `{c.GENERIC!r}`.
    n_rings
        Number of rings of neighbors for grid data. The argument is used if `coord_type={c.GRID!r}`.
    neigh_grid
        Number of neighboring tiles in a grid. The argument is used if `coord_type={c.GRID!r}`.
    n_neigh
        Number of neighborhoods to consider for non-grid data. The argument is used if `coord_type={c.GENERIC!r}`.
    delaunay
        Whether to compute the graph from Delaunay triangulation. The argument is used if `coord_type={c.GENERIC!r}`.
    radius
        Radius of neighbors for non-grid data. The argument is used if `coord_type={c.GENERIC!r}`.
    transform
        Type of adjacency matrix transform. Valid options are:

            - `{t.SPECTRAL.s!r}` - spectral transformation of the adjacency matrix.
            - `{t.COSINE.s!r}` - cosine transformation of the adjacency matrix.
            - `{t.NONE.v}` - no transformation of the adjacency matrix.
    set_diag
        Whether to set the diagonal to 1.0.
    key_added
        Key which controls where the results are saved.

    Returns
    -------
    Modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_connectivities']`` - spatial connectivity matrix.
        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_distances']`` - spatial distances matrix.
        - :attr:`anndata.AnnData.uns`  ``['{{key_added}}']`` - spatial neighbors dictionary.
    """
    _assert_positive(n_rings, name="n_rings")
    _assert_positive(n_neigh, name="n_neigh")
    _assert_positive(neigh_grid, name="neigh_grid")
    _assert_spatial_basis(adata, spatial_key)

    transform = Transform.NONE if transform is None else Transform(transform)
    if coord_type is None:
        coord_type = CoordType.GRID if Key.uns.spatial in adata.uns else CoordType.GENERIC
    else:
        coord_type = CoordType(coord_type)

    start = logg.info(f"Creating graph using `{coord_type}` coordinates and `{transform}` transform")

    coords = adata.obsm[spatial_key]
    if coord_type == CoordType.GRID:
        if n_rings > 1:
            Adj: csr_matrix = _build_connectivity(
                coords,
                n_neighbors=neigh_grid,
                neigh_correct=True,
                set_diag=True,
                delaunay=delaunay,
                return_distance=False,
            )
            Res = Adj
            Walk = Adj
            for i in range(n_rings - 1):
                Walk = Walk @ Adj
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SparseEfficiencyWarning)
                    Walk[Res.nonzero()] = 0.0
                Walk.eliminate_zeros()
                Walk.data[:] = i + 2.0
                Res = Res + Walk
            Adj = Res
            Adj.setdiag(float(set_diag))
            Adj.eliminate_zeros()

            Dst = Adj.copy()
            Adj.data[:] = 1.0
        else:
            Adj = _build_connectivity(
                coords, n_neighbors=neigh_grid, neigh_correct=True, delaunay=delaunay, set_diag=set_diag
            )
            Dst = Adj.copy()
    elif coord_type == CoordType.GENERIC:
        Adj, Dst = _build_connectivity(
            coords, n_neighbors=n_neigh, radius=radius, delaunay=delaunay, return_distance=True, set_diag=set_diag
        )
    else:
        raise NotImplementedError(coord_type)

    Dst.setdiag(0.0)
    Dst.eliminate_zeros()

    # check transform
    if transform == Transform.SPECTRAL:
        Adj = _transform_a_spectral(Adj)
    elif transform == Transform.COSINE:
        Adj = _transform_a_cosine(Adj)
    elif transform == Transform.NONE:
        pass
    else:
        raise NotImplementedError(f"Transform `{transform}` is not yet implemented.")

    neighs_key = Key.uns.spatial_neighs(key_added)
    conns_key = Key.obsp.spatial_conn(key_added)
    dists_key = Key.obsp.spatial_dist(key_added)

    neighbors_dict = {
        "connectivities_key": conns_key,
        "params": {"n_neighbors": n_neigh, "coord_type": coord_type.v, "radius": radius, "transform": transform.v},
        "distances_key": dists_key,
    }

    _save_data(adata, attr="obsp", key=conns_key, data=Adj)
    _save_data(adata, attr="obsp", key=dists_key, data=Dst, prefix=False)
    _save_data(adata, attr="uns", key=neighs_key, data=neighbors_dict, prefix=False, time=start)


def _build_connectivity(
    coords: np.ndarray,
    n_neighbors: int,
    radius: Optional[float] = None,
    delaunay: bool = False,
    neigh_correct: bool = False,
    set_diag: bool = False,
    return_distance: bool = False,
) -> Union[Tuple[csr_matrix, csr_matrix], csr_matrix]:
    """Build connectivity matrix from spatial coordinates."""
    N = coords.shape[0]

    dists_m = None
    if delaunay:
        tri = Delaunay(coords)
        col_lst = []
        row_lst = []
        dists_lst = []
        for i in np.arange(N):
            idx = np.argwhere(i == tri.simplices)[:, 0]
            idx_col = np.unique(np.setdiff1d(tri.simplices[idx.squeeze(), ...], i)).tolist()
            col_lst.append(idx_col)
            row_lst.append(np.repeat(i, len(idx_col)))
            dists_lst.append(
                euclidean_distances(
                    coords[idx_col, :],
                    coords[np.newaxis, i, :],
                )
            )

        col_indices = np.array(list(chain(*col_lst)))
        row_indices = np.array(list(chain(*row_lst)))
        dists = np.array(list(chain(*dists_lst))).squeeze()

    else:
        tree = NearestNeighbors(n_neighbors=n_neighbors, radius=radius or 1, metric="euclidean")
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
            col_indices = np.repeat(np.arange(N), n_neighbors or 6)
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


@njit
def outer(indices: np.ndarray, indptr: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    res = np.empty_like(indices, dtype=np.float64)
    start = 0
    for i in range(len(indptr) - 1):
        ixs = indices[indptr[i] : indptr[i + 1]]
        res[start : start + len(ixs)] = degrees[i] * degrees[ixs]
        start += len(ixs)

    return res


def _transform_a_spectral(a: spmatrix) -> spmatrix:
    if not isspmatrix_csr(a):
        a = a.tocsr()
    degrees = np.squeeze(np.array(np.sqrt(1.0 / a.sum(axis=0))))

    a = a.multiply(outer(a.indices, a.indptr, degrees))
    a.eliminate_zeros()

    return a


def _transform_a_cosine(a: spmatrix) -> spmatrix:
    return cosine_similarity(a, dense_output=False)
