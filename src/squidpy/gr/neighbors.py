"""Builder classes and helpers for spatial neighbor graph construction."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import chain
from typing import Any, cast

import numpy as np
from fast_array_utils import stats as fau_stats
from numba import njit, prange
from scipy.sparse import (
    SparseEfficiencyWarning,
    csr_array,
    csr_matrix,
    isspmatrix_csr,
    spmatrix,
)
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

from squidpy._constants._constants import CoordType, Transform
from squidpy._utils import NDArrayA
from squidpy._validators import assert_positive

__all__ = ["GraphBuilder", "KNNBuilder", "RadiusBuilder", "DelaunayBuilder", "GridBuilder"]


class GraphBuilder(ABC):
    """Base class for spatial graph construction strategies."""

    def __init__(
        self,
        transform: str | Transform | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        self.transform = Transform.NONE if transform is None else Transform(transform)
        self.set_diag = set_diag
        self.percentile = percentile

    @property
    @abstractmethod
    def coord_type(self) -> CoordType:
        """Coordinate system supported by this builder."""

    @property
    def legacy_params(self) -> dict[str, Any]:
        """Return parameters expressed in the legacy spatial_neighbors API."""
        return {
            "coord_type": self.coord_type,
            "n_neighs": 6,
            "radius": None,
            "delaunay": False,
            "n_rings": 1,
            "percentile": self.percentile,
            "transform": self.transform,
            "set_diag": self.set_diag,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata stored in adata.uns after graph construction."""
        params = self.legacy_params
        return {
            "n_neighbors": params["n_neighs"],
            "coord_type": self.coord_type.v,
            "radius": params["radius"],
            "transform": self.transform.v,
        }

    def build(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            Adj, Dst = self._build_graph(coords)

        self._apply_filters(Adj, Dst)
        self._apply_percentile(Adj, Dst)
        Adj.eliminate_zeros()
        Dst.eliminate_zeros()

        return self._apply_transform(Adj), Dst

    @abstractmethod
    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        """Construct raw adjacency and distance matrices."""

    def _apply_filters(self, Adj: csr_matrix, Dst: csr_matrix) -> None:
        """Apply builder-specific post-processing filters."""
        return None

    def _apply_percentile(self, Adj: csr_matrix, Dst: csr_matrix) -> None:
        if self.percentile is not None and self.coord_type == CoordType.GENERIC:
            threshold = np.percentile(Dst.data, self.percentile)
            Adj[Dst > threshold] = 0.0
            Dst[Dst > threshold] = 0.0

    def _apply_transform(self, Adj: csr_matrix) -> csr_matrix:
        if self.transform == Transform.SPECTRAL:
            return cast(csr_matrix, _transform_a_spectral(Adj))
        if self.transform == Transform.COSINE:
            return cast(csr_matrix, _transform_a_cosine(Adj))
        if self.transform == Transform.NONE:
            return Adj

        raise NotImplementedError(f"Transform `{self.transform}` is not yet implemented.")


class KNNBuilder(GraphBuilder):
    """Build a generic k-nearest-neighbor spatial graph."""

    def __init__(
        self,
        n_neighs: int = 6,
        transform: str | Transform | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        assert_positive(n_neighs, name="n_neighs")
        super().__init__(transform=transform, set_diag=set_diag, percentile=percentile)
        self.n_neighs = n_neighs

    @property
    def coord_type(self) -> CoordType:
        return CoordType.GENERIC

    @property
    def legacy_params(self) -> dict[str, Any]:
        return super().legacy_params | {"n_neighs": self.n_neighs}

    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        return cast(
            tuple[csr_matrix, csr_matrix],
            _build_connectivity(
                coords,
                n_neighs=self.n_neighs,
                return_distance=True,
                set_diag=self.set_diag,
            ),
        )


class _RadiusFilterBuilder(GraphBuilder):
    """Intermediate base for builders that support radius-interval pruning."""

    radius: float | tuple[float, float] | None
    n_neighs: int

    @property
    def coord_type(self) -> CoordType:
        return CoordType.GENERIC

    def _apply_filters(self, Adj: csr_matrix, Dst: csr_matrix) -> None:
        if isinstance(self.radius, Iterable):
            _filter_by_radius_interval(Adj, Dst, self.radius)


class RadiusBuilder(_RadiusFilterBuilder):
    """Build a generic radius-based spatial graph."""

    def __init__(
        self,
        radius: float | tuple[float, float],
        n_neighs: int = 6,
        transform: str | Transform | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        assert_positive(n_neighs, name="n_neighs")
        super().__init__(transform=transform, set_diag=set_diag, percentile=percentile)
        self.radius = radius
        self.n_neighs = n_neighs

    @property
    def legacy_params(self) -> dict[str, Any]:
        return super().legacy_params | {"n_neighs": self.n_neighs, "radius": self.radius}

    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        return cast(
            tuple[csr_matrix, csr_matrix],
            _build_connectivity(
                coords,
                n_neighs=self.n_neighs,
                radius=self.radius,
                return_distance=True,
                set_diag=self.set_diag,
            ),
        )


class DelaunayBuilder(_RadiusFilterBuilder):
    """Build a generic spatial graph from a Delaunay triangulation."""

    def __init__(
        self,
        radius: float | tuple[float, float] | None = None,
        n_neighs: int = 6,
        transform: str | Transform | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        assert_positive(n_neighs, name="n_neighs")
        super().__init__(transform=transform, set_diag=set_diag, percentile=percentile)
        self.radius = radius
        self.n_neighs = n_neighs

    @property
    def legacy_params(self) -> dict[str, Any]:
        return super().legacy_params | {"n_neighs": self.n_neighs, "radius": self.radius, "delaunay": True}

    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        return cast(
            tuple[csr_matrix, csr_matrix],
            _build_connectivity(
                coords,
                n_neighs=self.n_neighs,
                radius=self.radius,
                delaunay=True,
                return_distance=True,
                set_diag=self.set_diag,
            ),
        )


class GridBuilder(GraphBuilder):
    """Build a grid-based spatial graph."""

    def __init__(
        self,
        n_neighs: int = 6,
        n_rings: int = 1,
        delaunay: bool = False,
        transform: str | Transform | None = None,
        set_diag: bool = False,
    ) -> None:
        assert_positive(n_neighs, name="n_neighs")
        assert_positive(n_rings, name="n_rings")
        super().__init__(transform=transform, set_diag=set_diag, percentile=None)
        self.n_neighs = n_neighs
        self.n_rings = n_rings
        self.delaunay = delaunay

    @property
    def coord_type(self) -> CoordType:
        return CoordType.GRID

    @property
    def legacy_params(self) -> dict[str, Any]:
        return super().legacy_params | {
            "n_neighs": self.n_neighs,
            "delaunay": self.delaunay,
            "n_rings": self.n_rings,
        }

    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        return _build_grid(
            coords,
            n_neighs=self.n_neighs,
            n_rings=self.n_rings,
            delaunay=self.delaunay,
            set_diag=self.set_diag,
        )


# ---------------------------------------------------------------------------
# Private helpers used by the builder classes
# ---------------------------------------------------------------------------


def _filter_by_radius_interval(
    Adj: csr_matrix,
    Dst: csr_matrix,
    radius: Iterable[float],
) -> None:
    minn, maxx = sorted(radius)[:2]
    mask = (Dst.data < minn) | (Dst.data > maxx)
    a_diag = Adj.diagonal()

    Dst.data[mask] = 0.0
    Adj.data[mask] = 0.0
    Adj.setdiag(a_diag)


def _build_grid(
    coords: NDArrayA,
    n_neighs: int,
    n_rings: int,
    delaunay: bool = False,
    set_diag: bool = False,
) -> tuple[csr_matrix, csr_matrix]:
    if n_rings > 1:
        Adj: csr_matrix = _build_connectivity(
            coords,
            n_neighs=n_neighs,
            neigh_correct=True,
            set_diag=True,
            delaunay=delaunay,
            return_distance=False,
        )
        Res, Walk = Adj, Adj
        for i in range(n_rings - 1):
            Walk = Walk @ Adj
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
            coords,
            n_neighs=n_neighs,
            neigh_correct=True,
            delaunay=delaunay,
            set_diag=set_diag,
        )
        Dst = Adj.copy()

    Dst.setdiag(0.0)

    return Adj, Dst


def _build_connectivity(
    coords: NDArrayA,
    n_neighs: int,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    neigh_correct: bool = False,
    set_diag: bool = False,
    return_distance: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    N = coords.shape[0]
    if delaunay:
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        Adj = csr_matrix((np.ones_like(indices, dtype=np.float32), indices, indptr), shape=(N, N))

        if return_distance:
            # fmt: off
            dists = np.array(list(chain(*(
                euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
                for i in range(N)
                if len(indices[indptr[i] : indptr[i + 1]])
            )))).squeeze()
            Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
            # fmt: on
    else:
        r = 1 if radius is None else radius if isinstance(radius, int | float) else max(radius)
        tree = NearestNeighbors(n_neighbors=n_neighs, radius=r, metric="euclidean")
        tree.fit(coords)

        if radius is None:
            dists, col_indices = tree.kneighbors()
            dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
            row_indices = np.repeat(np.arange(N), n_neighs)
            if neigh_correct:
                dist_cutoff = np.median(dists) * 1.3  # there's a small amount of sway
                mask = dists < dist_cutoff
                row_indices, col_indices, dists = (
                    row_indices[mask],
                    col_indices[mask],
                    dists[mask],
                )
        else:
            dists, col_indices = tree.radius_neighbors()
            row_indices = np.repeat(np.arange(N), [len(x) for x in col_indices])
            dists = np.concatenate(dists)
            col_indices = np.concatenate(col_indices)

        Adj = csr_matrix(
            (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
            shape=(N, N),
        )
        if return_distance:
            Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

    # radius-based filtering needs same indices/indptr: do not remove 0s
    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    if return_distance:
        Dst.setdiag(0.0)
        return Adj, Dst

    return Adj


@njit
def _csr_bilateral_diag_scale_helper(
    mat: csr_array | csr_matrix,
    degrees: NDArrayA,
) -> NDArrayA:
    """
    Return an array F aligned with CSR non-zeros such that
    F[k] = d[i] * data[k] * d[j] for the k-th non-zero (i, j) in CSR order.

    Parameters
    ----------

    data : array of float
        CSR `data` (non-zero values).
    indices : array of int
        CSR `indices` (column indices).
    indptr : array of int
        CSR `indptr` (row pointer).
    degrees : array of float, shape (n,)
        Diagonal scaling vector.

    Returns
    -------
    array of float
        Length equals len(data). Entry-wise factors d_i * d_j * data[k]
    """

    res = np.empty_like(mat.data, dtype=np.float32)
    for i in prange(len(mat.indptr) - 1):
        ixs = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        res[mat.indptr[i] : mat.indptr[i + 1]] = degrees[i] * degrees[ixs] * mat.data[mat.indptr[i] : mat.indptr[i + 1]]

    return res


def symmetric_normalize_csr(adj: spmatrix) -> csr_matrix:
    """
    Return D^{-1/2} * A * D^{-1/2}, where D = diag(degrees(A)) and A = adj.


    Parameters
    ----------
    adj : scipy.sparse.csr_matrix

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    degrees = np.squeeze(np.array(np.sqrt(1.0 / fau_stats.sum(adj, axis=0))))
    if adj.shape[0] != len(degrees):
        raise ValueError("len(degrees) must equal number of rows of adj")
    res_data = _csr_bilateral_diag_scale_helper(adj, degrees)
    return csr_matrix((res_data, adj.indices, adj.indptr), shape=adj.shape)


def _transform_a_spectral(a: spmatrix) -> spmatrix:
    if not isspmatrix_csr(a):
        a = a.tocsr()
    if not a.nnz:
        return a

    return symmetric_normalize_csr(a)


def _transform_a_cosine(a: spmatrix) -> spmatrix:
    return cosine_similarity(a, dense_output=False)
