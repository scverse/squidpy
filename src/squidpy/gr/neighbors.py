"""Graph construction strategies for spatial neighbor graphs.

See the :doc:`/extensibility` guide for how to implement a custom builder.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Generic, TypeVar, cast

import numpy as np
from fast_array_utils import stats as fau_stats
from numba import njit, prange
from scipy.sparse import (
    SparseEfficiencyWarning,
    block_diag,
    csr_array,
    csr_matrix,
    isspmatrix_csr,
    spmatrix,
)
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

from squidpy._constants._constants import Transform
from squidpy._utils import NDArrayA
from squidpy._validators import assert_positive

__all__ = [
    "GraphMatrixT",
    "GraphBuilder",
    "GraphBuilderCSR",
    "GraphPostprocessor",
    "DistanceIntervalPostprocessor",
    "PercentilePostprocessor",
    "TransformPostprocessor",
    "KNNBuilder",
    "RadiusBuilder",
    "DelaunayBuilder",
    "GridBuilder",
]


CoordT = TypeVar("CoordT")
GraphMatrixT = TypeVar("GraphMatrixT")
GraphPostprocessor = Callable[[GraphMatrixT, GraphMatrixT], tuple[GraphMatrixT, GraphMatrixT]]


class GraphBuilder(ABC, Generic[CoordT, GraphMatrixT]):
    """Base class for spatial graph construction strategies.

    Custom builders must implement :meth:`build_graph`. Overriding
    :meth:`postprocessors` and :meth:`combine` is optional. Postprocessors can
    be provided directly via ``__init__`` or by overriding
    :meth:`postprocessors`.
    """

    def __init__(
        self,
        transform: str | Transform | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
        postprocessors: Sequence[GraphPostprocessor[GraphMatrixT]] = (),
    ) -> None:
        self.transform = Transform.NONE if transform is None else Transform(transform)
        self.set_diag = set_diag
        self.percentile = percentile
        self._postprocessors: list[GraphPostprocessor[GraphMatrixT]] = list(postprocessors)

    def build(self, coords: CoordT) -> tuple[GraphMatrixT, GraphMatrixT]:
        adj, dst = self.build_graph(coords)
        for postprocessor in self.postprocessors():
            adj, dst = postprocessor(adj, dst)
        return adj, dst

    @abstractmethod
    def build_graph(self, coords: CoordT) -> tuple[GraphMatrixT, GraphMatrixT]:
        """Construct raw adjacency and distance matrices."""

    def postprocessors(self) -> Sequence[GraphPostprocessor[GraphMatrixT]]:
        """Return post-build processing steps for ``(adj, dst)``."""
        return self._postprocessors

    def combine(
        self,
        mats: Sequence[tuple[GraphMatrixT, GraphMatrixT]],
        ixs: Sequence[int],
    ) -> tuple[GraphMatrixT, GraphMatrixT]:
        """Combine per-library results into a single graph.

        Override this only if the builder should support multi-library graph
        construction via ``library_key``.
        """
        raise NotImplementedError("Using `library_key` with this graph builder is not implemented yet.")


class GraphBuilderCSR(GraphBuilder[NDArrayA, csr_matrix], ABC):
    """CSR-based graph construction strategy.

    Specializes :class:`GraphBuilder` for sparse CSR matrix output. Adds
    SparseEfficiencyWarning suppression and multi-library ``library_key``
    combination. Built-in concrete builders
    (:class:`KNNBuilder`, :class:`RadiusBuilder`, :class:`DelaunayBuilder`, :class:`GridBuilder`)
    inherit from this class and declare their postprocessors explicitly in
    ``__init__`` using the reusable public postprocessor classes.

    Subclass this (not the generic :class:`GraphBuilder`) when implementing a builder
    that returns CSR matrices.

    See Also
    --------
    GraphBuilder : Generic builder interface for custom coordinate/matrix types.
    KNNBuilder : Example of a concrete CSR-based builder.
    """

    def build(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            return super().build(coords)

    @abstractmethod
    def build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        """Construct raw adjacency and distance matrices."""

    def combine(
        self,
        mats: Sequence[tuple[csr_matrix, csr_matrix]],
        ixs: Sequence[int],
    ) -> tuple[csr_matrix, csr_matrix]:
        order = cast(list[int], np.argsort(ixs).tolist())
        adj = block_diag([m[0] for m in mats], format="csr")[order, :][:, order]
        dst = block_diag([m[1] for m in mats], format="csr")[order, :][:, order]
        return cast(csr_matrix, adj), cast(csr_matrix, dst)


class KNNBuilder(GraphBuilderCSR):
    """Build a generic k-nearest-neighbor spatial graph.

    Each observation is connected to its k nearest neighbors. See
    :func:`~squidpy.gr.spatial_neighbors_knn` for the user-facing API or
    :func:`~squidpy.gr.spatial_neighbors_from_builder` for direct builder usage.
    """

    def __init__(
        self,
        n_neighs: int = 6,
        transform: str | Transform | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        assert_positive(n_neighs, name="n_neighs")
        postprocessors: list[GraphPostprocessor[csr_matrix]] = []
        if percentile is not None:
            postprocessors.append(PercentilePostprocessor(percentile))
        postprocessors.append(TransformPostprocessor(Transform.NONE if transform is None else Transform(transform)))
        super().__init__(
            transform=transform,
            set_diag=set_diag,
            percentile=percentile,
            postprocessors=postprocessors,
        )
        self.n_neighs = n_neighs

    def build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        N = coords.shape[0]
        tree = NearestNeighbors(n_neighbors=self.n_neighs, radius=1, metric="euclidean")
        tree.fit(coords)

        dists, col_indices = tree.kneighbors()
        dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
        row_indices = np.repeat(np.arange(N), self.n_neighs)

        adj = csr_matrix(
            (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
            shape=(N, N),
        )
        dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

        adj.setdiag(1.0 if self.set_diag else adj.diagonal())
        dst.setdiag(0.0)
        return adj, dst


class RadiusBuilder(GraphBuilderCSR):
    """Build a generic radius-based spatial graph.

    Two observations are connected when their Euclidean distance falls within
    the specified radius. See :func:`~squidpy.gr.spatial_neighbors_radius` for the
    user-facing API or :func:`~squidpy.gr.spatial_neighbors_from_builder` for
    direct builder usage.
    """

    def __init__(
        self,
        radius: float | tuple[float, float],
        transform: str | Transform | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        postprocessors: list[GraphPostprocessor[csr_matrix]] = []
        if isinstance(radius, tuple):
            postprocessors.append(DistanceIntervalPostprocessor(tuple(sorted(radius))))
        if percentile is not None:
            postprocessors.append(PercentilePostprocessor(percentile))
        postprocessors.append(TransformPostprocessor(Transform.NONE if transform is None else Transform(transform)))
        super().__init__(
            transform=transform,
            set_diag=set_diag,
            percentile=percentile,
            postprocessors=postprocessors,
        )
        self.radius = radius

    def build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        N = coords.shape[0]
        r = self.radius if isinstance(self.radius, int | float) else max(self.radius)
        tree = NearestNeighbors(radius=r, metric="euclidean")
        tree.fit(coords)

        dists, col_indices = tree.radius_neighbors()
        row_indices = np.repeat(np.arange(N), [len(x) for x in col_indices])
        dists = np.concatenate(dists)
        col_indices = np.concatenate(col_indices)

        adj = csr_matrix(
            (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
            shape=(N, N),
        )
        dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

        adj.setdiag(1.0 if self.set_diag else adj.diagonal())
        dst.setdiag(0.0)
        return adj, dst


class DelaunayBuilder(GraphBuilderCSR):
    """Build a generic point-cloud graph from a Delaunay triangulation.

    Delaunay triangulation connects observations into triangles such that no
    other observation lies inside the circumcircle of each triangle. Unlike
    ``GridBuilder(delaunay=True)``, this builder uses geometry-based connectivity
    and stores real Euclidean edge lengths.

    ``radius`` only controls post-construction edge pruning; the triangulation
    itself is unchanged. A tuple ``(min, max)`` keeps edges with Euclidean
    length in that interval. A scalar ``r`` is shorthand for ``(0.0, r)``,
    i.e. keep edges with length at most ``r``. ``None`` keeps every edge.

    See :func:`~squidpy.gr.spatial_neighbors_delaunay` for the user-facing API or
    :func:`~squidpy.gr.spatial_neighbors_from_builder` for direct builder usage.
    """

    def __init__(
        self,
        radius: float | tuple[float, float] | None = None,
        transform: str | Transform | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        if isinstance(radius, int | float):
            radius = (0.0, float(radius))
        postprocessors: list[GraphPostprocessor[csr_matrix]] = []
        if radius is not None:
            postprocessors.append(DistanceIntervalPostprocessor(tuple(sorted(radius))))
        if percentile is not None:
            postprocessors.append(PercentilePostprocessor(percentile))
        postprocessors.append(TransformPostprocessor(Transform.NONE if transform is None else Transform(transform)))
        super().__init__(
            transform=transform,
            set_diag=set_diag,
            percentile=percentile,
            postprocessors=postprocessors,
        )
        self.radius = radius

    def build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        N = coords.shape[0]
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        adj = csr_matrix((np.ones_like(indices, dtype=np.float32), indices, indptr), shape=(N, N))

        # fmt: off
        dists = np.array(list(chain(*(
            euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
            for i in range(N)
            if len(indices[indptr[i] : indptr[i + 1]])
        )))).squeeze()
        # fmt: on
        dst = csr_matrix((dists, indices, indptr), shape=(N, N))

        adj.setdiag(1.0 if self.set_diag else adj.diagonal())
        dst.setdiag(0.0)
        return adj, dst


class GridBuilder(GraphBuilderCSR):
    """Build a grid-based spatial graph.

    Assumes observations lie on an approximately regular lattice (e.g., Visium).
    When ``delaunay=True``, Delaunay triangulation is used only to derive the
    base connectivity; the distance matrix still encodes grid/ring distances,
    not Euclidean lengths.

    See :func:`~squidpy.gr.spatial_neighbors_grid` for the user-facing API or
    :func:`~squidpy.gr.spatial_neighbors_from_builder` for direct builder usage.
    """

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
        postprocessors = [TransformPostprocessor(Transform.NONE if transform is None else Transform(transform))]
        super().__init__(transform=transform, set_diag=set_diag, percentile=None, postprocessors=postprocessors)
        self.n_neighs = n_neighs
        self.n_rings = n_rings
        self.delaunay = delaunay

    def build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        if self.n_rings > 1:
            adj = self._base_adjacency(coords, set_diag=True)
            res, walk = adj, adj
            for i in range(self.n_rings - 1):
                walk = walk @ adj
                walk[res.nonzero()] = 0.0
                walk.eliminate_zeros()
                walk.data[:] = i + 2.0
                res = res + walk
            adj = res
            adj.setdiag(float(self.set_diag))
            adj.eliminate_zeros()

            dst = adj.copy()
            adj.data[:] = 1.0
        else:
            adj = self._base_adjacency(coords, set_diag=self.set_diag)
            dst = adj.copy()

        dst.setdiag(0.0)
        return adj, dst

    def _base_adjacency(self, coords: NDArrayA, *, set_diag: bool) -> csr_matrix:
        """KNN adjacency with median-distance correction for grid coordinates."""
        N = coords.shape[0]
        if self.delaunay:
            tri = Delaunay(coords)
            indptr, indices = tri.vertex_neighbor_vertices
            adj = csr_matrix((np.ones_like(indices, dtype=np.float32), indices, indptr), shape=(N, N))
        else:
            tree = NearestNeighbors(n_neighbors=self.n_neighs, radius=1, metric="euclidean")
            tree.fit(coords)
            dists, col_indices = tree.kneighbors()
            dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
            row_indices = np.repeat(np.arange(N), self.n_neighs)

            dist_cutoff = np.median(dists) * 1.3
            mask = dists < dist_cutoff
            row_indices, col_indices = row_indices[mask], col_indices[mask]

            adj = csr_matrix(
                (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
                shape=(N, N),
            )

        adj.setdiag(1.0 if set_diag else adj.diagonal())
        return adj


# ---------------------------------------------------------------------------
# Private helpers used by the builder classes
# ---------------------------------------------------------------------------


def _filter_by_radius_interval(
    adj: csr_matrix,
    dst: csr_matrix,
    radius: tuple[float, float],
) -> None:
    minn, maxx = radius
    mask = (dst.data < minn) | (dst.data > maxx)
    a_diag = adj.diagonal()

    dst.data[mask] = 0.0
    adj.data[mask] = 0.0
    adj.setdiag(a_diag)


@dataclass(frozen=True)
class DistanceIntervalPostprocessor:
    interval: tuple[float, float]

    def __call__(self, adj: csr_matrix, dst: csr_matrix) -> tuple[csr_matrix, csr_matrix]:
        _filter_by_radius_interval(adj, dst, self.interval)
        return adj, dst


@dataclass(frozen=True)
class PercentilePostprocessor:
    percentile: float

    def __call__(self, adj: csr_matrix, dst: csr_matrix) -> tuple[csr_matrix, csr_matrix]:
        threshold = np.percentile(dst.data, self.percentile)
        adj[dst > threshold] = 0.0
        dst[dst > threshold] = 0.0
        return adj, dst


@dataclass(frozen=True)
class TransformPostprocessor:
    transform: Transform

    def __call__(self, adj: csr_matrix, dst: csr_matrix) -> tuple[csr_matrix, csr_matrix]:
        adj.eliminate_zeros()
        dst.eliminate_zeros()

        if self.transform == Transform.SPECTRAL:
            return cast(csr_matrix, _transform_a_spectral(adj)), dst
        if self.transform == Transform.COSINE:
            return cast(csr_matrix, _transform_a_cosine(adj)), dst
        if self.transform == Transform.NONE:
            return adj, dst

        raise NotImplementedError(f"Transform `{self.transform}` is not yet implemented.")


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
