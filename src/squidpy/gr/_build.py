"""Functions for building graphs from spatial coordinates."""

from __future__ import annotations

import warnings
from collections.abc import Iterable  # noqa: F401
from functools import partial
from itertools import chain
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.utils import make_index_unique
from geopandas import GeoDataFrame
from numba import njit
from scanpy import logging as logg
from scipy.sparse import (
    SparseEfficiencyWarning,
    block_diag,
    csr_matrix,
    isspmatrix_csr,
    spmatrix,
)
from scipy.spatial import Delaunay
from shapely import LineString, MultiPolygon, Point, Polygon, distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from spatialdata import SpatialData
from spatialdata._core.centroids import get_centroids
from spatialdata._core.query.relational_query import (
    get_element_instances,
    match_element_to_table,
)
from spatialdata.models import SpatialElement, get_table_keys
from spatialdata.models.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    RasterSchema,
    ShapesModel,
    TableModel,
    get_axes_names,
    get_model,
)

from squidpy._constants._constants import CoordType, Transform
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_positive,
    _assert_spatial_basis,
    _save_data,
)

__all__ = ["spatial_neighbors"]


@d.dedent
@inject_docs(t=Transform, c=CoordType)
def spatial_neighbors(
    adata: AnnData | SpatialData,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    coord_type: str | CoordType | None = None,
    n_neighs: int = 6,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    n_rings: int = 1,
    percentile: float | None = None,
    transform: str | Transform | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> tuple[csr_matrix, csr_matrix] | None:
    """
    Create a graph from spatial coordinates.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
        If `adata` is a :class:`spatialdata.SpatialData`, the coordinates of the centroids will be stored in the
        `adata` with this key.
    elements_to_coordinate_systems
        A dictionary mapping element names of the SpatialData object to coordinate systems.
        The elements can be either Shapes or Labels. For compatibility, the spatialdata table must annotate
        all regions keys. Must not be `None` if `adata` is a :class:`spatialdata.SpatialData`.
    table_key
        Key in :attr:`spatialdata.SpatialData.tables` where the spatialdata table is stored. Must not be `None` if
        `adata` is a :class:`spatialdata.SpatialData`.
    mask_polygon
        The Polygon or MultiPolygon element.
    %(library_key)s
    coord_type
        Type of coordinate system. Valid options are:

            - `{c.GRID.s!r}` - grid coordinates.
            - `{c.GENERIC.s!r}` - generic coordinates.
            - `None` - `{c.GRID.s!r}` if ``spatial_key`` is in :attr:`anndata.AnnData.uns`
              with ``n_neighs = 6`` (Visium), otherwise use `{c.GENERIC.s!r}`.
    n_neighs
        Depending on the ``coord_type``:

            - `{c.GRID.s!r}` - number of neighboring tiles.
            - `{c.GENERIC.s!r}` - number of neighborhoods for non-grid data. Only used when ``delaunay = False``.
    radius
        Only available when ``coord_type = {c.GENERIC.s!r}``. Depending on the type:

            - :class:`float` - compute the graph based on neighborhood radius.
            - :class:`tuple` - prune the final graph to only contain edges in interval `[min(radius), max(radius)]`.
    delaunay
        Whether to compute the graph from Delaunay triangulation. Only used when ``coord_type = {c.GENERIC.s!r}``.
    n_rings
        Number of rings of neighbors for grid data. Only used when ``coord_type = {c.GRID.s!r}``.
    percentile
        Percentile of the distances to use as threshold. Only used when ``coord_type = {c.GENERIC.s!r}``.
    transform
        Type of adjacency matrix transform. Valid options are:

            - `{t.SPECTRAL.s!r}` - spectral transformation of the adjacency matrix.
            - `{t.COSINE.s!r}` - cosine transformation of the adjacency matrix.
            - `{t.NONE.v}` - no transformation of the adjacency matrix.
    set_diag
        Whether to set the diagonal of the spatial connectivities to `1.0`.
    key_added
        Key which controls where the results are saved if ``copy = False``.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the spatial connectivities and distances matrices.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_connectivities']`` - the spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_distances']`` - the spatial distances.
        - :attr:`anndata.AnnData.uns`  ``['{{key_added}}']`` - :class:`dict` containing parameters.
    """
    if isinstance(adata, SpatialData):
        assert elements_to_coordinate_systems is not None, (
            "Since `adata` is a :class:`spatialdata.SpatialData`, `elements_to_coordinate_systems` must not be `None`."
        )
        assert table_key is not None, (
            "Since `adata` is a :class:`spatialdata.SpatialData`, `table_key` must not be `None`."
        )
        elements, table = match_element_to_table(adata, list(elements_to_coordinate_systems), table_key)
        assert table.obs_names.equals(adata.tables[table_key].obs_names), (
            "The spatialdata table must annotate all elements keys. Some elements are missing, please check the `elements_to_coordinate_systems` dictionary."
        )
        regions, region_key, instance_key = get_table_keys(adata.tables[table_key])
        regions = [regions] if isinstance(regions, str) else regions
        ordered_regions_in_table = adata.tables[table_key].obs[region_key].unique()

        # TODO: remove this after https://github.com/scverse/spatialdata/issues/614
        remove_centroids = {}
        elem_instances = []
        for e in regions:
            schema = get_model(elements[e])
            element_instances = get_element_instances(elements[e]).to_series()
            if np.isin(0, element_instances.values) and (schema in (Labels2DModel, Labels3DModel)):
                element_instances = element_instances.drop(index=0)
                remove_centroids[e] = True
            else:
                remove_centroids[e] = False
            elem_instances.append(element_instances)

        element_instances = pd.concat(elem_instances)
        if (not np.all(element_instances.values == adata.tables[table_key].obs[instance_key].values)) or (
            not np.all(ordered_regions_in_table == regions)
        ):
            raise ValueError(
                "The spatialdata table must annotate all elements keys. Some elements are missing or not ordered correctly, please check the `elements_to_coordinate_systems` dictionary."
            )
        centroids = []
        for region_ in ordered_regions_in_table:
            cs = elements_to_coordinate_systems[region_]
            centroid = get_centroids(adata[region_], coordinate_system=cs)[["x", "y"]].compute()

            # TODO: remove this after https://github.com/scverse/spatialdata/issues/614
            if remove_centroids[region_]:
                centroid = centroid[1:].copy()
            centroids.append(centroid)

        adata.tables[table_key].obsm[spatial_key] = np.concatenate(centroids)
        adata = adata.tables[table_key]
        library_key = region_key

    _assert_positive(n_rings, name="n_rings")
    _assert_positive(n_neighs, name="n_neighs")
    _assert_spatial_basis(adata, spatial_key)

    transform = Transform.NONE if transform is None else Transform(transform)
    if coord_type is None:
        if radius is not None:
            logg.warning(
                f"Graph creation with `radius` is only available when `coord_type = {CoordType.GENERIC!r}` specified. "
                f"Ignoring parameter `radius = {radius}`."
            )
        coord_type = CoordType.GRID if Key.uns.spatial in adata.uns else CoordType.GENERIC
    else:
        coord_type = CoordType(coord_type)

    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        libs = adata.obs[library_key].cat.categories
        make_index_unique(adata.obs_names)
    else:
        libs = [None]

    start = logg.info(
        f"Creating graph using `{coord_type}` coordinates and `{transform}` transform and `{len(libs)}` libraries."
    )
    _build_fun = partial(
        _spatial_neighbor,
        spatial_key=spatial_key,
        coord_type=coord_type,
        n_neighs=n_neighs,
        radius=radius,
        delaunay=delaunay,
        n_rings=n_rings,
        transform=transform,
        set_diag=set_diag,
        percentile=percentile,
    )

    if library_key is not None:
        mats: list[tuple[spmatrix, spmatrix]] = []
        ixs: list[int] = []
        for lib in libs:
            ixs.extend(np.where(adata.obs[library_key] == lib)[0])
            mats.append(_build_fun(adata[adata.obs[library_key] == lib]))
        ixs = cast(list[int], np.argsort(ixs).tolist())
        Adj = block_diag([m[0] for m in mats], format="csr")[ixs, :][:, ixs]
        Dst = block_diag([m[1] for m in mats], format="csr")[ixs, :][:, ixs]
    else:
        Adj, Dst = _build_fun(adata)

    neighs_key = Key.uns.spatial_neighs(key_added)
    conns_key = Key.obsp.spatial_conn(key_added)
    dists_key = Key.obsp.spatial_dist(key_added)

    neighbors_dict = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": {
            "n_neighbors": n_neighs,
            "coord_type": coord_type.v,
            "radius": radius,
            "transform": transform.v,
        },
    }

    if copy:
        return Adj, Dst

    _save_data(adata, attr="obsp", key=conns_key, data=Adj)
    _save_data(adata, attr="obsp", key=dists_key, data=Dst, prefix=False)
    _save_data(adata, attr="uns", key=neighs_key, data=neighbors_dict, prefix=False, time=start)


def _spatial_neighbor(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    coord_type: str | CoordType | None = None,
    n_neighs: int = 6,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    n_rings: int = 1,
    transform: str | Transform | None = None,
    set_diag: bool = False,
    percentile: float | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    coords = adata.obsm[spatial_key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        if coord_type == CoordType.GRID:
            Adj, Dst = _build_grid(
                coords,
                n_neighs=n_neighs,
                n_rings=n_rings,
                delaunay=delaunay,
                set_diag=set_diag,
            )
        elif coord_type == CoordType.GENERIC:
            Adj, Dst = _build_connectivity(
                coords,
                n_neighs=n_neighs,
                radius=radius,
                delaunay=delaunay,
                return_distance=True,
                set_diag=set_diag,
            )
        else:
            raise NotImplementedError(f"Coordinate type `{coord_type}` is not yet implemented.")

    if coord_type == CoordType.GENERIC and isinstance(radius, Iterable):
        minn, maxx = sorted(radius)[:2]
        mask = (Dst.data < minn) | (Dst.data > maxx)
        a_diag = Adj.diagonal()

        Dst.data[mask] = 0.0
        Adj.data[mask] = 0.0
        Adj.setdiag(a_diag)

    if percentile is not None and coord_type == CoordType.GENERIC:
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0

    Adj.eliminate_zeros()
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

    return Adj, Dst


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
        Adj = csr_matrix((np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N))

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
            (np.ones_like(row_indices, dtype=np.float64), (row_indices, col_indices)),
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
def outer(indices: NDArrayA, indptr: NDArrayA, degrees: NDArrayA) -> NDArrayA:
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
    if not a.nnz:
        return a

    degrees = np.squeeze(np.array(np.sqrt(1.0 / a.sum(axis=0))))
    a = a.multiply(outer(a.indices, a.indptr, degrees))
    a.eliminate_zeros()

    return a


def _transform_a_cosine(a: spmatrix) -> spmatrix:
    return cosine_similarity(a, dense_output=False)


@d.dedent
def mask_graph(
    sdata: SpatialData,
    table_key: str,
    polygon_mask: Polygon | MultiPolygon,
    negative_mask: bool = False,
    spatial_key: str = Key.obsm.spatial,
    key_added: str = "mask",
    copy: bool = False,
) -> SpatialData:
    """
    Mask the graph based on a polygon mask.

    Given a spatial graph stored in :attr:`anndata.AnnData.obsp` ``['{{key_added}}_{{spatial_key}}_connectivities']`` and spatial coordinates stored in :attr:`anndata.AnnData.obsp` ``['{{spatial_key}}']``, it maskes the graph so that only edges fully contained in the polygons are kept.

    Parameters
    ----------
    sdata
        The spatial data object.
    table_key:
        The key of the table containing the spatial data.
    polygon_mask
        The :class:`shapely.Polygon` or :class:`shapely.MultiPolygon` to be used as mask.
    negative_mask
        Whether to keep the edges within the polygon mask or outside.
        Note that when ``negative_mask = True``, only the edges fully contained in the polygon are removed.
        If edges are partially contained in the polygon, they are kept.
    %(spatial_key)s
    key_added
        Key which controls where the results are saved if ``copy = False``.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the masked spatial connectivities and masked distances matrices.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_{{spatial_key}}_connectivities']`` - the spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_{{spatial_key}}_distances']`` - the spatial distances.
        - :attr:`anndata.AnnData.uns`  ``['{{key_added}}_{{spatial_key}}']`` - :class:`dict` containing parameters.

    Notes
    -----
    The `polygon_mask` must be in the same `coordinate_systems` of the spatial graph, but no check is performed to assess this.
    """
    # we could add this to arg, but I don't see use case for now
    neighs_key = Key.uns.spatial_neighs(spatial_key)
    conns_key = Key.obsp.spatial_conn(spatial_key)
    dists_key = Key.obsp.spatial_dist(spatial_key)

    # check polygon type
    if not isinstance(polygon_mask, Polygon | MultiPolygon):
        raise ValueError(f"`polygon_mask` should be of type `Polygon` or `MultiPolygon`, got {type(polygon_mask)}")

    # get elements
    table = sdata.tables[table_key]
    coords = table.obsm[spatial_key]
    Adj = table.obsp[conns_key]
    Dst = table.obsp[dists_key]

    # convert edges to lines
    lines_coords, idx_out = _get_lines_coords(Adj.indices, Adj.indptr, coords)
    lines_coords, idx_out = np.array(lines_coords), np.array(idx_out)
    lines_df = gpd.GeoDataFrame(geometry=list(map(LineString, lines_coords)))

    # check that lines overlap with the polygon
    filt_lines = lines_df.geometry.within(polygon_mask).values

    # ~ within index, and set that to 0
    if not negative_mask:
        # keep only the lines that are within the polygon
        filt_lines = ~filt_lines
    filt_idx_out = idx_out[filt_lines]

    # filter connectivities
    Adj[filt_idx_out[:, 0], filt_idx_out[:, 1]] = 0
    Adj.eliminate_zeros()

    # filter_distances
    Dst[filt_idx_out[:, 0], filt_idx_out[:, 1]] = 0
    Dst.eliminate_zeros()

    mask_conns_key = f"{key_added}_{conns_key}"
    mask_dists_key = f"{key_added}_{dists_key}"
    mask_neighs_key = f"{key_added}_{neighs_key}"

    neighbors_dict = {
        "connectivities_key": mask_conns_key,
        "distances_key": mask_dists_key,
        "unfiltered_graph_key": conns_key,
        "params": {
            "negative_mask": negative_mask,
            "table_key": table_key,
        },
    }

    if copy:
        return Adj, Dst

    # save back to spatialdata
    _save_data(table, attr="obsp", key=mask_conns_key, data=Adj)
    _save_data(table, attr="obsp", key=mask_dists_key, data=Dst, prefix=False)
    _save_data(table, attr="uns", key=mask_neighs_key, data=neighbors_dict, prefix=False)


@njit
def _get_lines_coords(indices: NDArrayA, indptr: NDArrayA, coords: NDArrayA) -> tuple[list[Any], list[Any]]:
    lines = []
    idx_out = []
    for i in range(len(indptr) - 1):
        ixs = indices[indptr[i] : indptr[i + 1]]
        for ix in ixs:
            lines.append([coords[i], coords[ix]])
            idx_out.append((i, ix))
    return lines, idx_out
