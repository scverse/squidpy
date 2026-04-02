"""Functions for building graphs from spatial coordinates."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, NamedTuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.utils import make_index_unique
from numba import njit
from scipy.sparse import (
    block_diag,
    csr_matrix,
    spmatrix,
)
from shapely import LineString, MultiPolygon, Polygon
from spatialdata import SpatialData
from spatialdata._core.centroids import get_centroids
from spatialdata._core.query.relational_query import get_element_instances, match_element_to_table
from spatialdata._logging import logger as logg
from spatialdata.models import get_table_keys
from spatialdata.models.models import (
    Labels2DModel,
    Labels3DModel,
    get_model,
)

from squidpy._constants._constants import CoordType, Transform
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA
from squidpy._validators import assert_positive
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_spatial_basis,
    _save_data,
)
from squidpy.gr.neighbors import (
    DelaunayBuilder,
    GraphBuilder,
    GridBuilder,
    KNNBuilder,
    RadiusBuilder,
)

__all__ = [
    "SpatialNeighborsResult",
    "spatial_neighbors",
    "spatial_neighbors_knn",
    "spatial_neighbors_radius",
    "spatial_neighbors_delaunay",
    "spatial_neighbors_grid",
]


class SpatialNeighborsResult(NamedTuple):
    """Result of spatial_neighbors function."""

    connectivities: csr_matrix
    distances: csr_matrix


def _validate_no_legacy_params(**kwargs: Any) -> None:
    conflicts = [k for k, v in kwargs.items() if v is not None]
    if conflicts:
        raise ValueError(
            "When `builder` is provided, graph-construction arguments must not be set. "
            f"Got non-default values for: {', '.join(conflicts)}."
        )


def _resolve_graph_builder(
    *,
    coord_type: str | CoordType | None,
    n_neighs: int,
    radius: float | tuple[float, float] | None,
    delaunay: bool,
    n_rings: int,
    percentile: float | None,
    transform: str | Transform | None,
    set_diag: bool,
    has_spatial_uns: bool = False,
) -> GraphBuilder:
    transform = Transform.NONE if transform is None else Transform(transform)
    if coord_type is None:
        if radius is not None:
            logg.warning(
                f"Graph creation with `radius` is only available when `coord_type = {CoordType.GENERIC!r}` specified. "
                f"Ignoring parameter `radius = {radius}`."
            )
        coord_type = CoordType.GRID if has_spatial_uns else CoordType.GENERIC
    else:
        coord_type = CoordType(coord_type)

    common: dict[str, Any] = {
        "n_neighs": n_neighs,
        "transform": transform,
        "set_diag": set_diag,
    }

    if coord_type == CoordType.GRID:
        if percentile is not None:
            raise ValueError(
                "`percentile` is not supported for grid coordinates. It only applies to generic (non-grid) graphs."
            )
        return GridBuilder(**common, n_rings=n_rings, delaunay=delaunay)
    if delaunay:
        return DelaunayBuilder(**common, radius=radius, percentile=percentile)
    if radius is not None:
        return RadiusBuilder(**common, radius=radius, percentile=percentile)
    return KNNBuilder(**common, percentile=percentile)


def _resolve_spatial_data(
    adata: AnnData | SpatialData,
    *,
    spatial_key: str,
    elements_to_coordinate_systems: dict[str, str] | None,
    table_key: str | None,
    library_key: str | None,
) -> tuple[AnnData, str | None]:
    """Resolve SpatialData to AnnData, returning (adata, library_key)."""
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
    return adata, library_key


@d.dedent
@inject_docs(t=Transform, c=CoordType)
def spatial_neighbors(
    adata: AnnData | SpatialData,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    coord_type: str | CoordType | None = None,
    n_neighs: int | None = None,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool | None = None,
    n_rings: int | None = None,
    percentile: float | None = None,
    transform: str | Transform | None = None,
    set_diag: bool | None = None,
    builder: GraphBuilder | None = None,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """
    Create a graph from spatial coordinates.

    .. deprecated:: 1.6.0
        The flat-parameter API of ``spatial_neighbors`` is deprecated and will
        be removed in squidpy v1.7.0.  Use one of the mode-specific functions
        instead:

        - :func:`spatial_neighbors_knn`
        - :func:`spatial_neighbors_radius`
        - :func:`spatial_neighbors_delaunay`
        - :func:`spatial_neighbors_grid`

        Passing a ``builder`` instance directly remains supported.

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
    %(library_key)s
    coord_type
        Type of coordinate system. Must not be set when ``builder`` is given.
        Valid options are:

            - `{c.GRID.s!r}` - grid coordinates.
            - `{c.GENERIC.s!r}` - generic coordinates.
            - `None` - `{c.GRID.s!r}` if ``spatial_key`` is in :attr:`anndata.AnnData.uns`
              with ``n_neighs = 6`` (Visium), otherwise use `{c.GENERIC.s!r}`.
    n_neighs
        Depending on the ``coord_type``:

            - `{c.GRID.s!r}` - number of neighboring tiles.
            - `{c.GENERIC.s!r}` - number of neighborhoods for non-grid data. Only used when ``delaunay = False``.

        Defaults to ``6`` when no ``builder`` is provided. Must not be set when ``builder`` is given.
    radius
        Only available when ``coord_type = {c.GENERIC.s!r}``. Must not be set when ``builder`` is given.
        Depending on the type:

            - :class:`float` - compute the graph based on neighborhood radius.
            - :class:`tuple` - prune the final graph to only contain edges in interval `[min(radius), max(radius)]`.
    delaunay
        Whether to compute the graph from Delaunay triangulation. Only used when ``coord_type = {c.GENERIC.s!r}``.
        Defaults to ``False`` when no ``builder`` is provided. Must not be set when ``builder`` is given.
    n_rings
        Number of rings of neighbors for grid data. Only used when ``coord_type = {c.GRID.s!r}``.
        Defaults to ``1`` when no ``builder`` is provided. Must not be set when ``builder`` is given.
    percentile
        Percentile of the distances to use as threshold. Only used when ``coord_type = {c.GENERIC.s!r}``.
        Must not be set when ``builder`` is given.
    transform
        Type of adjacency matrix transform. Must not be set when ``builder`` is given.
        Valid options are:

            - `{t.SPECTRAL.s!r}` - spectral transformation of the adjacency matrix.
            - `{t.COSINE.s!r}` - cosine transformation of the adjacency matrix.
            - `{t.NONE.v}` - no transformation of the adjacency matrix.
    set_diag
        Whether to set the diagonal of the spatial connectivities to `1.0`.
        Defaults to ``False`` when no ``builder`` is provided. Must not be set when ``builder`` is given.
    builder
        Advanced graph construction strategy. When provided, all other graph-construction
        arguments (``coord_type``, ``n_neighs``, ``radius``, ``delaunay``, ``n_rings``,
        ``percentile``, ``transform``, ``set_diag``) must be left as ``None``.
    key_added
        Key which controls where the results are saved if ``copy = False``.
    %(copy)s

    Notes
    -----
    ``spatial_neighbors`` has 4 graph-construction modes:

        - Grid mode:
          ``coord_type='grid'``. Uses ``n_neighs`` and ``n_rings``.
          ``radius`` is ignored. ``delaunay`` is forwarded to the
          underlying grid connectivity builder. This is the mode used
          for Visium-like grid coordinates.
        - Generic k-nearest-neighbor mode:
          ``coord_type='generic'``, ``delaunay=False``, ``radius=None``.
          Uses ``n_neighs``.
        - Generic radius mode:
          ``coord_type='generic'``, ``delaunay=False``, ``radius`` set.
          Uses ``radius`` and builds a radius-based neighbor graph.
          If ``radius`` is a tuple, the graph is built with the maximum
          radius and then pruned to the interval
          ``[min(radius), max(radius)]``.
        - Generic Delaunay mode:
          ``coord_type='generic'``, ``delaunay=True``.
          Builds a Delaunay triangulation graph. ``n_neighs`` is
          ignored by the triangulation and passing it is deprecated.
          If ``radius`` is a tuple, it is used only as a
          post-construction pruning interval.

    Across these modes:

        - ``percentile`` only affects generic graphs.
        - ``transform`` and ``set_diag`` apply to all modes.
        - If ``builder`` is provided, it determines the mode directly.
          All other graph-construction arguments must be left as
          ``None``.
        - By default, observations are not treated as their own
          neighbors. The distance matrix always has a zero diagonal.
          The connectivity matrix only gets a nonzero diagonal when
          ``set_diag=True``.

    Argument precedence
    -------------------
    When ``builder`` is not provided, the mode is resolved as follows:

        - If ``coord_type`` resolves to ``'grid'``, grid mode is used.
          In that case ``radius`` is ignored.
        - Otherwise, if ``delaunay=True``, Delaunay mode is used.
          ``n_neighs`` is ignored (deprecated).
          A tuple ``radius`` is only used afterward as a pruning
          interval. A scalar ``radius`` is ignored.
        - Otherwise, if ``radius`` is set, radius mode is used.
          In this mode ``n_neighs`` does not act as a second cutoff.
        - Otherwise, k-nearest-neighbor mode is used.

    Grid-specific behavior
    ----------------------
    Grid mode currently does not validate ``n_neighs`` to a fixed set
    such as ``{{4, 6}}``. Internally it first queries the
    ``n_neighs`` nearest candidates and then applies a distance-based
    correction tuned for grid-like coordinates. As a result:

        - values such as ``n_neighs=4`` and ``n_neighs=6`` are the
          intended square-grid and hex-grid choices, respectively;
        - other values are accepted for backward compatibility, but
          their geometric interpretation is not guaranteed to match a
          continuous ring on the grid;
        - no clockwise or other within-ring ordering is part of the
          public API.

    Returns
    -------
    If ``copy = True``, returns a :class:`~squidpy.gr.SpatialNeighborsResult` with the spatial connectivities and distances matrices.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_connectivities']`` - the spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_distances']`` - the spatial distances.
        - :attr:`anndata.AnnData.uns`  ``['{{key_added}}']`` - :class:`dict` containing parameters.
    """
    if builder is None:
        warnings.warn(
            "Calling `spatial_neighbors` without a `builder` argument is deprecated "
            "and will be removed in squidpy v1.7.0. Use one of the mode-specific "
            "functions instead: `spatial_neighbors_knn`, `spatial_neighbors_radius`, "
            "`spatial_neighbors_delaunay`, or `spatial_neighbors_grid`.",
            FutureWarning,
            stacklevel=2,
        )
    return _spatial_neighbors(
        adata,
        spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key,
        library_key=library_key,
        coord_type=coord_type,
        n_neighs=n_neighs,
        radius=radius,
        delaunay=delaunay,
        n_rings=n_rings,
        percentile=percentile,
        transform=transform,
        set_diag=set_diag,
        builder=builder,
        key_added=key_added,
        copy=copy,
    )


def _spatial_neighbors(
    adata: AnnData | SpatialData,
    *,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    coord_type: str | CoordType | None = None,
    n_neighs: int | None = None,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool | None = None,
    n_rings: int | None = None,
    percentile: float | None = None,
    transform: str | Transform | None = None,
    set_diag: bool | None = None,
    builder: GraphBuilder | None = None,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Internal implementation of spatial_neighbors (no deprecation warning)."""
    adata, library_key = _resolve_spatial_data(
        adata, spatial_key=spatial_key, elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key, library_key=library_key,
    )

    _assert_spatial_basis(adata, spatial_key)

    if builder is not None:
        _validate_no_legacy_params(
            coord_type=coord_type,
            n_neighs=n_neighs,
            radius=radius,
            delaunay=delaunay,
            n_rings=n_rings,
            percentile=percentile,
            transform=transform,
            set_diag=set_diag,
        )
    else:
        n_neighs = n_neighs if n_neighs is not None else 6
        delaunay = delaunay if delaunay is not None else False
        n_rings = n_rings if n_rings is not None else 1
        set_diag = set_diag if set_diag is not None else False

        assert_positive(n_rings, name="n_rings")
        assert_positive(n_neighs, name="n_neighs")

        builder = _resolve_graph_builder(
            coord_type=coord_type,
            n_neighs=n_neighs,
            radius=radius,
            delaunay=delaunay,
            n_rings=n_rings,
            percentile=percentile,
            transform=transform,
            set_diag=set_diag,
            has_spatial_uns=Key.uns.spatial in adata.uns,
        )

    return _run_spatial_neighbors(adata, builder=builder, spatial_key=spatial_key, library_key=library_key, key_added=key_added, copy=copy)


@d.dedent
def spatial_neighbors_knn(
    adata: AnnData | SpatialData,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    n_neighs: int = 6,
    percentile: float | None = None,
    transform: str | Transform | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Create a k-nearest-neighbor graph from spatial coordinates.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    n_neighs
        Number of nearest neighbors. Defaults to ``6``.
    %(graph_common_params)s
    %(copy)s

    Returns
    -------
    %(spatial_neighbors_returns)s
    """
    transform_enum = Transform.NONE if transform is None else Transform(transform)
    builder = KNNBuilder(
        n_neighs=n_neighs, percentile=percentile,
        transform=transform_enum, set_diag=set_diag,
    )
    return _spatial_neighbors(
        adata, spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key, library_key=library_key,
        builder=builder, key_added=key_added, copy=copy,
    )


@d.dedent
def spatial_neighbors_radius(
    adata: AnnData | SpatialData,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    radius: float | tuple[float, float] = 1.0,
    n_neighs: int = 6,
    percentile: float | None = None,
    transform: str | Transform | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Create a radius-based graph from spatial coordinates.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    radius
        Neighborhood radius.  If a :class:`tuple`, the graph is built with the
        maximum radius and then pruned to the interval ``[min(radius), max(radius)]``.
    n_neighs
        Number of nearest neighbors used internally by the radius graph builder.
        Defaults to ``6``.
    %(graph_common_params)s
    %(copy)s

    Returns
    -------
    %(spatial_neighbors_returns)s
    """
    transform_enum = Transform.NONE if transform is None else Transform(transform)
    builder = RadiusBuilder(
        n_neighs=n_neighs, radius=radius, percentile=percentile,
        transform=transform_enum, set_diag=set_diag,
    )
    return _spatial_neighbors(
        adata, spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key, library_key=library_key,
        builder=builder, key_added=key_added, copy=copy,
    )


@d.dedent
def spatial_neighbors_delaunay(
    adata: AnnData | SpatialData,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    radius: tuple[float, float] | None = None,
    percentile: float | None = None,
    transform: str | Transform | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Create a Delaunay triangulation graph from spatial coordinates.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    radius
        If a :class:`tuple`, used as a post-construction pruning interval
        ``[min(radius), max(radius)]``.
    %(graph_common_params)s
    %(copy)s

    Returns
    -------
    %(spatial_neighbors_returns)s
    """
    transform_enum = Transform.NONE if transform is None else Transform(transform)
    builder = DelaunayBuilder(
        radius=radius, percentile=percentile,
        transform=transform_enum, set_diag=set_diag,
    )
    return _spatial_neighbors(
        adata, spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key, library_key=library_key,
        builder=builder, key_added=key_added, copy=copy,
    )


@d.dedent
def spatial_neighbors_grid(
    adata: AnnData | SpatialData,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    n_neighs: int = 6,
    n_rings: int = 1,
    delaunay: bool = False,
    transform: str | Transform | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Create a grid-based graph from spatial coordinates.

    This is the mode used for Visium-like grid coordinates.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    n_neighs
        Number of neighboring tiles. Defaults to ``6``.
    n_rings
        Number of rings of neighbors. Defaults to ``1``.
    delaunay
        Whether to compute the grid graph from Delaunay triangulation.
    %(graph_common_params)s
    %(copy)s

    Returns
    -------
    %(spatial_neighbors_returns)s
    """
    assert_positive(n_rings, name="n_rings")
    assert_positive(n_neighs, name="n_neighs")
    transform_enum = Transform.NONE if transform is None else Transform(transform)
    builder = GridBuilder(
        n_neighs=n_neighs, n_rings=n_rings, delaunay=delaunay,
        transform=transform_enum, set_diag=set_diag,
    )
    return _spatial_neighbors(
        adata, spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key, library_key=library_key,
        builder=builder, key_added=key_added, copy=copy,
    )


def _run_spatial_neighbors(
    adata: AnnData,
    *,
    builder: GraphBuilder,
    spatial_key: str = Key.obsm.spatial,
    library_key: str | None = None,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Shared core: build the graph from a resolved builder and save results."""
    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        libs = adata.obs[library_key].cat.categories
        make_index_unique(adata.obs_names)
    else:
        libs = [None]

    start = logg.info(
        f"Creating graph using `{builder.coord_type}` coordinates and `{builder.transform}` transform and `{len(libs)}` libraries."
    )
    _build_fun = partial(
        _spatial_neighbor,
        spatial_key=spatial_key,
        builder=builder,
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
            "n_neighbors": getattr(builder, "n_neighs", 6),
            "coord_type": builder.coord_type.v,
            "radius": getattr(builder, "radius", None),
            "transform": builder.transform.v,
        },
    }

    if copy:
        return SpatialNeighborsResult(connectivities=Adj, distances=Dst)

    _save_data(adata, attr="obsp", key=conns_key, data=Adj)
    _save_data(adata, attr="obsp", key=dists_key, data=Dst, prefix=False)
    _save_data(adata, attr="uns", key=neighs_key, data=neighbors_dict, prefix=False, time=start)


def _spatial_neighbor(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    builder: GraphBuilder | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    if builder is None:
        raise ValueError("No graph builder was provided.")

    coords = adata.obsm[spatial_key]
    return builder.build(coords)


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
