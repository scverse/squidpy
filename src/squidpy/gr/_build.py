"""Functions for building graphs from spatial coordinates."""

from __future__ import annotations

import warnings
from typing import Any, Generic, Literal, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.utils import make_index_unique
from numba import njit
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
from squidpy._utils import NDArrayA, thread_map
from squidpy._validators import assert_positive
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_spatial_basis,
    _save_data,
    extract_adata_if_sdata,
)
from squidpy.gr.neighbors import (
    DelaunayBuilder,
    GraphBuilder,
    GraphMatrixT,
    GridBuilder,
    KNNBuilder,
    RadiusBuilder,
)

__all__ = [
    "GraphMatrixT",
    "SpatialNeighborsResult",
    "spatial_neighbors",
    "spatial_neighbors_from_builder",
    "spatial_neighbors_knn",
    "spatial_neighbors_radius",
    "spatial_neighbors_delaunay",
    "spatial_neighbors_grid",
]


class SpatialNeighborsResult(NamedTuple, Generic[GraphMatrixT]):
    """Result of spatial_neighbors function."""

    connectivities: GraphMatrixT
    distances: GraphMatrixT


def _resolve_graph_builder(
    *,
    coord_type: str | CoordType | None,
    n_neighs: int | None,
    radius: float | tuple[float, float] | None,
    delaunay: bool | None,
    n_rings: int | None,
    percentile: float | None,
    transform: str | Transform | None,
    set_diag: bool | None,
    has_spatial_uns: bool = False,
) -> GraphBuilder[Any, Any]:
    n_neighs_was_set = n_neighs is not None
    n_neighs = 6 if n_neighs is None else n_neighs
    delaunay = False if delaunay is None else delaunay
    n_rings = 1 if n_rings is None else n_rings
    set_diag = False if set_diag is None else set_diag

    assert_positive(n_rings, name="n_rings")
    assert_positive(n_neighs, name="n_neighs")

    transform = Transform.NONE if transform is None else Transform(transform)
    if coord_type is None:
        if radius is not None:
            logg.warning(
                "Graph creation with `radius` is only available for generic coordinates. "
                f"Ignoring parameter `radius = {radius}`."
            )
        coord_type = CoordType.GRID if has_spatial_uns else CoordType.GENERIC
    else:
        coord_type = CoordType(coord_type)

    common: dict[str, Any] = {"transform": transform, "set_diag": set_diag}

    if coord_type == CoordType.GRID:
        if percentile is not None:
            raise ValueError(
                "`percentile` is not supported for grid coordinates. It only applies to generic (non-grid) graphs."
            )
        return GridBuilder(n_neighs=n_neighs, **common, n_rings=n_rings, delaunay=delaunay)
    if delaunay:
        # TODO: below check should be removed when spatial_neighbors is deprecated
        if n_neighs_was_set:
            warnings.warn(
                "Parameter `n_neighs` is ignored when `delaunay=True` use `spatial_neighbors_delaunay` instead.",
                FutureWarning,
                stacklevel=3,
            )
        # Preserve the documented legacy contract: under the deprecated
        # `spatial_neighbors`, a scalar `radius` with `delaunay=True` is silently
        # ignored. The new `DelaunayBuilder(radius=r)` interprets a scalar as
        # the interval `(0.0, r)`, so we strip scalars here before delegating
        # and only forward tuple intervals to keep the legacy behavior intact.
        legacy_radius = radius if isinstance(radius, tuple) else None
        return DelaunayBuilder(**common, radius=legacy_radius, percentile=percentile)
    if radius is not None:
        # TODO: below check should be removed when spatial_neighbors is deprecated
        if n_neighs_was_set:
            warnings.warn(
                "Parameter `n_neighs` is ignored when `radius` is set use `spatial_neighbors_radius` instead.",
                FutureWarning,
                stacklevel=3,
            )
        return RadiusBuilder(**common, radius=radius, percentile=percentile)
    return KNNBuilder(n_neighs=n_neighs, **common, percentile=percentile)


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
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
    n_jobs: int = 1,
) -> SpatialNeighborsResult | None:
    """
    Create a graph from spatial coordinates.

    .. deprecated:: 1.7.0
        ``spatial_neighbors`` is deprecated and will be removed in squidpy
        v1.9.0. Use one of the mode-specific functions instead:

        - :func:`spatial_neighbors_knn`
        - :func:`spatial_neighbors_radius`
        - :func:`spatial_neighbors_delaunay`
        - :func:`spatial_neighbors_grid`
        - :func:`spatial_neighbors_from_builder`

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
        Type of coordinate system. Valid options are:

            - `{c.GRID.s!r}` - grid coordinates.
            - `{c.GENERIC.s!r}` - generic coordinates.
            - `None` - `{c.GRID.s!r}` if ``spatial_key`` is in :attr:`anndata.AnnData.uns`
              with ``n_neighs = 6`` (Visium), otherwise use `{c.GENERIC.s!r}`.
    n_neighs
        Depending on the ``coord_type``:

            - `{c.GRID.s!r}` - number of neighboring tiles.
            - `{c.GENERIC.s!r}` - number of neighborhoods for non-grid data. Only used when ``delaunay = False``.

        Defaults to ``6``.
    radius
        Only available when ``coord_type = {c.GENERIC.s!r}``.
        Depending on the type:

            - :class:`float` - compute the graph based on neighborhood radius.
            - :class:`tuple` - prune the final graph to only contain edges in interval `[min(radius), max(radius)]`.
    delaunay
        Whether to compute the graph from Delaunay triangulation. Only used when ``coord_type = {c.GENERIC.s!r}``.
        Defaults to ``False``.
    n_rings
        Number of rings of neighbors for grid data. Only used when ``coord_type = {c.GRID.s!r}``.
        Defaults to ``1``.
    percentile
        Percentile of the distances to use as threshold. Only used when ``coord_type = {c.GENERIC.s!r}``.
    transform
        Type of adjacency matrix transform.
        Valid options are:

            - `{t.SPECTRAL.s!r}` - spectral transformation of the adjacency matrix.
            - `{t.COSINE.s!r}` - cosine transformation of the adjacency matrix.
            - `{t.NONE.v}` - no transformation of the adjacency matrix.
    set_diag
        Whether to set the diagonal of the spatial connectivities to `1.0`.
    key_added
        Key which controls where the results are saved if ``copy = False``.
    %(copy)s
    %(n_jobs_libraries)s

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
          ``n_neighs`` is ignored and will throw a warning if passed.
          If ``radius`` is a tuple, the graph is built with the maximum
          radius and then pruned to the interval
          ``[min(radius), max(radius)]``.
        - Generic Delaunay mode:
          ``coord_type='generic'``, ``delaunay=True``.
          Builds a Delaunay triangulation graph. ``n_neighs`` is
          ignored by the triangulation and will throw a warning if passed.
          If ``radius`` is a tuple, it is used only as a
          post-construction pruning interval.

    Across these modes:

        - ``percentile`` only affects generic graphs.
        - ``transform`` and ``set_diag`` apply to all modes.
        - By default, observations are not treated as their own
          neighbors. The distance matrix always has a zero diagonal.
          The connectivity matrix only gets a nonzero diagonal when
          ``set_diag=True``.

    Argument precedence
    -------------------
    The mode is resolved as follows:

        - If ``coord_type`` resolves to ``'grid'``, grid mode is used.
          In that case ``radius`` is ignored.
        - Otherwise, if ``delaunay=True``, Delaunay mode is used.
          ``n_neighs`` is ignored (deprecated, removed in v1.9.0).
          A tuple ``radius`` is only used afterward as a pruning
          interval. A scalar ``radius`` is ignored.
        - Otherwise, if ``radius`` is set, radius mode is used.
          In this mode ``n_neighs`` is ignored (deprecated, removed in v1.9.0).
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
    warnings.warn(
        "Calling `spatial_neighbors` is deprecated and will be removed in squidpy "
        "v1.9.0. Use `spatial_neighbors_knn`, `spatial_neighbors_radius`, "
        "`spatial_neighbors_delaunay`, `spatial_neighbors_grid`, or "
        "`spatial_neighbors_from_builder` instead.",
        FutureWarning,
        stacklevel=2,
    )
    adata, library_key = _prepare_spatial_neighbors_input(
        adata,
        spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key,
        library_key=library_key,
    )
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

    return _run_spatial_neighbors(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
        n_jobs=n_jobs,
    )


def _resolve_data(
    data: AnnData | SpatialData,
    elements_to_coordinate_systems: dict[str, str] | None,
    table_key: str | None = None,
    spatial_key: str = Key.obsm.spatial,
    library_key: str | None = None,
) -> tuple[AnnData, str | None]:
    if not isinstance(data, SpatialData):
        return data, library_key

    assert elements_to_coordinate_systems is not None, (
        "Since input is a :class:`spatialdata.SpatialData`, `elements_to_coordinate_systems` must not be `None`."
    )
    table = extract_adata_if_sdata(data, table_key=table_key)
    elements, matched_table = match_element_to_table(data, list(elements_to_coordinate_systems), table_key)
    assert matched_table.obs_names.equals(table.obs_names), (
        "The spatialdata table must annotate all elements keys. Some elements are missing, please check the `elements_to_coordinate_systems` dictionary."
    )
    regions, region_key, instance_key = get_table_keys(table)
    regions = [regions] if isinstance(regions, str) else regions
    ordered_regions_in_table = table.obs[region_key].unique()

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
    if (not np.all(element_instances.values == table.obs[instance_key].values)) or (
        not np.all(ordered_regions_in_table == regions)
    ):
        raise ValueError(
            "The spatialdata table must annotate all elements keys. Some elements are missing or not ordered correctly, please check the `elements_to_coordinate_systems` dictionary."
        )
    centroids = []
    for region_ in ordered_regions_in_table:
        cs = elements_to_coordinate_systems[region_]
        centroid = get_centroids(data[region_], coordinate_system=cs)[["x", "y"]].compute()

        # TODO: remove this after https://github.com/scverse/spatialdata/issues/614
        if remove_centroids[region_]:
            centroid = centroid[1:].copy()
        centroids.append(centroid)

    table.obsm[spatial_key] = np.concatenate(centroids)
    return table, region_key


@d.dedent
def spatial_neighbors_from_builder(
    data: AnnData | SpatialData,
    builder: GraphBuilder[Any, Any],
    *,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    key_added: str = "spatial",
    copy: bool = False,
    n_jobs: int = 1,
) -> SpatialNeighborsResult | None:
    """Create a graph from spatial coordinates using an explicit builder instance.

    This function is the bridge between the high-level API (e.g.,
    :func:`spatial_neighbors_knn`, :func:`spatial_neighbors_radius`) and advanced
    customization via builder classes. Use this when you need to:

    - Stack or chain builder behaviors
    - Pass pre-configured builder instances multiple times
    - Implement custom builders (see :doc:`/extensibility`)

    Parameters
    ----------
    %(adata)s
    builder
        Graph construction strategy to execute. Built-in builders subclass
        {{class}}`~squidpy.gr.neighbors.GraphBuilderCSR`, while custom backends
        can implement the more generic
        {{class}}`~squidpy.gr.neighbors.GraphBuilder` interface directly.
        Reusable post-build operations are also exposed via
        :class:`~squidpy.gr.neighbors.DistanceIntervalPostprocessor`,
        :class:`~squidpy.gr.neighbors.PercentilePostprocessor`, and
        :class:`~squidpy.gr.neighbors.TransformPostprocessor`.
        Custom builders only need to implement multi-library support when using
        ``library_key``; otherwise leaving
        :meth:`~squidpy.gr.neighbors.GraphBuilder.combine` unimplemented is fine.
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    key_added
        Key which controls where the results are saved if ``copy = False``.
    %(copy)s
    %(n_jobs_libraries)s

    Returns
    -------
    %(spatial_neighbors_returns)s

    See Also
    --------
    spatial_neighbors_knn : k-nearest-neighbor graphs (wraps :class:`~squidpy.gr.neighbors.KNNBuilder`).
    spatial_neighbors_radius : radius-based graphs (wraps :class:`~squidpy.gr.neighbors.RadiusBuilder`).
    spatial_neighbors_delaunay : Delaunay triangulation graphs (wraps :class:`~squidpy.gr.neighbors.DelaunayBuilder`).
    spatial_neighbors_grid : grid-based graphs (wraps :class:`~squidpy.gr.neighbors.GridBuilder`).
    squidpy.gr.neighbors.GraphBuilder : Base builder interface. Inherit from this or :class:`~squidpy.gr.neighbors.GraphBuilderCSR` to implement custom graph construction.
    """
    adata, library_key = _prepare_spatial_neighbors_input(
        data,
        spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key,
        library_key=library_key,
    )
    return _run_spatial_neighbors(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
        n_jobs=n_jobs,
    )


def _prepare_spatial_neighbors_input(
    data: AnnData | SpatialData,
    *,
    spatial_key: str,
    elements_to_coordinate_systems: dict[str, str] | None,
    table_key: str | None,
    library_key: str | None,
) -> tuple[AnnData, str | None]:
    """Resolve input data and validate the requested spatial basis."""
    adata, library_key = _resolve_data(
        data,
        spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key,
        library_key=library_key,
    )
    _assert_spatial_basis(adata, spatial_key)
    return adata, library_key


@d.dedent
def spatial_neighbors_knn(
    data: AnnData | SpatialData,
    *,
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
    n_jobs: int = 1,
) -> SpatialNeighborsResult | None:
    """Create a k-nearest-neighbor graph from spatial coordinates.

    Each observation is connected to its ``n_neighs`` nearest observations in
    Euclidean space. This mode is typically most useful for continuous
    coordinates, where you want to control neighborhood size directly.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    n_neighs
        Number of nearest neighbors. Defaults to ``6``. Smaller values produce a
        sparser, more local graph; larger values connect broader neighborhoods.
    %(graph_common_params)s
    %(copy)s
    %(n_jobs_libraries)s

    Returns
    -------
    %(spatial_neighbors_returns)s

    See Also
    --------
    spatial_neighbors_from_builder : Use :class:`~squidpy.gr.neighbors.KNNBuilder` directly for advanced customization.
    squidpy.gr.neighbors.KNNBuilder : k-nearest-neighbor builder class.
    """
    transform_enum = Transform.NONE if transform is None else Transform(transform)
    builder = KNNBuilder(
        n_neighs=n_neighs,
        percentile=percentile,
        transform=transform_enum,
        set_diag=set_diag,
    )
    adata, library_key = _prepare_spatial_neighbors_input(
        data,
        spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key,
        library_key=library_key,
    )
    return _run_spatial_neighbors(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
        n_jobs=n_jobs,
    )


@d.dedent
def spatial_neighbors_radius(
    data: AnnData | SpatialData,
    *,
    radius: float | tuple[float, float],
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    percentile: float | None = None,
    transform: str | Transform | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
    n_jobs: int = 1,
) -> SpatialNeighborsResult | None:
    """Create a radius-based graph from spatial coordinates.

    Two observations are connected when their Euclidean distance falls within the
    requested radius. This mode is useful when a physical interaction scale is
    more meaningful than a fixed number of neighbors.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    radius
        Neighborhood radius.  If a :class:`tuple`, the graph is built with the
        maximum radius and then pruned to the interval ``[min(radius), max(radius)]``.
        In practice, a single value defines a disk around each observation,
        whereas a tuple defines an annulus by keeping only edges within the
        specified distance interval.
    %(graph_common_params)s
    %(copy)s
    %(n_jobs_libraries)s

    Returns
    -------
    %(spatial_neighbors_returns)s

    See Also
    --------
    spatial_neighbors_from_builder : Use :class:`~squidpy.gr.neighbors.RadiusBuilder` directly for advanced customization.
    squidpy.gr.neighbors.RadiusBuilder : radius-based builder class.
    """
    transform_enum = Transform.NONE if transform is None else Transform(transform)
    builder = RadiusBuilder(
        radius=radius,
        percentile=percentile,
        transform=transform_enum,
        set_diag=set_diag,
    )
    adata, library_key = _prepare_spatial_neighbors_input(
        data,
        spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key,
        library_key=library_key,
    )
    return _run_spatial_neighbors(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
        n_jobs=n_jobs,
    )


@d.dedent
def spatial_neighbors_delaunay(
    data: AnnData | SpatialData,
    *,
    spatial_key: str = Key.obsm.spatial,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    table_key: str | None = None,
    library_key: str | None = None,
    radius: float | tuple[float, float] | None = None,
    percentile: float | None = None,
    transform: str | Transform | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
    n_jobs: int = 1,
) -> SpatialNeighborsResult | None:
    """Create a Delaunay triangulation graph from spatial coordinates.

    Delaunay triangulation connects observations into triangles such that no
    other observation lies inside the circumcircle of each triangle. In
    practice, this yields an adaptive geometry-driven graph rather than one
    based on a fixed ``k`` or radius, and ``dst`` stores Euclidean edge lengths.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    radius
        Post-construction edge pruning of the Delaunay graph. The triangulation
        itself is never changed; only the resulting edges are filtered.

            - :class:`tuple` ``(min, max)``: keep edges with Euclidean length
              in ``[min(radius), max(radius)]``.
            - :class:`float` / :class:`int` ``r``: shorthand for ``(0.0, r)``,
              i.e. keep edges with length at most ``r``.
            - ``None``: keep every Delaunay edge.
    %(graph_common_params)s
    %(copy)s
    %(n_jobs_libraries)s

    Returns
    -------
    %(spatial_neighbors_returns)s

    See Also
    --------
    spatial_neighbors_from_builder : Use :class:`~squidpy.gr.neighbors.DelaunayBuilder` directly for advanced customization.
    squidpy.gr.neighbors.DelaunayBuilder : Delaunay triangulation builder class.
    """
    transform_enum = Transform.NONE if transform is None else Transform(transform)
    builder = DelaunayBuilder(
        radius=radius,
        percentile=percentile,
        transform=transform_enum,
        set_diag=set_diag,
    )
    adata, library_key = _prepare_spatial_neighbors_input(
        data,
        spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key,
        library_key=library_key,
    )
    return _run_spatial_neighbors(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
        n_jobs=n_jobs,
    )


@d.dedent
def spatial_neighbors_grid(
    data: AnnData | SpatialData,
    *,
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
    n_jobs: int = 1,
) -> SpatialNeighborsResult | None:
    """Create a grid-based graph from spatial coordinates.

    This is the mode used for Visium-like grid coordinates.
    It assumes observations lie on an approximately regular lattice, so it is
    usually not appropriate for continuous coordinates such as Xenium point
    clouds. On irregular coordinates, the resulting graph and ring distances may
    not have a meaningful grid interpretation.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(sdata_params)s
    %(library_key)s
    n_neighs
        Number of neighboring tiles used to form the base grid connectivity.
        Defaults to ``6`` for Visium-like grid coordinates.
    n_rings
        Number of rings of neighbors. Defaults to ``1``. ``n_rings=1`` keeps
        only immediate neighbors; larger values add progressively more distant
        shells and encode the shell number in ``dst``. For example,
        ``n_neighs=3`` with ``n_rings=2`` on a Visium-like grid starts from a
        sparse three-neighbor base graph and then adds a second graph-distance
        ring relative to that base connectivity.
    delaunay
        Whether to derive the base grid connectivity from a Delaunay triangulation.
        This is still grid mode: unlike :func:`spatial_neighbors_delaunay`, the
        resulting distance matrix encodes grid or ring distances rather than
        Euclidean edge lengths. In practice, this changes how the first-ring
        connectivity is inferred, but not the meaning of the resulting
        distances.
    %(graph_common_params)s
    %(copy)s
    %(n_jobs_libraries)s

    Returns
    -------
    %(spatial_neighbors_returns)s

    See Also
    --------
    spatial_neighbors_from_builder : Use :class:`~squidpy.gr.neighbors.GridBuilder` directly for advanced customization.
    squidpy.gr.neighbors.GridBuilder : grid-based builder class.
    """
    assert_positive(n_rings, name="n_rings")
    assert_positive(n_neighs, name="n_neighs")
    transform_enum = Transform.NONE if transform is None else Transform(transform)
    builder = GridBuilder(
        n_neighs=n_neighs,
        n_rings=n_rings,
        delaunay=delaunay,
        transform=transform_enum,
        set_diag=set_diag,
    )
    adata, library_key = _prepare_spatial_neighbors_input(
        data,
        spatial_key=spatial_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        table_key=table_key,
        library_key=library_key,
    )
    return _run_spatial_neighbors(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
        n_jobs=n_jobs,
    )


def _run_spatial_neighbors(
    adata: AnnData,
    builder: GraphBuilder[Any, Any],
    *,
    spatial_key: str = Key.obsm.spatial,
    library_key: str | None = None,
    key_added: str = "spatial",
    copy: bool = False,
    n_jobs: int = 1,
) -> SpatialNeighborsResult | None:
    """Shared core: build the graph from a resolved builder and save results."""
    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        libs = adata.obs[library_key].cat.categories
        make_index_unique(adata.obs_names)
    else:
        libs = [None]

    start = logg.info(f"Creating graph using `{builder.transform}` transform and `{len(libs)}` libraries.")
    if library_key is not None:
        # Extract the per-library coordinate arrays once.
        # Subsetting the full AnnData inside the loop recomputes the library mask repeatedly and
        # creates a view per library, which dominates the runtime for many cells.
        # Slicing the coordinate array by precomputed category codes is far cheaper and,
        # because the resulting arrays are small, makes the per-library graph construction cheap to parallelize.
        codes = adata.obs[library_key].cat.codes.to_numpy()
        coords = adata.obsm[spatial_key]
        per_lib_coords: list[np.ndarray] = []
        idxs: list[int] = []
        for code in range(len(libs)):
            idx = np.where(codes == code)[0]
            per_lib_coords.append(np.ascontiguousarray(coords[idx]))
            idxs.extend(idx.tolist())

        mats = thread_map(
            builder.build,
            per_lib_coords,
            n_jobs=n_jobs,
            unit="library",
        )
        adj, dst = builder.combine(mats, idxs)
    else:
        adj, dst = builder.build(adata.obsm[spatial_key])

    neighs_key = Key.uns.spatial_neighs(key_added)
    conns_key = Key.obsp.spatial_conn(key_added)
    dists_key = Key.obsp.spatial_dist(key_added)

    neighbors_dict = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": builder.uns_params(),
    }

    if copy:
        return SpatialNeighborsResult(connectivities=adj, distances=dst)

    _save_data(adata, attr="obsp", key=conns_key, data=adj)
    _save_data(adata, attr="obsp", key=dists_key, data=dst, prefix=False)
    _save_data(adata, attr="uns", key=neighs_key, data=neighbors_dict, prefix=False, time=start)
    return None


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

    table = extract_adata_if_sdata(sdata, table_key=table_key)
    coords = table.obsm[spatial_key]
    adj = table.obsp[conns_key]
    dst = table.obsp[dists_key]

    # convert edges to lines
    lines_coords, idx_out = _get_lines_coords(adj.indices, adj.indptr, coords)
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
    adj[filt_idx_out[:, 0], filt_idx_out[:, 1]] = 0
    adj.eliminate_zeros()

    # filter_distances
    dst[filt_idx_out[:, 0], filt_idx_out[:, 1]] = 0
    dst.eliminate_zeros()

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
        return adj, dst

    # save back to spatialdata
    _save_data(table, attr="obsp", key=mask_conns_key, data=adj)
    _save_data(table, attr="obsp", key=mask_dists_key, data=dst, prefix=False)
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
