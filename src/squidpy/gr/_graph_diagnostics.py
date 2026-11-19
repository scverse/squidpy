"""Diagnostics for spatial graphs: a narrow build-time warning and :func:`graph_summary`."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import KDTree
from spatialdata import SpatialData

from squidpy._compat import SKIP_OWN_FRAMES
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d
from squidpy._utils import NDArrayA
from squidpy.gr._utils import extract_adata_if_sdata
from squidpy.gr.neighbors import DelaunayBuilder, RadiusBuilder

__all__ = ["SpatialGraphWarning", "graph_summary"]

#: Build-time warning thresholds, applied per library. Calibrated on public data, see :class:`SpatialGraphWarning`.
ISOLATED_FRAC_THRESHOLD = 0.25
MEAN_DEGREE_THRESHOLD = 1.0
_MAX_LIBRARIES_IN_MESSAGE = 3
_LIBRARY_COLUMN_HINTS = ("library", "sample", "slide", "fov", "region", "batch")


class SpatialGraphWarning(UserWarning):
    """Warning for a spatial graph in which many cells have no neighbour.

    The ``spatial_neighbors_*`` functions and :func:`squidpy.gr.mask_graph` emit it when, in
    any library, at least 25% of the cells have no edge in either direction, or the mean number
    of neighbours is below 1. The graph build also emits it when ``adata.uns["spatial"]`` lists
    several libraries but no ``library_key`` was passed.
    A radius below the typical cell spacing, or coordinates in other units than the radius
    (e.g. pixels vs micrometres), are the usual causes. The message names the library, the
    radius and the median nearest-neighbour distance. :func:`squidpy.gr.graph_summary`
    reports the full numbers.

    The thresholds come from graphs built at 1.5x, 2x and 3x the median nearest-neighbour
    distance on seqFISH, MERFISH, Slide-seqV2, IMC, MIBI-TOF and sparse Xenium sections
    (subcutaneous adipose and lung). None of the calibration graphs reaches the thresholds,
    and sparse adipose tissue at 1.5x the nearest-neighbour distance comes closest.

    Silence it with the standard :mod:`warnings` filters::

        import warnings
        import squidpy as sq

        warnings.filterwarnings("ignore", category=sq.gr.SpatialGraphWarning)
    """


def _degrees(adj: csr_matrix) -> tuple[NDArrayA, NDArrayA]:
    """Return off-diagonal out- and in-degrees of a CSR adjacency matrix."""
    # ponytail: counts stored entries, so explicit zeros count as edges; squidpy builders eliminate them
    n = adj.shape[0]
    diag = (adj.diagonal() != 0).astype(np.int64)
    out_deg = np.diff(adj.indptr) - diag
    in_deg = np.bincount(adj.indices, minlength=n) - diag
    return out_deg, in_deg


def _nn_distances(coords: NDArrayA) -> NDArrayA:
    """Return the distance from each point to its nearest other point."""
    if len(coords) < 2:
        return np.full(len(coords), np.nan)
    dist, _ = KDTree(coords).query(coords, k=2)
    return dist[:, 1]


def _builder_radius(builder: Any) -> float | None:
    """Return the largest radius a builder prunes to, or ``None``."""
    if not isinstance(builder, RadiusBuilder | DelaunayBuilder) or builder.radius is None:
        return None
    return float(np.max(builder.radius))


def _check_graph(
    adj: csr_matrix,
    coords: NDArrayA,
    codes: NDArrayA,
    categories: Sequence[str],
    *,
    builder: Any,
    key_added: str,
) -> dict[str, Any]:
    """Warn when a freshly built graph is degenerate and return scalars for ``uns[key]['summary']``.

    ``codes`` gives each cell's library index into ``categories`` (all zeros without ``library_key``).
    """
    out_deg, in_deg = _degrees(adj)
    isolated = (out_deg == 0) & (in_deg == 0)
    n_lib = np.bincount(codes, minlength=len(categories))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_isolated = np.bincount(codes, weights=isolated, minlength=len(categories)) / n_lib
        mean_degree = np.bincount(codes, weights=out_deg, minlength=len(categories)) / n_lib

    bad = np.flatnonzero(
        (n_lib > 0) & ((frac_isolated >= ISOLATED_FRAC_THRESHOLD) | (mean_degree < MEAN_DEGREE_THRESHOLD))
    )

    # The nearest-neighbour query costs about a third of a radius build on 1M cells,
    # so it only runs for the libraries named in the warning.
    radius = _builder_radius(builder)
    nn_median = np.full(len(categories), np.nan)
    if radius is not None:
        for i in bad[:_MAX_LIBRARIES_IN_MESSAGE]:
            if n_lib[i] > 1:
                nn_median[i] = np.median(_nn_distances(coords[codes == i]))
    if len(bad):
        parts = []
        for i in bad[:_MAX_LIBRARIES_IN_MESSAGE]:
            lib = "" if len(categories) == 1 else f"library {categories[i]!r}: "
            part = f"{lib}{frac_isolated[i]:.1%} of cells have no neighbour (mean degree {mean_degree[i]:.2f})"
            if radius is not None and not np.isnan(nn_median[i]):
                part += f", radius {radius:g} vs median nearest-neighbour distance {nn_median[i]:.3g}"
            parts.append(part)
        if len(bad) > _MAX_LIBRARIES_IN_MESSAGE:
            parts.append(f"and {len(bad) - _MAX_LIBRARIES_IN_MESSAGE} more libraries")
        hint = (
            " Distances are in your coordinate units; check that the radius and the coordinates use the same units (e.g. pixels vs micrometres)."
            if radius is not None
            else ""
        )
        warnings.warn(
            f"Spatial graph {key_added!r}: {'; '.join(parts)}.{hint} See `squidpy.gr.graph_summary`.",
            SpatialGraphWarning,
            skip_file_prefixes=SKIP_OWN_FRAMES,
        )

    summary: dict[str, Any] = {
        "n_obs": int(adj.shape[0]),
        "n_edges": int(out_deg.sum()),
        "mean_degree": float(out_deg.mean()) if adj.shape[0] else 0.0,
        "n_isolated": int(isolated.sum()),
    }
    return summary


def _warn_multiple_libraries(adata: AnnData, library_key: str | None) -> None:
    spatial_uns = adata.uns.get(Key.uns.spatial)
    if library_key is None and isinstance(spatial_uns, dict) and len(spatial_uns) > 1:
        warnings.warn(
            f"`adata.uns[{Key.uns.spatial!r}]` lists {len(spatial_uns)} libraries, but no `library_key` was given, "
            "so the graph can connect cells from different libraries. Pass `library_key` to build one graph per library.",
            SpatialGraphWarning,
            skip_file_prefixes=SKIP_OWN_FRAMES,
        )


def _find_params(adata: AnnData, connectivity_key: str) -> dict[str, Any]:
    for value in adata.uns.values():
        if isinstance(value, dict) and value.get("connectivities_key") == connectivity_key:
            return dict(value.get("params", {}))
    return {}


def _possible_library_columns(adata: AnnData) -> list[str]:
    return [
        str(c)
        for c in adata.obs.columns
        if any(h in str(c).lower() for h in _LIBRARY_COLUMN_HINTS)
        and (isinstance(adata.obs[c].dtype, pd.CategoricalDtype) or adata.obs[c].dtype == object)
        and adata.obs[c].nunique() > 1
    ]


@d.dedent
def graph_summary(
    data: AnnData | SpatialData,
    *,
    connectivity_key: str = Key.obsp.spatial_conn(),
    spatial_key: str = Key.obsm.spatial,
    library_key: str | None = None,
    radii: Sequence[float] | None = None,
    table_key: str | None = None,
) -> pd.DataFrame:
    """Summarize a spatial graph per library.

    Use it after building a graph, or before choosing a radius, to compare the number of
    neighbours per cell with the spacing between cells. It works on any graph in
    :attr:`anndata.AnnData.obsp`, including graphs built outside squidpy, and it reports
    numbers without picking a radius.

    Parameters
    ----------
    data
        Annotated data object, or a :class:`spatialdata.SpatialData` object.
    connectivity_key
        Key in :attr:`anndata.AnnData.obsp` of the graph to summarize.
    %(spatial_key)s
    library_key
        Categorical column in :attr:`anndata.AnnData.obs` with one category per library.
        If `None`, the ``library_key`` stored when squidpy built the graph is used, if any.
        Cells with a missing library are left out of the per-library rows but counted in the
        ``"all"`` row.
    radii
        Radii at which to count neighbours directly from the coordinates, without building
        a graph. Each radius adds one row per library.
    %(table_key)s

    Returns
    -------
    A :class:`pandas.DataFrame` with one row per library and one row for all cells
    (``library == "all"``), plus one such row per radius in ``radii``. The ``radius`` column
    is `NaN` for rows that describe the stored graph. Columns:

        - ``n_obs``, ``n_edges``: cells and directed off-diagonal edges.
        - ``degree_mean``, ``degree_sd``, ``degree_median``, ``degree_min``, ``degree_max``:
          neighbours per cell (out-degree).
        - ``n_isolated``, ``frac_isolated``: cells with no edge in either direction.
        - ``n_in_degree_zero``: cells that are no cell's neighbour. In a directed kNN graph,
          an aggregation over neighbours never uses such a cell.
        - ``n_components``, ``largest_component_frac``: weakly connected components.
        - ``reciprocity``: fraction of edges whose reverse edge also exists.
        - ``nn_dist_median``: median distance to the nearest other cell, in the units of
          ``spatial_key``. squidpy doesn't know the units.
        - ``n_duplicate_coords``: cells that share their position with another cell.
        - ``n_cross_library_edges``: edges between libraries, only when ``library_key`` is set.

    Rows from ``radii`` fill ``degree_mean``, ``degree_sd``, ``n_isolated`` and
    ``frac_isolated``. If no ``library_key`` is known, ``df.attrs["possible_library_columns"]``
    lists :attr:`anndata.AnnData.obs` columns that look like library labels.

    Notes
    -----
    The graph build warns with :class:`~squidpy.gr.SpatialGraphWarning` only when cells have
    too few neighbours. A graph with too many neighbours is not flagged, e.g., coordinates
    stored in millimetres with a radius meant in micrometres, which connects each cell to
    thousands of others. Compare ``nn_dist_median`` with your radius to catch an over-connected
    graph.

    The function doesn't read the values stored in ``adata.uns[...]["summary"]`` at build
    time, because they go stale when cells are subset.

    Examples
    --------
    >>> import squidpy as sq
    >>> adata = sq.datasets.imc()
    >>> sq.gr.spatial_neighbors_radius(adata, radius=15.0)
    >>> sq.gr.graph_summary(adata, radii=[5, 10, 20])  # doctest: +SKIP
    """
    adata = extract_adata_if_sdata(data, table_key=table_key) if isinstance(data, SpatialData) else data
    if connectivity_key not in adata.obsp:
        raise KeyError(f"`{connectivity_key}` not found in `adata.obsp`.")
    adj = adata.obsp[connectivity_key]
    adj = csr_matrix(adj) if not issparse(adj) or adj.format != "csr" else adj

    params = _find_params(adata, connectivity_key)
    library_key = library_key if library_key is not None else params.get("library_key")
    if library_key is not None:
        libs = pd.Categorical(adata.obs[library_key])
        categories, codes = [str(c) for c in libs.categories], libs.codes.astype(np.int64)
    else:
        categories, codes = ["all"], np.zeros(adata.n_obs, dtype=np.int64)
    coords = np.asarray(adata.obsm[spatial_key], dtype=float) if spatial_key in adata.obsm else None

    out_deg, in_deg = _degrees(adj)
    offdiag = adj.copy()
    offdiag.setdiag(0)
    offdiag.eliminate_zeros()
    mutual = offdiag.multiply(offdiag.T)
    mutual_per_row = np.diff(csr_matrix(mutual).indptr)
    _, comp = connected_components(offdiag, directed=True, connection="weak")
    rows = np.repeat(np.arange(adj.shape[0]), np.diff(offdiag.indptr))
    cross = codes[rows] != codes[offdiag.indices]

    nn = np.full(adata.n_obs, np.nan)
    if coords is not None:
        for i in range(len(categories)):
            mask = codes == i
            nn[mask] = _nn_distances(coords[mask])

    def row(name: str, mask: NDArrayA) -> dict[str, Any]:
        n = int(mask.sum())
        deg, iso = out_deg[mask], (out_deg[mask] == 0) & (in_deg[mask] == 0)
        n_edges = int(deg.sum())
        sizes = np.bincount(comp[mask]) if n else np.array([0])
        res = {
            "library": name,
            "radius": np.nan,
            "n_obs": n,
            "n_edges": n_edges,
            "degree_mean": deg.mean() if n else np.nan,
            "degree_sd": deg.std() if n else np.nan,
            "degree_median": np.median(deg) if n else np.nan,
            "degree_min": deg.min() if n else np.nan,
            "degree_max": deg.max() if n else np.nan,
            "n_isolated": int(iso.sum()),
            "frac_isolated": iso.mean() if n else np.nan,
            "n_in_degree_zero": int((in_deg[mask] == 0).sum()),
            "n_components": int((sizes > 0).sum()),
            "largest_component_frac": sizes.max() / n if n else np.nan,
            "reciprocity": mutual_per_row[mask].sum() / n_edges if n_edges else np.nan,
            "nn_dist_median": np.nanmedian(nn[mask]) if n and not np.isnan(nn[mask]).all() else np.nan,
            "n_duplicate_coords": int((nn[mask] == 0).sum()),
        }
        if library_key is not None:
            res["n_cross_library_edges"] = int(cross[mask[rows]].sum())
        return res

    masks = [(c, codes == i) for i, c in enumerate(categories)]
    if len(categories) > 1 or library_key is not None:
        masks.append(("all", np.ones(adata.n_obs, dtype=bool)))
    records = [row(name, mask) for name, mask in masks]

    if radii is not None:
        if coords is None:
            raise KeyError(f"`radii` needs coordinates in `adata.obsm[{spatial_key!r}]`.")
        counts = {r: np.zeros(adata.n_obs, dtype=np.int64) for r in radii}
        for i in range(len(categories)):
            idx = np.flatnonzero(codes == i)
            if not len(idx):
                continue
            tree = KDTree(coords[idx])
            for r in radii:
                counts[r][idx] = tree.query_radius(coords[idx], r=r, count_only=True) - 1  # minus the cell itself
        for r in radii:
            for name, mask in masks:
                c = counts[r][mask]
                records.append(
                    {
                        "library": name,
                        "radius": float(r),
                        "n_obs": int(mask.sum()),
                        "degree_mean": c.mean() if len(c) else np.nan,
                        "degree_sd": c.std() if len(c) else np.nan,
                        "n_isolated": int((c == 0).sum()),
                        "frac_isolated": (c == 0).mean() if len(c) else np.nan,
                    }
                )

    df = pd.DataFrame.from_records(records)
    if library_key is None:
        df.attrs["possible_library_columns"] = _possible_library_columns(adata)
    return df
