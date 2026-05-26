"""Materialise a stitched labels element from a stitch_tile_cuts result.

Companion to :func:`squidpy.experimental.tl.stitch_tile_cuts`.  Takes the
piece-to-group mapping from ``stitch_group_id`` in the QC table and writes
a new labels element where stitched pieces share a single ID.  The original
labels element is untouched.
"""

from __future__ import annotations

import copy as _copy
from collections.abc import Callable

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from scipy.ndimage import binary_closing
from skimage.measure import regionprops
from skimage.morphology import disk as morph_disk
from spatialdata._logging import logger as logg
from spatialdata.models import Labels2DModel, TableModel
from spatialdata.transformations import get_transformation

from squidpy.experimental.utils._labels import resolve_labels_array

__all__ = ["make_stitched_labels"]


def _build_lookup(adata_obs: pd.DataFrame, dtype: np.dtype) -> np.ndarray:
    """Build an int->int LUT from ``label_id`` to ``stitch_group_id``.

    LUT covers ``[0, max_label_id]``; unmapped indices keep their own value
    (identity), so background (0) and any cells absent from the QC table are
    preserved.

    Raises
    ------
    ValueError
        If ``stitch_group_id`` (or ``label_id``) values exceed the labels'
        dtype range -- silent truncation here would alias unrelated cells.
    """
    label_ids = adata_obs["label_id"].astype(np.int64).to_numpy()
    group_ids = adata_obs["stitch_group_id"].astype(np.int64).to_numpy()
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        worst = max(int(label_ids.max(initial=0)), int(group_ids.max(initial=0)))
        if worst > info.max:
            raise ValueError(
                f"label_id / stitch_group_id values up to {worst} exceed the labels "
                f"dtype range {dtype} (max {info.max}); cannot build a safe LUT."
            )
    max_id = int(label_ids.max(initial=0))
    lut = np.arange(max_id + 1, dtype=dtype)
    lut[label_ids] = group_ids.astype(dtype)
    return lut


def _apply_lut(labels_da: xr.DataArray, lut: np.ndarray) -> da.Array | np.ndarray:
    """Lazily remap a labels DataArray via ``np.take`` over its dask blocks.

    Returns a bare array (dask or numpy) rather than a DataArray so the caller
    can re-parse via Labels2DModel without colliding transformation metadata.
    """
    src = labels_da.data
    if isinstance(src, da.Array):
        return src.map_blocks(lambda block, _lut=lut: _lut[block], dtype=lut.dtype)
    return lut[np.asarray(src)]


def _join_stitched_labels(
    labels_arr: da.Array | np.ndarray,
    stitched_group_ids: set[int],
    close_radius: int = 3,
) -> np.ndarray:
    """Morphologically close gaps between pieces of each stitched group.

    The basic LUT remap leaves stitched groups as multi-component regions
    (the cut stripe between pieces stays at 0).  This pass fills only those
    background pixels that lie inside the closed convex-ish hull of each
    stitched group, so each group becomes a single connected component.
    Other cells' pixels are never overwritten.

    Forces materialisation of the labels array; cost is O(image_size) for
    the regionprops pass plus O(stitched_groups x bbox) for the per-group
    closing.  Both are bounded for typical workloads.
    """
    if hasattr(labels_arr, "compute"):
        out = np.asarray(labels_arr.compute())
    else:
        out = np.asarray(labels_arr).copy()
    if not stitched_group_ids:
        return out

    # Single regionprops pass to locate every label's bbox.
    bboxes = {int(r.label): r.bbox for r in regionprops(out)}
    pad = close_radius + 2
    H, W = out.shape[-2], out.shape[-1]
    structure = morph_disk(close_radius)

    for gid in stitched_group_ids:
        bbox = bboxes.get(int(gid))
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        r0 = max(r0 - pad, 0)
        c0 = max(c0 - pad, 0)
        r1 = min(r1 + pad, H)
        c1 = min(c1 + pad, W)
        crop = out[r0:r1, c0:c1]
        mask = crop == int(gid)
        if not mask.any():
            continue
        closed = binary_closing(mask, structure=structure)
        # Only fill genuine background pixels -- never overwrite another cell.
        fill = closed & ~mask & (crop == 0)
        if fill.any():
            crop[fill] = int(gid)
            out[r0:r1, c0:c1] = crop
    return out


_BUILTIN_STRATEGIES: dict[str, Callable[[pd.Series], object]] = {
    "sum": lambda s: s.sum(),
    "min": lambda s: s.min(),
    "max": lambda s: s.max(),
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "first": lambda s: s.iloc[0],
}

# Vectorised counterparts: ``f(block) -> 1-D array of length n_cols``.  Used
# in :func:`_aggregate_X` to avoid an O(groups*cols) Python loop when the
# user passes a built-in strategy name.  Callable strategies fall back to
# the per-column path.
_BUILTIN_X_REDUCERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sum": lambda b: b.sum(axis=0),
    "min": lambda b: b.min(axis=0),
    "max": lambda b: b.max(axis=0),
    "mean": lambda b: b.mean(axis=0),
    "median": lambda b: np.median(b, axis=0),
    "first": lambda b: b[0],
}

# Columns whose value is shared across all members of a stitch group; we always
# take the first member's value rather than aggregating.
_GROUP_INVARIANT_COLS = frozenset({"stitch_group_id", "is_stitched", "n_pieces", "stitch_confidence", "region"})


def _resolve_strategy(strategy: str | Callable[[pd.Series], object]) -> Callable[[pd.Series], object]:
    if callable(strategy):
        return strategy
    if strategy not in _BUILTIN_STRATEGIES:
        raise ValueError(
            f"Unknown merge_strategy {strategy!r}. Use one of {sorted(_BUILTIN_STRATEGIES)} or pass a callable."
        )
    return _BUILTIN_STRATEGIES[strategy]


def _aggregate_X(
    X,
    group_indices: list[np.ndarray],
    strategy: str | Callable[[pd.Series], object],
) -> np.ndarray:
    """Aggregate ``X`` row-blocks into one row per group, column-wise.

    Built-in string strategies use vectorised numpy reductions.  Callable
    strategies fall back to a per-column ``pd.Series`` loop -- slow on wide
    matrices but expressive.  Sparse input is densified once up front.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    n_groups = len(group_indices)
    out = np.empty((n_groups, X.shape[1]), dtype=X.dtype)
    reducer = None
    if isinstance(strategy, str) and strategy in _BUILTIN_X_REDUCERS:
        reducer = _BUILTIN_X_REDUCERS[strategy]

    if reducer is not None:
        for i, idx in enumerate(group_indices):
            out[i] = reducer(X[idx])
    else:
        # Resolve callable lazily so non-builtin strings raise upstream.
        strategy_fn = _resolve_strategy(strategy)
        for i, idx in enumerate(group_indices):
            block = X[idx]
            for c in range(X.shape[1]):
                out[i, c] = strategy_fn(pd.Series(block[:, c]))
    return out


def _collapse_groups(
    adata: ad.AnnData,
    new_labels_key: str,
    merge_strategy: str | Callable[[pd.Series], object],
) -> ad.AnnData:
    """Collapse each stitch group into a single row.

    Output has one row per unique ``stitch_group_id``: unstitched cells (their
    own group) keep their row unchanged, stitched groups (size 2-4) collapse
    via ``merge_strategy``.  ``.obs`` columns, ``.uns``, ``.var`` and ``.X``
    are preserved/aggregated; ``spatialdata_attrs`` and the ``region`` column
    are rewritten to point at the new labels element.

    Aggregation rules:
    - ``label_id``: rewritten to the group id (matches new labels element).
    - ``stitch_group_id``, ``is_stitched``, ``n_pieces``, ``stitch_confidence``,
      ``region``: members agree -> first value.
    - Other numeric obs columns and all of ``X``: ``merge_strategy`` (default
      ``"sum"``).  Built-ins: ``sum``, ``min``, ``max``, ``mean``, ``median``,
      ``first``.  A callable receives a :class:`pandas.Series` and returns a
      scalar; it's applied column-wise to both ``.obs`` and ``.X``.
    - Non-numeric obs columns: ``"first"`` regardless of ``merge_strategy``
      (sum/mean don't make sense for strings/categoricals).

    Note: ``merge_strategy="sum"`` is the right default for additive features
    (area, intensity, count) but wrong for centroids, scores, fractions.
    Override accordingly for those.

    .. warning::
        ``.obsm``, ``.obsp``, ``.layers`` are passed through but not
        aggregated.  If their row dimensions become inconsistent with the new
        ``n_obs``, downstream tools may complain.  Drop them if not needed.
    """
    obs = adata.obs
    if "stitch_group_id" not in obs.columns:
        raise ValueError("AnnData missing 'stitch_group_id'; run stitch_tile_cuts first.")
    if "label_id" not in obs.columns:
        raise ValueError("AnnData missing 'label_id'.")

    strategy_fn = _resolve_strategy(merge_strategy)
    group_ids = obs["stitch_group_id"].astype(int).to_numpy()
    unique_groups = np.unique(group_ids)
    # Positional indices per group (preserves order of first appearance via sort).
    indices_by_group: list[np.ndarray] = [np.where(group_ids == g)[0] for g in unique_groups]

    # ---- Aggregate obs ----
    out_rows: dict[str, list] = {col: [] for col in obs.columns}
    for gid, idx in zip(unique_groups, indices_by_group, strict=True):
        members = obs.iloc[idx]
        for col in obs.columns:
            series = members[col]
            if col == "label_id":
                out_rows[col].append(int(gid))
            elif col in _GROUP_INVARIANT_COLS or not pd.api.types.is_numeric_dtype(series):
                out_rows[col].append(series.iloc[0])
            else:
                out_rows[col].append(strategy_fn(series))

    new_obs = pd.DataFrame(out_rows)
    # Preserve dtypes where possible (groupby+agg sometimes loses categorical).
    for col in new_obs.columns:
        try:
            new_obs[col] = new_obs[col].astype(obs[col].dtype)
        except (TypeError, ValueError):
            pass
    # Update the region column to point at the new labels element.
    if "region" in new_obs.columns:
        new_obs["region"] = pd.Categorical([new_labels_key] * len(new_obs))
    new_obs.index = [f"group_{gid}" for gid in unique_groups]

    # ---- Aggregate X ----
    if adata.X is not None and adata.X.shape[1] > 0:
        new_X = _aggregate_X(adata.X, indices_by_group, merge_strategy)
    else:
        new_X = np.empty((len(unique_groups), 0), dtype=np.float32)

    # ---- Preserve var / uns / pass-through obsm-style fields ----
    new_uns = _copy.deepcopy(dict(adata.uns))
    new_uns["spatialdata_attrs"] = {
        "region": new_labels_key,
        "region_key": "region",
        "instance_key": "label_id",
    }
    out = ad.AnnData(X=new_X, obs=new_obs, var=adata.var.copy(), uns=new_uns)

    # Warn if there are row-dimensioned fields we didn't aggregate; user can
    # decide whether to drop them.
    skipped = [name for name in ("obsm", "obsp", "layers") if getattr(adata, name, None)]
    if skipped:
        logg.warning(
            f"AnnData has {skipped}; these were not aggregated and the "
            "resulting table omits them.  Pass them through manually if needed."
        )

    return out


def make_stitched_labels(
    sdata: sd.SpatialData,
    labels_key: str,
    qc_table_key: str | None = None,
    labels_key_added: str | None = None,
    table_key_added: str | None = None,
    write_table: bool = True,
    merge_strategy: str | Callable[[pd.Series], object] = "sum",
    join_labels: bool = False,
    join_close_radius: int = 3,
    inplace: bool = True,
) -> dict[str, object] | None:
    """Materialise a stitched labels element from a stitch_tile_cuts result.

    Reads the ``stitch_group_id`` mapping in the QC table, builds a lazy
    int->int LUT, and registers a new labels element where each stitched
    group shares a single ID.  The original labels element is **not**
    modified.

    Optionally also writes a companion AnnData (``write_table=True``) with one
    row per unique ``stitch_group_id`` -- unstitched cells keep their row
    unchanged, stitched groups (size 2-4) collapse via ``merge_strategy``.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` with a labels element and a QC
        table that has been processed by
        :func:`squidpy.experimental.tl.stitch_tile_cuts`.
    labels_key
        Key in ``sdata.labels`` of the original labels element.
    qc_table_key
        Key of the QC table.  Defaults to ``"{labels_key}_qc"``.
    labels_key_added
        Key for the new labels element.  Defaults to
        ``"{labels_key}_stitched"``.  Existing element at this key is
        overwritten with a warning.
    table_key_added
        Key for the optional collapsed AnnData (one row per unique
        ``stitch_group_id``).  Defaults to ``"{labels_key_added}_table"``
        (must differ from the labels element key -- SpatialData requires
        unique names across element types).
    write_table
        If ``True``, also write the collapsed AnnData to
        ``sdata.tables[table_key_added]``.
    merge_strategy
        How to aggregate numeric ``.obs`` columns and ``.X`` across the
        2-4 pieces of each stitched cell.  String options:
        ``"sum"`` (default), ``"min"``, ``"max"``, ``"mean"``, ``"median"``,
        ``"first"``.  Callable: receives a :class:`pandas.Series` (one
        column of one group's members) and returns a scalar; applied
        column-wise.

        ``"sum"`` is the right default for additive features (area,
        intensity); for centroids, scores, or fractions, override with
        ``"mean"`` or pass a callable.

        Two classes of columns are **always** taken from the first member
        regardless of ``merge_strategy`` (including callables):

        - Group-invariant columns -- ``stitch_group_id``, ``is_stitched``,
          ``n_pieces``, ``stitch_confidence``, ``region`` -- because every
          member of a group already shares the same value.
        - Non-numeric columns (strings, categoricals, booleans) -- because
          ``sum`` / ``mean`` / etc. don't have a meaningful interpretation.
    join_labels
        If ``True``, morphologically close the gap between pieces of each
        stitched group so the resulting labels are single connected
        components instead of multi-component regions sharing an ID.  Only
        background pixels inside each group's closed hull are filled;
        other cells are never overwritten.  **Forces materialisation of
        the labels array** -- cost is O(image_size) plus O(stitched x bbox).
        Default ``False`` preserves the original gap pixels.
    join_close_radius
        Radius (px) of the disk structuring element used when
        ``join_labels=True``.  Default ``3`` matches the closing radius
        used during scoring; raise it if pieces remain disconnected after
        joining.
    inplace
        If ``True`` (default), write the new labels element (and table when
        ``write_table=True``) into ``sdata``.  If ``False``, return the
        materialised objects in a dict ``{"labels": ..., "table": ...}``
        without mutating ``sdata``; ``"table"`` is ``None`` when
        ``write_table=False``.
    """
    if labels_key not in sdata.labels:
        raise ValueError(f"Labels key '{labels_key}' not found in sdata.labels.")
    table_key = qc_table_key if qc_table_key is not None else f"{labels_key}_qc"
    if table_key not in sdata.tables:
        raise ValueError(f"QC table '{table_key}' not found in sdata.tables.")
    adata = sdata.tables[table_key]
    if "stitch_group_id" not in adata.obs.columns:
        raise ValueError(
            f"QC table '{table_key}' is missing 'stitch_group_id'; run squidpy.experimental.tl.stitch_tile_cuts first."
        )

    qc_params = adata.uns.get("tiling_qc", {})
    scale = qc_params.get("scale")
    labels_da = resolve_labels_array(sdata, labels_key, scale)

    lut = _build_lookup(adata.obs, labels_da.dtype)
    new_data = _apply_lut(labels_da, lut)
    if join_labels:
        stitched_gids = adata.obs.loc[adata.obs["is_stitched"].astype(bool), "stitch_group_id"].astype(int).unique()
        new_data = _join_stitched_labels(new_data, set(int(g) for g in stitched_gids), close_radius=join_close_radius)

    out_key = labels_key_added if labels_key_added is not None else f"{labels_key}_stitched"
    new_labels = Labels2DModel.parse(
        data=new_data,
        dims=("y", "x"),
        transformations=get_transformation(sdata.labels[labels_key], get_all=True),
    )
    new_table = None
    if write_table:
        collapsed = _collapse_groups(adata, out_key, merge_strategy)
        new_table = TableModel.parse(collapsed)

    if not inplace:
        return {"labels": new_labels, "table": new_table}

    if out_key in sdata.labels:
        logg.warning(f"Overwriting existing labels element '{out_key}'.")
    sdata.labels[out_key] = new_labels

    if new_table is not None:
        tbl_key = table_key_added if table_key_added is not None else f"{out_key}_table"
        if tbl_key in sdata.tables:
            logg.warning(f"Overwriting existing table '{tbl_key}'.")
        sdata.tables[tbl_key] = new_table
    return None
