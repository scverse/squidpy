"""Experimental feature extraction module.

Extracts per-cell features from segmentation masks using scikit-image
``regionprops`` and squidpy-specific metrics (summary statistics, GLCM
texture, colour histograms).  Large images are automatically tiled so
that each tile is processed independently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from spatialdata import SpatialData, rasterize
from spatialdata._logging import logger as logg
from spatialdata.models import TableModel, get_channel_names
from tqdm.auto import tqdm

from squidpy.experimental.im._tiling import (
    build_tile_specs,
    compute_cell_info,
    compute_cell_info_multiscale,
    compute_cell_info_tiled,
    extract_tile_lazy,
)

# ---------------------------------------------------------------------------
# Drop accounting
# ---------------------------------------------------------------------------


@dataclass
class DropReport:
    """Counters for cells/tiles excluded during a featurization run."""

    empty_tiles: int = 0

    def summary(self) -> str:
        if self.empty_tiles == 0:
            return "No empty tiles."
        return f"Skipped {self.empty_tiles} empty tile(s)."


__all__ = ["calculate_image_features"]

# ---------------------------------------------------------------------------
# Skimage property sets
# ---------------------------------------------------------------------------

_MASK_PROPS = frozenset(
    {
        "area",
        "area_filled",
        "area_convex",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "equivalent_diameter_area",
        "extent",
        "feret_diameter_max",
        "solidity",
        "euler_number",
        "centroid",
        "centroid_local",
        "perimeter",
        "perimeter_crofton",
        "inertia_tensor",
        "inertia_tensor_eigvals",
    }
)
_INTENSITY_PROPS = frozenset(
    {
        "intensity_max",
        "intensity_mean",
        "intensity_min",
        "intensity_std",
    }
)

# cp_measure flag names recognised by `_parse_features`.  These currently
# raise NotImplementedError so the error is "not implemented" rather than
# the confusing "unknown feature".
_CPMEASURE_FLAG_NAMES = frozenset(
    {
        "cpmeasure:intensity",
        "cpmeasure:sizeshape",
        "cpmeasure:texture",
        "cpmeasure:granularity",
        "cpmeasure:zernike",
        "cpmeasure:feret",
        "cpmeasure:radial",
        "cpmeasure:correlation",
        "cpmeasure:correlation_pearson",
        "cpmeasure:correlation_costes",
        "cpmeasure:correlation_manders_fold",
        "cpmeasure:correlation_rwc",
    }
)

# All known top-level feature group names (used for validation).
_ALL_FEATURES = (
    _CPMEASURE_FLAG_NAMES
    | {"skimage:label", "skimage:label+image"}
    | {"squidpy:summary", "squidpy:texture", "squidpy:color_hist"}
)


# ---------------------------------------------------------------------------
# Feature parsing
# ---------------------------------------------------------------------------


class _ParsedFeatures(NamedTuple):
    skimage_label_props: frozenset[str] | None
    skimage_intensity_props: frozenset[str] | None
    squidpy_summary: bool
    squidpy_texture: bool
    squidpy_color_hist: bool


def _parse_features(features: list[str] | str | None) -> _ParsedFeatures:
    """Parse user-facing feature names into structured config.

    ``features=None`` requires an explicit choice.  Any ``cpmeasure:*``
    flag raises ``NotImplementedError``.
    """
    if features is None:
        raise ValueError(
            "`features` must be specified explicitly.  "
            "Use e.g. `features=['skimage:label']` for skimage regionprops or "
            "`features=['squidpy:summary', 'squidpy:texture', 'squidpy:color_hist']` for squidpy-native features."
        )

    if isinstance(features, str):
        features = [features]

    label_props: set[str] | None = None
    intensity_props: set[str] | None = None
    sq_summary = False
    sq_texture = False
    sq_color_hist = False

    for f in features:
        if f in _CPMEASURE_FLAG_NAMES:
            raise NotImplementedError(f"cp_measure feature `{f}` is not yet implemented.")

        # skimage group-level
        if f == "skimage:label":
            if label_props is not None:
                raise ValueError("Mixing 'skimage:label' with 'skimage:label:<prop>' is ambiguous; pick one form.")
            label_props = set(_MASK_PROPS)
        elif f == "skimage:label+image":
            if intensity_props is not None:
                raise ValueError(
                    "Mixing 'skimage:label+image' with 'skimage:label+image:<prop>' is ambiguous; pick one form."
                )
            intensity_props = set(_INTENSITY_PROPS)

        # skimage fine-grained: "skimage:label:prop" or "skimage:label+image:prop"
        elif f.startswith("skimage:label:"):
            prop = f.split(":", 2)[2]
            if prop not in _MASK_PROPS:
                raise ValueError(f"Unknown skimage label property: '{prop}'. Available: {sorted(_MASK_PROPS)}")
            if label_props is not None and label_props >= _MASK_PROPS:
                raise ValueError("Mixing 'skimage:label' with 'skimage:label:<prop>' is ambiguous; pick one form.")
            label_props = (label_props or set()) | {prop}
        elif f.startswith("skimage:label+image:"):
            prop = f.split(":", 2)[2]
            if prop not in _INTENSITY_PROPS:
                raise ValueError(f"Unknown skimage intensity property: '{prop}'. Available: {sorted(_INTENSITY_PROPS)}")
            if intensity_props is not None and intensity_props >= _INTENSITY_PROPS:
                raise ValueError(
                    "Mixing 'skimage:label+image' with 'skimage:label+image:<prop>' is ambiguous; pick one form."
                )
            intensity_props = (intensity_props or set()) | {prop}

        # squidpy features
        elif f == "squidpy:summary":
            sq_summary = True
        elif f == "squidpy:texture":
            sq_texture = True
        elif f == "squidpy:color_hist":
            sq_color_hist = True

        else:
            # cp_measure flags get a specific NotImplementedError above; don't
            # advertise them in the "available" list since they always raise.
            supported = sorted(_ALL_FEATURES - _CPMEASURE_FLAG_NAMES)
            raise ValueError(
                f"Unknown feature: '{f}'. Available top-level features: {supported}, "
                f"or use 'skimage:label:property' / 'skimage:label+image:property' for individual properties."
            )

    return _ParsedFeatures(
        skimage_label_props=frozenset(label_props) if label_props else None,
        skimage_intensity_props=frozenset(intensity_props) if intensity_props else None,
        squidpy_summary=sq_summary,
        squidpy_texture=sq_texture,
        squidpy_color_hist=sq_color_hist,
    )


def _has_any_features(parsed: _ParsedFeatures) -> bool:
    return (
        parsed.skimage_label_props is not None
        or parsed.skimage_intensity_props is not None
        or parsed.squidpy_summary
        or parsed.squidpy_texture
        or parsed.squidpy_color_hist
    )


# ---------------------------------------------------------------------------
# Per-tile dispatcher
# ---------------------------------------------------------------------------


def _featurize_tile(
    tile_image: np.ndarray,
    tile_labels: np.ndarray,
    parsed: _ParsedFeatures,
    channel_names: list[str],
) -> pd.DataFrame:
    """Compute all requested features for a single tile.

    Parameters
    ----------
    tile_image
        ``(C, H, W)`` image tile.
    tile_labels
        ``(H, W)`` label tile with only owned cells.
    parsed
        Parsed feature configuration.
    channel_names
        Channel names for column naming.

    Returns
    -------
    DataFrame indexed by cell label ID with one column per feature.
    """
    cell_ids = np.unique(tile_labels)
    cell_ids = cell_ids[cell_ids != 0]
    if len(cell_ids) == 0:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []

    # --- skimage regionprops ---
    if parsed.skimage_label_props is not None or parsed.skimage_intensity_props is not None:
        df = _compute_skimage_features(
            tile_labels, tile_image, parsed.skimage_label_props, parsed.skimage_intensity_props, channel_names
        )
        if not df.empty:
            parts.append(df)

    # --- squidpy per-cell features ---
    if parsed.squidpy_summary or parsed.squidpy_texture or parsed.squidpy_color_hist:
        df = _compute_squidpy_per_cell(tile_labels, tile_image, parsed, channel_names)
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame(index=cell_ids)

    combined = pd.concat(parts, axis=1)
    combined = combined.reindex(cell_ids)
    return combined


# ---------------------------------------------------------------------------
# skimage regionprops
# ---------------------------------------------------------------------------


def _regionprops_to_row(region: Any, props: frozenset[str]) -> dict[str, float]:
    """Extract scalar features from a single regionprops object."""
    row: dict[str, float] = {}
    for prop in props:
        try:
            value = getattr(region, prop)
            arr = np.asarray(value)
            if arr.ndim == 0:
                row[prop] = float(arr)
            elif arr.ndim == 1:
                for i, v in enumerate(arr):
                    row[f"{prop}_{i}"] = float(v)
            elif arr.ndim == 2:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        row[f"{prop}_{i}x{j}"] = float(arr[i, j])
        except (ValueError, TypeError, AttributeError):
            continue
    return row


def _compute_skimage_features(
    labels: np.ndarray,
    image: np.ndarray,
    label_props: frozenset[str] | None,
    intensity_props: frozenset[str] | None,
    channel_names: list[str],
) -> pd.DataFrame:
    """Compute skimage regionprops features for all cells in a tile."""
    parts: list[pd.DataFrame] = []

    if label_props is not None:
        regions = measure.regionprops(labels)
        rows = {r.label: _regionprops_to_row(r, label_props) for r in regions}
        parts.append(pd.DataFrame.from_dict(rows, orient="index"))

    if intensity_props is not None:
        for ch_idx, ch_name in enumerate(channel_names):
            regions = measure.regionprops(labels, intensity_image=image[ch_idx])
            rows = {r.label: _regionprops_to_row(r, intensity_props) for r in regions}
            df = pd.DataFrame.from_dict(rows, orient="index")
            df = df.rename(columns=lambda c, _ch=ch_name: f"{c}_{_ch}")
            parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# squidpy per-cell features
# ---------------------------------------------------------------------------


def _compute_squidpy_per_cell(
    labels: np.ndarray,
    image: np.ndarray,
    parsed: _ParsedFeatures,
    channel_names: list[str],
) -> pd.DataFrame:
    """Compute squidpy features per cell within a tile."""
    regions = measure.regionprops(labels)
    n_channels = image.shape[0]
    rows: dict[int, dict[str, float]] = {}

    for region in regions:
        lid = region.label
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)
        cell_features: dict[str, float] = {}

        # Extract cell's bounding box from image
        img_crop = image[:, bbox[0] : bbox[2], bbox[1] : bbox[3]]
        mask_crop = labels[bbox[0] : bbox[2], bbox[1] : bbox[3]] == lid

        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx]
            ch_crop = img_crop[ch_idx].astype(np.float32)
            masked_vals = ch_crop[mask_crop]

            if len(masked_vals) == 0:
                continue

            if parsed.squidpy_summary:
                cell_features[f"summary_mean_{ch_name}"] = float(np.mean(masked_vals))
                cell_features[f"summary_std_{ch_name}"] = float(np.std(masked_vals))
                cell_features[f"summary_min_{ch_name}"] = float(np.min(masked_vals))
                cell_features[f"summary_max_{ch_name}"] = float(np.max(masked_vals))

            if parsed.squidpy_texture:
                cell_features.update(_glcm_features(ch_crop, mask_crop, ch_name))

            if parsed.squidpy_color_hist:
                cell_features.update(_histogram_features(masked_vals, ch_name))

        rows[lid] = cell_features

    return pd.DataFrame.from_dict(rows, orient="index")


def _glcm_features(channel_crop: np.ndarray, mask: np.ndarray, ch_name: str) -> dict[str, float]:
    """GLCM texture features for a single channel within a cell's bbox."""
    quant_levels = 32
    ch = channel_crop.copy()
    # Zero out non-cell pixels so they don't affect GLCM
    ch[~mask] = 0
    ch_min, ch_max = float(ch[mask].min()), float(ch[mask].max())
    if ch_max > ch_min:
        ch = (ch - ch_min) / (ch_max - ch_min)
    else:
        ch = np.zeros_like(ch)
    ch_q = np.clip((ch * (quant_levels - 1)).round().astype(np.uint8), 0, quant_levels - 1)
    ch_q[~mask] = 0

    try:
        glcm = graycomatrix(ch_q, distances=[1], angles=[0], levels=quant_levels, symmetric=True, normed=True)
        return {
            f"texture_contrast_{ch_name}": float(graycoprops(glcm, "contrast")[0, 0]),
            f"texture_dissimilarity_{ch_name}": float(graycoprops(glcm, "dissimilarity")[0, 0]),
            f"texture_homogeneity_{ch_name}": float(graycoprops(glcm, "homogeneity")[0, 0]),
            f"texture_energy_{ch_name}": float(graycoprops(glcm, "energy")[0, 0]),
            f"texture_ASM_{ch_name}": float(graycoprops(glcm, "ASM")[0, 0]),
            f"texture_correlation_{ch_name}": float(graycoprops(glcm, "correlation")[0, 0]),
        }
    except (ValueError, IndexError):
        return {}


def _histogram_features(masked_vals: np.ndarray, ch_name: str, bins: int = 16) -> dict[str, float]:
    """Per-cell intensity histogram features."""
    lo, hi = float(masked_vals.min()), float(masked_vals.max())
    hist, _ = np.histogram(masked_vals, bins=bins, range=(lo, hi if hi > lo else lo + 1))
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return {f"color_hist_bin{b}_{ch_name}": float(v) for b, v in enumerate(hist)}


# ---------------------------------------------------------------------------
# Input preparation (lazy — returns xarray DataArrays, not numpy)
# ---------------------------------------------------------------------------


def _resolve_da(node: xr.DataTree | xr.DataArray, scale: str | None) -> xr.DataArray:
    """Get a DataArray from a DataTree or single-scale element (stays lazy)."""
    if not isinstance(node, xr.DataTree):
        return node
    if scale is None:
        raise ValueError("Scale must be provided for DataTree data.")
    if scale not in node:
        raise ValueError(f"Scale '{scale}' not found. Available: {list(node.keys())}")
    return node[scale].ds["image"]


def _validate_inputs(
    sdata: SpatialData,
    image_key: str,
    labels_key: str | None,
    shapes_key: str | None,
    scale: str | None,
) -> None:
    """Run all input validation checks (no data loading)."""
    if image_key not in sdata.images:
        raise ValueError(f"Image key '{image_key}' not found, valid keys: {list(sdata.images.keys())}")
    if labels_key is None and shapes_key is None:
        raise ValueError("Provide either `labels_key` or `shapes_key`.")
    if labels_key is not None and shapes_key is not None:
        raise ValueError("Use either `labels_key` or `shapes_key`, not both.")
    if labels_key is not None and labels_key not in sdata.labels:
        raise ValueError(f"Labels key '{labels_key}' not found, valid keys: {list(sdata.labels.keys())}")
    if shapes_key is not None and shapes_key not in sdata.shapes:
        raise ValueError(f"Shapes key '{shapes_key}' not found, valid keys: {list(sdata.shapes.keys())}")
    if labels_key is not None and isinstance(sdata.labels[labels_key], xr.DataTree) and scale is None:
        raise ValueError("When using multi-scale labels, please specify the scale.")
    if isinstance(sdata.images[image_key], xr.DataTree) and scale is None:
        raise ValueError("When using multi-scale images, please specify the scale.")


def _prepare_lazy(
    sdata: SpatialData,
    image_key: str,
    labels_key: str | None,
    shapes_key: str | None,
    scale: str | None,
    channels: list[str] | None,
    align_mode: Literal["strict"],
) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    """Return lazy (dask-backed) image and labels DataArrays, plus channel names.

    Does NOT call ``.compute()`` — arrays stay lazy for on-demand tile reads.
    For the shapes→labels path, labels are materialized (rasterize returns
    an in-memory array) but wrapped in a DataArray for a uniform interface.
    """
    _validate_inputs(sdata, image_key, labels_key, shapes_key, scale)

    # Image DataArray (lazy)
    image_da = _resolve_da(sdata.images[image_key], scale)
    if "c" not in image_da.dims:
        image_da = image_da.expand_dims("c")

    # Labels DataArray (lazy for labels_key, materialized for shapes_key)
    if labels_key is not None:
        labels_da = _resolve_da(sdata.labels[labels_key], scale)
    else:
        logg.info("Converting shapes to labels.")
        img_shape = {d: image_da.sizes[d] for d in ("y", "x")}
        try:
            labels_result = rasterize(
                sdata.shapes[shapes_key],
                ["x", "y"],
                min_coordinate=[0, 0],
                max_coordinate=[img_shape["x"], img_shape["y"]],
                target_coordinate_system="global",
                target_unit_to_pixels=1.0,
                return_regions_as_labels=True,
            )
        except ValueError as e:
            raise ValueError(
                "Failed to rasterize shapes; geometries may be empty or unsupported. "
                "Filter out empty/non-polygon geometries or choose a different shapes_key."
            ) from e
        if isinstance(labels_result, xr.DataArray):
            labels_da = labels_result
        else:
            labels_da = xr.DataArray(np.asarray(labels_result), dims=["y", "x"])

    # Only strict, axis-aligned image/labels are supported.  The Literal narrows
    # align_mode statically; this guard catches callers passing the value
    # dynamically (e.g. from config).
    if align_mode != "strict":
        raise ValueError(f"`align_mode` must be 'strict'; got {align_mode!r}.")
    if labels_key is not None:
        if image_da.sizes.get("y") != labels_da.sizes.get("y") or image_da.sizes.get("x") != labels_da.sizes.get("x"):
            raise ValueError(
                f"Image (y={image_da.sizes.get('y')}, x={image_da.sizes.get('x')}) and labels "
                f"(y={labels_da.sizes.get('y')}, x={labels_da.sizes.get('x')}) have different "
                f"pixel grids.  Pre-align with `spatialdata.rasterize`."
            )

    # Resolve channel names through spatialdata's canonical accessor so we
    # honor c_coords set at parse time. Always cast to str.
    all_ch = [str(v) for v in get_channel_names(sdata.images[image_key])]
    if len(all_ch) != image_da.sizes["c"]:
        # Multiscale element where get_channel_names may report from a
        # different scale than image_da. Fall back to positional naming.
        all_ch = [str(i) for i in range(image_da.sizes["c"])]

    ch_names: list[str]
    if channels is not None:
        selected_idx: list[int] = []
        ch_names = []
        for ch in channels:
            if not isinstance(ch, str):
                raise TypeError(
                    f"channels must contain strings (channel names); got {type(ch).__name__} {ch!r}. "
                    f"Available channel names: {all_ch}."
                )
            if ch not in all_ch:
                raise ValueError(f"Channel '{ch}' not found. Available: {all_ch}")
            selected_idx.append(all_ch.index(ch))
            ch_names.append(ch)
        image_da = image_da.isel(c=selected_idx)
    else:
        ch_names = all_ch

    return image_da, labels_da, ch_names


def _compute_centroids(
    sdata: SpatialData,
    labels_key: str | None,
    labels_da: xr.DataArray,
    scale: str | None,
) -> dict:
    """Compute cell centroids using the most efficient strategy available."""
    # Multiscale labels → use coarsest scale
    if labels_key is not None and isinstance(sdata.labels[labels_key], xr.DataTree):
        logg.info("Computing centroids from coarse scale.")
        return compute_cell_info_multiscale(sdata.labels[labels_key], target_scale=scale or "scale0")

    # Small enough to fit in memory → direct regionprops
    n_pixels = labels_da.sizes.get("y", 1) * labels_da.sizes.get("x", 1)
    if n_pixels <= 4096 * 4096:
        lbl_np = labels_da.values
        if lbl_np.ndim > 2:
            lbl_np = lbl_np.squeeze()
        return compute_cell_info(lbl_np)

    # Large single-scale → tiled centroid computation
    logg.info("Computing centroids in tiled mode (large single-scale labels).")
    return compute_cell_info_tiled(labels_da)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def calculate_image_features(
    sdata: SpatialData,
    image_key: str,
    labels_key: str | None = None,
    shapes_key: str | None = None,
    scale: str | None = None,
    channels: list[str] | list[int] | None = None,
    features: list[str] | str | None = None,
    tile_size: int = 2048,
    overlap_margin: int | Literal["auto"] = "auto",
    align_mode: Literal["strict"] = "strict",
    adata_key_added: str = "morphology",
    invalid_as_zero: bool = True,
    n_jobs: int = 1,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Calculate per-cell features from segmentation masks.

    Uses scikit-image ``regionprops`` for morphological/intensity features
    and squidpy-specific per-cell metrics (summary statistics, GLCM texture,
    colour histograms).  Large images are automatically tiled into
    ``tile_size x tile_size`` chunks with overlap so that every cell is
    fully contained in exactly one tile.

    Parameters
    ----------
    sdata
        SpatialData object.
    image_key
        Key in ``sdata.images``.
    labels_key
        Key in ``sdata.labels`` with segmentation masks.
    shapes_key
        Key in ``sdata.shapes`` (rasterized to labels internally).
    scale
        Scale level for multi-scale data.
    channels
        Subset of channel names to use, matching those returned by
        :func:`spatialdata.models.get_channel_names`. ``None`` uses all
        channels. Integer indices are not accepted -- always pass names.
    features
        Which features to compute.  Required (``None`` is rejected).
        Accepts a list of strings:

        - ``"skimage:label"`` (all mask props), ``"skimage:label:area"``
          (single prop), ``"skimage:label+image"`` (all intensity props),
          ``"skimage:label+image:intensity_mean"`` (single prop)
        - ``"squidpy:summary"``, ``"squidpy:texture"``,
          ``"squidpy:color_hist"``

        ``cpmeasure:*`` flag names are recognised but currently raise
        ``NotImplementedError``.
    tile_size
        Side length of the tiling grid (pixels).
    overlap_margin
        Overlap around each tile to capture boundary cells.
        ``"auto"`` computes the minimum from the largest cell's bounding box.
    align_mode
        Only ``"strict"`` is supported: require image and labels to
        share the same pixel grid (same y/x sizes).  Raise otherwise.
    adata_key_added
        Key under which to store the result in ``sdata.tables``.
    invalid_as_zero
        Replace ``inf`` and ``NaN`` values with zero.
    n_jobs
        Number of parallel jobs for tile processing.
    inplace
        If ``True``, store result in ``sdata.tables``.  Otherwise return it.

    Returns
    -------
    :class:`~anndata.AnnData` when ``inplace=False``, otherwise ``None``.
    """
    # --- Parse & validate ---
    parsed = _parse_features(features)
    if not _has_any_features(parsed):
        raise ValueError("No valid features requested.")

    drop_report = DropReport()

    image_da, labels_da, channel_names = _prepare_lazy(
        sdata, image_key, labels_key, shapes_key, scale, channels, align_mode
    )

    # --- Warmup: compute centroids without materializing full arrays ---
    cell_info = _compute_centroids(sdata, labels_key, labels_da, scale)
    if not cell_info:
        logg.info(drop_report.summary())
        raise ValueError("No cells found in labels (all zeros).")

    H = int(labels_da.sizes.get("y", labels_da.shape[-2]))
    W = int(labels_da.sizes.get("x", labels_da.shape[-1]))

    # --- Tile ---
    specs = build_tile_specs((H, W), cell_info, tile_size=tile_size, overlap_margin=overlap_margin)
    total_tiles = len(specs)
    logg.info(f"Processing {total_tiles} tiles ({tile_size}x{tile_size}, margin={overlap_margin}).")

    # --- Process tiles (each worker materializes only its own ~2k x 2k crop) ---
    def _process_one(spec):
        tile_img, tile_lbl = extract_tile_lazy(image_da, labels_da, spec)
        return _featurize_tile(tile_img, tile_lbl, parsed, channel_names)

    log_every = max(1, total_tiles // 10)
    start_t = time.monotonic()
    tile_dfs: list[pd.DataFrame] = []
    results_iter = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator_unordered")(
        delayed(_process_one)(spec) for spec in specs
    )
    for done, df in enumerate(
        tqdm(results_iter, total=total_tiles, desc="Featurizing tiles", unit="tile"),
        start=1,
    ):
        if df.empty:
            drop_report.empty_tiles += 1
        else:
            tile_dfs.append(df)
        if done == 1 or done == total_tiles or done % log_every == 0:
            elapsed = time.monotonic() - start_t
            logg.info(f"Tile {done}/{total_tiles} done (elapsed {elapsed:.1f}s).")

    if not tile_dfs:
        logg.info(drop_report.summary())
        raise ValueError("No features computed for any tile.")

    # Sort by cell label for deterministic output.  inf/NaN handling happens
    # in one numpy pass below to avoid two extra full-table allocations.
    combined = pd.concat(tile_dfs, axis=0).sort_index()

    # --- Build AnnData ---
    # Exactly one of labels_key / shapes_key is set (enforced in _validate_inputs).
    region_key_value = labels_key or shapes_key

    arr = combined.to_numpy(dtype=np.float32, copy=True)
    if invalid_as_zero:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    adata = ad.AnnData(X=arr)
    adata.obs_names = [f"cell_{i}" for i in combined.index]
    adata.var_names = list(combined.columns)

    adata.uns["spatialdata_attrs"] = {
        "region": region_key_value,
        "region_key": "region",
        "instance_key": "label_id",
    }
    adata.obs["region"] = pd.Categorical.from_codes(np.zeros(len(adata), dtype=np.int8), categories=[region_key_value])

    if shapes_key is not None and len(sdata.shapes[shapes_key]) == len(adata):
        adata.obs["label_id"] = sdata.shapes[shapes_key].index.values
    else:
        adata.obs["label_id"] = combined.index.values

    logg.info(drop_report.summary())

    if inplace:
        sdata.tables[adata_key_added] = TableModel.parse(adata)
        return None
    return adata
