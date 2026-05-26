"""Experimental feature extraction module.

Extracts per-cell features from segmentation masks using cp_measure,
scikit-image regionprops, and squidpy-specific metrics.  Large images
are automatically tiled so that each tile is processed independently.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Literal, NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
import xarray as xr
from cp_measure.featurizer import featurize, make_featurizer_config
from joblib import Parallel, delayed
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import relabel_sequential
from spatialdata import SpatialData, rasterize
from spatialdata._logging import logger as logg
from spatialdata.models import TableModel, get_channel_names
from spatialdata.transformations import get_transformation
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
    """Counters for cells that were excluded during a featurization run.

    Emitted once at the end of ``calculate_image_features`` so users know
    why their cell count shrank.
    """

    outside_image_extent: int = 0
    partial_at_image_boundary: int = 0
    cp_measure_no_data: int = 0
    empty_tiles: int = 0
    other: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["Cell drop report:"]
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, int) and v > 0:
                lines.append(f"  {f.name}: {v}")
            elif isinstance(v, dict):
                for k, vv in v.items():
                    if vv:
                        lines.append(f"  {k}: {vv}")
        if len(lines) == 1:
            return "Cell drop report: no cells dropped."
        return "\n".join(lines)


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

# cp_measure feature name → make_featurizer_config keyword(s)
_CPMEASURE_FLAGS: dict[str, dict[str, bool]] = {
    "cpmeasure:intensity": {"intensity": True},
    "cpmeasure:sizeshape": {"sizeshape": True},
    "cpmeasure:texture": {"texture": True},
    "cpmeasure:granularity": {"granularity": True},
    "cpmeasure:zernike": {"zernike": True},
    "cpmeasure:feret": {"feret": True},
    "cpmeasure:radial": {"radial_distribution": True, "radial_zernikes": True},
    "cpmeasure:correlation": {
        "correlation_pearson": True,
        "correlation_costes": True,
        "correlation_manders_fold": True,
        "correlation_rwc": True,
    },
    "cpmeasure:correlation_pearson": {"correlation_pearson": True},
    "cpmeasure:correlation_costes": {"correlation_costes": True},
    "cpmeasure:correlation_manders_fold": {"correlation_manders_fold": True},
    "cpmeasure:correlation_rwc": {"correlation_rwc": True},
}

# All known top-level feature group names (used for validation)
_ALL_FEATURES = (
    set(_CPMEASURE_FLAGS.keys())
    | {"skimage:label", "skimage:label+image"}
    | {"squidpy:summary", "squidpy:texture", "squidpy:color_hist"}
)


# ---------------------------------------------------------------------------
# Feature parsing
# ---------------------------------------------------------------------------


class _ParsedFeatures(NamedTuple):
    cp_flags: dict[str, bool] | None  # kwargs for make_featurizer_config
    skimage_label_props: frozenset[str] | None
    skimage_intensity_props: frozenset[str] | None
    squidpy_summary: bool
    squidpy_texture: bool
    squidpy_color_hist: bool


def _parse_features(features: list[str] | str | None) -> _ParsedFeatures:
    """Parse user-facing feature names into structured config."""
    if features is None:
        # Default: all cp_measure features
        return _ParsedFeatures(
            cp_flags={},  # empty dict → all defaults (all True)
            skimage_label_props=None,
            skimage_intensity_props=None,
            squidpy_summary=False,
            squidpy_texture=False,
            squidpy_color_hist=False,
        )

    if isinstance(features, str):
        features = [features]

    cp_flags: dict[str, bool] = {}
    has_any_cp = False
    label_props: set[str] | None = None
    intensity_props: set[str] | None = None
    sq_summary = False
    sq_texture = False
    sq_color_hist = False

    for f in features:
        # cp_measure features
        if f in _CPMEASURE_FLAGS:
            has_any_cp = True
            cp_flags.update(_CPMEASURE_FLAGS[f])

        # skimage group-level
        elif f == "skimage:label":
            label_props = set(_MASK_PROPS)
        elif f == "skimage:label+image":
            intensity_props = set(_INTENSITY_PROPS)

        # skimage fine-grained: "skimage:label:prop" or "skimage:label+image:prop"
        elif f.startswith("skimage:label:"):
            prop = f.split(":", 2)[2]
            if prop not in _MASK_PROPS:
                raise ValueError(f"Unknown skimage label property: '{prop}'. Available: {sorted(_MASK_PROPS)}")
            label_props = (label_props or set()) | {prop}
        elif f.startswith("skimage:label+image:"):
            prop = f.split(":", 2)[2]
            if prop not in _INTENSITY_PROPS:
                raise ValueError(f"Unknown skimage intensity property: '{prop}'. Available: {sorted(_INTENSITY_PROPS)}")
            intensity_props = (intensity_props or set()) | {prop}

        # squidpy features
        elif f == "squidpy:summary":
            sq_summary = True
        elif f == "squidpy:texture":
            sq_texture = True
        elif f == "squidpy:color_hist":
            sq_color_hist = True

        else:
            raise ValueError(
                f"Unknown feature: '{f}'. Available top-level features: {sorted(_ALL_FEATURES)}, "
                f"or use 'skimage:label:property' / 'skimage:label+image:property' for individual properties."
            )

    return _ParsedFeatures(
        cp_flags=cp_flags if has_any_cp else None,
        skimage_label_props=frozenset(label_props) if label_props else None,
        skimage_intensity_props=frozenset(intensity_props) if intensity_props else None,
        squidpy_summary=sq_summary,
        squidpy_texture=sq_texture,
        squidpy_color_hist=sq_color_hist,
    )


def _has_any_features(parsed: _ParsedFeatures) -> bool:
    return (
        parsed.cp_flags is not None
        or parsed.skimage_label_props is not None
        or parsed.skimage_intensity_props is not None
        or parsed.squidpy_summary
        or parsed.squidpy_texture
        or parsed.squidpy_color_hist
    )


# ---------------------------------------------------------------------------
# cp_measure config builder
# ---------------------------------------------------------------------------


def _build_cp_config(cp_flags: dict[str, bool], channel_names: list[str]) -> dict:
    """Build a cp_measure featurizer config from parsed flags.

    When ``cp_flags`` is empty (the default-all case), every feature is
    enabled.  Otherwise, only the explicitly requested features are turned on.
    """
    if not cp_flags:
        # All defaults (everything True)
        return make_featurizer_config(channel_names)

    # Start with everything off, then enable requested features
    all_off = {
        "intensity": False,
        "texture": False,
        "granularity": False,
        "radial_distribution": False,
        "radial_zernikes": False,
        "sizeshape": False,
        "zernike": False,
        "feret": False,
        "correlation_pearson": False,
        "correlation_costes": False,
        "correlation_manders_fold": False,
        "correlation_rwc": False,
    }
    all_off.update(cp_flags)
    return make_featurizer_config(channel_names, **all_off)


# ---------------------------------------------------------------------------
# Per-tile feature computation
# ---------------------------------------------------------------------------


def _featurize_tile(
    tile_image: np.ndarray,
    tile_labels: np.ndarray,
    parsed: _ParsedFeatures,
    channel_names: list[str],
    *,
    cp_config: dict | None = None,
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
    cp_config
        Pre-built cp_measure featurizer config. When ``None`` (default), the
        config is built locally from ``parsed.cp_flags``. ``calculate_image_features``
        builds it once and reuses it across tiles; direct callers can rely on
        the fallback.

    Returns
    -------
    DataFrame indexed by cell label ID with one column per feature.
    """
    cell_ids = np.unique(tile_labels)
    cell_ids = cell_ids[cell_ids != 0]
    if len(cell_ids) == 0:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []

    # --- cp_measure features ---
    if cp_config is None and parsed.cp_flags is not None:
        cp_config = _build_cp_config(parsed.cp_flags, channel_names)
    if cp_config is not None:
        # cp_measure assumes dense 1..N IDs and index-errors on sparse IDs.
        contiguous_labels, _, inverse = relabel_sequential(tile_labels)
        masks_3d = contiguous_labels[np.newaxis, :, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, columns, rows = featurize(tile_image, masks_3d, cp_config)
        if data.shape[0] > 0:
            # cp_measure may return more rows than data; trim and remap.
            row_labels = [int(inverse[r[2]]) for r in rows[: data.shape[0]]]
            cp_df = pd.DataFrame(data, index=row_labels, columns=columns)
            parts.append(cp_df)

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


# ---------------------------------------------------------------------------
# Coordinate-system aware alignment
# ---------------------------------------------------------------------------


def _shared_coordinate_system(sdata: SpatialData, image_key: str, labels_key: str) -> str:
    img_t = get_transformation(sdata.images[image_key], get_all=True)
    lbl_t = get_transformation(sdata.labels[labels_key], get_all=True)
    shared = set(img_t) & set(lbl_t)
    if not shared:
        raise ValueError(
            f"Image '{image_key}' and labels '{labels_key}' share no coordinate "
            f"system (image: {sorted(img_t)}, labels: {sorted(lbl_t)})."
        )
    return "global" if "global" in shared else sorted(shared)[0]


def _relative_affine(sdata: SpatialData, image_key: str, labels_key: str, cs: str) -> np.ndarray:
    """Return the 3x3 affine that maps labels-pixel-coords to image-pixel-coords.

    Uses ``(x, y)`` axis order to match :mod:`spatialdata` convention.
    """
    t_img = get_transformation(sdata.images[image_key], to_coordinate_system=cs)
    t_lbl = get_transformation(sdata.labels[labels_key], to_coordinate_system=cs)
    # image_pixel <- global <- labels_pixel
    m_img_to_global = t_img.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    m_lbl_to_global = t_lbl.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    m_global_to_img = np.linalg.inv(m_img_to_global)
    return m_global_to_img @ m_lbl_to_global


def _rasterize_to_image_grid(element: Any, image_da: xr.DataArray, cs: str) -> xr.DataArray:
    """Rasterize a spatialdata element onto an image DataArray's pixel grid."""
    logg.warning(
        f"Materializing element onto image grid via spatialdata.rasterize in '{cs}'. "
        f"Lazy behavior is lost for this run."
    )
    img_h = int(image_da.sizes["y"])
    img_w = int(image_da.sizes["x"])
    result = rasterize(
        element,
        ["x", "y"],
        min_coordinate=[0, 0],
        max_coordinate=[img_w, img_h],
        target_coordinate_system=cs,
        target_unit_to_pixels=1.0,
        return_regions_as_labels=True,
    )
    if isinstance(result, xr.DataArray):
        return result
    return xr.DataArray(np.asarray(result), dims=["y", "x"])


def _is_close_identity(m: np.ndarray, atol: float = 1e-6) -> bool:
    return bool(np.allclose(m, np.eye(m.shape[0]), atol=atol))


def _decompose_pixel_translation(m: np.ndarray, atol: float = 1e-6) -> tuple[int, int] | None:
    """If ``m`` is identity-plus-integer-translation, return ``(tx, ty)``; else None.

    ``m`` is a 3x3 affine in (x, y) axis order.
    """
    rotscale = m[:2, :2]
    if not np.allclose(rotscale, np.eye(2), atol=atol):
        return None
    tx, ty = float(m[0, 2]), float(m[1, 2])
    if not (abs(tx - round(tx)) < atol and abs(ty - round(ty)) < atol):
        return None
    return int(round(tx)), int(round(ty))


def _align_to_image_grid(
    sdata: SpatialData,
    image_key: str,
    labels_key: str,
    image_da: xr.DataArray,
    labels_da: xr.DataArray,
    align_mode: Literal["strict", "rasterize"],
    drop_report: DropReport,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Crop image and labels to their pixel-grid overlap, honoring transforms.

    See module docstring of concern 3 fix for full semantics. Mutates
    ``drop_report`` to count cells dropped because they fall outside the
    overlap rectangle.
    """
    cs = _shared_coordinate_system(sdata, image_key, labels_key)
    m = _relative_affine(sdata, image_key, labels_key, cs)

    # Integer-pixel offset of labels relative to image. (tx, ty) means
    # labels pixel (0, 0) lands at image pixel (tx, ty) in (x, y) order.
    if _is_close_identity(m):
        tx, ty = 0, 0
    elif (decomposed := _decompose_pixel_translation(m)) is not None:
        tx, ty = decomposed
    elif align_mode == "strict":
        raise ValueError(
            f"Labels not aligned to image pixel grid in coordinate system '{cs}'. "
            f"Relative affine (x,y) =\n{m}\n"
            f"Pass align_mode='rasterize' to resample labels onto the image grid "
            f"(via spatialdata.rasterize), or pre-align with spatialdata.rasterize "
            f"in your pipeline."
        )
    else:
        labels_da = _rasterize_to_image_grid(sdata.labels[labels_key], image_da, cs)
        tx, ty = 0, 0

    # Determine overlap rectangle in image-pixel coords.
    img_h = int(image_da.sizes["y"])
    img_w = int(image_da.sizes["x"])
    lbl_h = int(labels_da.sizes.get("y", labels_da.shape[-2]))
    lbl_w = int(labels_da.sizes.get("x", labels_da.shape[-1]))

    # Labels pixel (i_y, i_x) in label coords maps to image pixel (i_y+ty, i_x+tx).
    img_y0 = max(0, ty)
    img_x0 = max(0, tx)
    img_y1 = min(img_h, lbl_h + ty)
    img_x1 = min(img_w, lbl_w + tx)
    if img_y1 <= img_y0 or img_x1 <= img_x0:
        raise ValueError(f"Image '{image_key}' and labels '{labels_key}' do not overlap in coordinate system '{cs}'.")

    lbl_y0 = img_y0 - ty
    lbl_x0 = img_x0 - tx
    lbl_y1 = img_y1 - ty
    lbl_x1 = img_x1 - tx

    image_crop = image_da.isel(y=slice(img_y0, img_y1), x=slice(img_x0, img_x1))
    labels_crop = labels_da.isel(y=slice(lbl_y0, lbl_y1), x=slice(lbl_x0, lbl_x1))

    # Count cells that fall (partially) outside the labels_crop.
    cells_inside, cells_partial, cells_outside = _classify_dropped_cells(labels_da, lbl_y0, lbl_x0, lbl_y1, lbl_x1)
    if cells_outside or cells_partial:
        drop_report.outside_image_extent += cells_outside
        drop_report.partial_at_image_boundary += cells_partial
        warnings.warn(
            f"Dropping {cells_outside} cells outside the image extent and "
            f"{cells_partial} cells partially outside. See end-of-run drop report.",
            UserWarning,
            stacklevel=2,
        )

    return image_crop, labels_crop


def _classify_dropped_cells(
    labels_da: xr.DataArray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> tuple[int, int, int]:
    """Return ``(fully_inside, partially_inside, fully_outside)`` cell counts.

    Uses per-cell bounding boxes computed via tile-streamed reads so the
    full label array is never materialized.
    """
    lbl_h = int(labels_da.sizes.get("y", labels_da.shape[-2]))
    lbl_w = int(labels_da.sizes.get("x", labels_da.shape[-1]))
    if y0 <= 0 and x0 <= 0 and y1 >= lbl_h and x1 >= lbl_w:
        return 0, 0, 0

    cell_info = compute_cell_info_tiled(labels_da)
    fully_inside = 0
    partial = 0
    fully_outside = 0
    for ci in cell_info.values():
        by0, bx0 = ci.bbox_y0, ci.bbox_x0
        by1, bx1 = by0 + ci.bbox_h, bx0 + ci.bbox_w
        if by1 <= y0 or by0 >= y1 or bx1 <= x0 or bx0 >= x1:
            fully_outside += 1
        elif by0 >= y0 and by1 <= y1 and bx0 >= x0 and bx1 <= x1:
            fully_inside += 1
        else:
            partial += 1
    return fully_inside, partial, fully_outside


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
    channels: list[str] | list[int] | None,
    align_mode: Literal["strict", "rasterize"],
    drop_report: DropReport,
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

    # Align labels to image pixel grid via SpatialData transformations.
    # For the shapes_key path, rasterize already targets the image grid, so
    # the transforms are identity and this is a cheap no-op.
    if labels_key is not None:
        image_da, labels_da = _align_to_image_grid(
            sdata, image_key, labels_key, image_da, labels_da, align_mode, drop_report
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
            if isinstance(ch, int):
                if ch < 0 or ch >= len(all_ch):
                    raise ValueError(f"Channel index {ch} out of range [0, {len(all_ch)}).")
                selected_idx.append(ch)
                ch_names.append(all_ch[ch])
            else:
                ch_str = str(ch)
                if ch_str not in all_ch:
                    raise ValueError(f"Channel '{ch}' not found. Available: {all_ch}")
                selected_idx.append(all_ch.index(ch_str))
                ch_names.append(ch_str)
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
    align_mode: Literal["strict", "rasterize"] = "strict",
    adata_key_added: str = "morphology",
    invalid_as_zero: bool = True,
    n_jobs: int = 1,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Calculate per-cell features from segmentation masks.

    Uses `cp_measure <https://github.com/afermg/cp_measure>`_ for
    CellProfiler-derived features, scikit-image ``regionprops`` for
    morphological/intensity features, and squidpy-specific per-cell
    metrics (summary statistics, GLCM texture, colour histograms).

    Large images are automatically tiled into ``tile_size × tile_size``
    chunks with overlap so that every cell is fully contained in exactly
    one tile.

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
        Subset of channels to use.  ``None`` uses all channels.
    features
        Which features to compute.  Accepts a list of strings:

        - ``"cpmeasure:intensity"``, ``"cpmeasure:sizeshape"``,
          ``"cpmeasure:texture"``, ``"cpmeasure:granularity"``,
          ``"cpmeasure:zernike"``, ``"cpmeasure:feret"``,
          ``"cpmeasure:radial"``, ``"cpmeasure:correlation"``
        - ``"skimage:label"`` (all mask props), ``"skimage:label:area"``
          (single prop), ``"skimage:label+image"`` (all intensity props),
          ``"skimage:label+image:intensity_mean"`` (single prop)
        - ``"squidpy:summary"``, ``"squidpy:texture"``,
          ``"squidpy:color_hist"``

        ``None`` enables all cp_measure features.
    tile_size
        Side length of the tiling grid (pixels).
    overlap_margin
        Overlap around each tile to capture boundary cells.
        ``"auto"`` computes the minimum from the largest cell's bounding box.
    align_mode
        How to handle image/labels coordinate-system alignment when their
        pixel grids do not match.

        * ``"strict"`` (default): require the relative transform between
          image and labels to be identity or an integer-pixel translation.
          Raise otherwise with a hint pointing to :func:`spatialdata.rasterize`.
        * ``"rasterize"``: silently resample labels onto the image pixel
          grid using :func:`spatialdata.rasterize` when the transforms are
          not pixel-aligned. Logs a warning because this materializes the
          full label grid in memory.
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
        sdata, image_key, labels_key, shapes_key, scale, channels, align_mode, drop_report
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

    # Build cp_measure config once; the same dict is reused for every tile.
    cp_config = _build_cp_config(parsed.cp_flags, channel_names) if parsed.cp_flags is not None else None

    # --- Process tiles (each worker materializes only its own ~2k x 2k crop) ---
    def _process_one(spec):
        tile_img, tile_lbl = extract_tile_lazy(image_da, labels_da, spec)
        return _featurize_tile(tile_img, tile_lbl, parsed, channel_names, cp_config=cp_config)

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

    combined = pd.concat(tile_dfs, axis=0)

    # --- Post-process ---
    if invalid_as_zero:
        combined = combined.replace([np.inf, -np.inf], 0).fillna(0)

    # Sort by cell label for deterministic output
    combined = combined.sort_index()

    # --- Build AnnData ---
    adata = ad.AnnData(X=combined.values.astype(np.float32))
    adata.obs_names = [f"cell_{i}" for i in combined.index]
    adata.var_names = list(combined.columns)

    region_key_value = labels_key if labels_key is not None else shapes_key
    adata.uns["spatialdata_attrs"] = {
        "region": region_key_value,
        "region_key": "region",
        "instance_key": "label_id",
    }
    adata.obs["region"] = pd.Categorical([region_key_value] * len(adata))

    if shapes_key is not None and len(sdata.shapes[shapes_key]) == len(adata):
        adata.obs["label_id"] = sdata.shapes[shapes_key].index.values
    else:
        adata.obs["label_id"] = combined.index.values

    logg.info(drop_report.summary())

    if inplace:
        sdata.tables[adata_key_added] = TableModel.parse(adata)
        return None
    return adata
