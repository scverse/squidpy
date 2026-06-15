"""Experimental feature extraction module.

Extracts per-cell features from segmentation masks using cp_measure,
scikit-image ``regionprops``, and squidpy-specific metrics (summary
statistics, GLCM texture, colour histograms).  Large images are
automatically tiled so that each tile is processed independently.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
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
    CellInfo,
    build_tile_specs,
    compute_cell_info,
    compute_cell_info_multiscale,
    compute_cell_info_tiled,
    extract_labels_tile_lazy,
    extract_tile_lazy,
    yx_size,
)

__all__ = ["calculate_image_features"]

# ---------------------------------------------------------------------------
# Drop accounting
# ---------------------------------------------------------------------------


@dataclass
class DropReport:
    """Counters for cells/tiles excluded during a featurization run."""

    outside_image_extent: int = 0
    partial_at_image_boundary: int = 0
    empty_tiles: int = 0

    def summary(self) -> str:
        counts = {
            "outside_image_extent": self.outside_image_extent,
            "partial_at_image_boundary": self.partial_at_image_boundary,
            "empty_tiles": self.empty_tiles,
        }
        lines = [f"  {name}: {n}" for name, n in counts.items() if n > 0]
        return "\n".join(["Cell drop report:", *lines]) if lines else "Cell drop report: no cells dropped."


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

# skimage morphology props that cp_measure's ``sizeshape`` group reproduces
# bit-identically (value-verified against cp_measure 0.1.19: per-cell relΔ == 0).
# Everything in _MASK_PROPS *except* these two has an exact cp_measure twin, so
# when a request also computes cp:sizeshape the overlap is dropped from skimage
# (cp_measure computes its whole group regardless and cannot be told to skip).
# Only these are genuinely skimage-only: local centroid, and skimage's feret.
_SKIMAGE_MORPH_ONLY = frozenset({"centroid_local", "feret_diameter_max"})

# squidpy-native texture/histogram tuning.
_GLCM_LEVELS = 32  # gray levels for the GLCM: balances texture resolution vs matrix sparsity
_HIST_BINS = 16  # bins for the per-cell intensity histogram

# cp_measure feature name -> make_featurizer_config keyword(s).
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

# cp_measure correlation features need >=2 channels (they correlate channel pairs).
_CP_CORRELATION_KEYS = frozenset(_CPMEASURE_FLAGS["cpmeasure:correlation"])

# All known top-level feature group names (used for validation).
_ALL_FEATURES = (
    set(_CPMEASURE_FLAGS)
    | {"skimage:morphology", "skimage:intensity"}
    | {"squidpy:summary", "squidpy:texture", "squidpy:color_hist"}
)


# ---------------------------------------------------------------------------
# Feature parsing
# ---------------------------------------------------------------------------


class _ParsedFeatures(NamedTuple):
    cp_flags: dict[str, bool] | None  # kwargs for make_featurizer_config
    skimage_morphology_props: frozenset[str] | None
    skimage_intensity_props: frozenset[str] | None
    squidpy_summary: bool
    squidpy_texture: bool
    squidpy_color_hist: bool


def _ambiguous_mix(group: str) -> str:
    return f"Mixing 'skimage:{group}' with 'skimage:{group}:<prop>' is ambiguous; pick one form."


def _dedupe_morphology_against_cp(
    morphology_props: frozenset[str] | None, cp_flags: dict[str, bool] | None
) -> frozenset[str] | None:
    """Drop skimage morphology props that cp_measure's ``sizeshape`` already yields.

    Only applies when ``sizeshape`` is part of the cp_measure request (an empty
    ``cp_flags`` means "all cp groups on"). Keeps the skimage-only props; returns
    ``None`` if nothing skimage-specific remains.
    """
    if morphology_props is None or cp_flags is None:
        return morphology_props
    sizeshape_on = not cp_flags or cp_flags.get("sizeshape", False)
    if not sizeshape_on:
        return morphology_props
    kept = morphology_props & _SKIMAGE_MORPH_ONLY
    if kept != morphology_props:
        logg.info(
            f"Dropping {len(morphology_props - kept)} skimage morphology prop(s) already "
            f"computed by cp_measure:sizeshape; keeping skimage-only {sorted(kept) or 'none'}."
        )
    return kept or None


def _parse_features(features: list[str] | str | None) -> _ParsedFeatures:
    """Parse user-facing feature names into structured config.

    ``features=None`` enables *all* features across every backend
    (cp_measure + skimage morphology/intensity + squidpy).  Since most of
    these need pixel data, :func:`calculate_image_features` raises a clear
    error when ``features=None`` is paired with no ``image_key``.
    """
    if features is None:
        # All features, all backends.  cp_flags={} means "all cp defaults on";
        # cp:sizeshape covers most skimage morphology, so dedupe to the
        # skimage-only props to avoid duplicate columns.
        return _ParsedFeatures(
            cp_flags={},
            skimage_morphology_props=_dedupe_morphology_against_cp(frozenset(_MASK_PROPS), {}),
            skimage_intensity_props=frozenset(_INTENSITY_PROPS),
            squidpy_summary=True,
            squidpy_texture=True,
            squidpy_color_hist=True,
        )

    if isinstance(features, str):
        features = [features]

    cp_flags: dict[str, bool] = {}
    has_any_cp = False
    morphology_props: set[str] | None = None
    intensity_props: set[str] | None = None
    sq_summary = False
    sq_texture = False
    sq_color_hist = False

    for f in features:
        if f in _CPMEASURE_FLAGS:
            has_any_cp = True
            cp_flags.update(_CPMEASURE_FLAGS[f])

        elif f == "skimage:morphology":
            if morphology_props is not None:
                raise ValueError(_ambiguous_mix("morphology"))
            morphology_props = set(_MASK_PROPS)
        elif f == "skimage:intensity":
            if intensity_props is not None:
                raise ValueError(_ambiguous_mix("intensity"))
            intensity_props = set(_INTENSITY_PROPS)

        # skimage fine-grained: "skimage:morphology:prop" or "skimage:intensity:prop"
        elif f.startswith("skimage:morphology:"):
            prop = f.split(":", 2)[2]
            if prop not in _MASK_PROPS:
                raise ValueError(f"Unknown skimage morphology property: '{prop}'. Available: {sorted(_MASK_PROPS)}")
            if morphology_props is not None and morphology_props >= _MASK_PROPS:
                raise ValueError(_ambiguous_mix("morphology"))
            morphology_props = (morphology_props or set()) | {prop}
        elif f.startswith("skimage:intensity:"):
            prop = f.split(":", 2)[2]
            if prop not in _INTENSITY_PROPS:
                raise ValueError(f"Unknown skimage intensity property: '{prop}'. Available: {sorted(_INTENSITY_PROPS)}")
            if intensity_props is not None and intensity_props >= _INTENSITY_PROPS:
                raise ValueError(_ambiguous_mix("intensity"))
            intensity_props = (intensity_props or set()) | {prop}

        elif f == "squidpy:summary":
            sq_summary = True
        elif f == "squidpy:texture":
            sq_texture = True
        elif f == "squidpy:color_hist":
            sq_color_hist = True

        else:
            raise ValueError(
                f"Unknown feature: '{f}'. Available top-level features: {sorted(_ALL_FEATURES)}, "
                f"or use 'skimage:morphology:property' / 'skimage:intensity:property' for individual properties."
            )

    cp = cp_flags if has_any_cp else None
    morph = frozenset(morphology_props) if morphology_props else None
    return _ParsedFeatures(
        cp_flags=cp,
        skimage_morphology_props=_dedupe_morphology_against_cp(morph, cp),
        skimage_intensity_props=frozenset(intensity_props) if intensity_props else None,
        squidpy_summary=sq_summary,
        squidpy_texture=sq_texture,
        squidpy_color_hist=sq_color_hist,
    )


def _has_any_features(parsed: _ParsedFeatures) -> bool:
    return (
        parsed.cp_flags is not None
        or parsed.skimage_morphology_props is not None
        or parsed.skimage_intensity_props is not None
        or parsed.squidpy_summary
        or parsed.squidpy_texture
        or parsed.squidpy_color_hist
    )


def _image_requiring_features(parsed: _ParsedFeatures) -> list[str]:
    """User-facing flags in the request that need pixel data (i.e. an image)."""
    flags = [
        (parsed.cp_flags is not None, "cpmeasure:*"),
        (parsed.skimage_intensity_props is not None, "skimage:intensity"),
        (parsed.squidpy_summary, "squidpy:summary"),
        (parsed.squidpy_texture, "squidpy:texture"),
        (parsed.squidpy_color_hist, "squidpy:color_hist"),
    ]
    return [name for cond, name in flags if cond]


# ---------------------------------------------------------------------------
# cp_measure config builder
# ---------------------------------------------------------------------------


def _build_cp_config(cp_flags: dict[str, bool], channel_names: list[str]) -> dict[str, Any]:
    """Build a cp_measure featurizer config from parsed flags.

    Empty ``cp_flags`` (the default-all case) enables every feature.
    Otherwise only the explicitly requested features are turned on.
    """
    if not cp_flags:
        return make_featurizer_config(channel_names)

    # Start from every known toggle off, then enable the requested ones.
    all_off = dict.fromkeys(set().union(*_CPMEASURE_FLAGS.values()), False)
    all_off.update(cp_flags)
    return make_featurizer_config(channel_names, **all_off)


# ---------------------------------------------------------------------------
# Per-tile dispatcher
# ---------------------------------------------------------------------------


def _featurize_tile(
    tile_image: np.ndarray | None,
    tile_labels: np.ndarray,
    parsed: _ParsedFeatures,
    channel_names: list[str],
    *,
    cp_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute all requested features for a single tile.

    ``tile_image`` is ``(C, H, W)`` or ``None`` for a morphology-only run;
    ``tile_labels`` is ``(H, W)`` with only owned cells.  ``cp_config`` is the
    pre-built cp_measure config (``None`` when no cp_measure features are
    requested).  Returns a DataFrame indexed by cell label ID, one column per
    feature.
    """
    cell_ids = np.unique(tile_labels)
    cell_ids = cell_ids[cell_ids != 0]
    if len(cell_ids) == 0:
        return pd.DataFrame()

    feature_blocks: list[pd.DataFrame] = []

    if cp_config is not None and tile_image is not None:
        feature_blocks.append(_compute_cpmeasure_features(tile_image, tile_labels, cp_config))

    if parsed.skimage_morphology_props is not None or parsed.skimage_intensity_props is not None:
        feature_blocks.append(
            _compute_skimage_features(
                tile_labels, tile_image, parsed.skimage_morphology_props, parsed.skimage_intensity_props, channel_names
            )
        )

    if parsed.squidpy_summary or parsed.squidpy_texture or parsed.squidpy_color_hist:
        feature_blocks.append(_compute_squidpy_per_cell(tile_labels, tile_image, parsed, channel_names))

    feature_blocks = [df for df in feature_blocks if not df.empty]
    if not feature_blocks:
        return pd.DataFrame(index=cell_ids)

    return pd.concat(feature_blocks, axis=1).reindex(cell_ids)


def _compute_cpmeasure_features(
    tile_image: np.ndarray, tile_labels: np.ndarray, cp_config: dict[str, Any]
) -> pd.DataFrame:
    """cp_measure features for a tile, indexed by original label ID.

    Columns keep cp_measure's raw CellProfiler names (CamelCase / ``__channel``).
    """
    # cp_measure assumes dense 1..N IDs and index-errors on sparse IDs.
    contiguous_labels, _, inverse = relabel_sequential(tile_labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, columns, rows = featurize(tile_image, contiguous_labels[np.newaxis, :, :], cp_config)
    if data.shape[0] == 0:
        return pd.DataFrame()
    row_labels = [int(inverse[r[2]]) for r in rows[: data.shape[0]]]
    return pd.DataFrame(data, index=row_labels, columns=columns)


# ---------------------------------------------------------------------------
# skimage regionprops
# ---------------------------------------------------------------------------


def _rename_intensity_col(col: str, channel_names: list[str]) -> str:
    """Map a multichannel ``regionprops_table`` column to ``<prop>_<channel>``.

    A multichannel ``intensity_image`` makes skimage suffix each intensity prop
    with the channel index (``intensity_mean-0``); rename that to the channel's
    name (``intensity_mean_DAPI``).
    """
    prop, _, idx = col.rpartition("-")
    return f"{prop}_{channel_names[int(idx)]}"


def _regionprops_table_to_df(table: dict[str, np.ndarray], rename: Callable[[str], str] | None = None) -> pd.DataFrame:
    """Build a label-indexed DataFrame from a ``regionprops_table`` dict.

    ``rename`` optionally maps each (non-label) column name.
    """
    index = table.pop("label")
    if rename is not None:
        table = {rename(col): vals for col, vals in table.items()}
    return pd.DataFrame(table, index=index)


def _compute_skimage_features(
    labels: np.ndarray,
    image: np.ndarray | None,
    morphology_props: frozenset[str] | None,
    intensity_props: frozenset[str] | None,
    channel_names: list[str],
) -> pd.DataFrame:
    """Compute skimage regionprops features for all cells in a tile.

    A single :func:`skimage.measure.regionprops_table` call covers morphology
    and intensity together (mask props ignore ``intensity_image``), so the cells
    are sliced once rather than once per group. Morphology props keep skimage's
    native flattened names (e.g. ``centroid-0``, ``inertia_tensor-0-0``);
    intensity props are computed per channel and renamed ``<prop>_<channel>``.
    ``image`` is only read for ``intensity_props`` and may be ``None`` for a
    morphology-only run.
    """
    props = [*sorted(morphology_props or []), *sorted(intensity_props or [])]
    if not props:
        return pd.DataFrame()

    # moveaxis -> (y, x, c) so skimage treats the last axis as channels.
    intensity_image = np.moveaxis(image, 0, -1) if intensity_props is not None else None
    table = measure.regionprops_table(labels, intensity_image=intensity_image, properties=["label", *props])

    # Rename only intensity columns (skimage suffixes them ``<prop>-<channel>``);
    # morphology columns (and spatial-dim suffixes like ``centroid-0``) stay as-is.
    def rename(col: str) -> str:
        return (
            _rename_intensity_col(col, channel_names)
            if (col.rpartition("-")[0] or col) in (intensity_props or ())
            else col
        )

    return _regionprops_table_to_df(table, rename)


# ---------------------------------------------------------------------------
# squidpy per-cell features
# ---------------------------------------------------------------------------


def _compute_squidpy_per_cell(
    labels: np.ndarray,
    image: np.ndarray,
    parsed: _ParsedFeatures,
    channel_names: list[str],
) -> pd.DataFrame:
    """Compute squidpy features per cell within a tile.

    Only reached when a squidpy feature is requested, which always requires an
    image (enforced by validation), so ``image`` is never ``None`` here.
    """
    regions = measure.regionprops(labels)
    n_channels = image.shape[0]
    rows: dict[int, dict[str, float]] = {}

    for region in regions:
        lid = region.label
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)
        cell_features: dict[str, float] = {}

        img_crop = image[:, bbox[0] : bbox[2], bbox[1] : bbox[3]]
        mask_crop = labels[bbox[0] : bbox[2], bbox[1] : bbox[3]] == lid

        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx]
            ch_crop = img_crop[ch_idx].astype(np.float32)
            masked_vals = ch_crop[mask_crop]

            if len(masked_vals) == 0:
                continue

            if parsed.squidpy_summary:
                for stat, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
                    cell_features[f"summary_{stat}_{ch_name}"] = float(fn(masked_vals))

            if parsed.squidpy_texture:
                cell_features.update(_glcm_features(ch_crop, mask_crop, ch_name))

            if parsed.squidpy_color_hist:
                cell_features.update(_histogram_features(masked_vals, ch_name))

        rows[lid] = cell_features

    return pd.DataFrame.from_dict(rows, orient="index")


def _glcm_features(channel_crop: np.ndarray, mask: np.ndarray, ch_name: str) -> dict[str, float]:
    """GLCM texture features for a single channel within a cell's bbox."""
    ch = channel_crop.copy()
    # Zero out non-cell pixels so they don't affect GLCM
    ch[~mask] = 0
    ch_min, ch_max = float(ch[mask].min()), float(ch[mask].max())
    if ch_max > ch_min:
        ch = (ch - ch_min) / (ch_max - ch_min)
    else:
        ch = np.zeros_like(ch)
    chan_quant = np.clip((ch * (_GLCM_LEVELS - 1)).round().astype(np.uint8), 0, _GLCM_LEVELS - 1)
    chan_quant[~mask] = 0

    try:
        glcm = graycomatrix(chan_quant, distances=[1], angles=[0], levels=_GLCM_LEVELS, symmetric=True, normed=True)
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


def _histogram_features(masked_vals: np.ndarray, ch_name: str) -> dict[str, float]:
    """Per-cell intensity histogram features."""
    lo, hi = float(masked_vals.min()), float(masked_vals.max())
    hist, _ = np.histogram(masked_vals, bins=_HIST_BINS, range=(lo, hi if hi > lo else lo + 1))
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return {f"color_hist_bin{b}_{ch_name}": float(v) for b, v in enumerate(hist)}


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
    """Return the 3x3 affine mapping labels-pixel-coords to image-pixel-coords.

    Uses ``(x, y)`` axis order to match :mod:`spatialdata` convention.
    """
    t_img = get_transformation(sdata.images[image_key], to_coordinate_system=cs)
    t_lbl = get_transformation(sdata.labels[labels_key], to_coordinate_system=cs)
    # image_pixel <- global <- labels_pixel
    m_img_to_global = t_img.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    m_lbl_to_global = t_lbl.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    m_global_to_img = np.linalg.inv(m_img_to_global)
    return m_global_to_img @ m_lbl_to_global


def _rasterize_to_grid(element: Any, image_da: xr.DataArray, cs: str) -> xr.DataArray:
    """Rasterize a spatialdata element onto ``image_da``'s pixel grid (as labels)."""
    result = rasterize(
        element,
        ["x", "y"],
        min_coordinate=[0, 0],
        max_coordinate=[int(image_da.sizes["x"]), int(image_da.sizes["y"])],
        target_coordinate_system=cs,
        target_unit_to_pixels=1.0,
        return_regions_as_labels=True,
    )
    if isinstance(result, xr.DataArray):
        return result
    return xr.DataArray(np.asarray(result), dims=["y", "x"])


def _rasterize_to_image_grid(element: Any, image_da: xr.DataArray, cs: str) -> xr.DataArray:
    """Rasterize labels onto the image grid, warning that laziness is lost."""
    warnings.warn(
        f"Materializing labels onto the image grid via spatialdata.rasterize in '{cs}'. "
        f"Lazy behavior is lost for this run.",
        UserWarning,
        stacklevel=2,
    )
    return _rasterize_to_grid(element, image_da, cs)


def _decompose_pixel_translation(affine: np.ndarray, atol: float = 1e-6) -> tuple[int, int] | None:
    """If ``affine`` is identity-plus-integer-translation, return ``(tx, ty)``; else None.

    ``affine`` is a 3x3 matrix in (x, y) axis order.
    """
    rotscale = affine[:2, :2]
    if not np.allclose(rotscale, np.eye(2), atol=atol):
        return None
    tx, ty = float(affine[0, 2]), float(affine[1, 2])
    if not (abs(tx - round(tx)) < atol and abs(ty - round(ty)) < atol):
        return None
    return int(round(tx)), int(round(ty))


def _classify_boundary_cells(
    labels_da: xr.DataArray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> tuple[list[int], int]:
    """Return ``(partial_cell_ids, n_fully_outside)`` for the crop rectangle.

    Streams per-cell bounding boxes from the (post-alignment) ``labels_da`` that
    gets tiled, so frames stay consistent and the full array is never
    materialized.  Returns early when the crop covers the whole array.  Partial
    cells (bbox straddling the crop edge) are returned by ID so the caller can
    drop them - their clipped pixels would otherwise yield wrong features.
    """
    lbl_h, lbl_w = yx_size(labels_da)
    if y0 <= 0 and x0 <= 0 and y1 >= lbl_h and x1 >= lbl_w:
        return [], 0

    partial_ids: list[int] = []
    fully_outside = 0
    for ci in compute_cell_info_tiled(labels_da).values():
        by0, bx0 = ci.bbox_y0, ci.bbox_x0
        by1, bx1 = by0 + ci.bbox_h, bx0 + ci.bbox_w
        if by1 <= y0 or by0 >= y1 or bx1 <= x0 or bx0 >= x1:
            fully_outside += 1
        elif not (by0 >= y0 and by1 <= y1 and bx0 >= x0 and bx1 <= x1):
            partial_ids.append(ci.label)
    return partial_ids, fully_outside


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

    Mutates ``drop_report`` to count cells dropped because they fall outside
    the overlap rectangle.  Under ``align_mode='strict'`` a non-pixel-aligned
    relative transform raises; under ``'rasterize'`` the labels are resampled
    onto the image grid.
    """
    cs = _shared_coordinate_system(sdata, image_key, labels_key)
    affine = _relative_affine(sdata, image_key, labels_key, cs)

    # Integer-pixel offset of labels relative to image. (tx, ty) means labels
    # pixel (0, 0) lands at image pixel (tx, ty) in (x, y) order. Identity
    # decomposes to (0, 0), so it needs no separate case.
    decomposed = _decompose_pixel_translation(affine)
    if decomposed is not None:
        tx, ty = decomposed
    elif align_mode == "strict":
        raise ValueError(
            f"Image '{image_key}' and labels '{labels_key}' have different pixel grids "
            f"in coordinate system '{cs}'. Relative affine (x,y) =\n{affine}\n"
            f"Pass align_mode='rasterize' to resample labels onto the image grid "
            f"(via spatialdata.rasterize), or pre-align with `spatialdata.rasterize`."
        )
    elif isinstance(sdata.labels[labels_key], xr.DataTree):
        # spatialdata.rasterize does not accept a multiscale element. Rather
        # than silently mis-resample, ask the user to pre-align or pass a
        # single-scale labels element.
        raise ValueError(
            f"align_mode='rasterize' is not supported for multiscale labels "
            f"('{labels_key}') under a non-integer transform in coordinate system '{cs}'. "
            f"Pre-align with `spatialdata.rasterize` and pass the resulting single-scale "
            f"labels, or supply single-scale labels."
        )
    else:
        labels_da = _rasterize_to_image_grid(sdata.labels[labels_key], image_da, cs)
        tx, ty = 0, 0

    # Overlap rectangle in image-pixel coords.
    img_h, img_w = yx_size(image_da)
    lbl_h, lbl_w = yx_size(labels_da)

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

    # Drop cells that fall (partially) outside the overlap: fully-outside cells
    # vanish with the crop; partial cells are zeroed so their clipped pixels
    # don't produce truncated features.
    partial_ids, cells_outside = _classify_boundary_cells(labels_da, lbl_y0, lbl_x0, lbl_y1, lbl_x1)
    if partial_ids:
        labels_crop = labels_crop.where(~labels_crop.isin(partial_ids), 0)
    if cells_outside or partial_ids:
        drop_report.outside_image_extent += cells_outside
        drop_report.partial_at_image_boundary += len(partial_ids)
        warnings.warn(
            f"Dropping {cells_outside} cells outside the image extent and "
            f"{len(partial_ids)} cells partially outside. See the end-of-run drop report.",
            UserWarning,
            stacklevel=2,
        )

    return image_crop, labels_crop


# ---------------------------------------------------------------------------
# Input preparation (lazy - returns xarray DataArrays, not numpy)
# ---------------------------------------------------------------------------


def _select_scale_array(element: xr.DataTree | xr.DataArray, scale: str | None) -> xr.DataArray:
    """Pick the single-scale DataArray from a (possibly multiscale) element (stays lazy)."""
    if not isinstance(element, xr.DataTree):
        return element
    if scale is None:
        raise ValueError("`scale` must be provided for multiscale (DataTree) elements.")
    if scale not in element:
        raise ValueError(f"Scale '{scale}' not found. Available: {list(element.keys())}")
    return element[scale].ds["image"]


def _validate_inputs(
    sdata: SpatialData,
    image_key: str | None,
    labels_key: str | None,
    shapes_key: str | None,
    scale: str | None,
) -> None:
    """Run structural input validation (no data loading).

    Feature-dependent rules (whether an image is required at all) live in
    :func:`calculate_image_features`, which has the parsed feature set.
    """
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
    if image_key is not None:
        if image_key not in sdata.images:
            raise ValueError(f"Image key '{image_key}' not found, valid keys: {list(sdata.images.keys())}")
        if isinstance(sdata.images[image_key], xr.DataTree) and scale is None:
            raise ValueError("When using multi-scale images, please specify the scale.")


def _prepare_lazy(
    sdata: SpatialData,
    image_key: str | None,
    labels_key: str | None,
    shapes_key: str | None,
    scale: str | None,
    channels: list[str] | None,
    align_mode: Literal["strict", "rasterize"],
    drop_report: DropReport,
) -> tuple[xr.DataArray | None, xr.DataArray, list[str]]:
    """Return lazy image and labels DataArrays, plus channel names.

    ``image_da`` is ``None`` (and ``channel_names`` empty) for a morphology-only
    run with no ``image_key``.  Does NOT call ``.compute()`` - arrays stay lazy
    for on-demand tile reads.  For the shapes->labels path, labels are
    materialized but wrapped in a DataArray for a uniform interface.
    """
    _validate_inputs(sdata, image_key, labels_key, shapes_key, scale)

    if align_mode not in ("strict", "rasterize"):
        raise ValueError(f"`align_mode` must be 'strict' or 'rasterize'; got {align_mode!r}.")

    image_da = None
    if image_key is not None:
        image_da = _select_scale_array(sdata.images[image_key], scale)
        if "c" not in image_da.dims:
            image_da = image_da.expand_dims("c")

    # Labels DataArray (lazy for labels_key, materialized for shapes_key).
    if labels_key is not None:
        labels_da = _select_scale_array(sdata.labels[labels_key], scale)
    else:
        # shapes_key requires an image to size the rasterization grid (enforced
        # by calculate_image_features), so image_da is not None here.
        logg.info("Converting shapes to labels.")
        try:
            labels_da = _rasterize_to_grid(sdata.shapes[shapes_key], image_da, "global")
        except ValueError as e:
            raise ValueError(
                "Failed to rasterize shapes; geometries may be empty or unsupported. "
                "Filter out empty/non-polygon geometries or choose a different shapes_key."
            ) from e

    # Align labels to the image pixel grid via SpatialData transformations.
    # Only meaningful with a real labels element + an image; the shapes->labels
    # path already rasterized onto the image grid (identity transform -> no-op).
    if image_da is not None and labels_key is not None:
        image_da, labels_da = _align_to_image_grid(
            sdata, image_key, labels_key, image_da, labels_da, align_mode, drop_report
        )

    if image_da is None:
        return image_da, labels_da, []

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
) -> dict[int, CellInfo]:
    """Compute cell centroids using the most efficient strategy available."""
    # Multiscale: the coarse-scale fast path is only valid when alignment did not
    # crop labels_da; after a crop, recompute from it so centroids and tiling
    # share one frame.
    if labels_key is not None and isinstance(sdata.labels[labels_key], xr.DataTree):
        full = _select_scale_array(sdata.labels[labels_key], scale)
        full_grid = (full.sizes.get("y"), full.sizes.get("x"))
        cur_grid = (labels_da.sizes.get("y"), labels_da.sizes.get("x"))
        if cur_grid == full_grid:
            logg.info("Computing centroids from coarse scale.")
            return compute_cell_info_multiscale(sdata.labels[labels_key], target_scale=scale or "scale0")
        logg.info("Computing centroids in tiled mode (aligned multiscale labels).")
        return compute_cell_info_tiled(labels_da)

    # Small enough to fit in memory - direct regionprops
    n_pixels = labels_da.sizes.get("y", 1) * labels_da.sizes.get("x", 1)
    if n_pixels <= 4096 * 4096:
        lbl_np = labels_da.values
        if lbl_np.ndim > 2:
            lbl_np = lbl_np.squeeze()
        return compute_cell_info(lbl_np)

    # Large single-scale - tiled centroid computation
    logg.info("Computing centroids in tiled mode (large single-scale labels).")
    return compute_cell_info_tiled(labels_da)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def calculate_image_features(
    sdata: SpatialData,
    *,
    image_key: str | None = None,
    labels_key: str | None = None,
    shapes_key: str | None = None,
    scale: str | None = None,
    channels: list[str] | None = None,
    features: list[str] | str | None = None,
    tile_size: int = 2048,
    overlap_margin: int | Literal["auto"] = "auto",
    align_mode: Literal["strict", "rasterize"] = "strict",
    key_added: str = "morphology",
    invalid_as_zero: bool = True,
    n_jobs: int = 1,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Calculate per-cell features from segmentation masks.

    Uses `cp_measure <https://github.com/afermg/cp_measure>`_ for
    CellProfiler-derived features, scikit-image ``regionprops`` for
    morphological/intensity features, and squidpy-specific per-cell metrics
    (summary statistics, GLCM texture, colour histograms).  Large images are
    automatically tiled into ``tile_size x tile_size`` chunks with overlap so
    that every cell is fully contained in exactly one tile.

    Parameters
    ----------
    sdata
        SpatialData object.
    image_key
        Key in ``sdata.images``. Optional: required only for intensity / squidpy
        / cp_measure features (and for ``shapes_key``). Morphology-only runs
        need no image.
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
        Which features to compute.  ``None`` (the default) computes *all*
        features across every backend; because that set includes intensity /
        texture features, an image is required (a clear error is raised if
        ``image_key`` is missing).  Otherwise a list of flag strings from three
        groups:

        - **cp_measure** -- ``"cpmeasure:intensity"``, ``"cpmeasure:sizeshape"``,
          ``"cpmeasure:texture"``, ``"cpmeasure:granularity"``,
          ``"cpmeasure:zernike"``, ``"cpmeasure:feret"``, ``"cpmeasure:radial"``,
          ``"cpmeasure:correlation"`` (or a single correlation kind via
          ``"cpmeasure:correlation_<pearson|costes|manders_fold|rwc>"``).
          Columns keep cp_measure's native CellProfiler names (e.g. ``Area``,
          ``Intensity_MeanIntensity__<channel>``). Correlation features need
          an image with >=2 channels.
        - **skimage regionprops** -- ``"skimage:morphology"`` (all shape props,
          from the mask alone) or ``"skimage:morphology:<prop>"`` for one
          (e.g. ``area``); ``"skimage:intensity"`` (all per-channel intensity
          props, needs an image) or ``"skimage:intensity:<prop>"`` for one
          (e.g. ``intensity_mean``). Morphology columns use skimage's native
          names (``area``, ``centroid-0``); intensity columns are suffixed with
          the channel name.
        - **squidpy per-cell** -- ``"squidpy:summary"`` (per-channel mean / std /
          min / max), ``"squidpy:texture"`` (per-channel GLCM contrast,
          dissimilarity, homogeneity, energy, ASM, correlation), and
          ``"squidpy:color_hist"`` (per-channel intensity histogram).

        If a request computes both ``"cpmeasure:sizeshape"`` and skimage
        morphology props, the overlapping skimage props (which cp_measure
        reproduces identically) are dropped to avoid duplicate columns; only
        the skimage-only props (``centroid_local``, ``feret_diameter_max``) are
        kept. cp_measure computes its groups all-or-nothing, so it wins.
    tile_size
        Side length of the tiling grid (pixels).
    overlap_margin
        Overlap around each tile to capture boundary cells.
        ``"auto"`` computes the minimum from the largest cell's bounding box.
    align_mode
        How to handle image/labels whose pixel grids do not match (via their
        SpatialData transformations).

        * ``"strict"`` (default): require the relative transform between image
          and labels to be identity or an integer-pixel translation; raise
          otherwise with a hint pointing to :func:`spatialdata.rasterize`.
        * ``"rasterize"``: resample labels onto the image pixel grid using
          :func:`spatialdata.rasterize` when not pixel-aligned (logs a warning
          because this materializes the full label grid). Not supported for
          multiscale labels under a non-integer transform -- pre-align instead.

        Cells falling outside the image/labels overlap are dropped and counted
        in the end-of-run drop report.
    key_added
        Key under which to store the result in ``sdata.tables``.
    invalid_as_zero
        Replace ``inf`` and ``NaN`` values with zero (cp_measure can emit NaN
        for undefined features on small cells).
    n_jobs
        Number of parallel jobs for tile processing.
    inplace
        If ``True``, store result in ``sdata.tables``.  Otherwise return it.

    Notes
    -----
    Cells dropped at the image boundary (under alignment) emit a ``UserWarning``
    as they are dropped, and an end-of-run summary of all drops (boundary cells
    and empty tiles) is logged at INFO level.

    Returns
    -------
    :class:`~anndata.AnnData` when ``inplace=False``, otherwise ``None``.

    Examples
    --------
    >>> import squidpy as sq
    >>> sq.experimental.im.calculate_image_features(
    ...     sdata,
    ...     image_key="image",
    ...     labels_key="cells",
    ...     features=["cpmeasure:sizeshape", "skimage:morphology", "squidpy:summary"],
    ... )  # doctest: +SKIP

    All features across every backend (needs an image):

    >>> sq.experimental.im.calculate_image_features(
    ...     sdata, image_key="image", labels_key="cells", features=None
    ... )  # doctest: +SKIP

    Morphology-only needs no image:

    >>> sq.experimental.im.calculate_image_features(
    ...     sdata, labels_key="cells", features=["skimage:morphology:area"]
    ... )  # doctest: +SKIP

    The per-cell table is stored in ``sdata.tables["morphology"]``.
    """
    # --- Parse & validate ---
    parsed = _parse_features(features)
    if not _has_any_features(parsed):
        raise ValueError(
            "No features requested. Pass a non-empty `features` list "
            "(e.g. ['skimage:morphology']), or `features=None` for all features."
        )

    # An image is needed only for intensity / squidpy features; morphology runs
    # from the labels alone.  Reject the cases that genuinely cannot proceed.
    if image_key is None:
        needs_image = _image_requiring_features(parsed)
        if needs_image:
            raise ValueError(f"Features {needs_image} require pixel data; pass `image_key`.")
        if shapes_key is not None:
            raise ValueError("`shapes_key` requires `image_key` (rasterization needs the image grid).")
        if channels is not None:
            raise ValueError("`channels` selection requires `image_key`.")

    drop_report = DropReport()

    image_da, labels_da, channel_names = _prepare_lazy(
        sdata, image_key, labels_key, shapes_key, scale, channels, align_mode, drop_report
    )

    # Correlation-only cp_measure with a single channel produces no features and
    # would crash cp_measure downstream; fail fast with an actionable message.
    if parsed.cp_flags and set(parsed.cp_flags) <= _CP_CORRELATION_KEYS and len(channel_names) < 2:
        raise ValueError(
            f"cp_measure correlation features require >=2 channels; got {len(channel_names)}. "
            f"Request other cp_measure features or pass an image with >=2 channels."
        )

    # Build cp_measure config once; the same dict is reused for every tile.
    cp_config = _build_cp_config(parsed.cp_flags, channel_names) if parsed.cp_flags is not None else None

    # --- Warmup: compute centroids without materializing full arrays ---
    cell_info = _compute_centroids(sdata, labels_key, labels_da, scale)
    if not cell_info:
        logg.info(drop_report.summary())
        raise ValueError("No cells found in labels (all zeros).")

    H, W = yx_size(labels_da)

    # --- Tile ---
    specs = build_tile_specs((H, W), cell_info, tile_size=tile_size, overlap_margin=overlap_margin)
    total_tiles = len(specs)
    logg.info(f"Processing {total_tiles} tiles ({tile_size}x{tile_size}, margin={overlap_margin}).")

    # --- Process tiles (each worker materializes only its own ~2k x 2k crop) ---
    def _process_one(spec):
        if image_da is None:
            tile_lbl = extract_labels_tile_lazy(labels_da, spec)
            return _featurize_tile(None, tile_lbl, parsed, channel_names, cp_config=cp_config)
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
        sdata.tables[key_added] = TableModel.parse(adata)
        return None
    return adata
