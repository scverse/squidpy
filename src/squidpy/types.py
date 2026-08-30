"""Squidpy's public parameter bags and result tuples.

A ``*Params`` is input a caller fills in; a ``*Result`` is what a function hands back.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, TypedDict

from squidpy._utils import RNGLike, SeedLike
from squidpy.experimental.utils._params import Default, defaults_of
from squidpy.gr._build import SpatialNeighborsResult
from squidpy.gr._nhood import NhoodEnrichmentResult

#: Pixels whose Ruderman Lab-L luminosity (normalised to ``[0, 1]``) exceeds this are
#: treated as near-white background and excluded when fitting stain statistics.
#: Semantics follow HistomicsTK's ``reinhard``, so luminosity thresholds from the H&E
#: literature transfer directly. Declared here, with the key it defaults, and
#: re-exported by ``squidpy.experimental.im._stain._constants``: this module must not
#: import from the implementation packages, whose ``__init__`` imports this one.
DEFAULT_LUMINOSITY_THRESHOLD: float = 0.8

#: Mean-absorbance (optical-density) cutoff selecting tissue pixels. One value for both
#: decomposition methods -- it is the same quantity, so it is declared once.
_OD_BETA: float = 0.15

__all__ = [
    # Parameters
    "BackgroundDetectionParams",
    "FelzenszwalbParams",
    "WekaParams",
    "ReinhardParams",
    "MacenkoParams",
    "VahadaneParams",
    "TilingQCParams",
    "StitchParams",
    # Results
    "NhoodEnrichmentResult",
    "SpatialNeighborsResult",
]


class BackgroundDetectionParams(TypedDict, total=False):
    """Which corners are background, and how large the corner boxes should be.

    If no corners are flagged ``True``, background is taken to be the bright side.
    """

    ymin_xmin_is_bg: Annotated[bool, Default(True)]
    """Whether the ``(ymin, xmin)`` corner is background."""

    ymax_xmin_is_bg: Annotated[bool, Default(True)]
    """Whether the ``(ymax, xmin)`` corner is background."""

    ymin_xmax_is_bg: Annotated[bool, Default(True)]
    """Whether the ``(ymin, xmax)`` corner is background."""

    ymax_xmax_is_bg: Annotated[bool, Default(True)]
    """Whether the ``(ymax, xmax)`` corner is background."""

    corner_size_pct: Annotated[float, Default(0.01)]
    """Corner box size as a fraction of height/width."""


_BACKGROUND_DEFAULTS: BackgroundDetectionParams = defaults_of(BackgroundDetectionParams)


class FelzenszwalbParams(TypedDict, total=False):
    """Size-aware superpixel defaults for felzenszwalb segmentation."""

    grid_rows: Annotated[int, Default(100)]
    """Target superpixel grid rows."""

    grid_cols: Annotated[int, Default(100)]
    """Target superpixel grid columns."""

    sigma_frac: Annotated[float, Default(0.008)]
    """Blur = this * short side, clipped to ``[1, 5]`` px."""

    scale_coef: Annotated[float, Default(0.25)]
    """``scale`` = coef * target_area."""

    min_size_coef: Annotated[float, Default(0.20)]
    """``min_size`` = coef * target_area."""


_FELZENSZWALB_DEFAULTS: FelzenszwalbParams = defaults_of(FelzenszwalbParams)


class WekaParams(TypedDict, total=False):
    """Parameters for WEKA-like trainable segmentation."""

    sigma_min: Annotated[float, Default(1.0)]
    """Smallest scale in the multiscale feature bank."""

    sigma_max: Annotated[float, Default(16.0)]
    """Largest scale in the multiscale feature bank."""

    edges: Annotated[bool, Default(True)]
    """Include edge features."""

    pseudo_tissue_percentile: Annotated[float, Default(90.0)]
    """Percentile of distance-from-bg to label as tissue."""

    pseudo_min_pixels: Annotated[int, Default(50)]
    """Minimum number of tissue pixels to seed."""

    rf_estimators: Annotated[int, Default(100)]
    """Number of trees in the random forest."""

    rf_max_depth: Annotated[int | None, Default(10)]
    """Maximum tree depth; ``None`` for unlimited."""

    rf_max_samples: Annotated[float, Default(0.05)]
    """Fraction of samples drawn to train each tree."""

    rng: Annotated[SeedLike | RNGLike | None, Default(None)]
    """Source of randomness; ``None`` draws from OS entropy."""

    refine_with_classifier: Annotated[bool, Default(True)]
    """Run the second-stage background refinement."""

    refine_n_samples_per_class: Annotated[int, Default(50_000)]
    """Training samples drawn per class in the refinement step."""

    refine_bg_prob_threshold: Annotated[float, Default(0.6)]
    """Only drop pixels whose background probability exceeds this."""

    border_margin_px: Annotated[int | Sequence[int], Default(0)]
    """Border ignored when seeding and predicting."""


_WEKA_DEFAULTS: WekaParams = defaults_of(WekaParams)


class ReinhardParams(TypedDict, total=False):
    """Tuning knobs for Reinhard stain normalization.

    Pass a mapping of these keys as ``method_params``; every key is optional and
    falls back to the default shown with it. Values are coerced and range-checked
    by ``validate_reinhard_params`` when resolved.
    """

    luminosity_threshold: Annotated[float, Default(DEFAULT_LUMINOSITY_THRESHOLD)]
    """Normalised Ruderman Lab-L cutoff in ``(0, 1]``; pixels brighter than this are excluded from the fit."""

    mask_background: Annotated[bool, Default(True)]
    """If ``True``, fit channel statistics over tissue pixels only; if ``False``, use every pixel (vanilla Reinhard)."""


_REINHARD_DEFAULTS: ReinhardParams = defaults_of(ReinhardParams)


class MacenkoParams(TypedDict, total=False):
    """Tuning knobs for Macenko stain-matrix fitting."""

    alpha: Annotated[float, Default(1.0)]
    """Angular percentile (deg) for the two stain directions; the extremes are taken at ``alpha`` / ``100 - alpha``."""

    beta: Annotated[float, Default(_OD_BETA)]
    """Mean-absorbance cutoff selecting tissue pixels (optical-density space)."""


_MACENKO_DEFAULTS: MacenkoParams = defaults_of(MacenkoParams)


class VahadaneParams(TypedDict, total=False):
    """Tuning knobs for Vahadane (sparse-NMF) stain-matrix fitting."""

    beta: Annotated[float, Default(_OD_BETA)]
    """Mean-absorbance cutoff selecting tissue pixels (optical-density space)."""

    lambda1: Annotated[float, Default(0.1)]
    """L1 sparsity regularisation on the concentration factor of the NMF."""

    n_iter: Annotated[int, Default(200)]
    """Maximum NMF iterations."""

    rng: Annotated[SeedLike | RNGLike | None, Default(None)]
    """Source of randomness for NMF initialisation tie-breaking; ``None`` draws from OS entropy."""


_VAHADANE_DEFAULTS: VahadaneParams = defaults_of(VahadaneParams)


class TilingQCParams(TypedDict, total=False):
    """Advanced tuning knobs for :func:`~squidpy.experimental.tl.calculate_tiling_qc`.

    Pass a mapping of these keys as ``tiling_qc_params``; every key is optional
    and falls back to the default shown with it. Values are coerced and
    range-checked by ``validate_qc_params`` when resolved.
    """

    distance_tol: Annotated[float, Default(0.75)]
    """Maximum perpendicular distance (pixels) from the fitted line for a contour point to count as straight."""

    min_area: Annotated[int, Default(20)]
    """Cells smaller than this (pixels at analysis resolution) are skipped (NaN scores)."""

    max_contour_points: Annotated[int, Default(500)]
    """Cap on contour resolution; longer contours are arc-length-resampled before the O(n^2) collinearity scan."""


_QC_DEFAULTS: TilingQCParams = defaults_of(TilingQCParams)


class StitchParams(TypedDict, total=False):
    """Advanced tuning knobs for :func:`~squidpy.experimental.tl.assign_stitch_groups`.

    Defaults work for typical 2D segmentation tiles produced by cellpose-like
    pipelines. Pass a mapping of these keys as ``stitch_params``; every key is
    optional and falls back to the default shown with it. Values are coerced and
    range-checked by ``validate_stitch_params`` when resolved. These are advanced
    knobs -- the defaults rarely need changing.
    """

    distance_tol: Annotated[float, Default(0.75)]
    """Sub-pixel tolerance for "lies on a bbox edge"."""

    min_edge_length: Annotated[float, Default(5.0)]
    """Absolute floor on cut-edge length (pixels)."""

    min_edge_length_ratio: Annotated[float, Default(0.4)]
    """Minimum cut-edge length relative to the cell's equivalent diameter."""

    min_edge_coverage: Annotated[float, Default(0.5)]
    """Minimum fraction of parallel-axis positions covered by near-edge contour points."""

    candidate_min_iou: Annotated[float, Default(0.2)]
    """Loose 1-D IoU floor at candidate enumeration."""

    close_radius: Annotated[int, Default(3)]
    """Morphological closing disk radius for the union mask. Also the length scale for
    ``gap_proximity`` (normalised by ``2 * close_radius``)."""


_STITCH_DEFAULTS: StitchParams = defaults_of(StitchParams)
