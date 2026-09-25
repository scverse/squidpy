"""Public ``*Params`` types for :mod:`squidpy.experimental`, and their defaults."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, TypedDict

from squidpy._params import Default
from squidpy._utils import RNGLike, SeedLike

__all__ = [
    "FelzenszwalbParams",
    "WekaParams",
    "ReinhardParams",
    "MacenkoParams",
    "VahadaneParams",
]


class FelzenszwalbParams(TypedDict, total=False):
    """Size-aware superpixel defaults for felzenszwalb segmentation.

    A :class:`~typing.TypedDict`: pass a plain :class:`dict` with any subset of these keys.
    """

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


class WekaParams(TypedDict, total=False):
    """Parameters for WEKA-like trainable segmentation.

    A :class:`~typing.TypedDict`: pass a plain :class:`dict` with any subset of these keys.
    """

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


class ReinhardParams(TypedDict, total=False):
    """Tuning knobs for Reinhard stain normalization.

    A :class:`~typing.TypedDict`: pass a plain :class:`dict` with any subset of these keys.
    """

    luminosity_threshold: Annotated[float, Default(0.8)]
    """Normalised Ruderman Lab-L cutoff in ``(0, 1]``; pixels brighter than this are treated as
    near-white background and excluded from the fit. Follows HistomicsTK's ``reinhard``, so
    thresholds from the H&E literature transfer directly."""

    mask_background: Annotated[bool, Default(True)]
    """If ``True``, fit channel statistics over tissue pixels only; if ``False``, use every pixel (vanilla Reinhard)."""


class _ODBetaParams(TypedDict, total=False):
    # the key both decomposition methods take: it is the same quantity, so it is declared once
    beta: Annotated[float, Default(0.15)]
    """Mean-absorbance cutoff selecting tissue pixels (optical-density space)."""


class MacenkoParams(_ODBetaParams, total=False):
    """Tuning knobs for Macenko stain-matrix fitting.

    A :class:`~typing.TypedDict`: pass a plain :class:`dict` with any subset of these keys.
    """

    alpha: Annotated[float, Default(1.0)]
    """Angular percentile (deg) for the two stain directions; the extremes are taken at ``alpha`` / ``100 - alpha``."""


class VahadaneParams(_ODBetaParams, total=False):
    """Tuning knobs for Vahadane (sparse-NMF) stain-matrix fitting.

    A :class:`~typing.TypedDict`: pass a plain :class:`dict` with any subset of these keys.
    """

    lambda1: Annotated[float, Default(0.1)]
    """L1 sparsity regularisation on the concentration factor of the NMF."""

    n_iter: Annotated[int, Default(200)]
    """Maximum NMF iterations."""

    rng: Annotated[SeedLike | RNGLike | None, Default(None)]
    """Source of randomness for NMF initialisation tie-breaking; ``None`` draws from OS entropy."""
