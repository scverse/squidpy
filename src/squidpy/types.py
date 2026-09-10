"""Public parameter bags, result schemas and the protocols they plug into."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, NamedTuple, Protocol, Self, TypedDict, overload, runtime_checkable

import numpy as np
import pandas as pd
from fast_array_utils.types import HasArrayNamespace as Array

from squidpy._utils import RNGLike, SeedLike
from squidpy.experimental.utils._params import Default, defaults_of

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
    "Clusterer",
    "SweepableClusterer",
    "ClusterAutoKResult",
    "BackgroundDetectionParams",
    "FelzenszwalbParams",
    "WekaParams",
    "ReinhardParams",
    "MacenkoParams",
    "VahadaneParams",
    "TilingQCParams",
    "StitchParams",
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
    """Tuning knobs for Reinhard stain normalization."""

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
    """Advanced tuning knobs for :func:`~squidpy.experimental.tl.calculate_tiling_qc`."""

    distance_tol: Annotated[float, Default(0.75)]
    """Maximum perpendicular distance (pixels) from the fitted line for a contour point to count as straight."""

    min_area: Annotated[int, Default(20)]
    """Cells smaller than this (pixels at analysis resolution) are skipped (NaN scores)."""

    max_contour_points: Annotated[int, Default(500)]
    """Cap on contour resolution; longer contours are arc-length-resampled before the O(n^2) collinearity scan."""


_QC_DEFAULTS: TilingQCParams = defaults_of(TilingQCParams)


class StitchParams(TypedDict, total=False):
    """Advanced tuning knobs for :func:`~squidpy.experimental.tl.assign_stitch_groups`.

    The defaults suit typical 2D segmentation tiles from cellpose-like pipelines.
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


@runtime_checkable
class Clusterer(Protocol):
    """Assigns one cluster label per observation.

    Structural on purpose, so that scikit-learn, cuML and hand-written estimators all
    qualify: :class:`~sklearn.mixture.GaussianMixture` is a
    :class:`~sklearn.base.DensityMixin` rather than a :class:`~sklearn.base.ClusterMixin`,
    and cuML does not import scikit-learn at all, so neither would pass a check against a
    base class.

    Runtime-checkable, so ``isinstance(estimator, Clusterer)`` answers whether something
    can be used as one -- by method *presence*, which is as far as
    :func:`~typing.runtime_checkable` goes.
    """

    def fit_predict(self, X: Array) -> Array:
        """Cluster *X*, observations as rows, and return one label per row."""
        ...


@runtime_checkable
class SweepableClusterer(Clusterer, Protocol):
    """A :class:`Clusterer` squidpy re-fits itself, setting the parameters of each fit.

    Required wherever one clusterer is fitted repeatedly:
    :func:`~squidpy.gr.sweep_auto_k` fits per candidate K and per run, and the niche
    pipeline fits per library. Every fit goes to a fresh :func:`~sklearn.base.clone`, so
    the estimator passed in is never mutated -- and cloning needs nothing beyond these two
    methods, so inheriting from scikit-learn is not required.
    """

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """The constructor parameters, as :func:`~sklearn.base.clone` reads them.

        Only the keys are read, to check up front that the parameters ``set_params`` will
        set are accepted at all.
        """
        ...

    @overload
    def set_params(self, *, n_components: int, random_state: int) -> Self: ...

    @overload
    def set_params(self, *, n_clusters: int, random_state: int) -> Self: ...

    @overload
    def set_params(self, *, random_state: int) -> Self: ...

    def set_params(self, **params: object) -> Self:
        """Set the number of clusters and the seed of the next fit; returns the estimator.

        The overloads are the contract on the parameters: ``random_state``, plus *one* of
        the two number-of-clusters spellings -- :class:`~sklearn.mixture.GaussianMixture`
        calls it ``n_components``, :class:`~sklearn.cluster.KMeans` calls it
        ``n_clusters``. The third is a re-fit at a fixed K, as the niche pipeline does per
        library.

        Which spelling an estimator takes is a property of its parameters rather than of
        its methods, so ``isinstance`` cannot see it and
        :func:`~squidpy.gr.sweep_auto_k` checks it before fitting instead.
        """
        ...


class ClusterAutoKResult(NamedTuple):
    """A sweep result."""

    #: Per-K diagnostics indexed by K, with the columns ``stability_mean``, ``stability_std``
    #: and ``nll``. Every fitted K has a row, but the ``+-1`` halo is never scored, so its
    #: stability is ``NaN``.
    table: pd.DataFrame

    #: Raw similarity values, of shape ``(n_scored_k, n_comparisons)``. Row ``i`` belongs to
    #: the ``i``-th scored K.
    stability: np.ndarray

    #: The scored K with the highest mean stability.
    best_k: int

    #: Number of runs actually performed, below ``max_runs`` if the sweep converged.
    n_runs: int

    #: Whether the sweep stopped early because the stability curve had settled.
    converged: bool

    #: Labeling of the best fit (lowest ``nll``) per K, for every fitted K.
    labels: dict[int, np.ndarray]
