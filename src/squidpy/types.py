"""Parameter and result types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, TypedDict

import numpy.typing as npt

from squidpy._params import Default
from squidpy._utils import RNGLike, SeedLike
from squidpy.gr._build import SpatialNeighborsResult
from squidpy.gr._nhood import NhoodEnrichmentResult

__all__ = [
    # Parameters
    "FelzenszwalbParams",
    "WekaParams",
    "ReinhardParams",
    "MacenkoParams",
    "VahadaneParams",
    "StalignImageParams",
    "StalignObsParams",
    "StalignVolumeParams",
    # Results
    "NhoodEnrichmentResult",
    "SpatialNeighborsResult",
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


class MacenkoParams(TypedDict, total=False):
    """Tuning knobs for Macenko stain-matrix fitting.

    A :class:`~typing.TypedDict`: pass a plain :class:`dict` with any subset of these keys.
    """

    alpha: Annotated[float, Default(1.0)]
    """Angular percentile (deg) for the two stain directions; the extremes are taken at ``alpha`` / ``100 - alpha``."""

    beta: Annotated[float, Default(0.15)]
    """Mean-absorbance cutoff selecting tissue pixels (optical-density space)."""


class VahadaneParams(TypedDict, total=False):
    """Tuning knobs for Vahadane (sparse-NMF) stain-matrix fitting.

    A :class:`~typing.TypedDict`: pass a plain :class:`dict` with any subset of these keys.
    """

    beta: Annotated[float, Default(0.15)]
    """Mean-absorbance cutoff selecting tissue pixels (optical-density space)."""

    lambda1: Annotated[float, Default(0.1)]
    """L1 sparsity regularisation on the concentration factor of the NMF."""

    n_iter: Annotated[int, Default(200)]
    """Maximum NMF iterations."""

    rng: Annotated[SeedLike | RNGLike | None, Default(None)]
    """Source of randomness for NMF initialisation tie-breaking; ``None`` draws from OS entropy."""


class StalignObsParams(TypedDict, total=False):
    """The LDDMM controls :func:`~squidpy.experimental.tl.stalign_align_obs` takes.

    Every key is optional and falls back to the default shown with it. The point-cloud
    path rasterizes both clouds into density images and then runs the same solver as the
    image path, so it takes everything that does plus the three rasterization knobs.
    """

    initial_affine: Annotated[npt.ArrayLike | None, Default(None)]
    """Homogeneous ``(3, 3)`` affine in ``(x, y)`` to start from."""

    initial_velocity: Annotated[npt.ArrayLike | None, Default(None)]
    """Continuation state from a prior fit, in the solver's row-column convention."""

    velocity_grid: Annotated[tuple[npt.ArrayLike, npt.ArrayLike] | None, Default(None)]
    """Axes ``initial_velocity`` lives on, row-column."""

    a: Annotated[float, Default(500.0)]
    """Sobolev kernel width -- the spatial scale the velocity field is smoothed over, so it
    sets how local a deformation can be."""

    p: Annotated[float, Default(2.0)]
    """Power of the regularisation operator."""

    expand: Annotated[float, Default(2.0)]
    """Padding factor sizing the velocity grid beyond the source extent."""

    nt: Annotated[int, Default(3)]
    """Integration steps along the deformation path."""

    niter: Annotated[int, Default(5000)]
    """Maximum iterations."""

    diffeo_start: Annotated[int, Default(0)]
    """Iteration at which the diffeomorphic part starts updating, letting the affine settle
    first."""

    epL: Annotated[float, Default(2e-08)]
    """Gradient-descent step size for the linear part. **Scale dependent.**"""

    epT: Annotated[float, Default(0.2)]
    """Gradient-descent step size for the translation. **Scale dependent.**"""

    epV: Annotated[float, Default(2000.0)]
    """Gradient-descent step size for the velocity field. **Scale dependent**, and the one to
    reach for first -- too large and the deformation overwhelms the affine."""

    sigmaM: Annotated[float, Default(1.0)]
    """Noise scale of the matching term."""

    sigmaB: Annotated[float, Default(2.0)]
    """Noise scale of the background term."""

    sigmaA: Annotated[float, Default(5.0)]
    """Noise scale of the artifact term."""

    sigmaR: Annotated[float, Default(500000.0)]
    """Noise scale of the regularisation term; larger penalises the velocity field less."""

    sigmaP: Annotated[float, Default(20.0)]
    """Noise scale of the landmark point-matching term."""

    muA: Annotated[npt.ArrayLike | None, Default(None)]
    """Fixed per-channel artifact means. ``None`` estimates them during fitting."""

    muB: Annotated[npt.ArrayLike | None, Default(None)]
    """Fixed per-channel background means. ``None`` estimates them during fitting."""

    tol: Annotated[float | None, Default(None)]
    """Relative objective improvement below which the fit stops early. ``None`` always runs
    ``niter``."""

    patience: Annotated[int, Default(25)]
    """Iterations the improvement must stay under ``tol`` before stopping."""

    dx: Annotated[float, Default(30.0)]
    """Grid spacing of the density rasters."""

    blur: Annotated[float | Sequence[float], Default((2.0, 1.0, 0.5))]
    """Gaussian blur scale(s) applied to the rasters."""

    raster_expand: Annotated[float, Default(1.1)]
    """Field-of-view padding factor for the rasters."""


class StalignImageParams(TypedDict, total=False):
    """The LDDMM controls :func:`~squidpy.experimental.tl.stalign_align_image` takes.

    Every key is optional and falls back to the default shown with it. The same schema as
    :class:`~squidpy.types.StalignObsParams` minus the rasterization knobs, and with the
    step sizes retuned: a kernel width in pixels is not one in cell coordinates.
    """

    initial_affine: Annotated[npt.ArrayLike | None, Default(None)]
    """Homogeneous ``(3, 3)`` affine in ``(x, y)`` to start from."""

    initial_velocity: Annotated[npt.ArrayLike | None, Default(None)]
    """Continuation state from a prior fit, in the solver's row-column convention."""

    velocity_grid: Annotated[tuple[npt.ArrayLike, npt.ArrayLike] | None, Default(None)]
    """Axes ``initial_velocity`` lives on, row-column."""

    a: Annotated[float, Default(20.0)]
    """Sobolev kernel width -- the spatial scale the velocity field is smoothed over, so it
    sets how local a deformation can be."""

    p: Annotated[float, Default(2.0)]
    """Power of the regularisation operator."""

    expand: Annotated[float, Default(2.0)]
    """Padding factor sizing the velocity grid beyond the source extent."""

    nt: Annotated[int, Default(3)]
    """Integration steps along the deformation path."""

    niter: Annotated[int, Default(200)]
    """Maximum iterations."""

    diffeo_start: Annotated[int, Default(100)]
    """Iteration at which the diffeomorphic part starts updating, letting the affine settle
    first."""

    epL: Annotated[float, Default(2e-08)]
    """Gradient-descent step size for the linear part. **Scale dependent.**"""

    epT: Annotated[float, Default(0.2)]
    """Gradient-descent step size for the translation. **Scale dependent.**"""

    epV: Annotated[float, Default(1.0)]
    """Gradient-descent step size for the velocity field. **Scale dependent**, and the one to
    reach for first -- too large and the deformation overwhelms the affine."""

    sigmaM: Annotated[float, Default(1.0)]
    """Noise scale of the matching term."""

    sigmaB: Annotated[float, Default(2.0)]
    """Noise scale of the background term."""

    sigmaA: Annotated[float, Default(5.0)]
    """Noise scale of the artifact term."""

    sigmaR: Annotated[float, Default(500000.0)]
    """Noise scale of the regularisation term; larger penalises the velocity field less."""

    sigmaP: Annotated[float, Default(20.0)]
    """Noise scale of the landmark point-matching term."""

    muA: Annotated[npt.ArrayLike | None, Default(None)]
    """Fixed per-channel artifact means. ``None`` estimates them during fitting."""

    muB: Annotated[npt.ArrayLike | None, Default(None)]
    """Fixed per-channel background means. ``None`` estimates them during fitting."""

    tol: Annotated[float | None, Default(None)]
    """Relative objective improvement below which the fit stops early. ``None`` always runs
    ``niter``."""

    patience: Annotated[int, Default(25)]
    """Iterations the improvement must stay under ``tol`` before stopping."""


class StalignVolumeParams(TypedDict, total=False):
    """The LDDMM controls :func:`~squidpy.experimental.tl.stalign_align_volume` takes.

    Every key is optional and falls back to the default shown with it. Two keys the rank-2
    schemas carry are absent: ``sigmaP``, because the rank-3 path has no point-matching
    energy, and ``initial_affine``, which is a named ``(4, 4)`` argument on the function
    rather than a solver key.
    """

    initial_velocity: Annotated[npt.ArrayLike | None, Default(None)]
    """Continuation state from a prior fit, in the solver's row-column convention."""

    velocity_grid: Annotated[tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike] | None, Default(None)]
    """Axes ``initial_velocity`` lives on, row-column."""

    a: Annotated[float, Default(500.0)]
    """Sobolev kernel width -- the spatial scale the velocity field is smoothed over, so it
    sets how local a deformation can be."""

    p: Annotated[float, Default(2.0)]
    """Power of the regularisation operator."""

    expand: Annotated[float, Default(1.25)]
    """Padding factor sizing the velocity grid beyond the source extent."""

    nt: Annotated[int, Default(3)]
    """Integration steps along the deformation path."""

    niter: Annotated[int, Default(5000)]
    """Maximum iterations."""

    diffeo_start: Annotated[int, Default(0)]
    """Iteration at which the diffeomorphic part starts updating, letting the affine settle
    first."""

    epL: Annotated[float, Default(1e-06)]
    """Gradient-descent step size for the linear part. **Scale dependent.**"""

    epT: Annotated[float, Default(10.0)]
    """Gradient-descent step size for the translation. **Scale dependent.**"""

    epV: Annotated[float, Default(1000.0)]
    """Gradient-descent step size for the velocity field. **Scale dependent**, and the one to
    reach for first -- too large and the deformation overwhelms the affine."""

    sigmaM: Annotated[float, Default(1.0)]
    """Noise scale of the matching term."""

    sigmaB: Annotated[float, Default(2.0)]
    """Noise scale of the background term."""

    sigmaA: Annotated[float, Default(5.0)]
    """Noise scale of the artifact term."""

    sigmaR: Annotated[float, Default(1000000.0)]
    """Noise scale of the regularisation term; larger penalises the velocity field less."""

    muA: Annotated[npt.ArrayLike | None, Default(None)]
    """Fixed per-channel artifact means. ``None`` estimates them during fitting."""

    muB: Annotated[npt.ArrayLike | None, Default(None)]
    """Fixed per-channel background means. ``None`` estimates them during fitting."""

    tol: Annotated[float | None, Default(None)]
    """Relative objective improvement below which the fit stops early. ``None`` always runs
    ``niter``."""

    patience: Annotated[int, Default(25)]
    """Iterations the improvement must stay under ``tol`` before stopping."""
