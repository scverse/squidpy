from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np

from squidpy.experimental.im._intensity_metrics import (
    brightness_mean,
    brightness_std,
    entropy,
    eosin_mean,
    eosin_std,
    fold_fraction,
    he_ratio,
    hematoxylin_mean,
    hematoxylin_std,
    tissue_fraction,
)
from squidpy.experimental.im._sharpness_metrics import (
    fft_high_freq_energy,
    haar_wavelet_energy,
    laplacian_variance,
    pop_variance,
    tenengrad_mean,
)

MetricFn = Callable[[np.ndarray], np.ndarray]


class InputKind(StrEnum):
    GRAYSCALE = "grayscale"  # (ty, tx) float32
    RGB = "rgb"  # (ty, tx, 3) float32 in [0,1]
    MASK = "mask"  # (ty, tx) binary float32


class QCMetric(StrEnum):
    # Sharpness (grayscale input)
    TENENGRAD = "tenengrad"
    VAR_OF_LAPLACIAN = "var_of_laplacian"
    VARIANCE = "variance"
    FFT_HIGH_FREQ_ENERGY = "fft_high_freq_energy"
    HAAR_WAVELET_ENERGY = "haar_wavelet_energy"
    # Intensity (grayscale input)
    BRIGHTNESS_MEAN = "brightness_mean"
    BRIGHTNESS_STD = "brightness_std"
    ENTROPY = "entropy"
    # Staining (RGB input, H&E only)
    HEMATOXYLIN_MEAN = "hematoxylin_mean"
    HEMATOXYLIN_STD = "hematoxylin_std"
    EOSIN_MEAN = "eosin_mean"
    EOSIN_STD = "eosin_std"
    HE_RATIO = "he_ratio"
    # Artifacts (RGB input, H&E only)
    FOLD_FRACTION = "fold_fraction"
    # Tissue coverage (mask input)
    TISSUE_FRACTION = "tissue_fraction"


_HNE_METRICS: set[QCMetric] = {
    QCMetric.HEMATOXYLIN_MEAN,
    QCMetric.HEMATOXYLIN_STD,
    QCMetric.EOSIN_MEAN,
    QCMetric.EOSIN_STD,
    QCMetric.HE_RATIO,
    QCMetric.FOLD_FRACTION,
}


# --- Registry ---

_METRIC_REGISTRY: dict[QCMetric, tuple[InputKind, MetricFn]] = {
    # Sharpness (grayscale)
    QCMetric.TENENGRAD: (InputKind.GRAYSCALE, tenengrad_mean),
    QCMetric.VAR_OF_LAPLACIAN: (InputKind.GRAYSCALE, laplacian_variance),
    QCMetric.VARIANCE: (InputKind.GRAYSCALE, pop_variance),
    QCMetric.FFT_HIGH_FREQ_ENERGY: (InputKind.GRAYSCALE, fft_high_freq_energy),
    QCMetric.HAAR_WAVELET_ENERGY: (InputKind.GRAYSCALE, haar_wavelet_energy),
    # Intensity (grayscale)
    QCMetric.BRIGHTNESS_MEAN: (InputKind.GRAYSCALE, brightness_mean),
    QCMetric.BRIGHTNESS_STD: (InputKind.GRAYSCALE, brightness_std),
    QCMetric.ENTROPY: (InputKind.GRAYSCALE, entropy),
    # Staining (RGB, H&E only)
    QCMetric.HEMATOXYLIN_MEAN: (InputKind.RGB, hematoxylin_mean),
    QCMetric.HEMATOXYLIN_STD: (InputKind.RGB, hematoxylin_std),
    QCMetric.EOSIN_MEAN: (InputKind.RGB, eosin_mean),
    QCMetric.EOSIN_STD: (InputKind.RGB, eosin_std),
    QCMetric.HE_RATIO: (InputKind.RGB, he_ratio),
    # Artifacts (RGB, H&E only)
    QCMetric.FOLD_FRACTION: (InputKind.RGB, fold_fraction),
    # Tissue coverage (mask)
    QCMetric.TISSUE_FRACTION: (InputKind.MASK, tissue_fraction),
}


def get_metric_info(metric: QCMetric) -> tuple[InputKind, MetricFn]:
    """Look up the input kind and callable for a QCMetric."""
    return _METRIC_REGISTRY[metric]
