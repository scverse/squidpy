from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np

from squidpy.experimental.im._intensity_metrics import (
    _brightness_mean,
    _brightness_std,
    _entropy,
    _eosin_mean,
    _eosin_std,
    _fold_fraction,
    _he_ratio,
    _hematoxylin_mean,
    _hematoxylin_std,
    _tissue_fraction,
)
from squidpy.experimental.im._sharpness_metrics import (
    _fft_high_freq_energy,
    _haar_wavelet_energy,
    _laplacian_variance,
    _pop_variance,
    _tenengrad_mean,
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
    QCMetric.TENENGRAD: (InputKind.GRAYSCALE, _tenengrad_mean),
    QCMetric.VAR_OF_LAPLACIAN: (InputKind.GRAYSCALE, _laplacian_variance),
    QCMetric.VARIANCE: (InputKind.GRAYSCALE, _pop_variance),
    QCMetric.FFT_HIGH_FREQ_ENERGY: (InputKind.GRAYSCALE, _fft_high_freq_energy),
    QCMetric.HAAR_WAVELET_ENERGY: (InputKind.GRAYSCALE, _haar_wavelet_energy),
    # Intensity (grayscale)
    QCMetric.BRIGHTNESS_MEAN: (InputKind.GRAYSCALE, _brightness_mean),
    QCMetric.BRIGHTNESS_STD: (InputKind.GRAYSCALE, _brightness_std),
    QCMetric.ENTROPY: (InputKind.GRAYSCALE, _entropy),
    # Staining (RGB, H&E only)
    QCMetric.HEMATOXYLIN_MEAN: (InputKind.RGB, _hematoxylin_mean),
    QCMetric.HEMATOXYLIN_STD: (InputKind.RGB, _hematoxylin_std),
    QCMetric.EOSIN_MEAN: (InputKind.RGB, _eosin_mean),
    QCMetric.EOSIN_STD: (InputKind.RGB, _eosin_std),
    QCMetric.HE_RATIO: (InputKind.RGB, _he_ratio),
    # Artifacts (RGB, H&E only)
    QCMetric.FOLD_FRACTION: (InputKind.RGB, _fold_fraction),
    # Tissue coverage (mask)
    QCMetric.TISSUE_FRACTION: (InputKind.MASK, _tissue_fraction),
}


def get_metric_info(metric: QCMetric) -> tuple[InputKind, MetricFn]:
    """Look up the input kind and callable for a QCMetric."""
    return _METRIC_REGISTRY[metric]
