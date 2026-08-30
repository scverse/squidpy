from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Literal

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


#: The metrics :func:`~squidpy.experimental.im.qc_image` can compute. A ``Literal`` rather
#: than an enum: the members were already their own strings under ``StrEnum``, every use was
#: as a dict key or an equality test, and callers write the string anyway. The enum only
#: added a name to import -- and a validation that rejected the plain string it compared
#: equal to.
QCMetric = Literal[
    # Sharpness (grayscale input)
    "tenengrad",
    "var_of_laplacian",
    "variance",
    "fft_high_freq_energy",
    "haar_wavelet_energy",
    # Intensity (grayscale input)
    "brightness_mean",
    "brightness_std",
    "entropy",
    # Staining (RGB input, H&E only)
    "hematoxylin_mean",
    "hematoxylin_std",
    "eosin_mean",
    "eosin_std",
    "he_ratio",
    # Artifacts (RGB input, H&E only)
    "fold_fraction",
    # Tissue coverage (mask input)
    "tissue_fraction",
]


_HNE_METRICS: set[QCMetric] = {
    "hematoxylin_mean",
    "hematoxylin_std",
    "eosin_mean",
    "eosin_std",
    "he_ratio",
    "fold_fraction",
}


# --- Registry ---

_METRIC_REGISTRY: dict[QCMetric, tuple[InputKind, MetricFn]] = {
    # Sharpness (grayscale)
    "tenengrad": (InputKind.GRAYSCALE, tenengrad_mean),
    "var_of_laplacian": (InputKind.GRAYSCALE, laplacian_variance),
    "variance": (InputKind.GRAYSCALE, pop_variance),
    "fft_high_freq_energy": (InputKind.GRAYSCALE, fft_high_freq_energy),
    "haar_wavelet_energy": (InputKind.GRAYSCALE, haar_wavelet_energy),
    # Intensity (grayscale)
    "brightness_mean": (InputKind.GRAYSCALE, brightness_mean),
    "brightness_std": (InputKind.GRAYSCALE, brightness_std),
    "entropy": (InputKind.GRAYSCALE, entropy),
    # Staining (RGB, H&E only)
    "hematoxylin_mean": (InputKind.RGB, hematoxylin_mean),
    "hematoxylin_std": (InputKind.RGB, hematoxylin_std),
    "eosin_mean": (InputKind.RGB, eosin_mean),
    "eosin_std": (InputKind.RGB, eosin_std),
    "he_ratio": (InputKind.RGB, he_ratio),
    # Artifacts (RGB, H&E only)
    "fold_fraction": (InputKind.RGB, fold_fraction),
    # Tissue coverage (mask)
    "tissue_fraction": (InputKind.MASK, tissue_fraction),
}


def get_metric_info(metric: QCMetric) -> tuple[InputKind, MetricFn]:
    """Look up the input kind and callable for a QCMetric."""
    return _METRIC_REGISTRY[metric]
