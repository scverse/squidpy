from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum

import numpy as np

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


# --- Intensity metrics (grayscale input) ---


def _brightness_mean(block: np.ndarray) -> np.ndarray:
    """Mean pixel intensity of a grayscale tile."""
    return np.array([[float(block.mean())]], dtype=np.float32)


def _brightness_std(block: np.ndarray) -> np.ndarray:
    """Standard deviation of pixel intensity of a grayscale tile."""
    return np.array([[float(block.std())]], dtype=np.float32)


def _entropy(block: np.ndarray) -> np.ndarray:
    """Shannon entropy of pixel intensity histogram."""
    arr = block.ravel()
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-10:
        return np.array([[0.0]], dtype=np.float32)
    normalized = (arr - lo) / (hi - lo)
    bins = np.clip((normalized * 255).astype(np.int32), 0, 255)
    counts = np.bincount(bins, minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    ent = -float(np.sum(probs * np.log2(probs)))
    return np.array([[ent]], dtype=np.float32)


# --- Staining metrics (RGB input, H&E only) ---


def _rgb_to_hed(block_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB tile to HED colour space using Beer-Lambert deconvolution.

    Parameters
    ----------
    block_rgb
        (ty, tx, 3) float32 array in [0, 1].

    Returns
    -------
    (ty, tx, 3) float64 array with channels H, E, D.
    """
    from skimage.color import rgb2hed

    rgb_clipped = np.clip(block_rgb, 0.0, 1.0)
    return rgb2hed(rgb_clipped)


# Thread-local 1-entry cache: avoids re-computing HED deconvolution when
# multiple HED metrics are called sequentially on the same dask block.
# Uses threading.local so each dask worker thread gets its own cache,
# and ctypes.data (the buffer's memory address) as the key so it remains
# valid and unique while the array is alive.
_hed_local = threading.local()


def _rgb_to_hed_cached(block_rgb: np.ndarray) -> np.ndarray:
    """Cached wrapper around ``_rgb_to_hed``."""
    key = block_rgb.ctypes.data
    cache = getattr(_hed_local, "cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    result = _rgb_to_hed(block_rgb)
    _hed_local.cache = (key, result)
    return result


def _hematoxylin_mean(block: np.ndarray) -> np.ndarray:
    """Mean hematoxylin channel intensity."""
    hed = _rgb_to_hed_cached(block)
    return np.array([[float(hed[..., 0].mean())]], dtype=np.float32)


def _hematoxylin_std(block: np.ndarray) -> np.ndarray:
    """Std of hematoxylin channel intensity."""
    hed = _rgb_to_hed_cached(block)
    return np.array([[float(hed[..., 0].std())]], dtype=np.float32)


def _eosin_mean(block: np.ndarray) -> np.ndarray:
    """Mean eosin channel intensity."""
    hed = _rgb_to_hed_cached(block)
    return np.array([[float(hed[..., 1].mean())]], dtype=np.float32)


def _eosin_std(block: np.ndarray) -> np.ndarray:
    """Std of eosin channel intensity."""
    hed = _rgb_to_hed_cached(block)
    return np.array([[float(hed[..., 1].std())]], dtype=np.float32)


def _he_ratio(block: np.ndarray) -> np.ndarray:
    """Ratio of hematoxylin to eosin mean intensity."""
    hed = _rgb_to_hed_cached(block)
    h_mean = float(np.abs(hed[..., 0]).mean())
    e_mean = float(np.abs(hed[..., 1]).mean())
    ratio = h_mean / (e_mean + 1e-10)
    return np.array([[ratio]], dtype=np.float32)


# --- Artifact metrics (RGB input, H&E only) ---


def _fold_fraction(block: np.ndarray) -> np.ndarray:
    """Fraction of pixels identified as tissue folds (high saturation, low value)."""
    from skimage.color import rgb2hsv

    rgb_clipped = np.clip(block, 0.0, 1.0)
    hsv = rgb2hsv(rgb_clipped)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    fold_mask = (sat > 0.4) & (val < 0.3)
    frac = float(fold_mask.sum()) / max(fold_mask.size, 1)
    return np.array([[frac]], dtype=np.float32)


# --- Tissue coverage (mask input) ---


def _tissue_fraction(block: np.ndarray) -> np.ndarray:
    """Fraction of pixels that are tissue (nonzero) in a binary mask tile."""
    return np.array([[float(block.mean())]], dtype=np.float32)


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
