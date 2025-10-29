from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numba
import numpy as np
from numba import njit
from scipy.fft import fft2, fftfreq

# One thread to avoid clashes with Dask
numba.set_num_threads(1)


MetricFn = Callable[[np.ndarray], np.ndarray]


class SharpnessMetric(str, Enum):
    TENENGRAD = "tenengrad"
    VAR_OF_LAPLACIAN = "var_of_laplacian"
    VARIANCE = "variance"
    FFT_HIGH_FREQ_ENERGY = "fft_high_freq_energy"
    HAAR_WAVELET_ENERGY = "haar_wavelet_energy"


def _ensure_f32_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("block must be 2D")
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


@njit(cache=True, fastmath=True)
def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True, fastmath=True)
def _tenengrad_mean(block: np.ndarray) -> np.ndarray:
    """Mean Tenengrad energy using Sobel 3×3."""
    h, w = block.shape
    gxk = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    gyk = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=np.float32)
    s = 0.0
    for i in range(h):
        for j in range(w):
            gx = 0.0
            gy = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ii = _clamp(i + di, 0, h - 1)
                    jj = _clamp(j + dj, 0, w - 1)
                    v = block[ii, jj]
                    gx += gxk[di + 1, dj + 1] * v
                    gy += gyk[di + 1, dj + 1] * v
            s += gx * gx + gy * gy
    mean_val = s / (h * w)
    return np.full_like(block, mean_val, dtype=np.float32)


@njit(cache=True, fastmath=True)
def _laplacian_variance(block: np.ndarray) -> np.ndarray:
    """Population variance of Laplacian response."""
    h, w = block.shape
    lk = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    n = h * w
    s = 0.0
    s2 = 0.0
    for i in range(h):
        for j in range(w):
            y = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ii = _clamp(i + di, 0, h - 1)
                    jj = _clamp(j + dj, 0, w - 1)
                    y += lk[di + 1, dj + 1] * block[ii, jj]
            s += y
            s2 += y * y
    mean = s / n
    # var = E[y^2] - (E[y])^2
    var = (s2 / n) - (mean * mean)
    var_val = var if var > 0.0 else 0.0
    return np.full_like(block, var_val, dtype=np.float32)


@njit(cache=True, fastmath=True)
def _pop_variance(block: np.ndarray) -> np.ndarray:
    """Population variance of pixel intensities."""
    h, w = block.shape
    n = h * w
    s = 0.0
    s2 = 0.0
    for i in range(h):
        for j in range(w):
            v = block[i, j]
            s += v
            s2 += v * v
    mean = s / n
    var = (s2 / n) - (mean * mean)
    var_val = var if var > 0.0 else 0.0
    return np.full_like(block, var_val, dtype=np.float32)


def _fft_high_freq_energy(block: np.ndarray) -> np.ndarray:
    x = _ensure_f32_2d(block).astype(np.float64, copy=False)
    m = float(x.mean())
    s = float(x.std())
    x = (x - m) / s if s > 0.0 else (x - m)

    F = fft2(x)
    mag2 = (F.real * F.real) + (F.imag * F.imag)

    h, w = x.shape
    fy = fftfreq(h)
    fx = fftfreq(w)
    ry, rx = np.meshgrid(fy, fx, indexing="ij")
    r = np.hypot(ry, rx)
    mask = r > 0.1

    total = float(mag2.sum())
    if not np.isfinite(total) or total <= 1e-12:
        ratio = 0.0
    else:
        hi = float(mag2[mask].sum())
        ratio = hi / total if np.isfinite(hi) else 0.0
        if ratio < 0.0:
            ratio = 0.0
        if ratio > 1.0:
            ratio = 1.0
    return np.full_like(block, ratio, dtype=np.float32)


def _haar_wavelet_energy(block: np.ndarray) -> np.ndarray:
    """Detail-band (LH+HL+HH) energy ratio of a single-level Haar transform."""
    x = _ensure_f32_2d(block).astype(np.float64, copy=False)
    m = float(x.mean())
    s = float(x.std())
    x = (x - m) / s if s > 0.0 else (x - m)

    h, w = x.shape
    if h % 2 == 1:
        x = np.vstack([x, x[-1:, :]])
        h += 1
    if w % 2 == 1:
        x = np.hstack([x, x[:, -1:]])
        w += 1

    cA_h = (x[::2, :] + x[1::2, :]) / 2.0
    cH_h = (x[::2, :] - x[1::2, :]) / 2.0

    cA = (cA_h[:, ::2] + cA_h[:, 1::2]) / 2.0  # LL
    cH = (cA_h[:, ::2] - cA_h[:, 1::2]) / 2.0  # LH
    cV = (cH_h[:, ::2] + cH_h[:, 1::2]) / 2.0  # HL
    cD = (cH_h[:, ::2] - cH_h[:, 1::2]) / 2.0  # HH

    total = float((cA * cA).sum() + (cH * cH).sum() + (cV * cV).sum() + (cD * cD).sum())
    if not np.isfinite(total) or total <= 1e-12:
        ratio = 0.0
    else:
        detail = float((cH * cH).sum() + (cV * cV).sum() + (cD * cD).sum())
        ratio = detail / total if np.isfinite(detail) else 0.0
        if ratio < 0.0:
            ratio = 0.0
        if ratio > 1.0:
            ratio = 1.0
    return np.full_like(block, ratio, dtype=np.float32)


_METRICS: dict[SharpnessMetric, MetricFn] = {
    SharpnessMetric.TENENGRAD: lambda a: _tenengrad_mean(_ensure_f32_2d(a)),
    SharpnessMetric.VAR_OF_LAPLACIAN: lambda a: _laplacian_variance(_ensure_f32_2d(a)),
    SharpnessMetric.VARIANCE: lambda a: _pop_variance(_ensure_f32_2d(a)),
    SharpnessMetric.FFT_HIGH_FREQ_ENERGY: _fft_high_freq_energy,
    SharpnessMetric.HAAR_WAVELET_ENERGY: _haar_wavelet_energy,
}


def _get_sharpness_metric_function(metric: str | SharpnessMetric) -> MetricFn:
    if isinstance(metric, str):
        try:
            metric = SharpnessMetric(metric.lower())
        except ValueError as e:
            avail = ", ".join(m.value for m in SharpnessMetric)
            raise ValueError(f"Unknown metric '{metric}'. Available: {avail}") from e
    return _METRICS[metric]
