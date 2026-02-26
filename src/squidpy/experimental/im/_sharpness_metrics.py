from __future__ import annotations

import numpy as np
from scipy.fft import fft2, fftfreq
from skimage.filters import laplace, sobel_h, sobel_v


def _to_f32_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("block must be 2D")
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


def _tenengrad_mean(block: np.ndarray) -> np.ndarray:
    """Mean Tenengrad energy (sum of squared Sobel gradients)."""
    b = _to_f32_2d(block)
    energy = sobel_h(b) ** 2 + sobel_v(b) ** 2
    return np.array([[float(energy.mean())]], dtype=np.float32)


def _laplacian_variance(block: np.ndarray) -> np.ndarray:
    """Population variance of Laplacian response."""
    b = _to_f32_2d(block)
    lap = laplace(b)
    var_val = float(np.var(lap))
    return np.array([[max(var_val, 0.0)]], dtype=np.float32)


def _pop_variance(block: np.ndarray) -> np.ndarray:
    """Population variance of pixel intensities."""
    b = _to_f32_2d(block)
    return np.array([[float(np.var(b))]], dtype=np.float32)


def _fft_high_freq_energy(block: np.ndarray) -> np.ndarray:
    x = _to_f32_2d(block).astype(np.float64, copy=False)
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
    return np.array([[ratio]], dtype=np.float32)


def _haar_wavelet_energy(block: np.ndarray) -> np.ndarray:
    """Detail-band (LH+HL+HH) energy ratio of a single-level Haar transform."""
    x = _to_f32_2d(block).astype(np.float64, copy=False)
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
    return np.array([[ratio]], dtype=np.float32)
