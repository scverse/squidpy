from __future__ import annotations

import threading

import numpy as np

# --- Intensity metrics (grayscale input) ---


def brightness_mean(block: np.ndarray) -> np.ndarray:
    """Mean pixel intensity of a grayscale tile."""
    return np.array([[float(block.mean())]], dtype=np.float32)


def brightness_std(block: np.ndarray) -> np.ndarray:
    """Standard deviation of pixel intensity of a grayscale tile."""
    return np.array([[float(block.std())]], dtype=np.float32)


def entropy(block: np.ndarray) -> np.ndarray:
    """Shannon entropy of pixel intensity histogram."""
    arr = block.ravel()
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-10:
        return np.array([[0.0]], dtype=np.float32)
    # Quantize to 256 bins directly without storing intermediate normalized array
    bins = np.clip(((arr - lo) * (255.0 / (hi - lo))).astype(np.int32), 0, 255)
    counts = np.bincount(bins, minlength=256)
    probs = counts[counts > 0].astype(np.float64)
    probs /= probs.sum()
    ent = -float(np.dot(probs, np.log2(probs)))
    return np.array([[ent]], dtype=np.float32)


# --- Staining metrics (RGB input, H&E only) ---


def rgb_to_hed(block_rgb: np.ndarray) -> np.ndarray:
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
    result = rgb_to_hed(block_rgb)
    _hed_local.cache = (key, result)
    return result


def hematoxylin_mean(block: np.ndarray) -> np.ndarray:
    """Mean hematoxylin channel intensity."""
    hed = _rgb_to_hed_cached(block)
    return np.array([[float(hed[..., 0].mean())]], dtype=np.float32)


def hematoxylin_std(block: np.ndarray) -> np.ndarray:
    """Std of hematoxylin channel intensity."""
    hed = _rgb_to_hed_cached(block)
    return np.array([[float(hed[..., 0].std())]], dtype=np.float32)


def eosin_mean(block: np.ndarray) -> np.ndarray:
    """Mean eosin channel intensity."""
    hed = _rgb_to_hed_cached(block)
    return np.array([[float(hed[..., 1].mean())]], dtype=np.float32)


def eosin_std(block: np.ndarray) -> np.ndarray:
    """Std of eosin channel intensity."""
    hed = _rgb_to_hed_cached(block)
    return np.array([[float(hed[..., 1].std())]], dtype=np.float32)


def he_ratio(block: np.ndarray) -> np.ndarray:
    """Ratio of hematoxylin to eosin mean intensity."""
    hed = _rgb_to_hed_cached(block)
    h_mean = float(np.abs(hed[..., 0]).mean())
    e_mean = float(np.abs(hed[..., 1]).mean())
    ratio = h_mean / (e_mean + 1e-10)
    return np.array([[ratio]], dtype=np.float32)


# --- Artifact metrics (RGB input, H&E only) ---


def fold_fraction(block: np.ndarray) -> np.ndarray:
    """Fraction of pixels identified as tissue folds.

    Uses HSV thresholds tuned for H&E staining: saturation > 0.4 and
    value < 0.3 captures the dark, saturated appearance of folded tissue.
    """
    from skimage.color import rgb2hsv

    rgb_clipped = np.clip(block, 0.0, 1.0)
    hsv = rgb2hsv(rgb_clipped)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    fold_mask = (sat > 0.4) & (val < 0.3)
    frac = float(fold_mask.sum()) / max(fold_mask.size, 1)
    return np.array([[frac]], dtype=np.float32)


# --- Tissue coverage (mask input) ---


def tissue_fraction(block: np.ndarray) -> np.ndarray:
    """Fraction of pixels that are tissue (nonzero) in a binary mask tile."""
    return np.array([[float(block.mean())]], dtype=np.float32)
