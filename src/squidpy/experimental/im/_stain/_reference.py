"""Slim container for a fitted stain reference.

Holds either a 3x3 stain matrix (Macenko/Vahadane, ships in PR 3) or a
pair of Ruderman Lab channel statistics (Reinhard, ships in PR 2). The
dataclass is intentionally minimal in this PR; cohort fields, persistence,
and provenance metadata land alongside their first consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

StainMethod = Literal["macenko", "vahadane", "reinhard"]
_DECOMPOSITION_METHODS: frozenset[str] = frozenset({"macenko", "vahadane"})
_VALID_METHODS: frozenset[str] = _DECOMPOSITION_METHODS | {"reinhard"}


def _coerce_finite(arr: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values.")
    return out


@dataclass(frozen=True)
class StainReference:
    """Container for a fitted stain reference.

    Parameters
    ----------
    method
        Fitting method: ``"macenko"``, ``"vahadane"``, or ``"reinhard"``.
    stain_matrix
        Shape ``(3, 3)`` unit-norm matrix in canonical order
        ``(H, E, complement)``. Required for decomposition methods.
    mu
        Shape ``(3,)`` Ruderman Lab channel means. Reinhard only.
    sigma
        Shape ``(3,)`` Ruderman Lab channel standard deviations. Reinhard
        only.
    background_intensity
        Shape ``(3,)`` per-channel white-point estimate. Required for
        decomposition methods (apply consumes it). Forbidden for Reinhard
        because Reinhard's color transfer operates in Ruderman Lab and
        does not model absorbance. There is no universal default; pass an
        estimate from your data (PR 3 ships the estimator).
    """

    method: StainMethod
    stain_matrix: np.ndarray | None = None
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    background_intensity: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected one of {sorted(_VALID_METHODS)}.")

        if self.method in _DECOMPOSITION_METHODS:
            if self.stain_matrix is None:
                raise ValueError(f"method={self.method!r} requires stain_matrix.")
            if self.mu is not None or self.sigma is not None:
                raise ValueError(f"method={self.method!r} forbids mu/sigma; pass them only for Reinhard.")
            if self.background_intensity is None:
                raise ValueError(f"method={self.method!r} requires background_intensity.")
            object.__setattr__(
                self,
                "stain_matrix",
                _coerce_finite(self.stain_matrix, shape=(3, 3), name="stain_matrix"),
            )
            bg = _coerce_finite(self.background_intensity, shape=(3,), name="background_intensity")
            if np.any(bg <= 0):
                raise ValueError("background_intensity must be strictly positive.")
            object.__setattr__(self, "background_intensity", bg)
        else:
            if self.mu is None or self.sigma is None:
                raise ValueError("method='reinhard' requires both mu and sigma.")
            if self.stain_matrix is not None:
                raise ValueError("method='reinhard' forbids stain_matrix.")
            if self.background_intensity is not None:
                raise ValueError(
                    "method='reinhard' forbids background_intensity; Reinhard's color "
                    "transfer is in Ruderman Lab and does not use a white point."
                )
            mu = _coerce_finite(self.mu, shape=(3,), name="mu")
            sigma = _coerce_finite(self.sigma, shape=(3,), name="sigma")
            if np.any(sigma <= 0):
                raise ValueError("sigma must be strictly positive.")
            object.__setattr__(self, "mu", mu)
            object.__setattr__(self, "sigma", sigma)
