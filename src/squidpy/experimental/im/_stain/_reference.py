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


@dataclass(frozen=True, eq=False)
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
    white_point
        Shape ``(3,)`` per-channel white-point estimate. Required for
        decomposition methods (apply consumes it). Forbidden for Reinhard
        because Reinhard's color transfer operates in Ruderman Lab and
        does not model absorbance. There is no universal default; pass an
        estimate from your data (see ``estimate_white_point``).
    max_concentrations
        Shape ``(2,)`` reference per-stain (H, E) 99th-percentile concentrations
        - a fitted characterization of the reference's staining strength.
        Decomposition only, and diagnostic: the colour-basis ``apply`` transfers
        stain colour, not amount, so it does not consume this. Optional; forbidden
        for Reinhard.
    """

    method: StainMethod
    stain_matrix: np.ndarray | None = None
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    white_point: np.ndarray | None = None
    max_concentrations: np.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        # The numpy-array fields make the dataclass-generated __eq__ raise
        # ("truth value of an array is ambiguous"), so compare explicitly:
        # equal method plus element-wise-equal arrays.
        if not isinstance(other, StainReference):
            return NotImplemented
        if self.method != other.method:
            return False
        return all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in ("stain_matrix", "mu", "sigma", "white_point", "max_concentrations")
        )

    # eq=False keeps the default identity-based __hash__ (the array fields are
    # unhashable, so a value-based hash is impossible); references remain usable
    # as set members / dict keys by identity.
    __hash__ = object.__hash__

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected one of {sorted(_VALID_METHODS)}.")

        if self.method in _DECOMPOSITION_METHODS:
            if self.stain_matrix is None:
                raise ValueError(f"method={self.method!r} requires stain_matrix.")
            if self.mu is not None or self.sigma is not None:
                raise ValueError(f"method={self.method!r} forbids mu/sigma; pass them only for Reinhard.")
            if self.white_point is None:
                raise ValueError(f"method={self.method!r} requires white_point.")
            object.__setattr__(
                self,
                "stain_matrix",
                _coerce_finite(self.stain_matrix, shape=(3, 3), name="stain_matrix"),
            )
            bg = _coerce_finite(self.white_point, shape=(3,), name="white_point")
            if np.any(bg <= 0):
                raise ValueError("white_point must be strictly positive.")
            object.__setattr__(self, "white_point", bg)
            if self.max_concentrations is not None:
                maxc = _coerce_finite(self.max_concentrations, shape=(2,), name="max_concentrations")
                if np.any(maxc <= 0):
                    raise ValueError("max_concentrations must be strictly positive.")
                object.__setattr__(self, "max_concentrations", maxc)
        else:
            if self.mu is None or self.sigma is None:
                raise ValueError("method='reinhard' requires both mu and sigma.")
            if self.stain_matrix is not None:
                raise ValueError("method='reinhard' forbids stain_matrix.")
            if self.white_point is not None:
                raise ValueError(
                    "method='reinhard' forbids white_point; Reinhard's color "
                    "transfer is in Ruderman Lab and does not use a white point."
                )
            if self.max_concentrations is not None:
                raise ValueError("method='reinhard' forbids max_concentrations.")
            mu = _coerce_finite(self.mu, shape=(3,), name="mu")
            sigma = _coerce_finite(self.sigma, shape=(3,), name="sigma")
            if np.any(sigma <= 0):
                raise ValueError("sigma must be strictly positive.")
            object.__setattr__(self, "mu", mu)
            object.__setattr__(self, "sigma", sigma)
