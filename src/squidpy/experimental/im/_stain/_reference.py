"""Persistent reference object for a fitted stain model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from squidpy.experimental.im._stain._constants import (
    DEFAULT_BACKGROUND_INTENSITY,
    STAIN_REFERENCE_SCHEMA_VERSION,
    StainMethod,
)

_DECOMPOSITION_METHODS: frozenset[StainMethod] = frozenset({StainMethod.MACENKO, StainMethod.VAHADANE})
_SCHEMA_TAG: str = "squidpy.stain_reference"
_ENCODED_NDARRAY_KEYS: frozenset[str] = frozenset({"__ndarray__", "dtype", "shape"})

_PERSISTED_ARRAY_FIELDS: tuple[str, ...] = (
    "stain_matrix",
    "max_concentrations",
    "mu",
    "sigma",
    "background_intensity",
)


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

    Holds either a stain matrix (for decomposition methods) or a pair of
    channel statistics (for the Reinhard statistical method). Round-trips
    through JSON via :meth:`save` and :meth:`load`. Frozen so it can be
    passed safely across worker processes.

    Parameters
    ----------
    method
        Fitting method. See :class:`StainMethod`.
    version
        Schema version. Set automatically; bump
        :data:`STAIN_REFERENCE_SCHEMA_VERSION` for migrations.
    stain_matrix
        Shape ``(3, 3)`` unit-norm matrix in canonical order
        ``(H, E, complement)``. Required for decomposition methods.
    max_concentrations
        Optional shape ``(2,)`` p99 concentrations of H and E used for
        dynamic-range matching during apply. Decomposition methods only.
    mu
        Shape ``(3,)`` Ruderman Lab channel means. Reinhard only.
    sigma
        Shape ``(3,)`` Ruderman Lab channel standard deviations. Reinhard
        only.
    background_intensity
        Shape ``(3,)`` per-channel white-point estimate.
    fit_metadata
        Free-form dict for provenance (squidpy version, image keys, pyramid
        level, seed, timestamp, aggregation rule).
    cohort_members
        Tuple of image keys aggregated into this reference. Cohort fits
        only.
    per_image_stats
        Raw per-image fits keyed by image. Cohort fits only.
    """

    method: StainMethod
    version: int = STAIN_REFERENCE_SCHEMA_VERSION
    stain_matrix: np.ndarray | None = None
    max_concentrations: np.ndarray | None = None
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    background_intensity: np.ndarray = field(default_factory=lambda: DEFAULT_BACKGROUND_INTENSITY.copy())
    fit_metadata: dict = field(default_factory=dict)
    cohort_members: tuple[str, ...] | None = None
    per_image_stats: dict | None = None

    def __post_init__(self) -> None:
        try:
            method = StainMethod(self.method)
        except ValueError as exc:
            raise ValueError(
                f"Unknown method {self.method!r}; expected one of {[m.value for m in StainMethod]}."
            ) from exc
        object.__setattr__(self, "method", method)

        if method in _DECOMPOSITION_METHODS:
            self._validate_decomposition()
        else:
            self._validate_reinhard()

        bg = _coerce_finite(self.background_intensity, shape=(3,), name="background_intensity")
        if np.any(bg <= 0):
            raise ValueError("background_intensity must be strictly positive.")
        object.__setattr__(self, "background_intensity", bg)

        if self.cohort_members is not None and not isinstance(self.cohort_members, tuple):
            object.__setattr__(self, "cohort_members", tuple(self.cohort_members))

    def _validate_decomposition(self) -> None:
        if self.stain_matrix is None:
            raise ValueError(f"method={self.method.value!r} requires stain_matrix.")
        if self.mu is not None or self.sigma is not None:
            raise ValueError(f"method={self.method.value!r} forbids mu/sigma; pass them only for Reinhard.")
        object.__setattr__(
            self,
            "stain_matrix",
            _coerce_finite(self.stain_matrix, shape=(3, 3), name="stain_matrix"),
        )
        if self.max_concentrations is not None:
            object.__setattr__(
                self,
                "max_concentrations",
                _coerce_finite(self.max_concentrations, shape=(2,), name="max_concentrations"),
            )

    def _validate_reinhard(self) -> None:
        if self.mu is None or self.sigma is None:
            raise ValueError("method='reinhard' requires both mu and sigma.")
        if self.stain_matrix is not None or self.max_concentrations is not None:
            raise ValueError("method='reinhard' forbids stain_matrix/max_concentrations.")
        mu = _coerce_finite(self.mu, shape=(3,), name="mu")
        sigma = _coerce_finite(self.sigma, shape=(3,), name="sigma")
        if np.any(sigma <= 0):
            raise ValueError("sigma must be strictly positive.")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)

    def save(self, path: str | Path) -> None:
        """Write the reference to a JSON file."""
        Path(path).write_text(json.dumps(_to_json(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> StainReference:
        """Load a reference previously written by :meth:`save`."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse StainReference JSON at {path!s}: {exc}") from exc
        return _from_json(payload)


def _encode_ndarray(arr: np.ndarray) -> dict[str, Any]:
    return {"__ndarray__": arr.tolist(), "dtype": str(arr.dtype), "shape": list(arr.shape)}


def _decode_ndarray(blob: dict[str, Any]) -> np.ndarray:
    return np.asarray(blob["__ndarray__"], dtype=np.dtype(blob["dtype"])).reshape(blob["shape"])


def _is_encoded_ndarray(value: Any) -> bool:
    return isinstance(value, dict) and _ENCODED_NDARRAY_KEYS.issubset(value.keys())


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _encode_ndarray(value)
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any) -> Any:
    if _is_encoded_ndarray(value):
        return _decode_ndarray(value)
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _to_json(ref: StainReference) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": _SCHEMA_TAG,
        "version": ref.version,
        "method": ref.method.value,
        "fit_metadata": _encode(ref.fit_metadata),
    }
    for name in _PERSISTED_ARRAY_FIELDS:
        value = getattr(ref, name)
        if value is not None:
            payload[name] = _encode_ndarray(value)
    if ref.cohort_members is not None:
        payload["cohort_members"] = list(ref.cohort_members)
    if ref.per_image_stats is not None:
        payload["per_image_stats"] = _encode(ref.per_image_stats)
    return payload


def _from_json(payload: dict[str, Any]) -> StainReference:
    if payload.get("schema") != _SCHEMA_TAG:
        raise ValueError("File is not a squidpy stain reference (missing schema marker).")
    version = int(payload.get("version", 1))
    if version > STAIN_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"StainReference file version {version} is newer than this squidpy "
            f"(supports up to {STAIN_REFERENCE_SCHEMA_VERSION}). Upgrade squidpy to load it."
        )
    kwargs: dict[str, Any] = {
        "method": payload["method"],
        "version": version,
        "fit_metadata": _decode(payload.get("fit_metadata", {})),
    }
    for name in _PERSISTED_ARRAY_FIELDS:
        if name in payload:
            kwargs[name] = _decode_ndarray(payload[name])
    if "cohort_members" in payload:
        kwargs["cohort_members"] = tuple(payload["cohort_members"])
    if "per_image_stats" in payload:
        kwargs["per_image_stats"] = _decode(payload["per_image_stats"])
    return StainReference(**kwargs)
