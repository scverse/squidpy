from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from squidpy.experimental.methods.registry import Registry


@dataclass
class _MeanShiftResult:
    """Toy result: a constant per-axis offset baked into ``transform``."""

    delta: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) + self.delta


def fit_mean_shift(ref: np.ndarray, query: np.ndarray) -> _MeanShiftResult:
    """Toy estimator function: fit the offset that maps the query centroid onto the ref centroid."""
    delta = np.asarray(ref, dtype=float).mean(0) - np.asarray(query, dtype=float).mean(0)
    return _MeanShiftResult(delta=delta, metadata={"method": "mean_shift"})


def test_fit_then_transform_round_trip() -> None:
    ref = np.array([[1.0, 1.0], [3.0, 3.0]])  # centroid (2, 2)
    query = np.array([[0.0, 0.0], [2.0, 2.0]])  # centroid (1, 1)

    result = fit_mean_shift(ref, query)

    np.testing.assert_allclose(result.delta, [1.0, 1.0])
    np.testing.assert_allclose(result.transform(query), query + 1.0)
    assert result.metadata == {"method": "mean_shift"}


def test_registry_register_get_keys() -> None:
    reg = Registry("demo")

    @reg.register("mean_shift")
    def _registered(ref: np.ndarray, query: np.ndarray) -> _MeanShiftResult:
        return fit_mean_shift(ref, query)

    assert reg.keys() == ("mean_shift",)
    assert reg.get("mean_shift") is _registered
    assert isinstance(reg.get("mean_shift")(np.ones((2, 2)), np.zeros((2, 2))), _MeanShiftResult)


def test_registry_unknown_key_lists_available() -> None:
    reg = Registry("demo")
    reg.register("a")(fit_mean_shift)

    with pytest.raises(ValueError, match=r"Unknown demo method 'b'. Available: \['a'\]"):
        reg.get("b")


def test_registry_rejects_duplicate_key() -> None:
    reg = Registry("demo")
    reg.register("dup")(fit_mean_shift)

    with pytest.raises(ValueError, match="already registered"):
        reg.register("dup")(fit_mean_shift)


def test_check_requirements_passes_when_none() -> None:
    reg = Registry("demo")
    # By default, registering without requires parameter does not wrap/check.
    reg.register("mean_shift")(fit_mean_shift)
    result = reg.get("mean_shift")(np.ones((2, 2)), np.zeros((2, 2)))
    assert isinstance(result, _MeanShiftResult)


def test_check_requirements_raises_for_missing_dependency() -> None:
    reg = Registry("demo")

    @reg.register("needs_ghost", requires=("squidpy_nonexistent_pkg_xyz",))
    def _needs_ghost(ref: np.ndarray, query: np.ndarray) -> _MeanShiftResult:
        return fit_mean_shift(ref, query)

    with pytest.raises(
        ImportError,
        match=r"Method 'needs_ghost' requires 'squidpy_nonexistent_pkg_xyz'.*squidpy\[squidpy_nonexistent_pkg_xyz\]",
    ):
        reg.get("needs_ghost")(np.ones((2, 2)), np.zeros((2, 2)))
