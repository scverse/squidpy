from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from squidpy.experimental.methods.registry import AlignMethod, Registry


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

    @reg.register("mean_shift", "obs")
    def _registered(ref: np.ndarray, query: np.ndarray) -> _MeanShiftResult:
        return fit_mean_shift(ref, query)

    assert reg.keys() == ("mean_shift",)
    assert reg.get("mean_shift").implementation("obs") is _registered
    assert isinstance(reg.get("mean_shift").obs(np.ones((2, 2)), np.zeros((2, 2))), _MeanShiftResult)


def test_one_method_accumulates_several_modalities() -> None:
    """A method is one record with a slot per modality, not one entry per modality."""
    reg = Registry("demo")
    reg.register("multi", "obs")(fit_mean_shift)
    reg.register("multi", "images")(fit_mean_shift)

    method = reg.get("multi")
    assert reg.keys() == ("multi",)
    assert method.supports() == ("obs", "images")
    assert method.landmarks is None


def test_unsupported_modality_says_what_is_supported() -> None:
    reg = Registry("demo")
    reg.register("obs_only", "obs")(fit_mean_shift)

    with pytest.raises(ValueError, match="'obs_only' does not support landmarks alignment. It supports: obs"):
        reg.get("obs_only").implementation("landmarks")


def test_supporting_filters_by_modality() -> None:
    reg = Registry("demo")
    reg.register("a", "obs")(fit_mean_shift)
    reg.register("b", "landmarks")(fit_mean_shift)
    reg.register("c", "obs")(fit_mean_shift)

    assert reg.supporting("obs") == ("a", "c")
    assert reg.supporting("landmarks") == ("b",)
    assert reg.supporting("images") == ()


def test_registry_unknown_key_lists_available() -> None:
    reg = Registry("demo")
    reg.register("a", "obs")(fit_mean_shift)

    with pytest.raises(ValueError, match=r"Unknown demo method 'b'. Available: \['a'\]"):
        reg.get("b")


def test_registry_rejects_duplicate_slot() -> None:
    reg = Registry("demo")
    reg.register("dup", "obs")(fit_mean_shift)

    with pytest.raises(ValueError, match="already has a obs estimator"):
        reg.register("dup", "obs")(fit_mean_shift)


def test_registry_rejects_unknown_modality() -> None:
    reg = Registry("demo")
    with pytest.raises(ValueError, match="Unknown modality 'obsm'"):
        reg.register("x", "obsm")  # type: ignore[arg-type]


def test_align_method_supports_reports_filled_slots() -> None:
    assert AlignMethod(name="empty").supports() == ()
    assert AlignMethod(name="x", obs=fit_mean_shift, landmarks=fit_mean_shift).supports() == ("obs", "landmarks")


def test_check_requirements_passes_when_none() -> None:
    reg = Registry("demo")
    # By default, registering without requires parameter does not wrap/check.
    reg.register("mean_shift", "obs")(fit_mean_shift)
    result = reg.get("mean_shift").implementation("obs")(np.ones((2, 2)), np.zeros((2, 2)))
    assert isinstance(result, _MeanShiftResult)


def test_check_requirements_raises_for_missing_dependency() -> None:
    reg = Registry("demo")

    @reg.register("needs_ghost", "obs", requires=("squidpy_nonexistent_pkg_xyz",))
    def _needs_ghost(ref: np.ndarray, query: np.ndarray) -> _MeanShiftResult:
        return fit_mean_shift(ref, query)

    with pytest.raises(
        ImportError,
        match=r"Method 'needs_ghost' requires 'squidpy_nonexistent_pkg_xyz'.*squidpy\[squidpy_nonexistent_pkg_xyz\]",
    ):
        reg.get("needs_ghost").implementation("obs")(np.ones((2, 2)), np.zeros((2, 2)))
