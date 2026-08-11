"""Unit tests for the shared estimator contracts in ``methods._common``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from squidpy.experimental.methods import AlignResult
from squidpy.experimental.methods._common import requires


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


def test_any_object_with_transform_satisfies_the_protocol() -> None:
    """The public functions are typed against `AlignResult`, not a concrete result."""
    assert isinstance(fit_mean_shift(np.ones((2, 2)), np.zeros((2, 2))), AlignResult)
    assert not isinstance(object(), AlignResult)


def test_requires_passes_through_when_installed() -> None:
    @requires("numpy")
    def fitted(ref: np.ndarray, query: np.ndarray) -> _MeanShiftResult:
        return fit_mean_shift(ref, query)

    assert isinstance(fitted(np.ones((2, 2)), np.zeros((2, 2))), _MeanShiftResult)


def test_requires_raises_with_install_hint_for_missing_dependency() -> None:
    @requires("squidpy_nonexistent_pkg_xyz")
    def needs_ghost(ref: np.ndarray, query: np.ndarray) -> _MeanShiftResult:
        return fit_mean_shift(ref, query)

    with pytest.raises(
        ImportError,
        match=r"`needs_ghost` requires 'squidpy_nonexistent_pkg_xyz'.*squidpy\[squidpy_nonexistent_pkg_xyz\]",
    ):
        needs_ghost(np.ones((2, 2)), np.zeros((2, 2)))


def test_requires_is_checked_at_call_time_not_import_time() -> None:
    """Decorating must not import the dependency -- that is the whole point."""

    @requires("squidpy_nonexistent_pkg_xyz")
    def never_called() -> None:  # pragma: no cover - the assertion is that this line is reached
        raise AssertionError

    assert callable(never_called)
