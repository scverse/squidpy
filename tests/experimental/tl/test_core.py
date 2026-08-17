"""Unit tests for the shared estimator contracts in ``methods._common``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from squidpy.experimental.tl import AlignResult


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


def test_importing_the_estimators_does_not_import_jax() -> None:
    """The optional dependency must stay unimported until a fit actually runs.

    Guards the reason the JAX imports sit inside the fit functions rather than at
    module scope: `import squidpy` must not pay for, or require, JAX.
    """
    import subprocess
    import sys

    probe = "import sys; import squidpy.experimental.tl; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", probe], check=True)
