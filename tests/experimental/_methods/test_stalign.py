"""Integration tests for the ported STalign estimator.

Tiny synthetic fixtures with ``niter=1`` keep these fast; they verify wiring
and shapes (dispatch -> JAX LDDMM -> StalignFitResult), not solver quality.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from squidpy.experimental._fit.align_samples import ALIGN_SAMPLES, StalignFitResult, fit_stalign
from squidpy.experimental._fit.align_samples._stalign_impl._tools import (
    STalignConfig,
    STalignPreprocessConfig,
    STalignRegistrationConfig,
)


def _points_xy() -> np.ndarray:
    return np.array(
        [
            [10.0, 1.0],
            [12.0, 1.0],
            [11.0, 2.0],
            [10.0, 3.0],
            [12.0, 3.0],
        ]
    )


def _tiny_config() -> STalignConfig:
    """Single-iteration LDDMM hyperparameters - the smallest possible solve."""
    return STalignConfig(
        preprocess=STalignPreprocessConfig(dx=0.5, blur=1.0),
        registration=STalignRegistrationConfig(a=1.0, expand=1.0, nt=1, niter=1, epV=1.0),
    )


def test_stalign_registered_in_align_family() -> None:
    assert "stalign" in ALIGN_SAMPLES.keys()
    assert ALIGN_SAMPLES.get("stalign") is fit_stalign


def test_stalign_fit_returns_displacement_result() -> None:
    ref, query = _points_xy(), _points_xy()

    result = fit_stalign(ref, query, config=_tiny_config())

    assert isinstance(result, StalignFitResult)
    assert result.deltas.shape == query.shape
    assert np.all(np.isfinite(result.deltas))
    assert result.affine is None
    assert result.metadata["method"] == "stalign"
    # Escape hatch: the full diffeomorphic map is preserved for power users.
    assert "stalign_result" in result.metadata


def test_stalign_transform_bakes_in_deltas() -> None:
    ref, query = _points_xy(), _points_xy()

    result = fit_stalign(ref, query, config=_tiny_config())

    np.testing.assert_allclose(result.transform(query), query + result.deltas)


def test_stalign_transform_rejects_wrong_shape() -> None:
    ref, query = _points_xy(), _points_xy()
    result = fit_stalign(ref, query, config=_tiny_config())

    with pytest.raises(ValueError, match="expects coordinates of shape"):
        result.transform(np.zeros((1, 2)))


def test_stalign_fit_with_landmarks() -> None:
    ref, query = _points_xy(), _points_xy()
    landmarks = ref[:3]

    result = fit_stalign(
        ref,
        query,
        config=_tiny_config(),
        landmarks_source=landmarks,
        landmarks_target=landmarks,
    )

    assert result.deltas.shape == query.shape
    assert "stalign_result" in result.metadata


def test_stalign_fit_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match=r"Expected `query` to be an \(N, 2\) array"):
        fit_stalign(_points_xy(), np.zeros((5, 3)), config=_tiny_config())
