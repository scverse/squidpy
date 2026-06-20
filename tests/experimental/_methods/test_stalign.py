"""Integration tests for the ported STalign estimator.

Tiny synthetic fixtures with ``niter=1`` keep these fast; they verify wiring
and shapes (dispatch -> JAX LDDMM -> StalignResult), not solver quality.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from squidpy.experimental._methods.align_samples import ALIGN_SAMPLES, fit_stalign
from squidpy.experimental._methods.align_samples._stalign_impl._tools import StalignResult

# Flat solver kwargs (assembled into the config internally) -- smallest possible solve.
_TINY = {"dx": 0.5, "blur": 1.0, "a": 1.0, "expand": 1.0, "nt": 1, "niter": 1, "epV": 1.0}


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


def test_stalign_registered_in_align_family() -> None:
    assert "stalign" in ALIGN_SAMPLES.keys()
    assert ALIGN_SAMPLES.get("stalign") is fit_stalign


def test_stalign_fit_returns_diffeomorphism() -> None:
    ref, query = _points_xy(), _points_xy()

    result = fit_stalign(ref, query, **_TINY)

    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.shape
    assert np.all(np.isfinite(np.asarray(result.aligned_points)))
    assert result.affine.shape == (3, 3)
    assert result.velocity.ndim == 4


def test_stalign_transform_matches_aligned_points() -> None:
    ref, query = _points_xy(), _points_xy()

    result = fit_stalign(ref, query, **_TINY)

    np.testing.assert_allclose(np.asarray(result.transform(query)), np.asarray(result.aligned_points), atol=1e-5)


def test_stalign_transform_accepts_arbitrary_points() -> None:
    ref, query = _points_xy(), _points_xy()
    result = fit_stalign(ref, query, **_TINY)

    out = result.transform(np.zeros((1, 2)))
    assert np.asarray(out).shape == (1, 2)


def test_stalign_transform_backward_inverts_forward() -> None:
    ref, query = _points_xy(), _points_xy()
    result = fit_stalign(ref, query, **_TINY)

    forward = result.transform(query, direction="forward")
    roundtrip = result.transform(forward, direction="backward")
    np.testing.assert_allclose(np.asarray(roundtrip), query, atol=1e-3)


def test_stalign_transform_rejects_non_2d() -> None:
    ref, query = _points_xy(), _points_xy()
    result = fit_stalign(ref, query, **_TINY)

    with pytest.raises(ValueError, match=r"Expected an \(N, 2\)"):
        result.transform(np.zeros((5, 3)))


def test_stalign_fit_with_landmarks() -> None:
    ref, query = _points_xy(), _points_xy()
    landmarks = ref[:3]

    result = fit_stalign(ref, query, landmarks_source=landmarks, landmarks_target=landmarks, **_TINY)

    assert result.aligned_points.shape == query.shape


def test_stalign_fit_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match=r"Expected `query` to have shape `\(n, 2\)`"):
        fit_stalign(_points_xy(), np.zeros((5, 3)), **_TINY)


def test_stalign_rejects_unknown_kwarg() -> None:
    ref, query = _points_xy(), _points_xy()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fit_stalign(ref, query, not_a_real_param=1.0, **_TINY)
