"""Integration tests for the ported STalign estimator.

Tiny synthetic fixtures with ``niter=1`` keep most of these fast; they verify wiring
and shapes (dispatch -> JAX LDDMM -> StalignResult), not solver quality.

The two tests at the bottom do check solver quality, cheaply. They exist because the
full numerical comparison against the original STalign lives in
``test_stalign_reference.py``, which is deselected by default and only runs on the
scheduled job -- leaving a fortnight in which a real regression could ship green.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from squidpy.experimental.methods import ALIGN_SAMPLES
from squidpy.experimental.methods.align_samples import StalignResult, fit_stalign

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


def test_lddmm_accepts_zero_iterations() -> None:
    """``niter=0`` means "evaluate the initial state and stop", not a crash.

    ``energy`` and the transformed landmarks used to be bound only inside the loop, so
    the return statement read unbound locals.
    """
    from squidpy.experimental.methods.align_samples._stalign_impl._core import lddmm
    from squidpy.experimental.methods.align_samples._stalign_impl._helpers import rasterize_cloud

    grid = rasterize_cloud(_points_xy()[:, ::-1], dx=0.5, blur=1.0, expand=1.1)
    result = lddmm(*grid, *grid, L=np.eye(2), T=np.zeros(2), niter=0, a=1.0, nt=1)

    assert result["v"].shape[0] == 1
    assert result["A"].shape == (3, 3)


def test_default_dtype_is_unchanged() -> None:
    """Guards the reference suite's blast radius.

    ``test_stalign_reference.py`` needs float64, but ``jax.config.update`` is
    process-global and every xdist worker imports every module in this directory. If
    that suite ever starts enabling x64 itself instead of reading ``JAX_ENABLE_X64``
    from the environment, these tests would silently stop testing float32 -- and this
    assertion is what would catch it.
    """
    import jax
    import jax.numpy as jnp

    from squidpy.experimental.methods.align_samples._stalign_impl._core import jax_dtype

    expected = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    assert jax_dtype() == expected


# --- solver quality, cheap enough to run on every PR ----------------------------------

# A cloud with enough structure to be alignable, and a query that is a known rigid
# transform of it. Small enough that a few hundred iterations run in about a second.
_SOLVE = {"dx": 8.0, "blur": 1.0, "a": 40.0, "nt": 3, "epV": 5e2}


def _blobs(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(scale=(18.0, 11.0), size=(120, 2)),
            rng.normal(scale=8.0, size=(60, 2)) + np.array([55.0, 34.0]),
        ]
    )


def test_lddmm_energy_decreases() -> None:
    """More iterations must buy a lower objective. Catches a broken gradient or step."""
    from squidpy.experimental.methods.align_samples._stalign_impl._core import lddmm
    from squidpy.experimental.methods.align_samples._stalign_impl._helpers import rasterize_cloud

    ref = _blobs()
    query = ref @ np.array([[np.cos(0.12), -np.sin(0.12)], [np.sin(0.12), np.cos(0.12)]]).T + np.array([6.0, -4.0])

    source = rasterize_cloud(query[:, ::-1], dx=_SOLVE["dx"], blur=_SOLVE["blur"], expand=1.1)
    target = rasterize_cloud(ref[:, ::-1], dx=_SOLVE["dx"], blur=_SOLVE["blur"], expand=1.1)
    solve = {k: v for k, v in _SOLVE.items() if k not in {"dx", "blur"}}

    common = {"L": np.eye(2), "T": np.zeros(2), **solve}
    first = lddmm(*source, *target, niter=1, **common)["E"]
    later = lddmm(*source, *target, niter=60, **common)["E"]

    assert float(later) < float(first), f"energy did not decrease: {float(first)} -> {float(later)}"


def test_lddmm_recovers_a_known_rigid_transform() -> None:
    """The query is a known rotation+translation of the reference; alignment must undo it.

    Asserts on the median rather than the mean: rasterisation at ``dx=8`` quantises the
    clouds, so a handful of points on the sparse outskirts stay poorly matched however
    good the fit is.
    """
    ref = _blobs()
    angle = 0.12
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    query = ref @ rotation.T + np.array([6.0, -4.0])

    result = fit_stalign(ref, query, niter=250, **_SOLVE)
    residual = np.linalg.norm(np.asarray(result.aligned_points) - ref, axis=1)

    before = np.median(np.linalg.norm(query - ref, axis=1))
    after = float(np.median(residual))
    assert after < before / 2.0, f"alignment barely improved: median {before:.2f} -> {after:.2f}"
    assert after < _SOLVE["dx"], f"median residual {after:.2f} exceeds one grid cell ({_SOLVE['dx']})"
