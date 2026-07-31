from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from squidpy.experimental.im._stain._reference import StainReference
from squidpy.experimental.im._stain._reinhard import (
    _SIGMA_FLOOR,
    ReinhardParams,
    _masked_channel_stats,
    _resolve_reinhard_params,
    apply_reinhard,
    fit_reinhard,
)


def _da(values: np.ndarray, *, chunked: bool) -> xr.DataArray:
    data = da.from_array(values, chunks=(3, 8, 8)) if chunked else values
    return xr.DataArray(data, dims=("c", "y", "x"))


@pytest.fixture
def rgb_a() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(60.0, 190.0, size=(3, 32, 32))


@pytest.fixture
def rgb_b() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.uniform(60.0, 190.0, size=(3, 32, 32))


class TestMaskedChannelStats:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_matches_numpy(self, chunked: bool) -> None:
        rng = np.random.default_rng(7)
        values = rng.uniform(0.0, 10.0, size=(3, 16, 16))
        mask_np = rng.random((16, 16)) > 0.3
        lab = _da(values, chunked=chunked)
        mask = xr.DataArray(mask_np, dims=("y", "x"))

        mu, sigma = _masked_channel_stats(lab, mask)

        sel = values[:, mask_np]
        np.testing.assert_allclose(mu, sel.mean(axis=1), atol=1e-6)
        np.testing.assert_allclose(sigma, sel.std(axis=1), atol=1e-6)

    def test_numpy_dask_identical(self) -> None:
        rng = np.random.default_rng(8)
        values = rng.uniform(0.0, 10.0, size=(3, 16, 16))
        mu_n, sigma_n = _masked_channel_stats(_da(values, chunked=False), None)
        mu_d, sigma_d = _masked_channel_stats(_da(values, chunked=True), None)
        np.testing.assert_allclose(mu_n, mu_d, atol=1e-6)
        np.testing.assert_allclose(sigma_n, sigma_d, atol=1e-6)

    def test_empty_mask_raises(self) -> None:
        lab = _da(np.ones((3, 8, 8)), chunked=False)
        mask = xr.DataArray(np.zeros((8, 8), dtype=bool), dims=("y", "x"))
        with pytest.raises(ValueError, match="zero tissue pixels"):
            _masked_channel_stats(lab, mask)


class TestFitReinhard:
    def test_returns_valid_reference(self, rgb_a: np.ndarray) -> None:
        ref = fit_reinhard(_da(rgb_a, chunked=False), ReinhardParams())
        assert ref.method == "reinhard"
        assert ref.mu.shape == (3,)
        assert ref.sigma.shape == (3,)
        assert np.all(np.isfinite(ref.mu))
        assert np.all(ref.sigma > 0)


class TestApplyReinhard:
    def test_idempotent_when_source_is_reference(self, rgb_a: np.ndarray) -> None:
        params = ReinhardParams()
        src = _da(rgb_a, chunked=False)
        ref = fit_reinhard(src, params)
        out = apply_reinhard(src, ref, params)
        np.testing.assert_allclose(out.values, rgb_a, atol=1e-4)

    def test_transfer_matches_reference_stats(self, rgb_a: np.ndarray, rgb_b: np.ndarray) -> None:
        params = ReinhardParams(mask_background=False)
        ref = fit_reinhard(_da(rgb_a, chunked=False), params)
        normalized = apply_reinhard(_da(rgb_b, chunked=False), ref, params)
        refit = fit_reinhard(normalized, params)
        np.testing.assert_allclose(refit.mu, ref.mu, atol=1e-4)
        np.testing.assert_allclose(refit.sigma, ref.sigma, atol=1e-4)

    def test_lazy_in_lazy_out(self, rgb_a: np.ndarray, rgb_b: np.ndarray) -> None:
        params = ReinhardParams()
        ref = fit_reinhard(_da(rgb_a, chunked=False), params)
        out = apply_reinhard(_da(rgb_b, chunked=True), ref, params)
        assert isinstance(out.data, da.Array)

    def test_degenerate_channel_no_nan(self, rgb_a: np.ndarray) -> None:
        params = ReinhardParams(mask_background=False)
        ref = fit_reinhard(_da(rgb_a, chunked=False), params)
        flat = rgb_a.copy()
        flat[0] = 128.0  # constant channel -> sigma_src == 0
        out = apply_reinhard(_da(flat, chunked=False), ref, params)
        assert np.all(np.isfinite(out.values))

    def test_sigma_floor_is_small(self) -> None:
        assert 0 < _SIGMA_FLOOR < 1e-3


class TestResolveReinhardParams:
    def test_none_returns_defaults(self) -> None:
        assert _resolve_reinhard_params(None) == ReinhardParams()

    def test_instance_passthrough(self) -> None:
        p = ReinhardParams(luminosity_threshold=0.5)
        assert _resolve_reinhard_params(p) is p

    def test_mapping(self) -> None:
        p = _resolve_reinhard_params({"luminosity_threshold": 0.6, "mask_background": False})
        assert p.luminosity_threshold == 0.6
        assert p.mask_background is False

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown `method_params`"):
            _resolve_reinhard_params({"nope": 1})

    def test_bad_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be ReinhardParams"):
            _resolve_reinhard_params(5)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_threshold_bounds(self, bad: float) -> None:
        with pytest.raises(ValueError, match="luminosity_threshold"):
            ReinhardParams(luminosity_threshold=bad)


def test_reference_is_stainreference(rgb_a: np.ndarray) -> None:
    ref = fit_reinhard(_da(rgb_a, chunked=False), ReinhardParams())
    assert isinstance(ref, StainReference)
