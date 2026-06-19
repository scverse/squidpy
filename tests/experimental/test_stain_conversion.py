from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from squidpy.experimental.im._stain._conversion import (
    lab_ruderman_to_rgb,
    rgb_to_lab_ruderman,
    rgb_to_sda,
    sda_to_rgb,
)

# Tests assert pure-white explicitly; the production API requires the
# caller to supply a background and offers no library-wide default.
_TEST_WHITE = np.array([255.0, 255.0, 255.0])


def _rgb_dataarray(values: np.ndarray, *, chunked: bool) -> xr.DataArray:
    if chunked:
        data = da.from_array(values, chunks=(3, 8, 8))
    else:
        data = values
    return xr.DataArray(data, dims=("c", "y", "x"))


@pytest.fixture
def random_rgb_patch() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.uniform(low=1.0, high=254.0, size=(3, 16, 16)).astype(np.float64)


class TestSdaRoundTrip:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_round_trip(self, random_rgb_patch: np.ndarray, chunked: bool) -> None:
        rgb = _rgb_dataarray(random_rgb_patch, chunked=chunked)
        recovered = sda_to_rgb(rgb_to_sda(rgb, _TEST_WHITE), _TEST_WHITE)
        np.testing.assert_allclose(recovered.values, random_rgb_patch, atol=1e-6)

    def test_white_maps_to_zero(self) -> None:
        white = xr.DataArray(np.full((3, 4, 4), 255.0), dims=("c", "y", "x"))
        sda = rgb_to_sda(white, _TEST_WHITE)
        np.testing.assert_allclose(sda.values, 0.0, atol=1e-6)

    def test_non_negative_on_valid_rgb(self, random_rgb_patch: np.ndarray) -> None:
        rgb = _rgb_dataarray(random_rgb_patch, chunked=False)
        sda = rgb_to_sda(rgb, _TEST_WHITE)
        assert float(sda.min()) >= -1e-9

    def test_uint8_promoted_to_float(self) -> None:
        rng = np.random.default_rng(seed=0)
        rgb = xr.DataArray(rng.integers(0, 255, size=(3, 8, 8), dtype=np.uint8), dims=("c", "y", "x"))
        sda = rgb_to_sda(rgb, _TEST_WHITE)
        assert np.issubdtype(sda.dtype, np.floating)

    def test_off_white_background_round_trip(self, random_rgb_patch: np.ndarray) -> None:
        bg = np.array([240.0, 250.0, 235.0])
        rgb = _rgb_dataarray(random_rgb_patch, chunked=False)
        recovered = sda_to_rgb(rgb_to_sda(rgb, bg), bg)
        np.testing.assert_allclose(recovered.values, random_rgb_patch, atol=1e-6)


class TestRudermanRoundTrip:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_round_trip(self, random_rgb_patch: np.ndarray, chunked: bool) -> None:
        rgb = _rgb_dataarray(random_rgb_patch, chunked=chunked)
        recovered = lab_ruderman_to_rgb(rgb_to_lab_ruderman(rgb))
        np.testing.assert_allclose(recovered.values, random_rgb_patch, atol=1e-4)


class TestLazinessContract:
    """Dask in -> dask out. The conversion must not eagerly compute."""

    def test_rgb_to_sda_stays_lazy(self, random_rgb_patch: np.ndarray) -> None:
        rgb = _rgb_dataarray(random_rgb_patch, chunked=True)
        sda = rgb_to_sda(rgb, _TEST_WHITE)
        assert isinstance(sda.data, da.Array)

    def test_sda_to_rgb_stays_lazy(self, random_rgb_patch: np.ndarray) -> None:
        rgb = _rgb_dataarray(random_rgb_patch, chunked=True)
        recovered = sda_to_rgb(rgb_to_sda(rgb, _TEST_WHITE), _TEST_WHITE)
        assert isinstance(recovered.data, da.Array)

    def test_ruderman_pair_stays_lazy(self, random_rgb_patch: np.ndarray) -> None:
        rgb = _rgb_dataarray(random_rgb_patch, chunked=True)
        recovered = lab_ruderman_to_rgb(rgb_to_lab_ruderman(rgb))
        assert isinstance(recovered.data, da.Array)

    def test_chunked_matches_unchunked(self, random_rgb_patch: np.ndarray) -> None:
        rgb_eager = _rgb_dataarray(random_rgb_patch, chunked=False)
        rgb_lazy = _rgb_dataarray(random_rgb_patch, chunked=True)
        sda_eager = rgb_to_sda(rgb_eager, _TEST_WHITE).values
        sda_lazy = rgb_to_sda(rgb_lazy, _TEST_WHITE).values
        np.testing.assert_allclose(sda_lazy, sda_eager, atol=1e-10)


class TestInputValidation:
    def test_missing_channel_dim_raises(self) -> None:
        arr = xr.DataArray(np.zeros((4, 4, 3)), dims=("y", "x", "channel"))
        with pytest.raises(ValueError, match="dimension named 'c'"):
            rgb_to_sda(arr, _TEST_WHITE)

    def test_wrong_channel_length_raises(self) -> None:
        arr = xr.DataArray(np.zeros((4, 4, 4)), dims=("y", "x", "c"))
        with pytest.raises(ValueError, match="3-channel RGB"):
            rgb_to_sda(arr, _TEST_WHITE)
