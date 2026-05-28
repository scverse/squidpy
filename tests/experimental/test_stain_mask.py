from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from squidpy.experimental.im._stain._mask import (
    absorbance_foreground_mask,
    luminosity_foreground_mask,
)

_WHITE = np.array([255.0, 255.0, 255.0])


def _rgb_dataarray(values: np.ndarray, *, chunked: bool) -> xr.DataArray:
    data = da.from_array(values, chunks=(3, 8, 8)) if chunked else values
    return xr.DataArray(data, dims=("c", "y", "x"))


class TestLuminosityForegroundMask:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_all_white_is_background(self, chunked: bool) -> None:
        white = np.full((3, 16, 16), 255.0)
        mask = luminosity_foreground_mask(_rgb_dataarray(white, chunked=chunked), 0.8)
        assert mask.dims == ("y", "x")
        assert not bool(mask.values.any())

    @pytest.mark.parametrize("chunked", [False, True])
    def test_all_black_is_tissue(self, chunked: bool) -> None:
        black = np.zeros((3, 16, 16))
        mask = luminosity_foreground_mask(_rgb_dataarray(black, chunked=chunked), 0.8)
        assert bool(mask.values.all())

    def test_half_split(self) -> None:
        values = np.zeros((3, 8, 16))
        values[:, :, 8:] = 255.0  # right half white
        mask = luminosity_foreground_mask(_rgb_dataarray(values, chunked=False), 0.8)
        assert bool(mask.values[:, :8].all())
        assert not bool(mask.values[:, 8:].any())

    def test_lazy_in_lazy_out(self) -> None:
        values = np.full((3, 16, 16), 100.0)
        mask = luminosity_foreground_mask(_rgb_dataarray(values, chunked=True), 0.8)
        assert isinstance(mask.data, da.Array)

    def test_non_three_channel_raises(self) -> None:
        values = np.zeros((2, 8, 8))
        with pytest.raises(ValueError, match="length 3"):
            luminosity_foreground_mask(xr.DataArray(values, dims=("c", "y", "x")), 0.8)


class TestAbsorbanceForegroundMask:
    def test_white_is_background_dark_is_tissue(self) -> None:
        values = np.full((3, 8, 16), 255.0)
        values[:, :, 8:] = 30.0  # dark right half = high absorbance = tissue
        mask = absorbance_foreground_mask(_rgb_dataarray(values, chunked=False), _WHITE)
        assert mask.dims == ("y", "x")
        assert not bool(mask.values[:, :8].any())
        assert bool(mask.values[:, 8:].all())

    def test_lazy_in_lazy_out(self) -> None:
        values = np.full((3, 16, 16), 50.0)
        mask = absorbance_foreground_mask(_rgb_dataarray(values, chunked=True), _WHITE)
        assert isinstance(mask.data, da.Array)

    def test_non_three_channel_raises(self) -> None:
        values = np.zeros((2, 8, 8))
        with pytest.raises(ValueError, match="length 3"):
            absorbance_foreground_mask(xr.DataArray(values, dims=("c", "y", "x")), _WHITE)
