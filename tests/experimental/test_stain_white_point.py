from __future__ import annotations

import numpy as np
import pytest
import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel, Labels2DModel

from squidpy.experimental.im import estimate_white_point
from squidpy.experimental.im._stain._conversion import dtype_max
from squidpy.experimental.im._stain._validation import StainFittingError
from squidpy.experimental.im._stain._white_point import default_white_point, validate_rgb_range


def _rgb(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, dims=("c", "y", "x"))


class TestDtypeMax:
    def test_known_dtypes(self) -> None:
        assert dtype_max(np.uint8) == 255.0
        assert dtype_max(np.uint16) == 65535.0
        assert dtype_max(np.float32) == 1.0


class TestDefaultWhitePoint:
    def test_uint8(self) -> None:
        rgb = _rgb(np.full((3, 8, 8), 200, dtype=np.uint8))
        np.testing.assert_array_equal(default_white_point(rgb), [255.0, 255.0, 255.0])

    def test_uint16(self) -> None:
        rgb = _rgb(np.full((3, 8, 8), 5000, dtype=np.uint16))
        np.testing.assert_array_equal(default_white_point(rgb), [65535.0] * 3)

    def test_float_unit_range(self) -> None:
        rgb = _rgb(np.full((3, 8, 8), 0.8, dtype=np.float32))
        np.testing.assert_array_equal(default_white_point(rgb), [1.0, 1.0, 1.0])


class TestValidateRgbRange:
    def test_passes_on_uint8(self) -> None:
        validate_rgb_range(_rgb(np.full((3, 8, 8), 200, dtype=np.uint8)))  # no raise

    def test_passes_on_float_unit_range(self) -> None:
        validate_rgb_range(_rgb(np.full((3, 8, 8), 0.8, dtype=np.float32)))  # no raise

    def test_raises_on_8bit_in_uint16(self) -> None:
        with pytest.raises(ValueError, match="8-bit data stored in"):
            validate_rgb_range(_rgb(np.full((3, 8, 8), 200, dtype=np.uint16)))

    def test_raises_on_0_255_float(self) -> None:
        with pytest.raises(ValueError, match="stored as float"):
            validate_rgb_range(_rgb(np.full((3, 8, 8), 200.0, dtype=np.float32)))


class TestEstimateWhitePoint:
    def _sdata(self, *, all_tissue: bool = False) -> sd.SpatialData:
        rng = np.random.default_rng(0)
        values = np.empty((3, 32, 32), dtype=np.uint8)
        values[0], values[1], values[2] = 240, 245, 250  # background
        values[:, :8, :8] = rng.integers(20, 60, size=(3, 8, 8))  # a darker tissue blob
        mask = np.zeros((32, 32), dtype=np.uint32)
        mask[:8, :8] = 1  # tissue = the blob
        if all_tissue:
            mask[:] = 1
        sdata = sd.SpatialData(images={"img": Image2DModel.parse(values, dims=("c", "y", "x"))})
        sdata.labels["img_tissue"] = Labels2DModel.parse(mask, dims=("y", "x"))
        return sdata

    def test_recovers_background_median(self) -> None:
        wp = estimate_white_point(self._sdata(), "img")
        assert wp.shape == (3,)
        np.testing.assert_allclose(wp, [240.0, 245.0, 250.0], atol=1.0)

    def test_raises_when_tissue_covers_all(self) -> None:
        with pytest.raises(StainFittingError, match="covers the whole image"):
            estimate_white_point(self._sdata(all_tissue=True), "img")

    def test_requires_a_tissue_mask(self) -> None:
        values = np.full((3, 16, 16), 240, dtype=np.uint8)
        sdata = sd.SpatialData(images={"img": Image2DModel.parse(values, dims=("c", "y", "x"))})
        with pytest.raises(KeyError, match="detect_tissue"):
            estimate_white_point(sdata, "img")
