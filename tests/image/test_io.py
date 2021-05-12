from typing import Dict, Tuple, Union, Optional
from pytest_mock import MockerFixture
import pytest

import numpy as np
import xarray as xr
import dask.array as da

import tifffile

from squidpy.im._io import _lazy_load_image, _infer_dimensions, _get_image_shape_dtype
from squidpy._constants._constants import InferDimensions


class TestIO:
    def _create_image(self, path: str, shape: Tuple[int, ...]):
        dtype = np.uint8 if len(shape) <= 3 else np.float32
        img = np.random.randint(0, 255, size=shape).astype(dtype)
        tifffile.imsave(path, img)

        return img

    @pytest.mark.parametrize(
        "shape",
        [
            (101, 64),
            (3, 64, 101),
            (64, 101, 4),
            (1, 101, 64, 1),
            (1, 101, 64, 3),
            (3, 101, 64, 1),
            (3, 101, 64, 4),
        ],
    )
    def test_get_shape(self, shape: Tuple[int, ...], tmpdir):
        path = str(tmpdir / "img.tiff")
        img = self._create_image(path, shape)

        expected_shape = shape
        if shape[-1] == 1:
            expected_shape = expected_shape[:-1]
        if len(shape) < 4:
            expected_shape = (1,) + expected_shape

        actual_shape, actual_dtype = _get_image_shape_dtype(path)
        np.testing.assert_array_equal(actual_shape, expected_shape)
        assert actual_dtype == img.dtype

    @pytest.mark.parametrize("infer_dim", ["default", "prefer_z", "prefer_channels"])
    @pytest.mark.parametrize(
        "shape", [(101, 64), (101, 64, 1), (3, 64, 101), (1, 101, 64, 3), (1, 101, 64, 1), (3, 101, 64, 1)]
    )
    def test_infer_dimensions(self, shape: Tuple[int, ...], infer_dim: str, mocker: MockerFixture):
        mocker.patch("squidpy.im._io._get_image_shape_dtype", return_value=(shape, np.uint8))
        infer_dim = InferDimensions(infer_dim)
        actual_shape, actual_dims, _ = _infer_dimensions("non_existent", infer_dim)
        print(shape, actual_shape, actual_dims)

        if len(shape) == 2:
            np.testing.assert_array_equal(actual_dims, ["y", "x", "z", "channels"])
            np.testing.assert_array_equal(actual_shape, shape + (1, 1))
        elif len(shape) == 3:
            if shape[0] <= shape[1] and shape[0] <= shape[2]:
                if infer_dim == InferDimensions.PREFER_Z:
                    np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
                else:
                    np.testing.assert_array_equal(actual_dims, ["channels", "y", "x", "z"])
            elif infer_dim == InferDimensions.PREFER_Z:
                np.testing.assert_array_equal(actual_dims, ["y", "x", "z", "channels"])
            else:
                np.testing.assert_array_equal(actual_dims, ["y", "x", "channels", "z"])
            np.testing.assert_array_equal(actual_shape, shape + (1,))
        elif len(shape) == 4:
            if infer_dim == InferDimensions.DEFAULT:
                if shape[0] == 1:
                    np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
                elif shape[-1] == 1:
                    np.testing.assert_array_equal(actual_dims, ["channels", "y", "x", "z"])
                else:
                    np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
            elif infer_dim == InferDimensions.PREFER_Z:
                np.testing.assert_array_equal(actual_dims, ["channels", "y", "x", "z"])
            else:
                np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
            np.testing.assert_array_equal(actual_shape, shape)

    @pytest.mark.parametrize("chunks", [100, (1, 100, 100, 3), "auto", None, {"y": 100, "x": 100}])
    def test_lazy_load_image(self, chunks: Optional[Union[int, Tuple[int, ...], str, Dict[str, int]]], tmpdir):
        path = str(tmpdir / "img.tiff")
        img = self._create_image(path, (256, 256, 3))

        res = _lazy_load_image(path, chunks=chunks)

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        if chunks not in (None, "auto"):
            np.testing.assert_array_equal(res.data.chunksize, [100, 100, 1, 3])

        np.testing.assert_array_equal(img, np.squeeze(res.values))
