from typing import Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pytest
import tifffile
import xarray as xr
from pytest_mock import MockerFixture
from skimage.io import imread
from squidpy._constants._constants import InferDimensions
from squidpy.im._io import _get_image_shape_dtype, _infer_dimensions, _lazy_load_image


class TestIO:
    @staticmethod
    def _create_image(path: str, shape: tuple[int, ...]):
        dtype = np.uint8 if len(shape) <= 3 else np.float32
        img = np.random.randint(0, 255, size=shape).astype(dtype)
        # set `photometric` to remove warnings
        tifffile.imwrite(path, img, photometric=tifffile.TIFF.PHOTOMETRIC.MINISBLACK)

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
    def test_get_shape(self, shape: tuple[int, ...], tmpdir):
        path = str(tmpdir / "img.tiff")
        img = self._create_image(path, shape)

        actual_shape, actual_dtype = _get_image_shape_dtype(path)
        np.testing.assert_array_equal(actual_shape, shape)
        assert actual_dtype == img.dtype, (actual_dtype, img.dtype)

    @pytest.mark.parametrize("infer_dim", list(InferDimensions))
    @pytest.mark.parametrize(
        "shape", [(101, 64), (101, 64, 3), (3, 64, 101), (1, 101, 64, 3), (1, 101, 64, 1), (3, 101, 64, 1)]
    )
    def test_infer_dimensions(self, shape: tuple[int, ...], infer_dim: str, mocker: MockerFixture):
        mocker.patch("squidpy.im._io._get_image_shape_dtype", return_value=(shape, np.uint8))
        infer_dim = InferDimensions(infer_dim)
        actual_shape, actual_dims, _, _ = _infer_dimensions("non_existent", infer_dim)

        if len(shape) == 2:
            np.testing.assert_array_equal(actual_dims, ["y", "x", "z", "channels"])
            np.testing.assert_array_equal(actual_shape, shape + (1, 1))
        elif len(shape) == 3:
            if shape[-1] == 3:
                if infer_dim == InferDimensions.Z_LAST:
                    np.testing.assert_array_equal(actual_dims, ["channels", "y", "x", "z"])
                else:
                    np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
                np.testing.assert_array_equal(actual_shape, (1,) + shape)
            else:
                if infer_dim == InferDimensions.Z_LAST:
                    np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
                else:
                    np.testing.assert_array_equal(actual_dims, ["channels", "y", "x", "z"])
                np.testing.assert_array_equal(actual_shape, shape + (1,))
        elif len(shape) == 4:
            if infer_dim == InferDimensions.DEFAULT:
                if shape[0] == 1:
                    np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
                elif shape[-1] == 1:
                    np.testing.assert_array_equal(actual_dims, ["channels", "y", "x", "z"])
                else:
                    np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
            elif infer_dim == InferDimensions.Z_LAST:
                np.testing.assert_array_equal(actual_dims, ["channels", "y", "x", "z"])
            else:
                np.testing.assert_array_equal(actual_dims, ["z", "y", "x", "channels"])
            np.testing.assert_array_equal(actual_shape, shape)

    @pytest.mark.parametrize("chunks", [100, (1, 100, 100, 3), "auto", None, {"y": 100, "x": 100}])
    def test_lazy_load_image(self, chunks: Optional[Union[int, tuple[int, ...], str, dict[str, int]]], tmpdir):
        path = str(tmpdir / "img.tiff")
        img = self._create_image(path, (256, 256, 3))

        res = _lazy_load_image(path, chunks=chunks)

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        if chunks not in (None, "auto"):
            np.testing.assert_array_equal(res.data.chunksize, [100, 100, 1, 3])

        np.testing.assert_array_equal(img, np.squeeze(res.values))

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
    def test_explicit_dimension_mismatch(self, tmpdir, n: int):
        path = str(tmpdir / "img.tiff")
        _ = self._create_image(path, (5, 100, 100, 2))

        with pytest.raises(ValueError, match="Image is `4` dimensional"):
            _ = _lazy_load_image(path, dims=tuple(str(i) for i in range(n)))

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_read_tiff_skimage(self, tmpdir, n: int):
        path = str(tmpdir / "img.tiff")
        img = self._create_image(path, (n, 100, 100))

        res = _lazy_load_image(path, dims=("c", "y", "x"))
        res_skimage = imread(path, plugin="tifffile")

        assert isinstance(res, xr.DataArray)
        assert isinstance(res_skimage, np.ndarray)

        np.testing.assert_array_equal(img, res.transpose("c", "y", "x", ...).values.squeeze(-1))
        if n in (3, 4):
            np.testing.assert_array_equal(res_skimage.shape, [100, 100, n])
            with pytest.raises(AssertionError, match="Arrays are not equal"):
                np.testing.assert_array_equal(res_skimage.shape, np.squeeze(res.shape))
        else:
            np.testing.assert_array_equal(res_skimage, res.transpose("c", "y", "x", ...).values.squeeze(-1))
