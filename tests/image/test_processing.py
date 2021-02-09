from typing import Union, Callable, Optional
import pytest

import numpy as np

from squidpy.im import process, ImageContainer
from squidpy._constants._pkg_constants import Key


class TestProcess:
    def test_invalid_layer(self, small_cont: ImageContainer):
        with pytest.raises(KeyError, match=r"Image layer `foobar` not found in"):
            process(small_cont, layer="foobar")

    @pytest.mark.parametrize("dy", [25, 0.3, None])
    @pytest.mark.parametrize("dx", [30, 0.5, None])
    def test_size(self, small_cont: ImageContainer, dy: Optional[Union[int, float]], dx: Optional[Union[int, float]]):
        res = process(small_cont, method="smooth", copy=True)
        key = Key.img.process("smooth", "image")

        assert res.shape == small_cont.shape
        np.testing.assert_array_equal(res[key].dims, small_cont["image"].dims)

    @pytest.mark.parametrize("method", ["smooth", "gray", lambda arr: arr])
    def test_method(self, small_cont: ImageContainer, method: Union[str, Callable[[np.ndarray], np.ndarray]]):
        res = process(small_cont, method=method, copy=True)
        key = Key.img.process(method, "image")

        assert isinstance(res, ImageContainer)
        assert key in res
        if callable(method):
            np.testing.assert_array_equal(small_cont["image"].values, res[key].values)
        else:
            assert not np.all(np.allclose(small_cont["image"].values, res[key].values))

    @pytest.mark.parametrize("method", ["smooth", "gray", lambda arr: arr[..., 0]])
    def test_channel_dim(self, small_cont: ImageContainer, method: Union[str, Callable[[np.ndarray], np.ndarray]]):
        res = process(small_cont, method=method, copy=True, channel_dim="foo")
        key = Key.img.process(method, "image")

        assert isinstance(res, ImageContainer)

        if method == "smooth":
            np.testing.assert_array_equal(res[key].dims, ["y", "x", "foo"])
        else:
            modifier = "_".join(key.split("_")[1:])  # will be e.g `foo_smooth`
            np.testing.assert_array_equal(res[key].dims, ["y", "x", f"foo_{modifier}"])

    def test_gray_not_rgb(self, small_cont_1c: ImageContainer):
        with pytest.raises(ValueError, match=r"Expected channel dimension to be `3`, found `1`."):
            process(small_cont_1c, method="gray")

    @pytest.mark.parametrize("key_added", [None, "foo"])
    def test_key_added(self, small_cont: ImageContainer, key_added: Optional[str]):
        res = process(small_cont, method="smooth", copy=False, layer_added=key_added, layer="image")

        assert res is None
        assert Key.img.process("smooth", "image", layer_added=key_added)

    def test_passing_kwargs(self, small_cont: ImageContainer):
        def dummy(arr: np.ndarray, sentinel: bool = False) -> np.ndarray:
            assert sentinel, "Sentinel not set."
            return arr

        res = process(small_cont, method=dummy, sentinel=True)
        key = Key.img.process(dummy, "image")

        assert res is None
        np.testing.assert_array_equal(small_cont[key].values, small_cont["image"].values)

    def test_copy(self, small_cont: ImageContainer):
        orig_keys = set(small_cont)
        res = process(small_cont, method="smooth", copy=True)

        assert isinstance(res, ImageContainer)
        assert set(small_cont) == orig_keys
