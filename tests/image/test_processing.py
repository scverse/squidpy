from typing import Callable, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import pytest
from pytest_mock import MockerFixture
from squidpy._constants._pkg_constants import Key
from squidpy.im import ImageContainer, process


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
            np.testing.assert_array_equal(res[key].dims, ["y", "x", "z", "foo"])
        else:
            modifier = "_".join(key.split("_")[1:])  # will be e.g `foo_smooth`
            np.testing.assert_array_equal(res[key].dims, ["y", "x", "z", f"foo_{modifier}"])

    def test_gray_not_rgb(self, small_cont_1c: ImageContainer):
        with pytest.raises(ValueError, match=r"Expected channel dimension to be `3`, found `1`."):
            process(small_cont_1c, method="gray")

    @pytest.mark.parametrize("key_added", [None, "foo"])
    def test_key_added(self, small_cont: ImageContainer, key_added: Optional[str]):
        res = process(small_cont, method="smooth", copy=False, layer_added=key_added, layer="image")

        assert res is None
        assert Key.img.process("smooth", "image", layer_added=key_added)

    def test_passing_kwargs(self, small_cont: ImageContainer):
        def func(arr: np.ndarray, sentinel: bool = False) -> np.ndarray:
            assert sentinel, "Sentinel not set."
            return arr

        res = process(small_cont, method=func, sentinel=True)
        key = Key.img.process(func, "image")

        assert res is None
        np.testing.assert_array_equal(small_cont[key].values, small_cont["image"].values)

    def test_apply_kwargs(self, small_cont: ImageContainer, mocker: MockerFixture):
        spy = mocker.spy(da, "map_overlap")
        res = process(
            small_cont,
            method=lambda _: _,
            apply_kwargs={"depth": {0: 10, 1: 10}},
            layer_added="foo",
            chunks={0: 10, 1: 10},
        )

        assert res is None
        spy.assert_called_once()
        np.testing.assert_array_equal(small_cont["foo"].values, small_cont["image"].values)

    @pytest.mark.parametrize("dask_input", [False, True])
    @pytest.mark.parametrize("chunks", [25, (50, 50, 1, 3), "auto"])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_dask_processing(
        self, small_cont: ImageContainer, dask_input: bool, chunks: Union[int, tuple[int, ...], str], lazy: bool
    ):
        def func(chunk: np.ndarray):
            if isinstance(chunks, tuple):
                np.testing.assert_array_equal(chunk.shape, chunks)
            elif isinstance(chunks, int):
                np.testing.assert_array_equal(chunk.shape, [chunks, chunks, 1, 3])

            return chunk

        # TODO assigning directly (small_cont[foo] = ... does not work, because z dim is not correctly parsed
        small_cont["foo"] = da.asarray(small_cont["image"].data) if dask_input else small_cont["image"].values
        assert isinstance(small_cont["foo"].data, da.Array if dask_input else np.ndarray)

        process(small_cont, method=func, layer="foo", layer_added="bar", chunks=chunks, lazy=lazy)

        if lazy:
            assert isinstance(small_cont["bar"].data, da.Array)
            small_cont.compute()
            assert isinstance(small_cont["foo"].data, np.ndarray)
        else:
            # make sure we didn't accidentally trigger foo's computation
            assert isinstance(small_cont["foo"].data, da.Array if dask_input else np.ndarray)

        assert isinstance(small_cont["bar"].data, np.ndarray)

    @pytest.mark.parametrize("library_id", [None, "3", ["1", "2"]])
    def test_library_id(self, cont_4d: ImageContainer, library_id: Optional[Union[str, Sequence[str]]]):
        def func(arr: np.ndarray):
            if library_id is None:
                assert arr.shape == cont_4d["image"].shape
            else:
                assert arr.shape == cont_4d["image"][..., 0, :].shape
            return arr + 1

        res = process(cont_4d, method=func, layer="image", layer_added="image", library_id=library_id, copy=True)
        orig_shape = list(cont_4d["image"].shape)

        if library_id is None:
            np.testing.assert_array_equal(orig_shape, res["image"].shape)
            np.testing.assert_array_equal(cont_4d["image"].coords, res["image"].coords)
            np.testing.assert_array_equal(cont_4d["image"] + 1, res["image"])
        else:
            if isinstance(library_id, str):
                library_id = [library_id]
            orig_shape[2] = len(library_id)
            np.testing.assert_array_equal(orig_shape, res["image"].shape)
            np.testing.assert_array_equal(res["image"].coords["z"], library_id)

            for lid in library_id:
                np.testing.assert_array_equal(cont_4d["image"].sel(z=lid) + 1, res["image"].sel(z=lid))

    def test_copy(self, small_cont: ImageContainer):
        orig_keys = set(small_cont)
        res = process(small_cont, method="smooth", copy=True)

        assert isinstance(res, ImageContainer)
        assert set(small_cont) == orig_keys
