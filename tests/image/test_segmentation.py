from typing import Callable, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import pytest
from pytest_mock import MockerFixture
from squidpy._constants._constants import SegmentationBackend
from squidpy._constants._pkg_constants import Key
from squidpy.im import (
    ImageContainer,
    SegmentationCustom,
    SegmentationWatershed,
    segment,
)
from squidpy.im._segment import _SEG_DTYPE


def dummy_segment(arr: np.ndarray) -> np.ndarray:
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
    return arr[..., 0].astype(np.uint32)


class TestGeneral:
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_input_ndim(self, ndim: int):
        img = np.zeros(shape=(10, 10))
        if ndim == 3:
            img = img[..., np.newaxis]
        sc = SegmentationCustom(dummy_segment)

        res = sc.segment(img)

        assert isinstance(res, np.ndarray)
        assert res.ndim == 3
        if ndim == 2:
            assert res.shape == img.shape + (1,)
        else:
            assert res.shape == img.shape

    def test_segment_invalid_shape(self):
        img = np.zeros(shape=(1, 10, 10, 2))
        sc = SegmentationCustom(dummy_segment)

        with pytest.raises(ValueError, match=r"Expected `2` or `3` dimensions"):
            sc.segment(img)

    def test_segment_container(self):
        img = ImageContainer(np.zeros(shape=(10, 10, 1)), layer="image")
        sc = SegmentationCustom(dummy_segment)

        res = sc.segment(img, layer="image", library_id=img["image"].z.values[0])

        assert isinstance(res, ImageContainer)
        assert res.shape == img.shape
        assert "image" in res
        assert res["image"].dims == img["image"].dims


class TestWatershed:
    @pytest.mark.parametrize("thresh", [None, 0.1, 0.5, 1.0])
    def test_threshold(self, thresh: Optional[float], mocker: MockerFixture):
        img = np.zeros((100, 200), dtype=np.float64)
        img[2:10, 2:10] = 1.0
        img[30:34, 10:16] = 1.0
        img = ImageContainer(img, layer="image")

        sw = SegmentationWatershed()
        spy = mocker.spy(sw, "_segment")

        res = sw.segment(img, layer="image", library_id=img["image"].z.values[0], fn_kwargs={"thresh": thresh})

        assert isinstance(res, ImageContainer)
        spy.assert_called_once()
        call = spy.call_args_list[0]

        assert call[1]["thresh"] == thresh


class TestHighLevel:
    def test_invalid_layer(self, small_cont: ImageContainer):
        with pytest.raises(KeyError, match=r"Image layer `foobar` not found in"):
            segment(small_cont, layer="foobar")

    @pytest.mark.parametrize("method", ["watershed", dummy_segment])
    def test_method(self, small_cont: ImageContainer, method: Union[str, Callable]):
        res = segment(small_cont, method=method, copy=True)

        assert isinstance(res, ImageContainer)
        assert res.shape == small_cont.shape

        if callable(method):
            method = SegmentationBackend.CUSTOM.s

        assert Key.img.segment(method) in res

        if method in ("log", "dog", "dog"):
            assert res[Key.img.segment(method)].values.max() <= 1

    @pytest.mark.parametrize("dy", [11, 0.5, None])
    @pytest.mark.parametrize("dx", [15, 0.1, None])
    def test_size(self, small_cont: ImageContainer, dy: Optional[Union[int, float]], dx: Optional[Union[int, float]]):
        res = segment(small_cont, size=(dy, dx), copy=True)

        assert isinstance(res, ImageContainer)
        assert res.shape == small_cont.shape

    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_channel(self, small_cont: ImageContainer, channel: int):
        segment(small_cont, copy=False, layer="image", channel=channel)

        assert Key.img.segment("watershed") in small_cont
        np.testing.assert_array_equal(
            list(small_cont[Key.img.segment("watershed")].dims),
            ["y", "x", "z", f"{small_cont['image'].dims[-1]}:{channel}"],
        )

    def test_all_channels(self, small_cont: ImageContainer):
        def func(arr: np.ndarray):
            assert arr.shape == (small_cont.shape + (n_channels,))
            return np.zeros(arr.shape[:2], dtype=np.uint8)

        n_channels = small_cont["image"].sizes["channels"]
        segment(small_cont, copy=False, layer="image", channel=None, method=func, layer_added="seg")

        np.testing.assert_array_equal(small_cont["seg"], np.zeros(small_cont.shape + (1, 1)))
        assert small_cont["seg"].dtype == _SEG_DTYPE

    @pytest.mark.parametrize("key_added", [None, "foo"])
    def test_key_added(self, small_cont: ImageContainer, key_added: Optional[str]):
        res = segment(small_cont, copy=False, layer="image", layer_added=key_added)

        assert res is None
        assert Key.img.segment("watershed", layer_added=key_added) in small_cont

    def test_passing_kwargs(self, small_cont: ImageContainer):
        def func(chunk: np.ndarray, sentinel: bool = False):
            assert sentinel, "Sentinel not set."
            return np.zeros(chunk[..., 0].shape, dtype=_SEG_DTYPE)

        segment(
            small_cont, method=func, layer="image", layer_added="bar", chunks=25, lazy=False, depth=None, sentinel=True
        )
        assert small_cont["bar"].values.dtype == _SEG_DTYPE
        np.testing.assert_array_equal(small_cont["bar"].values, 0)

    @pytest.mark.parametrize("dask_input", [False, True])
    @pytest.mark.parametrize("chunks", [25, (50, 50, 1), "auto"])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_dask_segment(
        self, small_cont: ImageContainer, dask_input: bool, chunks: Union[int, tuple[int, ...], str], lazy: bool
    ):
        def func(chunk: np.ndarray):
            if isinstance(chunks, tuple):
                np.testing.assert_array_equal(chunk.shape, [chunks[0] + 2 * d, chunks[1] + 2 * d, 1])
            elif isinstance(chunks, int):
                np.testing.assert_array_equal(chunk.shape, [chunks + 2 * d, chunks + 2 * d, 1])

            return np.zeros(chunk[..., 0].shape, dtype=_SEG_DTYPE)

        small_cont["foo"] = da.asarray(small_cont["image"].data) if dask_input else small_cont["image"].values
        d = 10  # overlap depth
        assert isinstance(small_cont["foo"].data, da.Array if dask_input else np.ndarray)

        segment(small_cont, method=func, layer="foo", layer_added="bar", chunks=chunks, lazy=lazy, depth={0: d, 1: d})

        if lazy:
            assert isinstance(small_cont["bar"].data, da.Array)
            small_cont.compute()
            assert isinstance(small_cont["foo"].data, np.ndarray)
        else:
            # make sure we didn't accidentally trigger foo's computation
            assert isinstance(small_cont["foo"].data, da.Array if dask_input else np.ndarray)

        assert isinstance(small_cont["bar"].data, np.ndarray)
        assert small_cont["bar"].values.dtype == _SEG_DTYPE
        np.testing.assert_array_equal(small_cont["bar"].values, 0)

    def test_copy(self, small_cont: ImageContainer):
        prev_keys = set(small_cont)
        res = segment(small_cont, copy=True, layer="image")

        assert isinstance(res, ImageContainer)
        assert set(small_cont) == prev_keys
        assert Key.img.segment("watershed") in res

    def test_parallelize(self, small_cont: ImageContainer):
        res1 = segment(small_cont, layer="image", n_jobs=1, copy=True)
        res2 = segment(small_cont, layer="image", n_jobs=2, copy=True)

        np.testing.assert_array_equal(
            res1[Key.img.segment("watershed")].values, res2[Key.img.segment("watershed")].values
        )

    @pytest.mark.parametrize("chunks", [25, 50])
    def test_blocking(self, small_cont: ImageContainer, chunks: int):
        def func(chunk: np.ndarray):
            labels = np.zeros(chunk[..., 0].shape, dtype=np.uint32)
            labels[0, 0] = 1
            return labels

        segment(small_cont, method=func, layer="image", layer_added="bar", chunks=chunks, lazy=False, depth=None)
        # blocks are label from top-left to bottom-right in an ascending order [0, num_blocks - 1]
        # lowest n bits are allocated for block, rest is for the label (i.e. for blocksize=25, we need 16 blocks ids
        # from [0, 15], which can be stored in 4 bits, then we just prepend 1 bit (see the above `func`, resulting
        # in unique 16 labels [10000, 11111]

        expected = np.zeros_like(small_cont["bar"].values)
        start = 16 if chunks == 25 else 4
        for i in range(0, 100, chunks):
            for j in range(0, 100, chunks):
                expected[i, j] = start
                start += 1

        assert small_cont["bar"].values.dtype == _SEG_DTYPE
        np.testing.assert_array_equal(small_cont["bar"].values, expected)

    @pytest.mark.parametrize("size", [None, 11])
    def test_watershed_works(self, size: Optional[int]):
        img_orig = np.zeros((100, 200, 30), dtype=np.float64)
        img_orig[2:10, 2:10] = 1.0
        img_orig[30:34, 10:16] = 1.0

        cont = ImageContainer(img_orig, layer="image_0")
        segment(
            img=cont,
            method="watershed",
            layer="image_0",
            layer_added="segment",
            size=size,
            channel=0,
            thresh=0.5,
        )
        # check that blobs are in segments
        assert np.mean(cont.data["segment"].values[img_orig[:, :, 0] > 0] > 0) > 0.5

        # for size=10, "fails with `size=10` due to border effects"
        # the reason why there is no test for it that inside tox, it "works" (i.e. the assertion passes)
        # but outside, the assertion fails, as it should

    @pytest.mark.parametrize("library_id", [None, "3", ["1", "2"]])
    def test_library_id(self, cont_4d: ImageContainer, library_id: Optional[Union[str, Sequence[str]]]):
        def func(arr: np.ndarray):
            assert arr.shape == cont_4d.shape + (1,)
            return np.ones(arr[..., 0].shape, dtype=_SEG_DTYPE)

        segment(cont_4d, method=func, layer="image", layer_added="image_seg", library_id=library_id, copy=False)

        np.testing.assert_array_equal(cont_4d["image"].coords, cont_4d["image_seg"].coords)
        if library_id is None:
            np.testing.assert_array_equal(1, cont_4d["image_seg"])
        else:
            if isinstance(library_id, str):
                library_id = [library_id]
            for lid in library_id:
                np.testing.assert_array_equal(1, cont_4d["image_seg"].sel(z=lid))
            for lid in set(cont_4d.library_ids) - set(library_id):
                # channels have been changed, apply sets to 0
                np.testing.assert_array_equal(0, cont_4d["image_seg"].sel(z=lid))
