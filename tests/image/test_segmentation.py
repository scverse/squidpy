from typing import Union, Callable, Optional
from pytest_mock import MockerFixture
import pytest

import numpy as np

import skimage

from squidpy.im import (
    segment,
    ImageContainer,
    SegmentationBlob,
    SegmentationCustom,
    SegmentationWatershed,
)
from squidpy._constants._constants import SegmentationBackend
from squidpy._constants._pkg_constants import Key


def dummy_segment(arr: np.ndarray) -> np.ndarray:
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
    return arr[..., 0]


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
        img = np.zeros(shape=(10, 10, 2))
        sc = SegmentationCustom(dummy_segment)

        with pytest.raises(ValueError, match=r"Expected only `1` channel, found `2`."):
            sc.segment(img)

    def test_segment_container(self):
        img = ImageContainer(np.zeros(shape=(10, 10, 1)), layer="image")
        sc = SegmentationCustom(dummy_segment)

        res = sc.segment(img, layer="image")

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

        res = sw.segment(img, layer="image", thresh=thresh)

        assert isinstance(res, ImageContainer)
        spy.assert_called_once()
        call = spy.call_args_list[0]

        assert call[1]["thresh"] == thresh


class TestBlob:
    @pytest.mark.parametrize("model", ["log", "dog", "doh", "watershed", "foo"])
    def test_model(self, model: str, mocker: MockerFixture):
        if model == "foo":
            with pytest.raises(ValueError, match=r"Invalid option `foo`"):
                SegmentationBlob(model)
        elif model == "watershed":
            with pytest.raises(NotImplementedError, match=r"Unknown blob model `watershed`."):
                SegmentationBlob(model)
        else:
            sb = SegmentationBlob(model)
            img = np.zeros((100, 200), dtype=np.float64)

            res = sb.segment(img)

            assert isinstance(res, np.ndarray)
            assert sb._model == getattr(skimage.feature, f"blob_{model}")

    def test_invert(self, mocker: MockerFixture):
        sb = SegmentationBlob("log")
        img = np.zeros((100, 200), dtype=np.float64)
        spy = mocker.spy(sb, "_model")

        res = sb.segment(img, invert=True)

        assert isinstance(res, np.ndarray)

        call = spy.call_args_list[0]

        np.testing.assert_array_equal(call[0][0], np.ones_like(img))


class TestHighLevel:
    def test_invalid_layer(self, small_cont: ImageContainer):
        with pytest.raises(KeyError, match=r"Image layer `foobar` not found in"):
            segment(small_cont, layer="foobar")

    @pytest.mark.parametrize("method", ["watershed", "log", "dog", "doh", dummy_segment])
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
            list(small_cont[Key.img.segment("watershed")].dims), ["y", "x", f"{small_cont['image'].dims[-1]}:{channel}"]
        )

    @pytest.mark.parametrize("key_added", [None, "foo"])
    def test_key_added(self, small_cont: ImageContainer, key_added: Optional[str]):
        res = segment(small_cont, copy=False, layer="image", layer_added=key_added)

        assert res is None
        assert Key.img.segment("watershed", layer_added=key_added) in small_cont

    def test_passing_kwargs(self):
        pass

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
