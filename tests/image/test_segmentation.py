from typing import Union, Callable, Optional
import pytest

import numpy as np

from squidpy.im._segment import segment
from squidpy.im._container import ImageContainer


class TestGeneral:
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_input_ndim(self, ndim: int):
        pass

    @pytest.mark.parametrize("ndims", [2, 3])
    def test_output_ndim(self, ndims: int):
        pass

    def test_segment_array(self):
        pass

    def test_custom_method(self):
        pass


class TestWatershed:
    @pytest.mark.parametrize("thresh", [None, 0.1, 0.5, 1.0])
    def test_threshold(self, thresh: Optional[float]):
        pass


class TestBlob:
    @pytest.mark.parametrize("model", ["log", "dog", "doh"])
    def test_model(self, model: str):
        pass

    def test_invert(self):
        pass


class TestHighLevel:
    def test_invalid_img_id(self):
        pass

    @pytest.mark.parametrize("method", ["watershed", "log", "dog", "doh", lambda arr: ...])
    def test_method(self, method: Union[str, Callable]):
        pass

    @pytest.mark.parametrize("dy", [10, 0.5, None])
    @pytest.mark.parametrize("dx", [15, 0.1, None])
    def test_size(self, dy: Optional[Union[int, float]], dx: Optional[Union[int, float]]):
        pass

    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_channel(self, channel: int):
        pass

    def test_image_id(self):
        pass

    @pytest.mark.parametrize("key_added", [None, "foo"])
    def test_key_added(self, key_added: Optional[str]):
        pass

    def test_passing_kwargs(self):
        pass

    def test_copy(self):
        pass

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_parallelize(self, n_jobs: int):
        pass

    # TODO: split this test
    @pytest.mark.parametrize("shape", [(100, 200, 3)])
    def test_segmentation_watershed(self, shape):
        img_orig = np.zeros(shape, dtype=np.float64)
        # Add blobs
        img_orig[2:10, 2:10] = 1.0
        img_orig[30:34, 10:16] = 1.0

        cont = ImageContainer(img_orig, img_id="image_0")
        segment(
            img=cont,
            method="watershed",
            img_id="image_0",
            key_added="segment",
            channel=0,
            thresh=0.5,
        )
        # Check that blobs are in segments:
        assert np.mean(cont.data["segment"].values[img_orig[:, :, 0] > 0] > 0) > 0.5

        # test segmentation with crops
        # TODO test fails with xs=ys=10 due to border effects!
        segment(
            img=cont,
            method="watershed",
            img_id="image_0",
            size=11,
            key_added="segment_crops",
            channel=0,
            thresh=0.5,
        )

        # Check that blobs are in segments:
        assert np.mean(cont.data["segment_crops"].values[img_orig[:, :, 0] > 0] > 0) > 0.5

        # test copy flag
        seg_img = segment(
            img=cont,
            method="watershed",
            img_id="image_0",
            size=11,
            key_added="segment_crops",
            channelx=0,
            thresh=0.5,
            copy=True,
        )
        assert (cont.data["segment_crops"].values == seg_img["segment_crops"].values).all()
