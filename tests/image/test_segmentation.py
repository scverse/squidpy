import pytest

import numpy as np

from squidpy.im._segment import segment
from squidpy.im._container import ImageContainer

# TODO: refactor all below


@pytest.mark.parametrize("shape", [(100, 200, 3)])
def test_segmentation_watershed(shape):
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
