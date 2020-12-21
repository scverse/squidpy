import pytest

import numpy as np

from squidpy.im.object import ImageContainer
from squidpy.im.processing import process_img


@pytest.mark.parametrize("xs, ys", [(None, None), (40, 40)])
@pytest.mark.parametrize("processing", ["smooth", "gray"])
def test_img_processing(xs, ys, processing):
    """Test skimage processing."""
    img_orig = np.zeros((100, 100, 3), dtype=np.uint8)

    cont = ImageContainer(img_orig, img_id="image_0")
    process_img(
        img=cont, processing=processing, img_id="image_0", xs=xs, ys=ys, key_added="processed", channel_id="proc"
    )
    if processing == "smooth":
        assert cont.data["image_0"].shape == cont.data["processed"].shape
    else:
        assert cont.data["image_0"].shape[:2] == cont.data["processed"].shape[:2]
