import pytest

import numpy as np

from squidpy.im.object import ImageContainer
from squidpy.im.processing import process_img


@pytest.mark.parametrize(("xs", "ys"), [(None, None), (40, 40)])
def test_img_processing(xs, ys):
    """Test skimage processing."""
    img_orig = np.zeros((3, 100, 100), dtype=np.uint8)

    cont = ImageContainer(img_orig, img_id="image_0")
    process_img(img=cont, processing="smooth", img_id="image_0", xs=xs, ys=ys, key_added="processed")
    assert cont.data["image_0"].shape == cont.data["processed"].shape
