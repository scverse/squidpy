import pytest

import numpy as np

from squidpy.im._process import process
from squidpy.im._container import ImageContainer


@pytest.mark.parametrize("size", [(None, None), (40, 40)])
@pytest.mark.parametrize("processing", ["smooth", "gray"])
def test_img_processing(size, processing):
    """Test skimage processing."""
    img_orig = np.zeros((100, 100, 3), dtype=np.uint8)

    cont = ImageContainer(img_orig, img_id="image_0")
    process(
        img=cont,
        size=size,
        processing=processing,
        img_id="image_0",
        key_added="processed",
    )
    if processing == "smooth":
        assert cont.data["image_0"].shape == cont.data["processed"].shape
    else:
        assert cont.data["image_0"].shape[:2] == cont.data["processed"].shape[:2]
