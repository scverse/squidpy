from typing import Union, Callable, Optional
import pytest

import numpy as np

from squidpy.im._process import process
from squidpy.im._container import ImageContainer


class TestProcess:
    def test_invalid_img_id(self):
        pass

    @pytest.mark.parametrize("dy", [25, 0.3, None])
    @pytest.mark.parametrize("dx", [30, 0.5, None])
    def test_size(self, dy: Optional[Union[int, float]], dx: Optional[Union[int, float]]):
        pass

    @pytest.mark.parametrize("method", ["smooth", "gray", lambda arr: ...])
    def test_method(self, method: Union[str, Callable[[np.ndarray, ...], np.ndarray]]):
        pass

    def test_channel_dim(self):
        pass

    def test_gray_not_rgb(self):
        pass

    @pytest.mark.parametrize("key_added", [None, "foo"])
    def test_key_added(self, key_added: Optional[str]):
        pass

    def test_passing_kwargs(self):
        pass

    def test_copy(self):
        pass


@pytest.mark.parametrize("size", [(None, None), (40, 40)])
@pytest.mark.parametrize("processing", ["smooth", "gray"])
def test_img_processing(size, processing):
    img_orig = np.zeros((100, 100, 3), dtype=np.uint8)

    cont = ImageContainer(img_orig, img_id="image_0")
    process(
        img=cont,
        size=size,
        method=processing,
        img_id="image_0",
        key_added="processed",
    )
    if processing == "smooth":
        assert cont.data["image_0"].shape == cont.data["processed"].shape
    else:
        assert cont.data["image_0"].shape[:2] == cont.data["processed"].shape[:2]
