import warnings

import pytest

import numpy as np

import rasterio.errors

from squidpy.im.object import ImageContainer
from squidpy.im.segment import segment


@pytest.mark.parametrize("shape", [(3, 100, 200)])
def test_segmentation_blob(shape):
    """Test skimage blob detection."""
    # ignore NotGeoreferencedWarning here
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    img_orig = np.zeros(shape, dtype=np.uint8)
    # Add blobs
    img_orig[:, 2:10, 2:10] = 1.0
    img_orig[:, 30:34, 10:16] = 1.0

    cont = ImageContainer(img_orig, img_id="image_0")
    segment(
        img=cont,
        model_group="watershed",
        img_id="image_0",
        key_added="segment",
        channel_idx=0,
        model_kwargs={"geq": False},
    )
    # Check that blobs are in segments:
    assert np.mean(cont.data["segment"].values[0][img_orig[0] > 0] > 0) > 0.5
