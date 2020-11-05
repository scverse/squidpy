import os

import pytest

import anndata as ad

import numpy as np

from spatial_tools.image.crop import crop_generator
from spatial_tools.image._utils import _round_odd, _round_even
from spatial_tools.image.object import ImageContainer


def test_crop_generator():
    """
    for simulated adata + image, generate crops.
    Make sure that the correct amount of crops are generated
    and that the crops have the correct content

    TODO
    """
    # load test data
    adata = ad.read(os.path.join(os.path.dirname(__file__), "_data/test_data.h5ad"))
    cont = ImageContainer(os.path.join(os.path.dirname(__file__), "_data/test_img.jpg"))

    i = 0
    expected_size = _round_even(adata.uns["spatial"]["V1_Adult_Mouse_Brain"]["scalefactors"]["spot_diameter_fullres"])
    for obs_id, crop in crop_generator(adata, cont):
        # crops have expected size?
        assert crop.shape[1] == expected_size
        assert crop.shape[2] == expected_size
        i += 1
    # expected number of crops are generated?
    assert i == adata.obsm["spatial"].shape[0]


def test_crop_img():
    """
    crop different sizes and scales. Check padding + correct crop location

    TODO currently done in test_image_objecct.test_crop
    """
    pass
