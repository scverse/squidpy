from anndata import AnnData

from squidpy.image._utils import _round_even
from squidpy.image.object import ImageContainer


# adata and cont are now in conftest.py
def test_crop_spot_generator(adata: AnnData, cont: ImageContainer):
    """
    for simulated adata + image, generate crops.
    Make sure that the correct amount of crops are generated
    and that the crops have the correct content

    TODO
    """
    i = 0
    expected_size = _round_even(adata.uns["spatial"]["V1_Adult_Mouse_Brain"]["scalefactors"]["spot_diameter_fullres"])
    for obs_id, crop in cont.crop_spot_generator(adata):
        # crops have expected size?
        assert crop.shape[1] == expected_size
        assert crop.shape[2] == expected_size
        assert obs_id == adata.obs.index[i]
        i += 1
    # expected number of crops are generated?
    assert i == adata.obsm["X_spatial"].shape[0]


def test_crop_img():
    """
    crop different sizes and scales. Check padding + correct crop location

    TODO currently done in test_image_objecct.test_crop
    """
    pass
