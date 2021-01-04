from anndata import AnnData

from squidpy.im.object import ImageContainer
from squidpy.constants._pkg_constants import Key


# adata and cont are now in conftest.py
def test_crop_spot_generator(adata: AnnData, cont: ImageContainer):
    """
    for simulated adata + im, generate crops.
    Make sure that the correct amount of crops are generated
    and that the crops have the correct content

    TODO
    """
    i = 0
    expected_size = adata.uns["spatial"]["V1_Adult_Mouse_Brain"]["scalefactors"]["spot_diameter_fullres"] // 2 * 2 + 1
    for obs_id, crop in cont.crop_spot_generator(adata):
        # crops have expected size?
        assert crop.shape[1] == expected_size
        assert crop.shape[2] == expected_size
        assert obs_id == adata.obs.index[i]
        i += 1
    # expected number of crops are generated?
    assert i == adata.obsm[Key.obsm.spatial].shape[0]


def test_crop_img():
    """
    crop different sizes and scales. Check padding + correct crop location

    TODO currently done in test_image_objecct.test_crop
    """
