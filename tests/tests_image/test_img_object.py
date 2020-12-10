import warnings

import pytest

from anndata import AnnData

import numpy as np

import rasterio.errors

from squidpy.im.object import ImageContainer
from squidpy.constants._pkg_constants import Key


@pytest.mark.parametrize("shape", [(100, 200, 3), (100, 200, 1), (10, 100, 200, 1)])
def test_image_loading(shape, tmpdir):
    """Initialize ImageObject with tiff / multipagetiff / numpy array and check that loaded data \
    fits the expected shape + content.
    """
    import tifffile

    # ignore NotGeoreferencedWarning here
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    img_orig = np.random.randint(low=0, high=255, size=shape, dtype=np.uint8)

    # load as np arrray
    if len(shape) <= 3:
        # load as np array
        cont = ImageContainer(img_orig)
        # check that contains same information
        assert (cont.data.image == img_orig).all()

    # save & load as tiff
    fname = tmpdir.mkdir("data").join("img.tif")
    tifffile.imsave(fname, img_orig)
    cont = ImageContainer(str(fname))
    if len(shape) > 3:
        # multi-channel tiff
        # check for existance of each im in multi-channel tiff
        # check that contains correct information
        assert (cont.data["image"] == img_orig[:, :, :, 0].transpose(1, 2, 0)).all()
    else:
        # check that contains same information
        assert (cont.data.image == img_orig).all()


@pytest.mark.parametrize(
    "shape1,shape2",
    [
        ((100, 200, 3), (100, 200, 1)),
        ((100, 200, 3), (100, 200)),
    ],
)
def test_add_img(shape1, shape2):
    """Add image to existing ImageObject and check result."""
    # create ImageContainer
    img_orig = np.random.randint(low=0, high=255, size=shape1, dtype=np.uint8)
    cont = ImageContainer(img_orig, img_id="img_orig")

    # add im
    img_new = np.random.randint(low=0, high=255, size=shape2, dtype=np.uint8)
    cont.add_img(img_new, img_id="img_new", channel_id="mask")

    assert "img_orig" in cont.data
    assert "img_new" in cont.data
    assert (np.squeeze(cont.data["img_new"]) == np.squeeze(img_new)).all()


def test_crop(tmpdir):
    """Check crop arguments:
    padding, masking, scaling, changing dtype,
    check that returned crops have correct shape.
    """
    # ignore NotGeoreferencedWarning here
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    # create ImageContainer
    xdim = 100
    ydim = 200
    img_orig = np.zeros((xdim, ydim, 10), dtype=np.uint8)
    # put a dot at y 20, x 50
    img_orig[20, 50, :] = range(10, 20)
    cont = ImageContainer(img_orig, img_id="image_0")

    # crop big crop
    crop = cont.crop_center(
        x=50,
        y=20,
        xr=150,
        yr=150,
        cval=5,
    )
    assert type(crop) == ImageContainer
    # shape is s x s x channels
    assert crop.data["image_0"].shape == (301, 301, 10)
    # check that values outside of img are padded with 5
    assert (crop.data["image_0"][0, 0, 0] == 5).all()
    assert (crop.data["image_0"][-1, -1, 0] == 5).all()
    assert crop.data["image_0"].dtype == np.uint8

    # compare with crop_corner
    crop2 = cont.crop_corner(x=-100, y=-130, xs=301, ys=301, cval=5)
    assert (crop2.data["image_0"] == crop.data["image_0"]).all()

    # crop small crop
    crop = cont.crop_center(
        x=50,
        y=20,
        xr=0,
        yr=0,
        cval=5,
    )
    assert type(crop) == ImageContainer
    assert crop.data["image_0"].shape == (1, 1, 10)
    # check that has cropped correct im
    assert (crop.data["image_0"][0, 0, :3] == [10, 11, 12]).all()
    assert crop.data["image_0"].dtype == np.uint8

    # crop with mask_circle
    crop = cont.crop_center(
        x=50,
        y=20,
        xr=5,
        yr=5,
        cval=5,
        mask_circle=True,
    )
    assert (crop.data["image_0"][1, 0, :] == 5).all()
    assert (crop.data["image_0"][2, 2, :] == 0).all()
    assert (crop.data["image_0"][7, 7, :] == 0).all()
    assert (crop.data["image_0"][9, 9, :] == 5).all()

    assert type(crop) == ImageContainer

    # crop image with several layers
    mask = np.random.randint(low=0, high=10, size=(xdim, ydim))
    cont.add_img(mask, img_id="image_1", channel_id="mask")
    crop = cont.crop_center(
        x=50,
        y=20,
        xr=0,
        yr=0,
        cval=5,
    )
    assert "image_1" in crop.data.keys()
    assert "image_0" in crop.data.keys()
    assert crop.data["image_1"].shape == (1, 1, 1)
    assert crop.data["image_0"].shape == (1, 1, 10)

    # crop with scaling
    crop = cont.crop_center(x=50, y=20, xr=10, yr=10, cval=5, scale=0.5)
    assert "image_1" in crop.data.keys()
    assert "image_0" in crop.data.keys()
    assert crop.data["image_1"].shape == (21 // 2, 21 // 2, 1)
    assert crop.data["image_0"].shape == (21 // 2, 21 // 2, 10)

    # crop casting to dtype
    img_orig = np.zeros((xdim, ydim, 10), dtype=np.uint16)
    cont = ImageContainer(img_orig, img_id="image_0")
    crop = cont.crop_center(x=50, y=20, xr=150, yr=150, cval=5, dtype="uint8")
    assert type(crop) == ImageContainer
    assert crop.data["image_0"].dtype == np.uint8


def test_generate_spot_crops(adata: AnnData, cont: ImageContainer):
    """
    for simulated adata + im, generate crops.
    Make sure that the correct amount of crops are generated
    and that the crops have the correct content
    """
    i = 0
    expected_size = adata.uns["spatial"]["V1_Adult_Mouse_Brain"]["scalefactors"]["spot_diameter_fullres"] // 2 * 2 + 1
    for crop, obs_id in cont.generate_spot_crops(adata):
        # crops have expected size?
        assert crop.shape[0] == expected_size
        assert crop.shape[1] == expected_size
        assert obs_id == adata.obs.index[i]
        i += 1
    # expected number of crops are generated?
    assert i == adata.obsm[Key.obsm.spatial].shape[0]


@pytest.mark.parametrize("ys,xs", [(10, 10), (None, None), (10, 20)])
def test_generate_equal_crops(ys, xs):
    """
    for simulated data, create equally sized crops.
    Make sure that the resulting crops have correct sizes.
    """
    img = ImageContainer(np.random.randint(low=0, high=300, size=(200, 500)), img_id="image")
    expected_size = [ys, xs]
    if ys is None or xs is None:
        expected_size = (200, 500)

    for crop, _x, _y in img.generate_equal_crops(xs=xs, ys=ys):
        assert crop.shape[0] == expected_size[0]
        assert crop.shape[1] == expected_size[1]


def test_uncrop_img(tmpdir):
    """Crop im and uncrop again and check equality."""
    # create ImageContainer
    xdim = 100
    ydim = 100
    # create ImageContainer
    img_orig = np.zeros((xdim, ydim, 10), dtype=np.uint8)
    # put a dot at y 20, x 50
    img_orig[20, 50, :] = range(10, 20)
    cont = ImageContainer(img_orig, img_id="image_0")

    # crop small crop
    xcoords = []
    ycoords = []
    crops = []
    for crop, x, y in cont.generate_equal_crops(xs=5, ys=5):
        crops.append(crop)
        xcoords.append(x)
        ycoords.append(y)
    a = ImageContainer.uncrop_img(crops, xcoords, ycoords, shape=cont.shape)

    # check that has cropped correct image
    assert np.max(np.abs(a.data["image_0"] - cont.data["image_0"])) == 0.0


def test_single_uncrop_img(tmpdir):
    """Crop image into one crop and uncrop again and check equality."""
    # create ImageContainer
    xdim = 100
    ydim = 100
    # create ImageContainer
    img_orig = np.zeros((xdim, ydim, 10), dtype=np.uint8)
    # put a dot at y 20, x 50
    img_orig[20, 50, :] = range(10, 20)
    cont = ImageContainer(img_orig, img_id="image_0")

    # crop small crop
    xcoords = []
    ycoords = []
    crops = []
    for crop, x, y in cont.generate_equal_crops(xs=xdim, ys=ydim):
        crops.append(crop)
        xcoords.append(x)
        ycoords.append(y)
    a = ImageContainer.uncrop_img(crops, xcoords, ycoords, shape=cont.shape)

    # check that has cropped correct image
    assert np.max(np.abs(a.data["image_0"] - cont.data["image_0"])) == 0.0
