import pytest
import numpy as np
import rasterio.errors
import warnings

from spatial_tools.image.crop import uncrop_img
from spatial_tools.image.object import ImageContainer


@pytest.mark.parametrize(
    "shape", [(3, 100, 200), (1, 100, 200), (10, 1, 100, 200)]
)
def test_image_loading(shape, tmpdir):
    """\
    initialize ImageObject with tiff / multipagetiff / numpy array and check that loaded data 
    fits the expected shape + content
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
    print(cont)
    if len(shape) > 3:
        # multi-channel tiff
        # check for existance of each image in multi-channel tiff
        # check that contains correct information
        assert (cont.data["image"] == img_orig[:, 0, :, :]).all()
    else:
        print(cont.data)
        # check that contains same information
        assert (cont.data.image == img_orig).all()


@pytest.mark.parametrize(
    "shape1,shape2",
    [
        ((3, 100, 200), (3, 100, 200)),
        ((100, 200), (3, 100, 200)),
        ((10, 3, 100, 200), (100, 200)),
    ],
)
def test_add_img(shape1, shape2):
    """\
    add image to existing ImageObject and check result
    """
    # create ImageContainer
    img_orig = np.random.randint(low=0, high=255, size=(3, 100, 200), dtype=np.uint8)
    cont = ImageContainer(img_orig, img_id="img_orig")

    # add image
    img_new = np.random.randint(low=0, high=255, size=(100, 200), dtype=np.uint8)
    cont.add_img(img_new, img_id="img_new")

    assert "img_orig" in cont.data
    assert "img_new" in cont.data
    assert (cont.data["img_new"] == img_new).all()


def test_crop(tmpdir):
    """\
    crop different img_ids and check result
    """
    import tifffile

    # ignore NotGeoreferencedWarning here
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    # create ImageContainer
    xdim = 100
    ydim = 200
    img_orig = np.zeros((10, xdim, ydim), dtype=np.uint8)
    # put a dot at y 20, x 50
    img_orig[:, 20, 50] = range(10, 20)
    cont = ImageContainer(img_orig, img_id="image_0")

    # crop big crop
    crop = cont.crop(
        x=50,
        y=20,
        xs=300,
        ys=300,
        cval=5,
        img_id="image_0",
    )
    # shape is s x s x len(img_id)/channels
    assert crop.shape == (10, 300, 300)
    # check that values outside of img are padded with 5
    assert (crop[:, 0, 0] == 5).all()
    assert (crop[:, -1, -1] == 5).all()

    # crop small crop
    crop = cont.crop(
        x=50.5,
        y=20.5,
        xs=1,
        ys=1,
        cval=5,
        img_id="image_0",
    )
    assert crop.shape == (10, 1, 1)
    # check that has cropped correct image
    assert (crop[:3, 0, 0] == [10, 11, 12]).all()

    # crop casting to dtype
    img_orig = np.zeros((10, xdim, ydim), dtype=np.uint16)
    cont = ImageContainer(img_orig, img_id="image_0")
    crop = cont.crop(
        x=50, y=20, xs=300, ys=300, cval=5, img_id="image_0", dtype="uint8"
    )
    assert crop.dtype == np.uint8


def test_uncrop_img(tmpdir):
    """\
    crop image and uncrop again and check equality
    """
    import tifffile

    # create ImageContainer
    xdim = 100
    ydim = 100
    # create ImageContainer
    img_orig = np.zeros((10, xdim, ydim), dtype=np.uint8)
    # put a dot at y 20, x 50
    img_orig[:, 20, 50] = range(10, 20)
    cont = ImageContainer(img_orig, img_id="image_0")

    # crop small crop
    crops, xcoord, ycoord = cont.crop_equally(xs=5, ys=5, img_id="image_0")
    a = uncrop_img(
        crops=crops,
        x=xcoord,
        y=ycoord,
        shape=cont.shape,
    )
    # check that has cropped correct image
    assert np.max(np.abs(a - cont.data["image_0"])) == 0.0


def test_single_uncrop_img(tmpdir):
    """\
    crop image into one crop and uncrop again and check equality
    """
    import tifffile

    # create ImageContainer
    xdim = 100
    ydim = 100
    # create ImageContainer
    img_orig = np.zeros((10, xdim, ydim), dtype=np.uint8)
    # put a dot at y 20, x 50
    img_orig[:, 20, 50] = range(10, 20)
    cont = ImageContainer(img_orig, img_id="image_0")

    # crop small crop
    crops, xcoord, ycoord = cont.crop_equally(xs=xdim, ys=ydim, img_id="image_0")
    a = uncrop_img(
        crops=crops,
        x=xcoord,
        y=ycoord,
        shape=cont.shape,
    )
    # check that has cropped correct image
    assert np.max(np.abs(a - cont.data["image_0"])) == 0.0
