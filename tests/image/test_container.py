from typing import Tuple, Union, Optional, Sequence
from pathlib import Path
import pytest

from anndata import AnnData

import numpy as np
import xarray as xr

import tifffile

from squidpy.im import ImageContainer
from squidpy.im._utils import CropCoords, _NULL_COORDS
from squidpy._constants._pkg_constants import Key


class TestContainerIO:
    def test_empty_initialization(self):
        img = ImageContainer()

        assert not len(img)
        assert img.shape == (0, 0)
        assert str(img)
        assert repr(img)

    def _test_initialize_from_dataset(self):
        dataset = xr.Dataset({"foo": xr.DataArray(np.zeros((100, 100, 3)))}, attrs={"foo": "bar"})
        img = ImageContainer._from_dataset(data=dataset)

        assert img.data is not dataset
        assert "foo" in img
        assert img.shape == (100, 100)
        np.testing.assert_array_equal(img.data.values(), dataset.values)
        assert img.data.attrs == dataset.attrs

    def test_save_load_zarr(self, tmpdir):
        img = ImageContainer(np.random.normal(size=(100, 100, 1)))
        img.data.attrs["scale"] = 42

        img.save(Path(tmpdir) / "foo")

        img2 = ImageContainer.load(Path(tmpdir) / "foo")

        np.testing.assert_array_equal(img["image"].values, img2["image"].values)
        np.testing.assert_array_equal(img.data.dims, img2.data.dims)
        np.testing.assert_array_equal(sorted(img.data.attrs.keys()), sorted(img2.data.attrs.keys()))
        for k, v in img.data.attrs.items():
            assert type(v) == type(img2.data.attrs[k])  # noqa: E721
            assert v == img2.data.attrs[k]

    def test_load_zarr_2_objects_can_overwrite_store(self, tmpdir):
        img = ImageContainer(np.random.normal(size=(100, 100, 1)))
        img.data.attrs["scale"] = 42

        img.save(Path(tmpdir) / "foo")

        img2 = ImageContainer.load(Path(tmpdir) / "foo")
        img2.data.attrs["sentinel"] = "foobar"
        img2["image"].values += 42
        img2.save(Path(tmpdir) / "foo")

        img3 = ImageContainer.load(Path(tmpdir) / "foo")

        assert "sentinel" in img3.data.attrs
        assert img3.data.attrs["sentinel"] == "foobar"

        np.testing.assert_array_equal(img3["image"].values, img2["image"].values)
        np.testing.assert_allclose(img3["image"].values - 42, img["image"].values)

    # TODO: add here: numpy.aray, xarray.array, invalid type,
    #  path to an existing file with known/unknown extension (JPEG and TIFF), path to a non-existent file
    @pytest.mark.parametrize("img", [])
    def test_add_img_types(self, img: Union[np.ndarray, xr.DataArray, str]):
        pass

    @pytest.mark.parametrize("array", [np.zeros((10, 10), dtype=np.uint8), np.random.rand(10, 10).astype(np.float32)])
    def test_load_2D_array(self, array: Union[np.ndarray, xr.DataArray]):
        img = ImageContainer(array)
        assert (img["image"].data == array).all()
        assert img["image"].data.dtype == array.dtype

        xarr = xr.DataArray(array)
        img = ImageContainer(xarr)
        assert (img["image"].data == array).all()
        assert img["image"].data.dtype == array.dtype

    def test_add_img_invalid_yx(self):
        pass

    def test_xarray_remapping_spatial_dims(self):
        pass

    @pytest.mark.parametrize("n_channels", [2, 3, 11])
    def test_add_img_number_of_channels(self, n_channels: int):
        img = ImageContainer()
        arr = np.random.rand(10, 10, n_channels)
        img.add_img(arr)
        assert img["image_0"].channels.shape == (n_channels,)

    @pytest.mark.parametrize("channel_dim", ["present", "absent"])
    def test_add_img_channel_dim(self, channel_dim: str):
        pass


class TestContainerCroppping:
    def test_padding_top_left(self, small_cont_1c: ImageContainer):
        crop = small_cont_1c.crop_center(0, 0, 10)
        data = crop["image"].data
        assert (data[:10, :10] == 0).all()
        assert data[10:, 10:].all()

    def test_padding_top_right(self, small_cont_1c: ImageContainer):
        crop = small_cont_1c.crop_center(0, small_cont_1c.shape[1], 10)
        data = crop["image"].data
        assert (data[:10, 10:] == 0).all()
        assert data[10:, :10].all()

    def test_padding_bottom_left(self, small_cont_1c: ImageContainer):
        crop = small_cont_1c.crop_center(small_cont_1c.shape[1], 0, 10)
        data = crop["image"].data
        assert (data[10:, :10] == 0).all()
        assert data[:10, 10:].any()

    def test_padding_bottom_right(self, small_cont_1c: ImageContainer):
        crop = small_cont_1c.crop_center(small_cont_1c.shape[1], small_cont_1c.shape[1], 10)
        data = crop["image"].data
        assert (data[10:, 10:] == 0).all()
        assert data[:10, :10].any()

    def test_padding_left_right(self, small_cont_1c: ImageContainer):
        dim1, dim2, _ = small_cont_1c["image"].data.shape
        crop = small_cont_1c.crop_center(dim1 // 2, 0, dim1 // 2)
        data = crop["image"].data
        assert (data[:, : dim2 // 2] == 0).all()
        crop = small_cont_1c.crop_center(dim1 // 2, dim2, dim1 // 2)
        data = crop["image"].data
        assert (data[:, dim2 // 2 :] == 0).all()

    def test_padding_top_bottom(self, small_cont_1c: ImageContainer):
        dim1, dim2, _ = small_cont_1c["image"].data.shape
        crop = small_cont_1c.crop_center(dim1, dim2 // 2, dim1 // 2)
        data = crop["image"].data
        assert (data[dim1 // 2 :, :] == 0).all()
        crop = small_cont_1c.crop_center(0, dim2 // 2, dim1 // 2)
        data = crop["image"].data
        assert (data[: dim2 // 2, :] == 0).all()

    def test_padding_all(self, small_cont_1c: ImageContainer):
        dim1, dim2, _ = small_cont_1c["image"].data.shape
        crop = small_cont_1c.crop_center(dim1 // 2, dim2 // 2, dim1)
        data = crop["image"].data
        assert (data[:, : dim2 // 2] == 0).all()
        assert (data[: dim2 // 2, :] == 0).all()

    @pytest.mark.parametrize("dy", [25, 0.3, None])
    @pytest.mark.parametrize("dx", [30, 0.5, None])
    def test_crop_corner_size(self, dy: Optional[Union[int, float]], dx: Optional[Union[int, float]]):
        pass

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
    def test_crop_corner_scale(self, scale: float):
        pass

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
    def test_test_crop_corner_cval(self, scale: float):
        pass

    @pytest.mark.parametrize("size", [(50, 50), (50, 49)])
    def test_crop_corner_mask_circle(self, size: Tuple[int, int]):
        pass

    @pytest.mark.parametrize("ry", [25, 0.3, None])
    @pytest.mark.parametrize("rx", [30, 0.5, None])
    def test_crop_center_radius(self, ry: Optional[Union[int, float]], rx: Optional[Union[int, float]]):
        pass

    @pytest.mark.parametrize("as_array", [False, True])
    def test_equal_crops_as_array(self, as_array: bool):
        pass

    @pytest.mark.parametrize("scale", [1, 0.5, 2])
    def test_spot_crops_spot_scale(self, scale: float):
        pass

    @pytest.mark.parametrize("obs_names", [None, ...])
    def test_spot_crops_obs_names(self, obs_names: Optional[Sequence[str]]):
        pass

    @pytest.mark.parametrize("return_obs", [False, True])
    @pytest.mark.parametrize("as_array", [False, True])
    def test_spot_crops_as_array_return_obs(self, as_array: bool, return_obs: bool):
        pass

    def test_spot_crops_mask_circle(self):
        pass

    def test_uncrop_preserves_shape(self):
        pass

    def test_uncrop_too_small_requested_shape(self):
        pass


class TestContainerUtils:
    @pytest.mark.parametrize("deep", [False, True])
    def test_copy(self, deep: bool):
        pass

    def test_get_default_size(self):
        pass

    def test_to_pixel_space(self):
        pass

    def test_apply_channel(self):
        pass

    def test_apply_inplace(self):
        pass

    def test_image_autoincrement(self):
        pass

    def test_repr_html(self):
        pass

    def test_repr(self):
        pass


class TestContainerShow:
    def test_channel(self):  # TODO: @hspitzer asked for some axis info
        pass

    def test_as_mask(self):
        pass


@pytest.mark.parametrize("shape", [(100, 200, 3), (100, 200, 1), (10, 100, 200, 1)])
def test_image_loading(shape, tmpdir):
    """Initialize ImageObject with tiff / multipagetiff / numpy array and check that loaded data \
    fits the expected shape + content.
    """
    img_orig = np.random.randint(low=0, high=255, size=shape, dtype=np.uint8)

    # load as np arrray
    if len(shape) <= 3:
        # load as np array
        cont = ImageContainer(img_orig)
        # check that contains same information
        assert (cont["image"] == img_orig).all()

    # save & load as tiff
    fname = tmpdir.mkdir("data").join("img.tif")
    tifffile.imsave(fname, img_orig)
    cont = ImageContainer(str(fname))

    if len(shape) > 3:
        # multi-channel tiff
        # check for existance of each im in multi-channel tiff
        # check that contains correct information
        assert (cont["image"] == img_orig[:, :, :, 0].transpose(1, 2, 0)).all()
    else:
        # check that contains same information
        assert (cont["image"] == img_orig).all()


@pytest.mark.parametrize(
    ("shape1", "shape2"),
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
    cont.add_img(img_new, img_id="img_new", channel_dim="mask")

    assert "img_orig" in cont.data
    assert "img_new" in cont.data
    assert (np.squeeze(cont.data["img_new"]) == np.squeeze(img_new)).all()
    np.testing.assert_array_equal(np.squeeze(cont.data["img_new"]), np.squeeze(img_new))


def test_crop(tmpdir):
    """Check crop arguments:
    padding, masking, scaling, changing dtype,
    check that returned crops have correct shape.
    """
    xdim = 100
    ydim = 200
    img_orig = np.zeros((xdim, ydim, 10), dtype=np.uint8)
    # put a dot at y 20, x 50
    img_orig[20, 50, :] = range(10, 20)
    cont = ImageContainer(img_orig, img_id="image_0")

    # crop big crop
    crop = cont.crop_center(
        y=50,
        x=20,
        radius=150,
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
    crop2 = cont.crop_corner(y=-100, x=-130, size=301, cval=5)
    np.testing.assert_array_equal(crop2.data["image_0"], crop.data["image_0"])

    # crop small crop
    crop = cont.crop_center(
        x=50,
        y=20,
        radius=0,
        cval=5,
    )
    assert type(crop) == ImageContainer
    assert crop.data["image_0"].shape == (1, 1, 10)
    # check that has cropped correct im
    assert (crop.data["image_0"][0, 0, :3] == [10, 11, 12]).all()
    assert crop.data["image_0"].dtype == np.uint8

    # crop with mask_circle
    crop = cont.crop_center(
        y=20,
        x=50,
        radius=5,
        cval=5,
        mask_circle=True,
    )
    np.testing.assert_array_equal(crop.data["image_0"][1, 0, :], 5)
    np.testing.assert_array_equal(crop.data["image_0"][2, 2, :], 0)
    np.testing.assert_array_equal(crop.data["image_0"][7, 7, :], 0)
    np.testing.assert_array_equal(crop.data["image_0"][9, 9, :], 5)

    assert type(crop) == ImageContainer

    # crop image with several layers
    mask = np.random.randint(low=0, high=10, size=(xdim, ydim))
    cont.add_img(mask, img_id="image_1", channel_dim="mask")
    crop = cont.crop_center(
        y=50,
        x=20,
        radius=0,
        cval=5,
    )
    assert "image_1" in crop.data.keys()
    assert "image_0" in crop.data.keys()
    assert crop.data["image_1"].shape == (1, 1, 1)
    assert crop.data["image_0"].shape == (1, 1, 10)

    # crop with scaling
    crop = cont.crop_center(y=50, x=20, radius=10, cval=5, scale=0.5)
    assert "image_1" in crop.data.keys()
    assert "image_0" in crop.data.keys()
    assert crop.data["image_1"].shape == (21 // 2, 21 // 2, 1)
    assert crop.data["image_0"].shape == (21 // 2, 21 // 2, 10)


def test_generate_spot_crops(adata: AnnData, cont: ImageContainer):
    """
    for simulated adata + im, generate crops.
    Make sure that the correct amount of crops are generated
    and that the crops have the correct content
    """
    i = 0
    expected_size = adata.uns["spatial"]["V1_Adult_Mouse_Brain"]["scalefactors"]["spot_diameter_fullres"] // 2 * 2 + 1
    for crop, obs_id in cont.generate_spot_crops(adata, return_obs=True):
        # crops have expected size?
        assert crop.shape[0] == expected_size
        assert crop.shape[1] == expected_size
        assert obs_id == adata.obs.index[i]
        i += 1
    # expected number of crops are generated?
    assert i == adata.obsm[Key.obsm.spatial].shape[0]


@pytest.mark.parametrize(("ys", "xs"), [(10, 10), (None, None), (10, 20)])
def test_generate_equal_crops(ys, xs):
    """
    for simulated data, create equally sized crops.
    Make sure that the resulting crops have correct sizes.
    """
    img = ImageContainer(np.random.randint(low=0, high=300, size=(200, 500)), img_id="image")
    expected_size = [ys, xs]
    if ys is None or xs is None:
        expected_size = (200, 500)

    for crop in img.generate_equal_crops(size=(ys, xs)):
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
    crops = []
    for crop in cont.generate_equal_crops(size=5):
        crops.append(crop)
    a = ImageContainer.uncrop(crops)

    # check that has cropped correct image
    np.testing.assert_array_equal(a["image_0"], cont["image_0"])


def test_single_uncrop_img(tmpdir):
    """Crop image into one crop and uncrop again and check equality."""
    # create ImageContainer
    ydim = 100
    xdim = 100
    # create ImageContainer
    img_orig = np.zeros((ydim, xdim, 10), dtype=np.uint8)
    # put a dot at y 20, x 50
    img_orig[20, 50, :] = range(10, 20)
    cont = ImageContainer(img_orig, img_id="image_0")

    # crop small crop
    crops = []
    for crop in cont.generate_equal_crops(size=(ydim, xdim)):
        crops.append(crop)
    a = ImageContainer.uncrop(crops)

    # check that has cropped correct image
    np.testing.assert_array_equal(a["image_0"], cont["image_0"])


def test_crop_metadata(cont: ImageContainer) -> None:
    crop = cont.crop_corner(0, 0, 50)

    assert cont.data.attrs["coords"] is _NULL_COORDS
    assert crop.data.attrs["coords"] == CropCoords(0, 0, 50, 50)
