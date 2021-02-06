from typing import Tuple, Union, Optional
from pathlib import Path
import pytest

from anndata import AnnData

import numpy as np
import xarray as xr

import tifffile

from squidpy.im import ImageContainer
from squidpy.im._utils import CropCoords, CropPadding, _NULL_COORDS
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


class TestContainerCropping:
    def test_padding_top_left(self, small_cont_1c: ImageContainer):
        crop = small_cont_1c.crop_center(0, 0, 10)

        data = crop["image"].data

        assert crop.shape == (21, 21)
        np.testing.assert_array_equal(data[:10, :10], 0)
        np.testing.assert_array_equal(data[10:, 10:] != 0, True)

    def test_padding_top_right(self, small_cont_1c: ImageContainer):
        crop = small_cont_1c.crop_center(0, small_cont_1c.shape[1], 10)
        data = crop["image"].data

        assert crop.shape == (21, 21)
        np.testing.assert_array_equal(data[:10, 10:], 0)
        np.testing.assert_array_equal(data[10:, :10] != 0, True)

    def test_padding_bottom_left(self, small_cont_1c: ImageContainer):
        crop = small_cont_1c.crop_center(small_cont_1c.shape[1], 0, 10)
        data = crop["image"].data

        assert crop.shape == (21, 21)
        np.testing.assert_array_equal(data[10:, :10], 0)
        np.testing.assert_array_equal(data[:10, 10:] != 0, True)

    def test_padding_bottom_right(self, small_cont_1c: ImageContainer):
        crop = small_cont_1c.crop_center(small_cont_1c.shape[1], small_cont_1c.shape[1], 10)
        data = crop["image"].data

        assert crop.shape == (21, 21)
        np.testing.assert_array_equal(data[10:, 10:], 0)
        np.testing.assert_array_equal(data[:10, :10] != 0, True)

    def test_padding_left_right(self, small_cont_1c: ImageContainer):
        dim1, dim2, _ = small_cont_1c["image"].data.shape

        crop = small_cont_1c.crop_center(dim1 // 2, 0, dim1 // 2)
        data = crop["image"].data
        np.testing.assert_array_equal(data[:, : dim2 // 2], 0)

        crop = small_cont_1c.crop_center(dim1 // 2, dim2, dim1 // 2)
        data = crop["image"].data
        np.testing.assert_array_equal(data[:, dim2 // 2 :], 0)

    def test_padding_top_bottom(self, small_cont_1c: ImageContainer):
        dim1, dim2, _ = small_cont_1c["image"].data.shape

        crop = small_cont_1c.crop_center(dim1, dim2 // 2, dim1 // 2)
        data = crop["image"].data
        np.testing.assert_array_equal(data[dim1 // 2 :, :], 0)

        crop = small_cont_1c.crop_center(0, dim2 // 2, dim1 // 2)
        data = crop["image"].data
        np.testing.assert_array_equal(data[: dim2 // 2, :], 0)

    def test_padding_all(self, small_cont_1c: ImageContainer):
        dim1, dim2, _ = small_cont_1c["image"].data.shape
        crop = small_cont_1c.crop_center(dim1 // 2, dim2 // 2, dim1)
        data = crop["image"].data

        np.testing.assert_array_equal(data[:, : dim2 // 2], 0)
        np.testing.assert_array_equal(data[: dim2 // 2, :], 0)

    @pytest.mark.parametrize("dy", [-10, 25, 0.3])
    @pytest.mark.parametrize("dx", [-10, 30, 0.5])
    def test_crop_corner_size(
        self, small_cont_1c: ImageContainer, dy: Optional[Union[int, float]], dx: Optional[Union[int, float]]
    ):
        crop = small_cont_1c.crop_corner(dy, dx, size=20)
        # original coordinates
        ody, odx = max(dy, 0), max(dx, 0)
        ody = int(ody * small_cont_1c.shape[0]) if isinstance(ody, float) else ody
        odx = int(odx * small_cont_1c.shape[1]) if isinstance(odx, float) else odx

        # crop coordinates
        cdy = 0 if isinstance(dy, float) or dy > 0 else dy
        cdx = 0 if isinstance(dx, float) or dx > 0 else dx
        cdy, cdx = abs(cdy), abs(cdx)

        assert crop.shape == (20, 20)
        cdata, odata = crop["image"].data, small_cont_1c["image"].data
        cdata = cdata[cdy:, cdx:]
        np.testing.assert_array_equal(cdata, odata[ody : ody + cdata.shape[0], odx : odx + cdata.shape[1]])

    @pytest.mark.parametrize("scale", [0, 0.5, 1.0, 1.5, 2.0])
    def test_crop_corner_scale(self, scale: float):
        shape_img = (50, 50)
        img = ImageContainer(np.zeros(shape_img))
        if scale <= 0:
            with pytest.raises(ValueError, match=r"Expected `scale` to be positive, found `0`."):
                img.crop_corner(10, 10, size=20, scale=scale)
        else:
            crop = img.crop_corner(10, 10, size=20, scale=scale)
            assert crop.shape == tuple(int(i * scale) for i in (20, 20))

    @pytest.mark.parametrize("cval", [0.5, 1.0, 2.0])
    def test_test_crop_corner_cval(self, cval: float):
        shape_img = (50, 50)
        img = ImageContainer(np.zeros(shape_img))
        crop = img.crop_corner(10, 10, cval=cval)
        np.testing.assert_array_equal(crop["image"].data[-10:, -10:], cval)

    @pytest.mark.parametrize("size", [(10, 10), (10, 11)])
    def test_crop_corner_mask_circle(self, small_cont_1c: ImageContainer, size: Tuple[int, int]):
        if size[0] != size[1]:
            with pytest.raises(ValueError, match=r"Masking circle is only"):
                small_cont_1c.crop_corner(0, 0, size=size, mask_circle=True, cval=np.nan)
        else:
            crop = small_cont_1c.crop_corner(0, 0, size=20, mask_circle=True, cval=np.nan)
            mask = (crop.data.x - 10) ** 2 + (crop.data.y - 10) ** 2 <= 10 ** 2

            assert crop.shape == (20, 20)
            np.testing.assert_array_equal(crop["image"].values[..., 0][~mask.values], np.nan)

    @pytest.mark.parametrize("ry", [-10, 25, 0.3])
    @pytest.mark.parametrize("rx", [-10, 30, 0.5])
    def test_crop_center_radius(self, ry: Optional[Union[int, float]], rx: Optional[Union[int, float]]):
        pass

    @pytest.mark.parametrize("as_array", [False, True, "image"])
    def test_equal_crops_as_array(self, small_cont: ImageContainer, as_array: bool):
        small_cont.add_img(np.random.normal(size=(small_cont.shape + (1,))), channel_dim="foobar")
        for crop in small_cont.generate_equal_crops(size=11, as_array=as_array):
            if as_array:
                if isinstance(as_array, bool):
                    assert isinstance(crop, dict)
                    for key in small_cont:
                        assert key in crop
                        assert crop[key].shape == (11, 11, small_cont[key].data.shape[-1])
                else:
                    assert isinstance(crop, np.ndarray)
                    assert crop.shape == (11, 11, small_cont[as_array].data.shape[-1])
            else:
                assert isinstance(crop, ImageContainer)
                for key in (Key.img.coords, Key.img.padding, Key.img.scale, Key.img.mask_circle):
                    assert key in crop.data.attrs, key
                assert crop.shape == (11, 11)

    @pytest.mark.parametrize("return_obs", [False, True])
    @pytest.mark.parametrize("as_array", [False, True, "baz"])
    def test_spot_crops_as_array_return_obs(
        self, adata: AnnData, cont: ImageContainer, as_array: bool, return_obs: bool
    ):
        cont.add_img(np.random.normal(size=(cont.shape + (4,))), channel_dim="foobar", img_id="baz")
        diameter = adata.uns["spatial"][Key.uns.library_id(adata, "spatial")]["scalefactors"]["spot_diameter_fullres"]
        radius = int(round(diameter // 2))
        size = (2 * radius + 1, 2 * radius + 1)

        for crop in cont.generate_spot_crops(adata, as_array=as_array, return_obs=return_obs, spatial_key="spatial"):
            crop, obs = crop if return_obs else (crop, None)
            if obs is not None:
                assert obs in adata.obs_names
                if not as_array:
                    assert Key.img.obs in crop.data.attrs

            if as_array is True:
                assert isinstance(crop, dict), type(crop)
                for key in cont:
                    assert key in crop
                    assert crop[key].shape == (*size, cont[key].data.shape[-1])
            elif isinstance(as_array, str):
                assert isinstance(crop, np.ndarray)
                assert crop.shape == (*size, cont[as_array].data.shape[-1])
            else:
                assert isinstance(crop, ImageContainer)
                assert crop.shape == size

    @pytest.mark.parametrize("n_names", [None, 4])
    def test_spot_crops_obs_names(self, adata: AnnData, cont: ImageContainer, n_names: Optional[int]):
        crops = list(cont.generate_spot_crops(adata, obs_names=None if n_names is None else adata.obs_names[:n_names]))

        if n_names is None:
            assert len(crops) == adata.n_obs
        else:
            assert len(crops) == n_names

    @pytest.mark.parametrize("spot_scale", [1, 0.5, 2])
    @pytest.mark.parametrize("scale", [1, 0.5, 2])
    def test_spot_crops_spot_scale(self, adata: AnnData, cont: ImageContainer, scale: float, spot_scale: float):
        diameter = adata.uns["spatial"][Key.uns.library_id(adata, "spatial")]["scalefactors"]["spot_diameter_fullres"]
        radius = int(round(diameter // 2) * spot_scale)
        size = int((2 * radius + 1) * scale), int((2 * radius + 1) * scale)

        for crop in cont.generate_spot_crops(adata, spot_scale=spot_scale, scale=scale):
            assert crop.shape == size

    @pytest.mark.parametrize("preserve", [False, True])
    def test_preserve_dtypes(self, cont: ImageContainer, preserve: bool):
        assert np.issubdtype(cont["image"].dtype, np.uint8)

        crop = cont.crop_corner(-10, -10, 20, cval=-5, preserve_dtypes=preserve)

        if preserve:
            assert np.issubdtype(crop["image"].dtype, np.uint8)
            # we specifically use 0, otherwise overflow would happend and the value would be 256 - 5
            np.testing.assert_array_equal(crop["image"][:10, :10], 0)
        else:
            assert np.issubdtype(crop["image"].dtype, np.signedinteger)
            np.testing.assert_array_equal(crop["image"][:10, :10], -5)

    def test_spot_crops_mask_circle(self, adata: AnnData, cont: ImageContainer):
        for crop in cont.generate_spot_crops(adata, cval=np.nan, mask_circle=True, preserve_dtypes=False):
            assert crop.shape[0] == crop.shape[1]
            c = crop.shape[0] // 2
            mask = (crop.data.x - c) ** 2 + (crop.data.y - c) ** 2 <= c ** 2

            np.testing.assert_array_equal(crop["image"].values[..., 0][~mask.values], np.nan)

    def test_uncrop_preserves_shape(self, small_cont_1c: ImageContainer):
        small_cont_1c.add_img(np.random.normal(size=(small_cont_1c.shape + (4,))), channel_dim="foobar", img_id="baz")
        crops = list(small_cont_1c.generate_equal_crops(size=13))

        uncrop = ImageContainer.uncrop(crops)

        np.testing.assert_array_equal(small_cont_1c.shape, uncrop.shape)
        for key in small_cont_1c:
            np.testing.assert_array_equal(uncrop[key], small_cont_1c[key])

    def test_uncrop_too_small_requested_shape(self, small_cont_1c: ImageContainer):
        crops = list(small_cont_1c.generate_equal_crops(size=13))
        with pytest.raises(ValueError, match=r"Requested final image shape"):
            ImageContainer.uncrop(crops, shape=(small_cont_1c.shape[0] - 1, small_cont_1c.shape[1] - 1))

    @pytest.mark.parametrize("dy", [-10, 0])
    def test_crop_metadata(self, small_cont_1c: ImageContainer, dy: int):
        crop = small_cont_1c.crop_corner(dy, 0, 50, mask_circle=True)

        assert small_cont_1c.data.attrs[Key.img.coords] is _NULL_COORDS
        assert crop.data.attrs[Key.img.coords] == CropCoords(0, 0, 50, 50 + dy)
        assert crop.data.attrs[Key.img.padding] == CropPadding(x_pre=0, y_pre=abs(dy), x_post=0, y_post=0)
        assert crop.data.attrs[Key.img.mask_circle]


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
