import subprocess
from collections import defaultdict
from html.parser import HTMLParser
from itertools import permutations
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import anndata as ad
import dask.array as da
import imageio.v3 as iio
import numpy as np
import pytest
import squidpy as sq
import tifffile
import xarray as xr
from anndata import AnnData
from PIL import Image
from pytest_mock import MockerFixture
from squidpy._constants._pkg_constants import Key
from squidpy.im import ImageContainer
from squidpy.im._coords import _NULL_COORDS, CropCoords, CropPadding


class SimpleHTMLValidator(HTMLParser):  # modified from CellRank
    def __init__(self, n_expected_rows: int, expected_tags: Set[str], **kwargs: Any):
        super().__init__(**kwargs)
        self._cnt = defaultdict(int)
        self._n_rows = 0

        self._n_expected_rows = n_expected_rows
        self._expected_tags = expected_tags

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self._cnt[tag] += 1
        self._n_rows += tag == "strong"

    def handle_endtag(self, tag: str) -> None:
        self._cnt[tag] -= 1

    def validate(self) -> None:
        assert self._n_rows == self._n_expected_rows
        assert set(self._cnt.keys()) == self._expected_tags
        if len(self._cnt):
            assert set(self._cnt.values()) == {0}


class TestContainerIO:
    def test_empty_initialization(self):
        img = ImageContainer()

        assert not len(img)
        assert isinstance(img.data, xr.Dataset)
        assert img.shape == (0, 0)
        assert str(img)
        assert repr(img)

    @pytest.mark.parametrize("on_init", [False, True])
    def test_lazy_load(self, on_init: bool, tmpdir):
        img_orig = np.random.randint(low=0, high=255, size=(100, 100, 1), dtype=np.uint8)
        if on_init:
            fname = str(tmpdir / "tmp.tiff")
            tifffile.imwrite(fname, img_orig)
            img = ImageContainer(fname, lazy=True)
        else:
            img = ImageContainer(da.from_array(img_orig), lazy=True)

        assert len(img) == 1
        for key in img:
            value = img[key].data
            assert isinstance(value, da.Array)
            np.testing.assert_array_equal(np.squeeze(value.compute()), np.squeeze(img_orig))

    def _test_initialize_from_dataset(self):
        dataset = xr.Dataset({"foo": xr.DataArray(np.zeros((100, 100, 3)))}, attrs={"foo": "bar"})
        img = ImageContainer._from_dataset(data=dataset)

        assert img.data is not dataset
        assert "foo" in img
        assert img.shape == (100, 100)
        np.testing.assert_array_equal(img.data.values(), dataset.values)
        assert img.data.attrs == dataset.attrs

    @pytest.mark.skip(reason="Sometimes fails to load image")
    def test_save_load_zarr(self, tmpdir):
        img = ImageContainer(np.random.normal(size=(100, 100, 1)))
        img.data.attrs["scale"] = 42
        img.save(Path(tmpdir) / "foo.zarr")

        img2 = ImageContainer.load(Path(tmpdir) / "foo.zarr")
        proc = None
        try:
            # start a simple http-server
            proc = subprocess.Popen(
                f"python -m http.server 8080 --bind 127.0.0.1 --directory {tmpdir}".split(" "),
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            img3 = ImageContainer.load("http://127.0.0.1:8080/foo.zarr")

            for test_img in [img2, img3]:
                np.testing.assert_array_equal(img["image"].values, test_img["image"].values)
                np.testing.assert_array_equal(img["image"].dims, test_img["image"].dims)
                assert img.data.dims == test_img.data.dims
                np.testing.assert_array_equal(sorted(img.data.attrs.keys()), sorted(img2.data.attrs.keys()))
                for k, v in img.data.attrs.items():
                    assert v == test_img.data.attrs[k]
        finally:
            if proc is not None:
                proc.terminate()
                proc.wait()

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

    @pytest.mark.parametrize(
        ("shape1", "shape2"),
        [
            ((100, 200, 3), (100, 200, 1)),
            ((100, 200, 3), (100, 200)),
        ],
    )
    def test_add_img(self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]):
        img_orig = np.random.randint(low=0, high=255, size=shape1, dtype=np.uint8)
        cont = ImageContainer(img_orig, layer="img_orig")

        img_new = np.random.randint(low=0, high=255, size=shape2, dtype=np.uint8)
        cont.add_img(img_new, layer="img_new", channel_dim="mask")

        assert "img_orig" in cont
        assert "img_new" in cont
        np.testing.assert_array_equal(np.squeeze(cont.data["img_new"]), np.squeeze(img_new))

    def test_add_img_invalid_zdim(self, cont: ImageContainer):
        with pytest.raises(ValueError, match=r"Expected image to have `1` Z-dimension\(s\), found `10`."):
            cont.add_img(np.random.normal(size=(*cont.shape, 10, 3)), dims=["y", "x", "z", "channels"])

    @pytest.mark.parametrize("ext", ["jpg", "png"])
    @pytest.mark.parametrize("shape", [(100, 200, 3), (100, 200, 1)])
    def test_load_ext(self, shape: Tuple[int, ...], ext: str, tmpdir):
        fname = tmpdir / f"tmp.{ext}"

        if shape == (100, 200, 1):
            img = np.random.randint(256, size=(100, 200), dtype=np.uint8)
            img_orig = Image.fromarray(img)
        else:
            img_orig = np.random.randint(low=0, high=255, size=shape, dtype=np.uint8)
        iio.imwrite(str(fname), img_orig)

        gt = iio.imread(str(fname))  # because of compression, we load again
        cont = ImageContainer(str(fname))

        np.testing.assert_array_equal(cont["image"].values.squeeze(), gt.squeeze())

    @pytest.mark.parametrize("shape", [(100, 200, 3), (100, 200, 1), (10, 100, 200, 1)])
    def test_load_tiff(self, shape: Tuple[int, ...], tmpdir):
        img_orig = np.random.randint(low=0, high=255, size=shape, dtype=np.uint8)
        fname = tmpdir / "tmp.tiff"
        tifffile.imwrite(fname, img_orig)

        cont = ImageContainer(str(fname))

        if len(shape) > 3:  # multi-channel tiff
            np.testing.assert_array_equal(np.squeeze(cont["image"]), img_orig[..., 0].transpose(1, 2, 0))
        else:
            np.testing.assert_array_equal(np.squeeze(cont["image"]), np.squeeze(img_orig))

    @pytest.mark.parametrize("dims", [("y", "x", "z", "c"), ("foo", "bar", "faa", "baz")])
    def test_load_netcdf(self, tmpdir, dims: Tuple[str, ...]):
        arr = np.random.normal(size=(100, 10, 1, 4))
        ds = xr.Dataset({"quux": xr.DataArray(arr, dims=dims)})
        fname = tmpdir / "tmp.nc"
        ds.to_netcdf(str(fname))

        if "foo" in dims:
            with pytest.raises(ValueError, match=r"Expected to find"):
                _ = ImageContainer(str(fname))
        else:
            cont = ImageContainer(str(fname))

            assert len(cont) == 1
            assert "quux" in cont
            np.testing.assert_array_equal(cont["quux"], ds["quux"])

    @pytest.mark.parametrize(
        "array", [np.zeros((10, 10, 3), dtype=np.uint8), np.random.rand(10, 10, 1).astype(np.float32)]
    )
    def test_array_dtypes(self, array: Union[np.ndarray, xr.DataArray]):
        img = ImageContainer(array)
        np.testing.assert_array_equal(np.squeeze(img["image"].data), np.squeeze(array))
        assert img["image"].data.dtype == array.dtype

        img = ImageContainer(xr.DataArray(array))
        np.testing.assert_array_equal(np.squeeze(img["image"].data), np.squeeze(array))
        assert img["image"].data.dtype == array.dtype

    def test_add_img_invalid_yx(self, small_cont_1c: ImageContainer):
        arr = xr.DataArray(np.empty((small_cont_1c.shape[0] - 1, small_cont_1c.shape[1])), dims=["y", "x"])
        with pytest.raises(ValueError, match=r".*cannot reindex or align along dimension"):
            small_cont_1c.add_img(arr)

    def test_xarray_remapping_spatial_dims(self):
        cont = ImageContainer(np.empty((100, 10)))
        cont.add_img(xr.DataArray(np.empty((100, 10)), dims=["foo", "bar"]), layer="baz")

        assert "baz" in cont
        assert len(cont) == 2
        assert cont["baz"].dims == ("y", "x", "z", "channels")

    @pytest.mark.parametrize("n_channels", [2, 3, 9])
    def test_add_img_number_of_channels(self, n_channels: int):
        img = ImageContainer()
        arr = np.random.rand(10, 10, n_channels)  # if n_channels >= 10, it would fail
        img.add_img(arr)
        assert img["image_0"].channels.shape == (n_channels,)

    @pytest.mark.parametrize("n_channels", [1, 3])
    @pytest.mark.parametrize("channel_dim", ["channels", "foo"])
    def test_add_img_channel_dim(self, small_cont_1c: ImageContainer, channel_dim: str, n_channels: int):
        arr = np.random.normal(size=(*small_cont_1c.shape, 1, n_channels))
        dims = ["y", "x", "z", channel_dim]
        expected_channel_dim = small_cont_1c._get_next_channel_id("channels")

        small_cont_1c.add_img(arr, dims=dims, layer="bar")
        assert len(small_cont_1c) == 2
        assert "bar" in small_cont_1c
        if channel_dim == "channels" and n_channels == 3:
            assert small_cont_1c["bar"].dims == ("y", "x", "z", expected_channel_dim)
        else:
            assert small_cont_1c["bar"].dims == ("y", "x", "z", channel_dim)
        np.testing.assert_array_equal(np.squeeze(small_cont_1c["bar"]), np.squeeze(arr))

    def test_add_img_does_not_load_other_lazy_layers(self, small_cont_1c: ImageContainer):
        img = np.random.normal(size=small_cont_1c.shape + (2,))
        lazy_img = da.from_array(img)

        for i in range(3):
            small_cont_1c.add_img(lazy_img, lazy=True, layer=f"lazy_{i}")
        small_cont_1c.add_img(lazy_img, lazy=False, layer="eager")

        for i in range(3):
            assert isinstance(small_cont_1c[f"lazy_{i}"].data, da.Array)
            np.testing.assert_array_equal(np.squeeze(small_cont_1c[f"lazy_{i}"].values), np.squeeze(img))
        assert isinstance(small_cont_1c["eager"].data, np.ndarray)
        np.testing.assert_array_equal(np.squeeze(small_cont_1c["eager"].values), np.squeeze(img))

    @pytest.mark.parametrize("copy", [False, True])
    def test_add_img_copy(self, small_cont_1c: ImageContainer, copy: bool):
        img = np.random.normal(size=small_cont_1c.shape + (1,))

        small_cont_1c.add_img(img, copy=copy, layer="foo")
        small_cont_1c.add_img(img, copy=copy, layer="bar")

        if copy:
            assert not np.shares_memory(small_cont_1c["foo"], small_cont_1c["bar"])
        else:
            assert np.shares_memory(small_cont_1c["foo"], small_cont_1c["bar"])

        np.testing.assert_array_equal(np.squeeze(small_cont_1c["foo"].values), np.squeeze(img))
        np.testing.assert_array_equal(np.squeeze(small_cont_1c["bar"].values), np.squeeze(img))

    def test_delete(self, small_cont_1c: ImageContainer):
        assert len(small_cont_1c) == 1
        del small_cont_1c["image"]

        assert len(small_cont_1c) == 0

        with pytest.raises(KeyError, match=r"'image'"):
            del small_cont_1c["image"]

    @pytest.mark.parametrize("img_key", [None, "hires", "lowres"])
    def test_read_from_adata(self, adata: AnnData, img_key: Optional[str]):
        img = sq.im.ImageContainer.from_adata(adata, img_key=img_key)
        if img_key is None:
            img_key = "hires"
        shape = ((100, 100) if img_key == "hires" else (88, 49)) + (1, 3)

        assert isinstance(img, ImageContainer)
        assert img_key in img

        np.testing.assert_array_equal(img[img_key].shape, shape)

    @pytest.mark.parametrize("scale", [None, 42])
    def test_read_from_adata_scalefactor(self, adata: AnnData, scale: Optional[int]):
        img_key = "lowres"
        library_id = Key.uns.library_id(adata, Key.uns.spatial)
        del adata.uns[Key.uns.spatial][library_id]["scalefactors"][f"tissue_{img_key}_scalef"]
        if scale is None:
            kwargs = {}
            scale = 1.0
        else:
            kwargs = {"scale": scale}

        img = sq.im.ImageContainer.from_adata(adata, img_key=img_key, **kwargs, copy=False)

        assert img.data.attrs["scale"] == scale
        assert np.shares_memory(img[img_key].values, adata.uns[Key.uns.spatial][library_id]["images"][img_key])


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
        dim1, dim2, _, _ = small_cont_1c["image"].data.shape

        crop = small_cont_1c.crop_center(dim1 // 2, 0, dim1 // 2)
        data = crop["image"].data
        np.testing.assert_array_equal(data[:, : dim2 // 2], 0)

        crop = small_cont_1c.crop_center(dim1 // 2, dim2, dim1 // 2)
        data = crop["image"].data
        np.testing.assert_array_equal(data[:, dim2 // 2 :], 0)

    def test_padding_top_bottom(self, small_cont_1c: ImageContainer):
        dim1, dim2, _, _ = small_cont_1c["image"].data.shape

        crop = small_cont_1c.crop_center(dim1, dim2 // 2, dim1 // 2)
        data = crop["image"].data
        np.testing.assert_array_equal(data[dim1 // 2 :, :], 0)

        crop = small_cont_1c.crop_center(0, dim2 // 2, dim1 // 2)
        data = crop["image"].data
        np.testing.assert_array_equal(data[: dim2 // 2, :], 0)

    def test_padding_all(self, small_cont_1c: ImageContainer):
        dim1, dim2, _, _ = small_cont_1c["image"].data.shape
        crop = small_cont_1c.crop_center(dim1 // 2, dim2 // 2, dim1)
        data = crop["image"].data

        np.testing.assert_array_equal(data[:, : dim2 // 2], 0)
        np.testing.assert_array_equal(data[: dim2 // 2, :], 0)

    @pytest.mark.parametrize("as_dask", [False, True])
    def test_lazy_scale(self, as_dask: bool):
        arr = np.empty((50, 50))
        scale = np.pi
        img = ImageContainer(da.from_array(arr) if as_dask else arr)

        crop = img.crop_corner(0, 0, size=20, scale=scale)

        assert crop.shape == tuple(round(i * scale) for i in (20, 20))
        if as_dask:
            assert isinstance(crop["image"].data, da.Array)
            crop.compute()

        assert isinstance(crop["image"].data, np.ndarray)

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
            assert crop.shape == tuple(round(i * scale) for i in (20, 20))

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
            mask = (crop.data.x - 10) ** 2 + (crop.data.y - 10) ** 2 <= 10**2

            assert crop.shape == (20, 20)
            np.testing.assert_array_equal(crop["image"].values[..., 0][~mask.values], np.nan)

    @pytest.mark.parametrize("ry", [23, 1.0])
    @pytest.mark.parametrize("rx", [30, 0.5])
    def test_crop_center_radius(
        self, small_cont_1c: ImageContainer, ry: Optional[Union[int, float]], rx: Optional[Union[int, float]]
    ):
        crop = small_cont_1c.crop_center(0, 0, radius=(ry, rx))
        sy = int(ry * small_cont_1c.shape[0]) if isinstance(ry, float) else ry
        sx = int(rx * small_cont_1c.shape[1]) if isinstance(rx, float) else rx

        assert crop.shape == (2 * sy + 1, 2 * sx + 1)

    @pytest.mark.parametrize("squeeze", [False, True])
    @pytest.mark.parametrize("as_array", [False, True, "image", ["image", "baz"]])
    def test_equal_crops_as_array(self, small_cont: ImageContainer, as_array: bool, squeeze: bool):
        def assert_shape(expected: xr.DataArray, actual: Union[np.ndarray, xr.DataArray]):
            expected_shape = list(expected.shape)
            expected_shape[:2] = [11, 11]  # because crop is 11x11
            if squeeze:
                assert actual.shape == np.squeeze(np.empty(expected_shape)).shape
            else:
                assert actual.shape == tuple(expected_shape)

        small_cont.add_img(np.random.normal(size=(small_cont.shape + (1, 1))), channel_dim="foobar", layer="baz")
        for crop in small_cont.generate_equal_crops(size=11, as_array=as_array, squeeze=squeeze):
            if as_array:
                if isinstance(as_array, bool):
                    assert isinstance(crop, dict)
                    for key in small_cont:
                        assert key in crop
                        assert_shape(small_cont[key].data, crop[key])
                elif isinstance(as_array, str):
                    assert isinstance(crop, np.ndarray)
                    assert_shape(small_cont[as_array].data, crop)
                else:
                    assert isinstance(crop, tuple)
                    assert len(crop) == len(as_array)
                    for key, data in zip(as_array, crop):
                        assert isinstance(data, np.ndarray)
                        assert_shape(small_cont[key].data, data)
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
        cont.add_img(np.random.normal(size=(cont.shape + (4,))), channel_dim="foobar", layer="baz")
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
        obs = adata.obs_names[:n_names] if isinstance(n_names, int) else adata.obs_names
        crops = list(cont.generate_spot_crops(adata, obs_names=obs))

        assert len(crops) == len(obs)
        for crop, o in zip(crops, obs):
            assert crop.data.attrs[Key.img.obs] == o

    @pytest.mark.parametrize("spot_scale", [1, 0.5, 2])
    @pytest.mark.parametrize("scale", [1, 0.5, 2])
    def test_spot_crops_spot_scale(self, adata: AnnData, cont: ImageContainer, scale: float, spot_scale: float):
        diameter = adata.uns["spatial"][Key.uns.library_id(adata, "spatial")]["scalefactors"]["spot_diameter_fullres"]
        radius = int(round(diameter // 2) * spot_scale)
        size = int((2 * radius + 1) * scale), int((2 * radius + 1) * scale)

        for crop in cont.generate_spot_crops(adata, spot_scale=spot_scale, scale=scale):
            assert crop.shape == size

    def test_spot_crops_with_scaled(self, adata: AnnData, cont: ImageContainer):
        # test generating spot crops with differently scaled images
        # crop locations should be the same when scaling spot crops or scaling cont beforehand
        gen1 = cont.generate_spot_crops(adata, scale=0.5)
        gen2 = cont.crop_corner(100, 100, cont.shape).generate_spot_crops(adata, scale=0.5)
        gen3 = cont.crop_corner(0, 0, cont.shape, scale=0.5).generate_spot_crops(adata)
        gen4 = cont.crop_corner(0, 0, cont.shape, scale=0.5).generate_spot_crops(adata, scale=0.5)

        # check that coords of generated crops are the same
        for c1, c2, c3, c4 in zip(gen1, gen2, gen3, gen4):
            # upscale c4
            c4 = c4.crop_corner(0, 0, c4.shape, scale=2)
            # need int here, because when generating spot crops from scaled images,
            # we need to center the spot crop on an actual pixel
            # this results in slighly different crop coords for the scaled cont
            assert int(c1.data.attrs["coords"].x0) == c3.data.attrs["coords"].x0
            assert int(c1.data.attrs["coords"].y0) == c3.data.attrs["coords"].y0
            assert c1.data.attrs["coords"].x0 == c2.data.attrs["coords"].x0
            assert c1.data.attrs["coords"].y0 == c2.data.attrs["coords"].y0
            assert c4.data.attrs["coords"].x0 == c3.data.attrs["coords"].x0
            assert c4.data.attrs["coords"].y0 == c3.data.attrs["coords"].y0

    def test_spot_crops_with_cropped(self, adata: AnnData, cont: ImageContainer):
        # crops should be the same when cropping from cropped cont or original cont
        # (as long as cropped cont contains all spots)
        cont_cropped = cont.crop_corner(100, 100, cont.shape)
        for c1, c2 in zip(cont.generate_spot_crops(adata), cont_cropped.generate_spot_crops(adata)):
            assert np.all(c1["image"].data == c2["image"].data)

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
            mask = (crop.data.x - c) ** 2 + (crop.data.y - c) ** 2 <= c**2

            np.testing.assert_array_equal(crop["image"].values[..., 0][~mask.values], np.nan)

    @pytest.mark.parametrize("diameter", [13, 17])
    def test_spot_crops_diameter(self, adata: AnnData, cont: ImageContainer, diameter: int):
        adata.uns[Key.uns.spatial] = {"bar": {"scalefactors": {"foo": diameter}}}
        for crop in cont.generate_spot_crops(adata, spot_diameter_key="foo"):
            assert crop.shape[0] == crop.shape[1]
            assert crop.shape[0] == diameter

    def test_uncrop_preserves_shape(self, small_cont_1c: ImageContainer):
        small_cont_1c.add_img(np.random.normal(size=(small_cont_1c.shape + (4,))), channel_dim="foobar", layer="baz")
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

    def test_chain_cropping(self, small_cont_seg: ImageContainer):
        # first crop
        c1 = small_cont_seg.crop_corner(10, 0, (60, 60))
        # test that have 1s and 2s in correct location
        assert np.all(c1["segmented"][10:20, 10:20] == 1)
        assert np.all(c1["segmented"][40:50, 30:40] == 2)
        # crop first crop
        c2 = c1.crop_corner(10, 10, (60, 60))
        assert np.all(c2["segmented"][:10, :10] == 1)
        assert np.all(c2["segmented"][30:40, 20:30] == 2)

        # uncrop c1 and c2 and check that are the same
        img1 = ImageContainer.uncrop([c1], small_cont_seg.shape)
        img2 = ImageContainer.uncrop([c2], small_cont_seg.shape)
        assert np.all(img1["segmented"].data == img2["segmented"].data)

    def test_chain_cropping_with_scale(self, small_cont_seg: ImageContainer):
        c1 = small_cont_seg.crop_corner(0, 0, (100, 100), scale=0.5)
        c2 = c1.crop_corner(10, 0, (50, 50), scale=2)
        img2 = ImageContainer.uncrop([c2], small_cont_seg.shape)
        # test that the points are in the right place after down + upscaling + cropping
        assert img2["segmented"][55, 35] == 2
        assert img2["segmented"][25, 15] == 1


class TestContainerUtils:
    def test_iter(self, small_cont_1c: ImageContainer):
        expected = list(small_cont_1c.data.keys())
        actual = list(small_cont_1c)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("deep", [False, True])
    def test_copy(self, deep: bool):
        cont = ImageContainer(np.random.normal(size=(10, 10)))
        sentinel = object()
        cont.data.attrs["sentinel"] = sentinel

        copy = cont.copy(deep=deep)

        if deep:
            assert not np.shares_memory(copy["image"].values, cont["image"].values)
            assert copy.data.attrs["sentinel"] is not sentinel
        else:
            assert np.shares_memory(copy["image"].values, cont["image"].values)
            assert copy.data.attrs["sentinel"] is sentinel

    def test_get_size(self):
        cont = ImageContainer(np.empty((10, 10)))

        ry, rx = cont._get_size(None)
        assert (ry, rx) == cont.shape

        ry, rx = cont._get_size((None, 1))
        assert (ry, rx) == (cont.shape[0], 1)

        ry, rx = cont._get_size((-1, None))
        assert (ry, rx) == (-1, cont.shape[1])

    @pytest.mark.parametrize("sx", [-1, -1.0, 0.5, 10])
    @pytest.mark.parametrize("sy", [-1, -1.0, 0.5, 10])
    def test_to_pixel_space(self, sy: Union[int, float], sx: Union[int, float]):
        cont = ImageContainer(np.empty((10, 10)))

        if (isinstance(sy, float) and sy < 0) or (isinstance(sx, float) and sx < 0):
            with pytest.raises(ValueError, match=r"Expected .* to be in interval `\[0, 1\]`.*"):
                cont._convert_to_pixel_space((sy, sx))
        else:
            ry, rx = cont._convert_to_pixel_space((sy, sx))
            if isinstance(sy, int):
                assert ry == sy
            else:
                assert ry == int(cont.shape[0] * sy)

            if isinstance(sx, int):
                assert rx == sx
            else:
                assert rx == int(cont.shape[1] * sx)

    @pytest.mark.parametrize("channel", [None, 0])
    @pytest.mark.parametrize("copy", [False, True])
    def test_apply(self, copy: bool, channel: Optional[int]):
        cont = ImageContainer(np.random.normal(size=(100, 100, 3)))
        orig = cont.copy()

        res = cont.apply(lambda arr: arr + 42, channel=channel, copy=copy)

        if copy:
            assert isinstance(res, ImageContainer)
            data = res["image"]
        else:
            assert res is None
            assert len(cont) == 1
            data = cont["image"]

        if channel is None:
            np.testing.assert_allclose(data.values, orig["image"].values + 42)
        else:
            np.testing.assert_allclose(data.values[..., 0], orig["image"].values[..., channel] + 42)

    @pytest.mark.parametrize("depth", [None, (30, 30, 0)])
    def test_apply_overlap(self, small_cont: ImageContainer, mocker: MockerFixture, depth: Optional[Tuple[int, ...]]):
        if depth is None:
            kwargs = {}
            spy = mocker.spy(da, "map_blocks")
        else:
            kwargs = {"depth": depth}
            spy = mocker.spy(da, "map_overlap")
        _ = small_cont.apply(lambda arr: arr + 1, chunks=15, **kwargs)

        spy.assert_called_once()

    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("chunks", [25, (50, 50, 1, 3), "auto"])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_apply_dask(
        self, small_cont: ImageContainer, copy: bool, chunks: Union[int, Tuple[int, ...], str], lazy: bool
    ):
        def func(chunk: np.ndarray) -> np.ndarray:
            if isinstance(chunks, tuple):
                np.testing.assert_array_equal(chunk.shape, chunks)
            elif isinstance(chunks, int):
                np.testing.assert_array_equal(chunk.shape, [chunks, chunks, 1, 3])
            return chunk

        cont = small_cont.apply(func, chunks=chunks, lazy=lazy, copy=copy, layer="image", new_layer="foo")
        if copy:
            assert isinstance(cont, ImageContainer)
            assert len(cont) == 1
        else:
            assert cont is None
            cont = small_cont
            assert len(cont) == 2

        if lazy:
            assert isinstance(cont["foo"].data, da.Array)
        else:
            assert isinstance(cont["foo"].data, np.ndarray)

    @pytest.mark.parametrize("as_dask", [False, True])
    def test_apply_passes_correct_array_type(self, as_dask: bool):
        def func(arr: Union[np.ndarray, da.Array]):
            if as_dask:
                assert isinstance(arr, da.Array)
            else:
                assert isinstance(arr, np.ndarray)
            assert arr.shape == (100, 100, 1, 3)
            return arr

        img = np.random.normal(size=(100, 100, 1, 3))
        cont = ImageContainer(da.from_array(img) if as_dask else img, dims=("y", "x", "z", "channels"))

        res = cont.apply(func, lazy=True, chunks=None, copy=True)
        if as_dask:
            assert isinstance(res["image"].data, da.Array)
        else:
            assert isinstance(res["image"].data, np.ndarray)
            assert not np.shares_memory(cont["image"].data, res["image"].data)

    def test_apply_wrong_number_of_dim(self):
        def func(arr: np.ndarray) -> float:
            assert arr.shape == (100, 100, 1, 3)
            assert arr.dtype == np.float64
            return np.sum(arr)

        cont = ImageContainer(
            np.random.normal(size=(100, 100, 1, 3)).astype(np.float64), dims=("y", "x", "z", "channels")
        )
        with pytest.raises(ValueError, match=r", found `0`."):
            cont.apply(func)

    @pytest.mark.parametrize("drop_unselected", [False, True])
    def test_apply_different_functions(self, cont_4d: ImageContainer, drop_unselected: bool):
        res = cont_4d.apply({"2": lambda arr: arr + 3, "1": lambda arr: arr + 1}, copy=True, drop=drop_unselected)

        assert len(res) == 1
        assert res["image"].shape == (*cont_4d.shape, 3 - drop_unselected, cont_4d["image"].shape[-1])
        if drop_unselected:
            # original are in order "0", "1", "2"
            assert res.library_ids == ["1", "2"]
        else:
            assert res.library_ids == cont_4d.library_ids

    def test_apply_modifies_channels(self, cont_4d: ImageContainer):
        with pytest.raises(ValueError, match="Unable to stack an array"):
            cont_4d.apply({"3": lambda arr: arr, "1": lambda arr: np.ones(arr.shape[:2] + (11,))}, copy=True, drop=True)

    def test_key_completions(self):
        cont = ImageContainer(np.random.normal(size=(100, 100, 3)))
        cont.add_img(np.random.normal(size=(100, 100, 3)), layer="alpha")

        np.testing.assert_array_equal(cont._ipython_key_completions_(), sorted(cont))

    def test_image_autoincrement(self, small_cont_1c: ImageContainer):
        assert len(small_cont_1c) == 1
        for _ in range(20):
            small_cont_1c.add_img(np.empty(small_cont_1c.shape))

        assert len(small_cont_1c) == 21
        for i in range(20):
            assert f"image_{i}" in small_cont_1c

    def test_rename(self, small_cont_1c: ImageContainer):
        new_cont = small_cont_1c.rename("image", "foo")

        assert new_cont is small_cont_1c
        assert len(new_cont) == len(small_cont_1c)
        assert "foo" in new_cont
        assert "image" not in new_cont

    @pytest.mark.parametrize("size", [0, 10, 20])
    def test_repr_html(self, size: int):
        cont = ImageContainer()
        for _ in range(size):
            cont.add_img(np.empty((10, 10)))

        validator = SimpleHTMLValidator(
            n_expected_rows=min(size, 10), expected_tags=set() if not size else {"p", "em", "strong"}
        )
        validator.feed(cont._repr_html_())
        validator.validate()

    def test_repr(self):
        cont = ImageContainer()

        assert "shape=(0, 0)" in repr(cont)
        assert "layers=[]" in repr(cont)
        assert repr(cont) == str(cont)


class TestZStacks:
    @pytest.mark.parametrize("library_ids", [None, ["1", "2", "3"]])
    @pytest.mark.parametrize("init_lid", [True, False])
    def test_concat(self, library_ids, init_lid):
        arrs = [np.zeros((10, 10)), np.zeros((10, 10)) + 1, np.zeros((10, 10)) + 2]
        imgs = [ImageContainer(arr, library_id=str(i) if init_lid else None) for i, arr in enumerate(arrs)]

        if not init_lid and library_ids is None:
            with pytest.raises(ValueError, match=r"Found non-unique library ids"):
                _ = ImageContainer.concat(imgs, library_ids=library_ids)
        else:
            img = ImageContainer.concat(imgs, library_ids=library_ids)
            if library_ids is None:
                library_ids = [img.data.coords["z"].values[0] for img in imgs]

            assert img["image"].shape == (10, 10, 3, 1)
            assert np.all(img.data.coords["z"] == library_ids)

    @pytest.mark.parametrize("library_id", [None, "1", "2", "3"])
    def test_crop_corner_library_ids(self, library_id):
        arrs = [np.zeros((10, 10)), np.zeros((10, 10)) + 1, np.zeros((10, 10)) + 2]
        img = ImageContainer.concat([ImageContainer(arr) for arr in arrs], library_ids=["1", "2", "3"])
        crop = img.crop_corner(0, 0, (10, 10), library_id=library_id)
        if library_id is None:
            assert crop["image"].shape == (10, 10, 3, 1)
            assert (crop["image"].values == img["image"].values).all()
        else:
            assert crop["image"].shape == (10, 10, 1, 1)
            assert (crop["image"].values == img["image"].sel(z=library_id).values).all()

    def test_generate_spot_crops(self):
        # TODO could probably divide this test in several smaller tests
        # build adata to crop from img
        crop_coords = np.array([[0, 0], [0, 4], [0, 8], [4, 0], [4, 4], [4, 8], [8, 0], [8, 4], [8, 8]])
        # for library_id 1
        adata1 = AnnData(
            np.zeros((len(crop_coords), 1)),
            uns={"spatial": {"1": {"scalefactors": {"spot_diameter_fullres": 5}}}},
            obsm={"spatial": crop_coords},
            dtype=float,
        )
        # for library_id 2 (with larger scalefactor)
        adata2 = AnnData(
            np.zeros((len(crop_coords), 1)),
            uns={"spatial": {"2": {"scalefactors": {"spot_diameter_fullres": 7}}}},
            obsm={"spatial": crop_coords},
            dtype=float,
        )
        # concatenate
        adata = ad.concat({"1": adata1, "2": adata2}, uns_merge="unique", label="library_id")
        adata.obs_names_make_unique()

        cont1 = ImageContainer(np.zeros((10, 10, 3)) + 1, library_id="1")
        cont2 = ImageContainer(np.zeros((10, 10, 3)) + 2, library_id="2")
        cont_comb = ImageContainer.concat([cont1, cont2])

        # test that crops from library_id 1 are as expected
        els = list(cont_comb.generate_spot_crops(adata[adata.obs["library_id"] == "1"], library_id="1"))
        for el in els:
            assert el.shape == (5, 5)
        res = ImageContainer.uncrop(els)
        assert (res.data["image"].sel(z="1").values == cont_comb.data["image"].sel(z="1").values).all()

        # test that crops from library_id 2 are as expected
        els = list(cont_comb.generate_spot_crops(adata[adata.obs["library_id"] == "2"], library_id="2"))
        for el in els:
            assert el.shape == (7, 7)
        res = ImageContainer.uncrop(els)
        assert (res.data["image"].sel(z="2").values == cont_comb.data["image"].sel(z="2").values).all()

        # test that cropping from multiple library_ids works
        els = list(cont_comb.generate_spot_crops(adata, library_id="library_id"))
        for i, el in enumerate(els):
            if i < 9:
                assert el.shape == (5, 5)
            else:
                assert el.shape == (7, 7)

        # test than can pass library_id as well - this results in cropping from another library id
        els = list(cont_comb.generate_spot_crops(adata, library_id="2"))
        for el in els:
            assert el.shape == (7, 7)
        res = ImageContainer.uncrop(els)
        assert (res.data["image"].sel(z="2").values == cont_comb.data["image"].sel(z="2").values).all()

    @pytest.mark.parametrize("channel", [None, 0])
    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("library_id", [["l1"], ["l2"], ["l1", "l2", "l3"], None])
    def test_apply(self, copy: bool, channel: Optional[int], library_id: Optional[Union[list[str], str]]):
        cont = ImageContainer(
            np.random.normal(size=(100, 100, 3, 2)), dims=("y", "x", "z", "channels"), library_id=["l1", "l2", "l3"]
        )
        orig = cont.copy()

        if library_id is None:
            library_ids = ["l1", "l2", "l3"]
            func = lambda arr: arr + 42  # noqa: E731
        else:
            library_ids = library_id
            func = {lid: lambda arr: arr + 42 for lid in library_ids}

        res = cont.apply(func, channel=channel, copy=copy, drop=False)

        if copy:
            assert isinstance(res, ImageContainer)
            data = res["image"]
        else:
            assert res is None
            assert len(cont) == 1
            data = cont["image"]

        if channel is None:
            for lid in ["l1", "l2", "l3"]:
                if lid in library_ids:
                    np.testing.assert_allclose(data.sel(z=lid).values, orig["image"].sel(z=lid).values + 42)
                else:
                    np.testing.assert_allclose(data.sel(z=lid).values, orig["image"].sel(z=lid).values)
        else:
            for lid in ["l1", "l2", "l3"]:
                if lid in library_ids:
                    np.testing.assert_allclose(
                        data.sel(z=lid).values[..., 0], orig["image"].sel(z=lid).values[..., channel] + 42
                    )
                else:
                    np.testing.assert_allclose(
                        data.sel(z=lid).values[..., 0], orig["image"].sel(z=lid).values[..., channel]
                    )


class TestExplicitDims:
    @pytest.mark.parametrize("dims", list(permutations(["y", "x", "z", "c"])))
    def test_explicit_dims(self, dims: Tuple[str, str, str, str]):
        shape = (2, 3, 4, 5)
        img = ImageContainer(np.random.normal(size=shape), dims=dims)

        for d, s in zip(dims, shape):
            assert img.data.dims[d] == s

    @pytest.mark.parametrize("missing", ["y", "x", "z"])
    @pytest.mark.parametrize("ndim", [2, 3, 4])
    def test_required_dim_missing(self, missing: str, ndim: int):
        shape = (2, 3)
        if ndim >= 3:
            shape += (4,)
        if ndim >= 4:
            shape += (5,)
        dims = (
            "a" if missing == "y" else "y",
            "b" if missing == "x" else "x",
            "c" if missing == "z" else "z",
            "channels",
        )
        dims = dims[:ndim]

        if ndim in (2, 3) and missing == "z":
            img = ImageContainer(np.random.normal(size=shape), dims=dims)
            for d, s in zip(dims, shape):
                assert img.data.dims[d] == s
            assert img.data.dims["z"] == 1
        else:
            with pytest.raises(ValueError, match=rf"Expected to find `\[{missing!r}\]` dimension\(s\)"):
                _ = ImageContainer(np.random.normal(size=shape), dims=dims)

    @pytest.mark.parametrize("dims", ["z_last", "channels_last", ("x", "y")])
    def test_2D_array(self, dims: str):
        shape = (2, 3)
        img = ImageContainer(np.random.normal(size=shape), dims=dims)

        if isinstance(dims, str):
            dims = ("channels", "z", "y", "x")
            shape = (1, 1) + shape
        for d, s in zip(dims, shape):
            assert img.data.dims[d] == s

    @pytest.mark.parametrize("dims", ["z_last", "channels_last", ("x", "y", "z"), ("y", "x", "c")])
    def test_3D_array(self, dims: str):
        shape = (2, 3, 4)
        img = ImageContainer(np.random.normal(size=shape), dims=dims)

        if isinstance(dims, str):
            dims = (("channels", "z") if dims == "z_last" else ("z", "channels")) + ("y", "x")
            shape = (1,) + shape
        for d, s in zip(dims, shape):
            assert img.data.dims[d] == s

    @pytest.mark.parametrize("dims", ["z_last", "channels_last", ("z", "y", "x", "c")])
    def test_4D_array(self, dims: str):
        shape = (2, 3, 4, 5)
        img = ImageContainer(np.random.normal(size=shape), dims=dims)

        if isinstance(dims, str):
            dims = (("channels", "z") if dims == "z_last" else ("z", "channels")) + ("y", "x")
        for d, s in zip(dims, shape):
            assert img.data.dims[d] == s


class TestLibraryIds:
    def test_empty_container(self):
        img = ImageContainer()
        np.testing.assert_array_equal(img.library_ids, [])

    def test_default(self):
        img = ImageContainer(np.random.normal(size=(2, 3, 4, 5)), dims=["z", "y", "x", "c"])
        np.testing.assert_array_equal(img.library_ids, ["0", "1"])

    @pytest.mark.parametrize("z", [1, 2])
    def test_explicit(self, z: int):
        library_id = ["foo", "bar"]
        if z == 1:
            library_id = library_id[0]

        img = ImageContainer(np.random.normal(size=(z, 3, 4, 5)), dims=["z", "y", "x", "c"], library_id=library_id)
        np.testing.assert_array_equal(img.library_ids, np.ravel([library_id]))

    @pytest.mark.parametrize("unique", [False, True])
    @pytest.mark.parametrize("typ", [list, dict])
    def test_set(self, typ: type, unique: bool):
        if unique:
            new_ids = ["0", "1"] if typ == list else {"a": "0", "b": "1"}
        else:
            new_ids = ["0", "0"] if typ == list else {"a": "b"}
        img = ImageContainer(np.random.normal(size=(2, 3, 4, 5)), dims=["z", "y", "x", "c"], library_id=["a", "b"])

        if not unique:
            with pytest.raises(ValueError, match=r"Remapped library ids"):
                img.library_ids = new_ids
        else:
            img.library_ids = new_ids
            np.testing.assert_array_equal(
                img.library_ids, new_ids if isinstance(new_ids, list) else [new_ids[old] for old in ("a", "b")]
            )

    def test_get_from_xarray(self):
        library_id = ["a", "b"]
        tmp = ImageContainer(np.random.normal(size=(2, 3, 4, 5)), dims=["z", "y", "x", "c"], library_id=library_id)
        img = ImageContainer(tmp["image"])

        np.testing.assert_array_equal(tmp.library_ids, library_id)
        np.testing.assert_array_equal(img.library_ids, library_id)

    @pytest.mark.parametrize("library_id", ["a", ["b", "a"], ["c"]])
    @pytest.mark.parametrize("empty", [False, True])
    def test_get_library_ids(self, empty: bool, library_id: Union[str, Sequence[str]]):
        img = ImageContainer()
        if not empty:
            img.add_img(np.random.normal(size=(2, 3, 4, 5)), dims=["z", "y", "x", "c"], library_id=["a", "b"])

        if empty:
            np.testing.assert_array_equal(np.ravel([library_id]), img._get_library_ids(library_id))
        else:
            if "c" in library_id:
                with pytest.raises(ValueError, match=r"Invalid library ids have been selected"):
                    _ = img._get_library_ids(library_id, allow_new=False)
            else:
                np.testing.assert_array_equal(np.ravel([library_id]), img._get_library_ids(library_id))


class TestCroppingExtra:
    def test_big_crop(self, cont_dot: ImageContainer):
        crop = cont_dot.crop_center(
            y=50,
            x=20,
            radius=150,
            cval=5,
        )

        np.testing.assert_array_equal(crop.data["image_0"].shape, (301, 301, 1, 10))
        # check that values outside of img are padded with 5
        np.testing.assert_array_equal(crop.data["image_0"][0, 0, 0, 0], 5)
        np.testing.assert_array_equal(crop.data["image_0"][-1, -1, 0, 0], 5)
        assert crop.data["image_0"].dtype == np.uint8

        # compare with crop_corner
        crop2 = cont_dot.crop_corner(y=-100, x=-130, size=301, cval=5)
        np.testing.assert_array_equal(crop2.data["image_0"], crop.data["image_0"])

    def test_crop_smapp(self, cont_dot: ImageContainer):
        crop = cont_dot.crop_center(
            x=50,
            y=20,
            radius=0,
            cval=5,
        )

        np.testing.assert_array_equal(crop.data["image_0"].shape, (1, 1, 1, 10))
        np.testing.assert_array_equal(crop.data["image_0"][0, 0, 0, :3], [10, 11, 12])
        assert crop.data["image_0"].dtype == np.uint8

    @pytest.mark.parametrize("s", [1, 2])
    def test_crop_mask_circle(self, cont_dot: ImageContainer, s: int):
        # crop with mask_circle
        crop = cont_dot.crop_center(
            y=20,
            x=50,
            radius=5,
            cval=5,
            mask_circle=True,
            scale=s,
        )

        np.testing.assert_array_equal(crop.data["image_0"][1 * s, 0 * s, 0, :], 5)
        np.testing.assert_array_equal(crop.data["image_0"][2 * s, 2 * s, 0, :], 0)
        np.testing.assert_array_equal(crop.data["image_0"][7 * s, 7 * s, 0, :], 0)
        np.testing.assert_array_equal(crop.data["image_0"][9 * s, 9 * s, 0, :], 5 if s == 1 else 0)
        np.testing.assert_array_equal(np.array([11, 11]) * s, crop.shape)
        assert crop.data.coords == cont_dot.data.coords

    def test_crop_multiple_images(self, cont_dot: ImageContainer):
        mask = np.random.randint(low=0, high=10, size=cont_dot.shape)
        cont_dot.add_img(mask, layer="image_1", channel_dim="mask")

        crop = cont_dot.crop_center(
            y=50,
            x=20,
            radius=0,
            cval=5,
        )

        assert "image_0" in crop
        assert "image_1" in crop
        np.testing.assert_array_equal(crop.data["image_0"].shape, (1, 1, 1, 10))
        np.testing.assert_array_equal(crop.data["image_1"].shape, (1, 1, 1, 1))

    def test_crop_scale(self, cont_dot: ImageContainer):
        # crop with scaling
        mask = np.random.randint(low=0, high=10, size=cont_dot.shape)
        cont_dot.add_img(mask, layer="image_1", channel_dim="mask")

        crop = cont_dot.crop_center(y=50, x=20, radius=10, cval=5, scale=0.5)

        assert "image_0" in crop
        assert "image_1" in crop
        np.testing.assert_array_equal(crop.data["image_0"].shape, (21 // 2, 21 // 2, 1, 10))
        np.testing.assert_array_equal(crop.data["image_1"].shape, (21 // 2, 21 // 2, 1, 1))


class TestPileLine:
    @pytest.mark.parametrize("lazy", [False, True])
    def test_pipeline_inplace(self, small_cont: ImageContainer, lazy: bool):
        chunks = 25 if lazy else None
        c1 = sq.im.process(small_cont, method="smooth", copy=False, layer_added="foo", chunks=chunks, lazy=lazy)
        c2 = sq.im.process(
            small_cont, method="gray", copy=False, layer="foo", layer_added="bar", chunks=chunks, lazy=lazy
        )
        c3 = sq.im.segment(
            small_cont,
            method="watershed",
            copy=False,
            layer="bar",
            thresh=0.3,
            layer_added="baz",
            chunks=chunks,
            lazy=lazy,
        )

        assert c1 is None
        assert c2 is None
        assert c3 is None
        np.testing.assert_array_equal(sorted(small_cont), sorted(["image", "foo", "bar", "baz"]))
        for key in small_cont:
            if key != "image":
                if lazy:
                    assert isinstance(small_cont[key].data, da.Array)
                else:
                    assert isinstance(small_cont[key].data, np.ndarray)

        tmp = small_cont.compute()
        assert tmp is small_cont

        for key in small_cont:
            assert isinstance(small_cont[key].data, np.ndarray)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_pipeline_copy(self, small_cont: ImageContainer, lazy: bool):
        chunks = 13 if lazy else None

        c1 = sq.im.process(small_cont, method="smooth", copy=True, layer_added="foo", chunks=chunks, lazy=lazy)
        c2 = sq.im.process(c1, method="gray", copy=True, layer="foo", layer_added="bar", chunks=chunks, lazy=lazy)
        c3 = sq.im.segment(
            c2,
            method="watershed",
            copy=True,
            layer="bar",
            thresh=0.3,
            layer_added="baz",
            chunks=chunks,
            lazy=lazy,
        )
        assert len(small_cont) == 1
        assert len(c1) == 1
        assert len(c2) == 1

        for key, cont in zip(["foo", "bar", "baz"], [c1, c2, c3]):
            if lazy:
                assert isinstance(cont[key].data, da.Array)
            else:
                assert isinstance(cont[key].data, np.ndarray)

        for key, cont in zip(["foo", "bar", "baz"], [c1, c2, c3]):
            cont.compute()
            assert isinstance(cont[key].data, np.ndarray)

    @pytest.mark.parametrize("dim_name", ["channels", "z"])
    def test_loading_bwd_compat_no_zdim(self, dim_name: str, tmpdir):
        ds = xr.Dataset({"foo": xr.DataArray(np.random.normal(size=(64, 64, 3)), dims=("x", "y", dim_name))})
        ds.to_zarr(Path(tmpdir))

        if dim_name == "z":
            with pytest.raises(ValueError, match=r".*z.*exists"):
                _ = ImageContainer.load(str(tmpdir))
        else:
            img = ImageContainer.load(str(tmpdir))
            assert img.data.dims == {"x": 64, "y": 64, "z": 1, dim_name: 3}
