from os import PathLike
from typing import Any, List, Tuple, Union, Iterable, Iterator, Optional
from pathlib import Path

from scanpy import logging as logg
from anndata import AnnData

import numpy as np
import xarray as xr

import skimage
from imageio import imread
from skimage.transform import rescale

from squidpy._docs import d
from squidpy.im._utils import _num_pages, _unique_order_preserving
from squidpy.constants._pkg_constants import Key

Pathlike_t = Union[str, Path]


# helper function for scaling one xarray
def _scale_xarray(arr, scale):
    dtype = arr.dtype
    dims = arr.dims

    # rescale only in x and y dim
    scales = np.ones(len(dims))
    scales[np.in1d(dims, ["y", "x"])] = scale

    arr = rescale(arr.values, scales, preserve_range=True, order=1)
    arr = arr.astype(dtype)
    # recreate DataArray
    arr = xr.DataArray(arr, dims=dims)

    return arr


@d.dedent  # trick to overcome not top-down order
@d.dedent
class ImageContainer:
    """
    Container for in memory `np.ndarray`/`xr.DataArray` or on-disk tiff/jpg images.

    Wraps :class:`xarray.Dataset` to store several image layers with the same x and y dimensions in one object.
    Dimensions of stored images are `(y, x, <channel_dimension>, ...)`.
    The channel dimension may vary between image layers.

    Allows for lazy and chunked reading via :mod:`rasterio` and :mod:`dask` if the input is a tiff image).
    An instance of this class is given to all image processing functions, along with an :mod:`anndata` instance,
    if necessary.

    Parameters
    ----------
    %(add_img.parameters)s

    Raises
    ------
    %(add_img.raises)s
    """

    data: xr.Dataset

    def __init__(
        self,
        img: Optional[Union[Pathlike_t, np.ndarray]] = None,
        img_id: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        self.data = xr.Dataset()
        chunks = kwargs.pop("chunks", None)
        if img is not None:
            if chunks is not None:
                chunks = {"x": chunks, "y": chunks}
            self.add_img(img, img_id, chunks=chunks, **kwargs)

    def __repr__(self):
        s = f"ImageContainer object with {len(self.data.keys())} layers\n"
        for layer in self.data.keys():
            s += f"    {layer}: "
            s += ", ".join(f"{dim} ({shape})" for dim, shape in zip(self.data[layer].dims, self.data[layer].shape))
            s += "\n"
        return s

    def __getitem__(self, key):
        return self.data[key]

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape (y, x)."""  # noqa: D402
        return self.data.dims["y"], self.data.dims["x"]  # TODO changed shape, need to catch all calls to img.shape

    @property
    def nchannels(self) -> int:
        """Number of channels."""  # noqa: D401
        # TODO this might fail, if we name channels sth else than "channels"
        return self.data.dims["channels"]

    @classmethod
    def open(cls, fname: Pathlike_t, lazy: bool = True, chunks: Optional[int] = None) -> "ImageContainer":
        """
        Initialize using a previously saved netcdf file.

        Parameters
        ----------
        fname
            Path to the saved .nc file.
        lazy
            Use :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`.
        """
        self = cls()
        self.data = xr.open_dataset(fname, chunks=chunks)
        if not lazy:
            self.data.load()
        return self

        def __repr__(self):
            s = f"ImageContainer object with {len(self.data.keys())} layers\n"
            for layer in self.data.keys():
                s += f"    {layer}: "
                s += ", ".join(f"{dim} ({shape})" for dim, shape in zip(self.data[layer].dims, self.data[layer].shape))
                s += "\n"
            return s

    def save(self, fname: Pathlike_t) -> None:
        """
        Save dataset as netcdf file.

        Parameters
        ----------
        fname
            Path to the saved .nc file.

        Returns
        -------
        None
            Nothing, just saves the data.
        """
        self.data.to_netcdf(fname, mode="a")

    @d.get_sections(base="add_img", sections=["Parameters", "Raises"])
    def add_img(
        self,
        img: Union[Pathlike_t, np.ndarray, xr.DataArray],
        img_id: Optional[str] = None,
        channel_id: str = "channels",
        lazy: bool = True,
        chunks: Optional[int] = None,
    ) -> None:
        """
        Add layer from in memory `np.ndarray`/`xr.DataArray` or on-disk tiff/jpg image.

        For :mod:`numpy` arrays, assume that dims are: ``(y, x, channel_id)``.

        NOTE: lazy loading via :mod:`dask` is not supported for on-disk jpg files.
        They will be loaded in memory.

        Parameters
        ----------
        img
            :mod:`numpy` array or path to image file.
        img_id
            Key (name) to be used for img.
            If not specified, DataArrays will be named "image".
        channel_id
            Name of the channel dimension. Default is "channels".
        lazy
            Use :mod:`rasterio` or :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`, used in call to `xr.open_rasterio` for tiff images.

        Returns
        -------
        None
            Nothing, just adds img to `.data`

        Raises
        ------
        :class:`ValueError`
            If ``img`` is a :class:`np.ndarray` and has more than 3 dimensions.
        """
        img = self._load_img(img=img, channel_id=channel_id, chunks=chunks)
        if img_id is None:
            img_id = "image"
        # add to data
        logg.info(f"Adding `{img_id}` into object")
        self.data[img_id] = img
        if lazy:
            # load in memory
            self.data.load()

    def _load_img(
        self, img: Union[Pathlike_t, np.ndarray], channel_id: str = "channels", chunks: Optional[int] = None
    ) -> xr.DataArray:
        """
        Load image as :mod:`xarray`.

        See :meth:`add_img` for more details.
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) > 3:
                raise ValueError(f"Img has more than 3 dimensions. img.shape is {img.shape}.")
            dims = ["y", "x", channel_id]
            if len(img.shape) == 2:
                # add channel dimension
                img = img[:, :, np.newaxis]
            xr_img = xr.DataArray(img, dims=dims)
        elif isinstance(img, xr.DataArray):
            assert "x" in img.dims
            assert "y" in img.dims
            xr_img = img
        elif isinstance(img, (str, PathLike)):
            img = str(img)
            ext = img.split(".")[-1]
            if ext in ("tif", "tiff"):  # TODO: constants
                # get the number of pages in the file
                num_pages = _num_pages(img)
                # read all pages using rasterio
                xr_img_byband = []
                for i in range(1, num_pages + 1):
                    data = xr.open_rasterio(f"GTIFF_DIR:{i}:{img}", chunks=chunks, parse_coordinates=False)
                    data = data.rename({"band": channel_id})
                    xr_img_byband.append(data)
                xr_img = xr.concat(xr_img_byband, dim=channel_id)
                print(xr_img.dims)
                xr_img = xr_img.transpose("y", "x", ...)
                # TODO transpose xr_img to have channels as last dim
            elif ext in ("jpg", "jpeg"):  # TODO: constants
                img = imread(img)
                dims = ["y", "x", channel_id]
                xr_img = xr.DataArray(img, dims=dims)
            else:
                raise NotImplementedError(f"Files with extension `{ext}`.")
        else:
            raise ValueError(img)
        return xr_img

    # TODO fix docstrings!
    @d.get_sections(base="crop_corner", sections=["Parameters", "Returns"])
    def crop_corner(
        self,
        x: int,
        y: int,
        xs: int,
        ys: int,
        scale: float = 1.0,
        mask_circle: bool = False,
        cval: float = 0.0,
        dtype: Optional[str] = None,
    ) -> xr.DataArray:
        """
        Extract a crop from upper left corner coordinates `x` and `y`.

        The crop will be extracted right and down from x, y.

        Parameters
        ----------
        x
            X coord of crop (in pixel space).
        y
            Y coord of crop (in pixel space).
        xs
            Width of the crops in pixels.
        ys
            Height of the crops in pixels.
        scale
            Resolution of the crop (smaller -> smaller image).
            Rescaling is done using `skimage.transform.rescale`.
        mask_circle
            Mask crop to a circle.
        cval
            The value outside image boundaries or the mask.
        dtype
            Type to which the output should be (safely) cast. If `None`, don't recast.
            Currently supported dtypes: 'uint8'. TODO: pass actualy types instead of strings.

        Returns
        -------
        :class:`ImageContainer`
            cropped ImageContainer
        """
        # get conversion function
        if dtype is not None:
            if dtype == "uint8":
                convert = skimage.util.img_as_ubyte
            else:
                raise NotImplementedError(dtype)

        # TODO: rewrite assertions to "normal" errors so they can be more easily tested against
        assert y < self.data.y.shape[0], f"y ({y}) is outsize of image range ({self.data.y.shape[0]})"
        assert x < self.data.x.shape[0], f"x ({x}) is outsize of image range ({self.data.x.shape[0]})"

        assert xs > 0, "im size cannot be 0"
        assert ys > 0, "im size cannot be 0"

        # get crop coords
        x0 = x
        x1 = x + xs
        y0 = y
        y1 = y + ys

        # make coords fit the image
        crop_x0 = max(x0, 0)
        crop_y0 = max(y0, 0)
        crop_x1 = min(self.data.x.shape[0], x1)
        crop_y1 = min(self.data.y.shape[0], y1)

        # create cropped xr.Dataset
        crop = self.data.isel(x=slice(crop_x0, crop_x1), y=slice(crop_y0, crop_y1))

        # pad crop if necessary
        if [x0, x1, y0, y1] != [crop_x0, crop_x1, crop_y0, crop_y1]:
            print("padding crop")
            crop = crop.pad(
                x=(abs(x0 - crop_x0), abs(x1 - crop_x1)),
                y=(abs(y0 - crop_y0), abs(y1 - crop_y1)),
                mode="constant",
                constant_values=cval,
            )

        # scale crop
        crop = crop.map(_scale_xarray, scale=scale)

        # mask crop
        if mask_circle:
            assert xs == ys, "Crop has to be square to use mask_circle."
            c = crop.x.shape[0] // 2
            crop = crop.where((crop.x - c) ** 2 + (crop.y - c) ** 2 <= c ** 2, other=cval)

        # convert to dtype
        if dtype is not None:
            crop = crop.map(convert)

        # return crop as ImageContainer
        crop_cont = ImageContainer()
        crop_cont.data = crop
        return crop_cont

    @d.dedent
    def crop_center(
        self,
        x: int,
        y: int,
        xr: int = 100,
        yr: int = 100,
        scale: float = 1.0,
        mask_circle: bool = False,
        cval: float = 0.0,
        dtype: Optional[str] = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Extract a crop based on coordinates `x` and `y`.

        The extracted crop will be centered on x, y, and have shape `yr*2+1, xr*2+1`.

        Parameters
        ----------
        %(crop_corner.parameters)s
        TODO remove xs and xs from docstring
        xr
            Radius of the crop in pixels.
        yr
            Radius of the crop in pixels.

        Returns
        -------
        %(crop_corner.returns)s
        """
        # move from center to corner
        x = x - xr
        y = y - yr

        # calculate size
        xs = xr * 2 + 1
        ys = yr * 2 + 1

        return self.crop_corner(x=x, y=y, xs=xs, ys=ys, **kwargs)

    @d.dedent
    def crop_equally(
        self,
        xs: Optional[int] = None,
        ys: Optional[int] = None,
        img_id: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Tuple[List[xr.DataArray], np.ndarray, np.ndarray]:
        """
        Decompose an image into equally sized crops.

        Parameters
        ----------
        %(width_height)s
        kwargs
            Keyword arguments for :meth:`squidpy.im.ImageContainer.crop_corner`.

        Returns
        -------
        :class:`tuple`
            Triple of the following:

                - crops of shape ``(channels, y, x)``.
                - x-positions of the crops.
                - y-positions of the crops.
        """
        if xs is None:
            xs = self.shape[0]
        if ys is None:
            ys = self.shape[1]

        unique_xcoord = np.arange(start=0, stop=(self.data.dims["x"] // xs) * xs, step=xs)
        unique_ycoord = np.arange(start=0, stop=(self.data.dims["y"] // ys) * ys, step=ys)

        xcoords = np.repeat(unique_xcoord, len(unique_ycoord))
        ycoords = np.tile(unique_ycoord, len(unique_xcoord))

        crops = [self.crop_corner(x=x, y=y, xs=xs, ys=ys, img_id=img_id, **kwargs) for x, y in zip(xcoords, ycoords)]
        return crops, xcoords, ycoords

    @d.dedent
    def crop_spot_generator(
        self,
        adata: AnnData,
        dataset_name: Optional[str] = None,
        size: float = 1.0,
        obs_ids: Optional[Iterable[Any]] = None,
        **kwargs,
    ) -> Iterator[Tuple[Union[int, str], xr.DataArray]]:
        """
        Iterate over all obs_ids defined in adata and extract crops from images.

        Implemented for 10x spatial datasets.

        Parameters
        ----------
        %(adata)s
        dataset_name
            Name of the spatial data in adata (if not specified, take first one).
        size
            Amount of context (1.0 means size of spot, larger -> more context).
        obs_ids
            Observations from :attr:`adata.obs_names` for which to generate the crops.
        kwargs
            Keyword arguments for :meth:`crop_center`.

        Yields
        ------
        :class:`tuple`
            Tuple of the following:

                - obs_id of spot from ``adata``.
                - crop of shape ``(channels, y, x)``.
        """
        if dataset_name is None:
            dataset_name = list(adata.uns[Key.uns.spatial].keys())[0]

        # TODO: expose key?
        xcoord = adata.obsm[Key.obsm.spatial][:, 0]
        ycoord = adata.obsm[Key.obsm.spatial][:, 1]
        spot_diameter = adata.uns[Key.uns.spatial][dataset_name]["scalefactors"]["spot_diameter_fullres"]
        r = int(round(spot_diameter * size // 2))
        # TODO: could also use round_odd and add 0.5 for xcoord and ycoord

        obs_ids, seen = _unique_order_preserving(adata.obs.index if obs_ids is None else obs_ids)
        indices = [i for i, obs in enumerate(adata.obs.index) if obs in seen]

        for i, obs_id in zip(indices, obs_ids):
            crop = self.crop_center(x=xcoord[i], y=ycoord[i], xr=r, yr=r, **kwargs)
            yield obs_id, crop
