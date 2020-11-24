import os
from typing import List, Tuple, Union, Iterator, Optional

from scanpy import logging as logg
from anndata import AnnData

import numpy as np
import xarray as xr

from imageio import imread

from ._utils import _num_pages
from ..constants._pkg_constants import SPATIAL_M, SPATIAL_U


class ImageContainer:
    """
    Container for in memory or on-disk tiff or jpg images.

    Allows for lazy and chunked reading via :mod:`rasterio` and :mod:`dask` (if input is a tiff image).
    An instance of this class is given to all image processing functions, along with an :mod:`anndata` instance,
    if necessary.
    """

    # the Attributes and Methods were not necessary, will be autogenerated

    data: xr.Dataset

    def __init__(
        self,
        img: Optional[Union[str, os.PathLike, np.ndarray]],
        img_id: Optional[Union[str, List[str]]] = None,
        lazy: bool = True,
        chunks: Optional[int] = None,
    ):
        """
        Set up ImageContainer from numpy array or on-disk tiff / jpg.

        Processes image as in memory numpy array or uses xarrays rasterio reading functions to load from disk
        (with caching) if image is a file path.
        If chunks are specified, the xarray is wrapped in a dask lazy dask array using the chunk size.

        Parameters
        ----------
        img
            An array or a path to tiff file.
        img_id
            Key (name) to be used for img. For multi-page tiffs this should be a list.
            If not specified, DataArrays will be named 'image_{i}'.
        lazy
            Use :mod:`rasterio` or :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`.
        """
        if chunks is not None:
            chunks = {"x": chunks, "y": chunks}
        self._chunks = chunks
        self._lazy = lazy
        self.data = xr.Dataset()
        if img is not None:
            self.add_img(img, img_id)

    def __repr__(self):
        s = f"ImageContainer object with {len(self.data.keys())} layers\n"
        for layer in self.data.keys():
            s += f"    {layer}: "
            s += ", ".join(f"{dim} ({shape})" for dim, shape in zip(self.data[layer].dims, self.data[layer].shape))
            s += "\n"
        return s

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape."""
        return self.data.dims["x"], self.data.dims["y"]

    @property
    def nchannels(self) -> int:
        """Number of channels."""  # noqa: D401
        return self.data.dims["channels"]

    @classmethod
    def open(cls, fname: str, lazy: bool = True, chunks: Optional[int] = None) -> "ImageContainer":
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
        self = cls(img=None, lazy=lazy, chunks=chunks)
        self.data = xr.open_dataset(fname, chunks=self._chunks)
        if not self._lazy:
            self.data.load()
        return self

    def save(self, fname: str) -> None:
        """
        Save dataset as netcdf file.

        Parameters
        ----------
        fname
            Path to the saved .nc file.

        Returns
        -------
        None
            TODO.
        """
        self.data.to_netcdf(fname, mode="a")

    def add_img(
        self,
        img: Union[str, os.PathLike, np.ndarray, xr.DataArray],
        img_id: Union[str, List[str]] = None,
        channel_id: str = "channels",
    ) -> None:
        """
        Add layer from numpy image / tiff file.

        For numpy arrays, assume that dims are: channels, y, x
        The added image has to have the same number of channels as the original image, or no channels.

        Parameters
        ----------
        img
            Numpy array or path to image file.
        img_id
            Key (name) to be used for img. For multi-page tiffs this should be a list.
            If not specified, DataArrays will be named "image".

        Returns
        -------
        None
            TODO.

        Raises
        ------
        :class:`ValueError`
            If ``img_id`` is neither a string nor a list.
        """
        img = self._load_img(img=img, channel_id=channel_id)
        if img_id is None:
            img_id = "image"
        # add to data
        logg.info("adding %s into object" % img_id)
        self.data[img_id] = img
        if not self._lazy:
            # load in memory
            self.data.load()

    def _load_img(self, img: Union[str, os.PathLike, np.ndarray], channel_id: str = "channels") -> xr.DataArray:
        """
        Load img as :mod:`xarray`.

        Supports numpy arrays and (multi-page) tiff files, and jpg files
        For :mod:`numpy` arrays, assume that dims are: `'channels, y, x'`

        NOTE: lazy loading via :mod:`dask` is currently not supported for on-disk jpg files.
        They will be loaded in memory.

        Parameters
        ----------
        img
            :mod:`numpy` array or path to image file.

        Returns
        -------
        DataArray containing loaded image.

        Raises
        ------
        ValueError:
            if img is a :class:`np.ndarray` and has more than 3 dimensions.
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) > 3:
                raise ValueError(f"img has more than 3 dimensions. img.shape is {img.shape}")
            dims = [channel_id, "y", "x"]
            if len(img.shape) == 2:
                dims = ["y", "x"]
            xr_img = xr.DataArray(img, dims=dims)
        elif isinstance(img, xr.DataArray):
            assert "x" in img.dims
            assert "y" in img.dims
            xr_img = img
        elif isinstance(img, (str, os.PathLike)):
            ext = os.path.splitext(img)[-1]
            if ext in (".tif", ".tiff"):
                # get the number of pages in the file
                num_pages = _num_pages(img)
                # read all pages using rasterio
                xr_img_byband = []
                for i in range(1, num_pages + 1):
                    data = xr.open_rasterio(f"GTIFF_DIR:{i}:{img}", chunks=self._chunks, parse_coordinates=False)
                    data = data.rename({"band": channel_id})
                    xr_img_byband.append(data)
                xr_img = xr.concat(xr_img_byband, dim=channel_id)
            elif ext in (".jpg", ".jpeg"):
                img = imread(img)
                # jpeg has channels as last dim - transpose
                img = img.transpose(2, 0, 1)
                dims = [channel_id, "y", "x"]
                xr_img = xr.DataArray(img, dims=dims)
            else:
                raise NotImplementedError(f"Files with extension {ext}")
        else:
            raise ValueError(img)
        return xr_img

    def crop_corner(
        self,
        x: int,
        y: int,
        xs: int = 100,
        ys: int = 100,
        img_id: Optional[str] = None,
        centred: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        """
        Extract a crop from upper left corner coordinates `x` and `y` of `img_id`.

        The crop will be extracted right and down from x, y.

        Parameters
        ----------
        x
            X coord of crop (in pixel space).
        y
            Y coord of crop (in pixel space).
        xs
            Width of the crop in pixels.
        ys
            Height of the crop in pixels.
        img_id
            id of the image layer to be cropped.
        scale
            Default is 1.0.
            Resolution of the crop (smaller -> smaller image).
        mask_circle
            Default is False.
            Mask crop to a circle.
        cval
            Default is 0
            The value outside image boundaries or the mask.
        dtype
            Optional, type to which the output should be (safely) cast.
            Currently supported dtypes: 'uint8'.

        Returns
        -------
        :class:`xarray.DataArray`
            Data with dimensions: channels, y, x.
        """
        from .crop import crop_img

        if img_id is None:
            img_id = list(self.data.keys())[0]

        img = self.data.data_vars[img_id]
        return crop_img(img=img, x=x, y=y, xs=xs, ys=ys, **kwargs)

    def crop_center(
        self,
        x: int,
        y: int,
        xr: int = 100,
        yr: int = 100,
        img_id: Optional[str] = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Extract a crop based on coordinates `x` and `y` of `img_id`.

        The extracted crop will be centered on x, y, and have shape `yr*2+1, xr*2+1`.

        Parameters
        ----------
        x
            X coord of crop (in pixel space). Can be float (ie. int+0.5) if model is centered and if x+xs/2 is integer.
        y
            Y coord of crop (in pixel space). Can be float (ie. int+0.5) if model is centered and if y+ys/2 is integer.
        xr
            Radius of the crop in pixels.
        yr
            Height of the crop in pixels.
        img_id
            id of the image layer to be cropped.
        scale
            Default is 1.0.
            Resolution of the crop (smaller -> smaller image).
        mask_circle
            Default is False.
            Mask crop to a circle.
        cval
            Default is 0
            The value outside image boundaries or the mask.
        dtype
            Optional, type to which the output should be (safely) cast.
            Currently supported dtypes: 'uint8'.

        Returns
        -------
        :class:`xarray.DataArray`
            Data with dimensions: channels, y, x.
        """
        # move from center to corner
        x = x - xr
        y = y - yr

        # calculate size
        xs = xr * 2 + 1
        ys = yr * 2 + 1

        return self.crop_corner(x=x, y=y, xs=xs, ys=ys, img_id=img_id, **kwargs)

    def crop_equally(
        self,
        xs: Optional[int] = None,
        ys: Optional[int] = None,
        img_id: Optional[Union[str, List[str]]] = None,
        **_kwargs,
    ) -> Tuple[List[xr.DataArray], np.ndarray, np.ndarray]:
        """
        Decompose image into equally sized crops.

        Parameters
        ----------
        xs
            Width of the crops in pixels. Defaults to image size if None.
        ys
            Height of the crops in pixels. Defaults to image size if None.
        cval
            Default is 0
            The value outside image boundaries or the mask.
        dtype
            Optional, type to which the output should be (safely) cast.
            Currently supported dtypes: 'uint8'.

        Returns
        -------
        :class:`tuple`

            - List[xr.DataArray] with dimensions: channels, y, x: crops.
            - :class:`np.ndarray`: length number of crops: x positions of crops.
            - :class:`np.ndarray`: length number of crops: y positions of crops.
        """
        if xs is None:
            xs = self.shape[0]
        if ys is None:
            ys = self.shape[1]
        unique_xcoord = np.arange(start=0, stop=(self.data.dims["x"] // xs) * xs, step=xs)
        unique_ycoord = np.arange(start=0, stop=(self.data.dims["y"] // ys) * ys, step=ys)
        xcoords = np.repeat(unique_xcoord, len(unique_ycoord))
        ycoords = np.tile(unique_xcoord, len(unique_ycoord))
        crops = [self.crop_corner(x=x, y=y, xs=xs, ys=ys, img_id=img_id) for x, y in zip(xcoords, ycoords)]
        return crops, xcoords, ycoords

    def crop_spot_generator(self, adata: AnnData, **kwargs) -> Iterator[Tuple[int, str]]:
        """
        Iterate over all obs_ids defined in adata and extract crops from img.

        Implemented for 10x spatial datasets.

        Parameters
        ----------
        adata
            Spatial dataset (including coords in adata.obsm['spatial']).
        dataset_name
            Name of the spatial data in adata (if not specified, take first one).
        size
            Default is 1.0.
            Amount of context (1.0 means size of spot, larger -> more context).
        scale
            Default is 1.0.
            Resolution of the crop (smaller -> smaller image).
        mask_circle
            Mask crop to a circle.
        cval
            Default is 0
            The value outside image boundaries or the mask.
        dtype
            Type to which the output should be (safely) cast.
            Currently supported dtypes: 'uint8'.

        Yields
        ------
        :class:`tuple`

            - Union[int, str]: obs_id of spot from adata.
            - :class:`xarray.DataArray` with dimensions channels, y, x: crop.
        """
        dataset_name = kwargs.get("dataset_name", None)
        if dataset_name is None:
            dataset_name = list(adata.uns[SPATIAL_U].keys())[0]
        xcoord = adata.obsm[SPATIAL_M][:, 0]
        ycoord = adata.obsm[SPATIAL_M][:, 1]
        spot_diameter = adata.uns[SPATIAL_U][dataset_name]["scalefactors"]["spot_diameter_fullres"]
        size = kwargs.get("size", 1)
        r = int(round(spot_diameter * size // 2))

        obs_ids = adata.obs.index.tolist()
        for i, obs_id in enumerate(obs_ids):
            crop = self.crop_center(x=xcoord[i], y=ycoord[i], xr=r, yr=r, **kwargs)
            yield (obs_id, crop)
