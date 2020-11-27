from os import PathLike
from typing import List, Tuple, Union, Iterator, Optional
from pathlib import Path

from scanpy import logging as logg
from anndata import AnnData

import numpy as np
import xarray as xr

from imageio import imread

from squidpy._docs import d
from squidpy.image._utils import _round_even
from squidpy.constants._pkg_constants import Key
from ._utils import _num_pages

Pathlike_t = Union[str, Path]


class ImageContainer:
    """
    Container for in memory or on-disk tiff or jpg images.

    Allows for lazy and chunked reading via :mod:`rasterio` and :mod:`dask` (if input is a tiff image).
    An instance of this class is given to all image processing functions, along with an :mod:`anndata` instance,
    if necessary.
    """

    data: xr.Dataset

    def __init__(
        self,
        img: Optional[Union[Pathlike_t, np.ndarray]] = None,
        img_id: Optional[Union[str, List[str]]] = None,
        lazy: bool = True,
        chunks: Optional[int] = None,
    ):
        """
        Set up ImageContainer from numpy array or on-disk tiff / jpg.

        Processes image as in memory :class:`numpy.array` or uses :mod`xarray`'s :mod:`rasterio` reading functions to
        load from disk (with caching) if ``img`` is a file path.
        If chunks are specified, the :mod:`xarray` is wrapped in a :mod:`dask`.

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

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape."""
        return self.data.dims["x"], self.data.dims["y"]

    @property
    def nchannels(self) -> int:
        """Number of channels."""  # noqa: D401
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
        self = cls(img=None, lazy=lazy, chunks=chunks)
        self.data = xr.open_dataset(fname, chunks=self._chunks)
        if not self._lazy:
            self.data.load()
        return self

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
            TODO.
        """
        self.data.to_netcdf(fname, mode="a")

    def add_img(
        self,
        img: Union[Pathlike_t, np.ndarray, xr.DataArray],
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
        channel_id
            TODO.

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
        logg.info(f"Adding `{img_id}` into object")
        self.data[img_id] = img
        if not self._lazy:
            # load in memory
            self.data.load()

    def _load_img(self, img: Union[str, np.ndarray], channel_id: str = "channels") -> xr.DataArray:
        """
        Load img as :mod:`xarray`.

        Supports numpy arrays and (multi-page) tiff files, and jpg files
        For :mod:`numpy` arrays, assume that dims are: ``(channels, y, x)``.

        NOTE: lazy loading via :mod:`dask` is currently not supported for on-disk jpg files.
        They will be loaded in memory.

        Parameters
        ----------
        img
            :mod:`numpy` array or path to image file.
        channel_id
            TODO.

        Returns
        -------
        :class:`xarray.DataArray`
            Array containing the loaded image.

        Raises
        ------
        :class:`ValueError`
            If ``img`` is a :class:`np.ndarray` and has more than 3 dimensions.
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
        elif isinstance(img, (str, PathLike)):
            img = str(img)
            ext = img.split(".")[-1]
            if ext in ("tif", "tiff"):  # TODO: constants
                # get the number of pages in the file
                num_pages = _num_pages(img)
                # read all pages using rasterio
                xr_img_byband = []
                for i in range(1, num_pages + 1):
                    data = xr.open_rasterio(f"GTIFF_DIR:{i}:{img}", chunks=self._chunks, parse_coordinates=False)
                    data = data.rename({"band": channel_id})
                    xr_img_byband.append(data)
                xr_img = xr.concat(xr_img_byband, dim=channel_id)
            elif ext in ("jpg", "jpeg"):  # TODO: constants
                img = imread(img)
                # jpeg has channels as last dim - transpose
                img = img.transpose(2, 0, 1)
                dims = [channel_id, "y", "x"]
                xr_img = xr.DataArray(img, dims=dims)
            else:
                raise NotImplementedError(f"Files with extension `{ext}`.")
        else:
            raise ValueError(img)
        return xr_img

    @d.dedent
    def crop(
        self,
        x: float,
        y: float,
        xs: int = 100,
        ys: int = 100,
        img_id: Optional[str] = None,
        centred: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        """
        Extract a crop based on coordinates `x` and `y` of `img_id`.

        Centred on x, y if centred is True, else right and down from x, y.

        Parameters
        ----------
        x
            X coord of crop (in pixel space). Can be float (ie. int+0.5) if model is centered and if x+xs/2 is integer.
        y
            Y coord of crop (in pixel space). Can be float (ie. int+0.5) if model is centered and if y+ys/2 is integer.
        %(width_height)s
        img_id
            id of the image layer to be cropped.
        centred
            Whether the crop coordinates are centred.
        kwargs
            Keyword arguments for :func:`squidpy.image.crop_img`.

        Returns
        -------
        :class:`xarray.DataArray`
            Array of shape ``(channels, y, x)``.
        """
        from .crop import crop_img

        # TODO: TODO: when scaling, will return a numpy array instead of an xarray

        if img_id is None:
            img_id = list(self.data.keys())[0]

        img = self.data.data_vars[img_id]
        if centred:
            assert (xs / 2.0 + x) % 1 == 0, "x and xs/2 have to add up to an integer to use centred model"
            assert (ys / 2.0 + y) % 1 == 0, "y and ys/2 have to add up to an integer to use centred model"
            x = int(x - xs // 2)  # move from centre to corner
            y = int(y - ys // 2)  # move from centre to corner
        else:
            assert x % 1.0 == 0, "x needs to be integer"
            assert y % 1.0 == 0, "x needs to be integer"
            x = int(x)
            y = int(y)
        return crop_img(img=img, x=x, y=y, xs=xs, ys=ys, **kwargs)

    @d.dedent
    def crop_equally(
        self,
        xs: Optional[int] = None,
        ys: Optional[int] = None,
        img_id: Optional[Union[str, List[str]]] = None,
        **_kwargs,
    ) -> Tuple[List[xr.DataArray], np.ndarray, np.ndarray]:
        """
        Decompose an image into equally sized crops.

        Parameters
        ----------
        %(width_height)s
        _kwargs
            TODO: unused.

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
        ycoords = np.tile(unique_xcoord, len(unique_ycoord))

        # TODO: outdated docs or why _kwargs are not passed?
        crops = [self.crop(x=x, y=y, xs=xs, ys=ys, img_id=img_id, centred=False) for x, y in zip(xcoords, ycoords)]

        return crops, xcoords, ycoords

    @d.dedent
    def crop_spot_generator(
        self, adata: AnnData, dataset_name: Optional[str] = None, size: float = 1.0, **kwargs
    ) -> Iterator[Tuple[Union[int, str], xr.DataArray]]:
        """
        Iterate over all obs_ids defined in adata and extract crops from img.

        Implemented for 10x spatial datasets.

        Parameters
        ----------
        %(adata)s
        dataset_name
            Name of the spatial data in adata (if not specified, take first one).
        size
            Amount of context (1.0 means size of spot, larger -> more context).
        kwargs
            Keyword arguments for :func:`squidpy.image.crop_img`.

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
        s = int(_round_even(spot_diameter * size))
        # TODO: could also use round_odd and add 0.5 for xcoord and ycoord

        obs_ids = adata.obs.index.tolist()
        for i, obs_id in enumerate(obs_ids):
            crop = self.crop(x=xcoord[i], y=ycoord[i], xs=s, ys=s, **kwargs)
            yield obs_id, crop
