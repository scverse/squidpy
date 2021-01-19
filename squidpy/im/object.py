from typing import Any, List, Tuple, Union, TypeVar, Iterable, Iterator, Optional

try:
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Literal

from pathlib import Path

from scanpy import logging as logg
from anndata import AnnData

import numpy as np
import xarray as xr

from imageio import imread
import skimage

from squidpy._docs import d
from squidpy._utils import _unique_order_preserving
from squidpy.im._utils import _num_pages, CropCoords, _scale_xarray
from squidpy.im.feature_mixin import FeatureMixin
from squidpy._constants._pkg_constants import Key

Pathlike_t = Union[str, Path]
Interactive = TypeVar("Interactive")  # cannot import because of cyclic dependecies


@d.dedent  # trick to overcome not top-down order
@d.dedent
class ImageContainer(FeatureMixin):
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

    def __init__(
        self,
        img: Optional[Union[Pathlike_t, np.ndarray]] = None,
        img_id: str = "image",
        **kwargs: Any,
    ):
        self._data: xr.Dataset = xr.Dataset()
        self._data.attrs["crop"] = None

        chunks = kwargs.pop("chunks", None)
        if img is not None:
            if chunks is not None:
                chunks = {"x": chunks, "y": chunks}
            self.add_img(img, img_id=img_id, chunks=chunks, **kwargs)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__} object with {len(self.data.keys())} layer(s)\n"
        for layer in self.data.keys():
            s += f"    {layer}: "
            s += ", ".join(f"{dim} ({shape})" for dim, shape in zip(self.data[layer].dims, self.data[layer].shape))
            s += "\n"
        return s

    def __getitem__(self, key: str) -> xr.DataArray:
        return self.data[key]

    @property
    def data(self) -> xr.Dataset:
        """Underlying data container."""
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape `(y, x)`."""
        return self.data.dims["y"], self.data.dims["x"]  # TODO changed shape, need to catch all calls to img.shape

    @property
    def nchannels(self) -> int:
        """Number of channels."""  # noqa: D401
        # TODO this might fail, if we name channels sth else than "channels"
        return self.data.dims["channels"]  # type: ignore[no-any-return]

    @classmethod
    def open(cls, fname: Pathlike_t, lazy: bool = True, chunks: Optional[int] = None) -> "ImageContainer":  # noqa: A003
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
        self._data = xr.open_dataset(fname, chunks=chunks)
        if not lazy:
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
        Nothing, just saves the data to ``fname``.
        """
        self.data.to_netcdf(fname, mode="a")

    @d.get_sections(base="add_img", sections=["Parameters", "Raises"])
    def add_img(
        self,
        img: Union[Pathlike_t, np.ndarray, xr.DataArray],
        img_id: str = "image",
        channel_id: str = "channels",
        lazy: bool = True,
        chunks: Optional[int] = None,
    ) -> None:
        """
        Add layer from in memory `np.ndarray`/`xr.DataArray` or on-disk tiff/jpg image.

        For :mod:`numpy` arrays, we assume that dims are: `(y, x, channel_id)`.

        NOTE: lazy loading via :mod:`dask` is not supported for on-disk jpg files.
        They will be loaded in memory.

        NOTE: multi-page tiffs will be loaded in one DataArray, with the concatenated channel dimensions.

        Parameters
        ----------
        img
            :mod:`numpy` array or path to image file.
        img_id
            Key (name) to be used for img.
        channel_id
            Name of the channel dimension.
        lazy
            Use :mod:`rasterio` or :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`, used in call to :func:`xarray.open_rasterio` for tiff images.

        Returns
        -------
        Nothing, just adds loaded img to ``data`` with key ``img_id``.

        Raises
        ------
        ValueError
            If ``img`` is a :class:`np.ndarray` and has more than 3 dimensions.
        """
        img = self._load_img(img=img, channel_id=channel_id, chunks=chunks)
        # add to data
        logg.info(f"Adding `{img_id}` into object")
        self.data[img_id] = img
        # TODO: is this correct?
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
        # TODO: singlemethoddispatch
        if isinstance(img, np.ndarray):
            if len(img.shape) > 3:
                raise ValueError(f"Img has more than 3 dimensions. img.shape is `{img.shape}`.")
            dims = ["y", "x", channel_id]
            if len(img.shape) == 2:
                # add channel dimension
                img = img[:, :, np.newaxis]
            xr_img = xr.DataArray(img, dims=dims)
        elif isinstance(img, xr.DataArray):
            assert "x" in img.dims
            assert "y" in img.dims
            xr_img = img
        elif isinstance(img, (str, Path)):
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
                xr_img = xr_img.transpose("y", "x", ...)
                # TODO transpose xr_img to have channels as last dim
            elif ext in ("jpg", "jpeg"):  # TODO: constants
                img = imread(img)
                dims = ["y", "x", channel_id]
                xr_img = xr.DataArray(img, dims=dims)
            else:
                raise ValueError(f"Files with extension `{ext}`.")
        else:
            raise TypeError(img)

        return xr_img

    @d.get_sections(base="crop_corner", sections=["Parameters", "Returns"])
    @d.dedent
    def crop_corner(
        self,
        x: int,
        y: int,
        xs: int,
        ys: int,
        scale: float = 1.0,
        mask_circle: bool = False,
        cval: float = 0,
        dtype: Optional[str] = None,
        preserve_dtype: bool = True,
    ) -> "ImageContainer":
        """
        Extract a crop from upper left corner coordinates ``x`` and ``y``.

        The crop will be extracted right and down from ``x``, ``y``.

        Parameters
        ----------
        %(xy_coord)s
        %(width_height)s
        scale
            Resolution of the crop (smaller -> smaller image).
            Rescaling is done using :func:`skimage.transform.rescale`.
        mask_circle
            Mask crop to a circle.
        cval
            The value outside image boundaries or the mask.
        dtype
            Type to which the output should be (safely) cast. If `None`, don't recast.
            Currently supported dtypes: 'uint8'.
        preserve_dtype
            Make sure that dtype of all layers of the crop stays the same.
            Depending on type of ``cval``, types might be promoted to :class:`float` otherwise.

        Returns
        -------
        Cropped image.
        """
        # get conversion function
        if dtype is not None:
            if dtype == "uint8":
                convert = skimage.util.img_as_ubyte
            else:
                raise NotImplementedError(dtype)

        # TODO: rewrite assertions to "normal" errors so they can be more easily tested against
        assert y < self.data.y.shape[0], f"y ({y}) is outside of image range ({self.data.y.shape[0]})"
        assert x < self.data.x.shape[0], f"x ({x}) is outside of image range ({self.data.x.shape[0]})"

        assert xs > 0, "im size cannot be 0"
        assert ys > 0, "im size cannot be 0"

        # get dtypes before cropping
        if preserve_dtype and dtype is None:
            orig_dtypes = {key: arr.dtype for key, arr in self.data.items()}

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
        crop = self.data.isel(x=slice(crop_x0, crop_x1), y=slice(crop_y0, crop_y1)).copy(deep=False)
        crop.attrs["crop"] = CropCoords(crop_x0, crop_y0, crop_x1, crop_y1)

        # pad crop if necessary
        if [x0, x1, y0, y1] != [crop_x0, crop_x1, crop_y0, crop_y1]:
            crop = crop.pad(
                x=(abs(x0 - crop_x0), abs(x1 - crop_x1)),
                y=(abs(y0 - crop_y0), abs(y1 - crop_y1)),
                mode="constant",
                constant_values=cval,
            )

        # scale crop
        if scale != 1:
            crop = crop.map(_scale_xarray, scale=scale)

        # mask crop
        if mask_circle:
            assert xs == ys, "Crop has to be square to use mask_circle."
            c = crop.x.shape[0] // 2
            crop = crop.where((crop.x - c) ** 2 + (crop.y - c) ** 2 <= c ** 2, other=cval)

        # convert dtypes
        if preserve_dtype and dtype is None:
            for key in orig_dtypes.keys():
                crop[key] = crop[key].astype(orig_dtypes[key])
        elif dtype is not None:
            crop = crop.map(convert)

        # return crop as ImageContainer
        # TODO: add class method from_xarray?
        crop_cont = ImageContainer()
        crop_cont._data = crop
        return crop_cont

    @d.dedent
    def crop_center(
        self,
        x: int,
        y: int,
        xr: int,
        yr: int,
        **kwargs: Any,
    ) -> "ImageContainer":
        """
        Extract a crop based on coordinates ``x`` and ``y``.

        The extracted crop will be centered on ``x``, ``y``, and have shape ``(yr * 2 + 1, xr * 2 + 1)``.

        Parameters
        ----------
        %(xy_coord)s
        xr
            Radius of the crop in pixels.
        yr
            Radius of the crop in pixels.
        kwargs
            Keyword arguments are passed to :meth:`crop_corner`.

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

        return self.crop_corner(x=x, y=y, xs=xs, ys=ys, **kwargs)  # type: ignore[no-any-return]

    @d.dedent
    def generate_equal_crops(
        self,
        xs: Optional[int] = None,
        ys: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple["ImageContainer", int, int]]:
        """
        Decompose an image into equally sized crops.

        Parameters
        ----------
        %(width_height)s
        kwargs
            Keyword arguments for :meth:`squidpy.im.ImageContainer.crop_corner`.

        Returns
        -------
        Triple of the following:

            - crop of size ``(ys, xs)``.
            - x-position of the crop.
            - y-position of the crop.
        """
        if xs is None:
            xs = self.data.x.shape[0]
        if ys is None:
            ys = self.data.y.shape[0]

        # TODO: if equal length of self, generate copy

        # TODO: selecting coords like this means that we will have a small border at the right and bottom
        # where pixels are not selected. Depending on the crops size, this can be a substantial amount
        # of the image. Should use stop=(self.data.dims["x"] // xs) * (xs + 1) - 1
        # need to implement partial assembly of crops in uncrop_img to make it work.
        unique_xcoord = np.arange(start=0, stop=(self.data.dims["x"] // xs) * xs, step=xs)
        unique_ycoord = np.arange(start=0, stop=(self.data.dims["y"] // ys) * ys, step=ys)

        xcoords = np.repeat(unique_xcoord, len(unique_ycoord))
        ycoords = np.tile(unique_ycoord, len(unique_xcoord))

        for x, y in zip(xcoords, ycoords):
            yield self.crop_corner(x=x, y=y, xs=xs, ys=ys, **kwargs), x, y

    @d.dedent
    def generate_spot_crops(
        self,
        adata: AnnData,
        dataset_name: Optional[str] = None,
        size: float = 1.0,
        obs_ids: Optional[Iterable[Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple["ImageContainer", Union[int, str]]]:
        """
        Iterate over all ``obs_ids`` in :attr:`adata.obs_names` and extract crops from images.

        Implemented for 10x spatial datasets.

        Parameters
        ----------
        %(adata)s
        dataset_name
            Name of the spatial data in ``adata`` (if not specified, take first one).
        size
            Amount of context (1.0 means size of spot, larger -> more context).
        obs_ids
            Observations from :attr:`adata.obs_names` for which to generate the crops.
        kwargs
            Keyword arguments for :meth:`crop_center`.

        Yields
        ------
        Tuple of the following:

            - crop of size ``(ys, xs)``.
            - obs_id of spot from ``adata``.
        """
        # TODO: I'd just require to supply dataset name
        if dataset_name is None:
            dataset_name = list(adata.uns[Key.uns.spatial].keys())[0]

        # TODO: expose key?
        xcoord = adata.obsm[Key.obsm.spatial][:, 0]
        ycoord = adata.obsm[Key.obsm.spatial][:, 1]
        spot_diameter = adata.uns[Key.uns.spatial][dataset_name]["scalefactors"]["spot_diameter_fullres"]
        r = int(round(spot_diameter * size // 2))

        obs_ids, seen = _unique_order_preserving(adata.obs.index if obs_ids is None else obs_ids)
        indices = [i for i, obs in enumerate(adata.obs.index) if obs in seen]

        for i, obs_id in zip(indices, obs_ids):
            crop = self.crop_center(x=xcoord[i], y=ycoord[i], xr=r, yr=r, **kwargs)
            yield crop, obs_id  # type: ignore[misc]

    @classmethod
    @d.get_sections(base="uncrop_img", sections=["Parameters", "Returns"])
    def uncrop_img(
        cls,
        crops: List["ImageContainer"],
        x: np.ndarray,
        y: np.ndarray,
        shape: Tuple[int, int],
    ) -> "ImageContainer":
        """
        Re-assemble image from crops and their positions.

        Fills remaining positions with zeros. Positions are given as upper right corners.

        Parameters
        ----------
        crops
            List of image crops.
        x
            X coord of crop in pixel space.
        y
            Y coord of crop in pixel space.
        shape
            Shape of full image ``(y, x)``.

        Returns
        -------
        Assembled image with shape ``(y, x)``.
        """
        # TODO: maybe more descriptive names (y==height, x==width)? + extract to constants...
        # TODO: rewrite asserts
        # TODO: expose remaining positions default value
        assert np.max(y) < shape[0], f"y ({y}) is outside of image range ({shape[0]})"
        assert np.max(x) < shape[1], f"x ({x}) is outside of image range ({shape[1]})"

        # check if can trivially return crop
        if len(crops) == 1:
            if crops[0].shape[:2] == shape:
                # have only one crop with already correct shape
                return crops[0]

        # create resulting dataset
        img_ids = crops[0].data.keys()
        data = xr.Dataset()
        for image_id in img_ids:
            orig_img = crops[0].data[image_id]
            # get shape for this DataArray
            cur_shape = [shape[0], shape[1]] + list(orig_img.shape[2:])
            data[image_id] = xr.DataArray(np.zeros(cur_shape, dtype=orig_img.dtype), dims=orig_img.dims)

        # fill data with crops
        for c, x, y in zip(crops, x, y):
            x0 = x
            x1 = x + c.data[list(img_ids)[0]].x.shape[0]
            y0 = y
            y1 = y + c.data[list(img_ids)[0]].y.shape[0]
            # TODO: rewrite asserts
            assert x0 >= 0, f"x ({x0}) is outside of image range (0)"
            assert y0 >= 0, f"x ({y0}) is outside of image range (0)"
            assert x1 <= shape[1], f"x ({x1}) is outside of image range ({shape[1]})"
            assert y1 <= shape[0], f"y ({y1}) is outside of image range ({shape[0]})"
            for image_id in img_ids:
                data[image_id][y0:y1, x0:x1] = c[image_id]

        self = cls()
        self._data = data

        return self

    @d.get_sections(base="_interactive", sections=["Parameters"])
    @d.dedent
    def interactive(
        self,
        adata: AnnData,
        spatial_key: str = Key.obsm.spatial,
        library_id: Optional[str] = None,
        cmap: str = "viridis",
        palette: Optional[str] = None,
        blending: Literal["opaque", "translucent", "adidtive"] = "opaque",
        key_added: str = "shapes",
    ) -> Interactive:
        """
        Launch :mod:`napari` viewer.

        Parameters
        ----------
        %(adata)s
        %(spatial_key)s
        library_id
            Key in :attr:`anndata.AnnData.uns` ['spatial'] used to get the spot diameter.
        cmap
            Colormap for continuous variables.
        palette
            Colormap for categorical variables in :attr:`anndata.AnnData.obs`. If `None`, use :mod:`scanpy`'s default.
        blending
            Method which determines how RGB and alpha values of :class:`napari.layers.Shapes` are mixed.
        key_added
            Key where to store :class:`napari.layers.Shapes` which can be exported by pressing `SHIFT-E`:

                - :attr:`anndata.AnnData.obs` ``['{layer_name}_{key_added}']`` - boolean mask containing the selected
                  cells.
                - :attr:`anndata.AnnData.uns` ``['{layer_name}_{key_added}']['meshes']`` - list of :class:`numpy.array`,
                  defining a mesh in the spatial coordinates.

            See :mod:`napari`'s `tutorial <https://napari.org/tutorials/fundamentals/shapes.html>`__ for more
            information about different mesh types, such as circles, squares etc.

        Returns
        -------
        Interactive view of this container. Screenshot of the canvas can be taken by
        :meth:`squidpy.pl.Interactive.screenshot`.
        """
        from squidpy.pl import Interactive

        return Interactive(  # type: ignore[no-any-return]
            img=self,
            adata=adata,
            spatial_key=spatial_key,
            library_id=library_id,
            cmap=cmap,
            palette=palette,
            blending=blending,
            key_added=key_added,
        ).show()
