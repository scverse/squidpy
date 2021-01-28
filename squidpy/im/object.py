from copy import copy, deepcopy
from types import MappingProxyType
from typing import (
    Any,
    List,
    Tuple,
    Union,
    Mapping,
    TypeVar,
    Iterable,
    Iterator,
    Optional,
)
from tqdm.auto import tqdm

from skimage.transform import rescale

from squidpy.gr._utils import (
    _assert_in_range,
    _assert_positive,
    _assert_spatial_basis,
    _assert_non_empty_sequence,
)

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

from squidpy._docs import d
from squidpy.im._utils import _num_pages, CropCoords, _NULL_PADDING, _open_rasterio
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
        self._data.attrs["coords"] = None
        self._data.attrs["padding"] = None
        self._data.attrs["scale"] = 1

        chunks = kwargs.pop("chunks", None)
        if img is not None:
            if chunks is not None:
                chunks = {"x": chunks, "y": chunks}
            self.add_img(img, img_id=img_id, chunks=chunks, **kwargs)

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
        if not lazy:
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
                    data = _open_rasterio(f"GTIFF_DIR:{i}:{img}", chunks=chunks, parse_coordinates=False)
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
        yx: Tuple[int, int],
        dydx: Tuple[int, int],
        scale: float = 1.0,
        cval: Union[int, float] = 0,
        mask_circle: bool = False,
    ) -> "ImageContainer":
        """
        Extract a crop from upper left corner coordinates ``x`` and ``y``.

        The crop will be extracted right and down from ``x``, ``y``.

        Parameters
        ----------
        TODO
            rename me.
        TODO
            rename me.
        scale
            Resolution of the crop (smaller -> smaller image).
            Rescaling is done using :func:`skimage.transform.rescale`.
        cval
            The value outside image boundaries or the mask.
        mask_circle
            Mask crop to a circle.

        Returns
        -------
        Cropped image.
        """
        self._assert_not_empty()

        (y, x), (ys, xs) = yx, dydx
        _assert_positive(ys, name="height")
        _assert_positive(xs, name="width")

        orig = CropCoords(x0=x, y0=y, x1=x + xs, y1=y + ys)

        ymin, xmin = self.data.dims["y"], self.data.dims["x"]
        coords = CropCoords(
            x0=min(max(x, 0), xmin), y0=min(max(y, 0), ymin), x1=min(x + xs, xmin), y1=min(y + ys, ymin)
        )

        if not coords.dy:
            raise ValueError("Height is empty.")
        if not coords.dx:
            raise ValueError("Width is empty.")

        crop = self.data.isel(x=slice(coords.x0, coords.x1), y=slice(coords.y0, coords.y1)).copy(deep=False)
        crop.attrs["orig"] = orig
        crop.attrs["coords"] = coords

        if orig != coords:
            padding = orig - coords
            crop = crop.pad(
                y=(padding.y_pre, padding.y_post),
                x=(padding.x_pre, padding.x_post),
                mode="constant",
                constant_values=cval,
            )
            crop.attrs["padding"] = padding
        else:
            crop.attrs["padding"] = _NULL_PADDING

        crop = self._post_process(data=crop, scale=scale, cval=cval, mask_circle=mask_circle)

        return self.from_array(crop)

    def _post_process(
        self,
        data: xr.Dataset,
        scale: Union[int, float] = 1,
        cval: Union[int, float] = 1,
        mask_circle: bool = False,
        **_: Any,
    ) -> xr.Dataset:
        _assert_positive(scale, name="scale")
        if scale != 1:
            attrs = data.attrs
            data = data.map(
                lambda arr: xr.DataArray(
                    rescale(arr, scale=scale, preserve_range=True, order=1, multichannel=True).astype(arr.dtype),
                    dims=arr.dims,
                )
            )
            data.attrs = {**attrs, "scale": scale}

        # TODO: does not update crop metadata (scale), should it?
        if mask_circle:
            # TODO: ignore/raise/ellipse?
            if data.dims["y"] != data.dims["x"]:
                logg.warning("Crops is not square TODO")
                # assert xs == ys, "Crop has to be square to use mask_circle."
            c = min(data.x.shape[0], data.y.shape[0]) // 2
            data = data.where((data.x - c) ** 2 + (data.y - c) ** 2 <= c ** 2, other=cval)

            for key, arr in self.data.items():
                data[key] = data[key].astype(arr.dtype, copy=False)

        return data

    @d.dedent
    def crop_center(
        self,
        yx: Tuple[int, int],
        ryrx: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ) -> "ImageContainer":
        """
        TODO. Extract a crop based on coordinates ``x`` and ``y``.

        The extracted crop will be centered on ``x``, ``y``, and have shape ``(yr * 2 + 1, xr * 2 + 1)``.

        Parameters
        ----------
        yx
            TODO - rename.
        ryrx
            TODO - rename.
        kwargs
            Keyword arguments are passed to :meth:`crop_corner`.

        Returns
        -------
        %(crop_corner.returns)s
        """
        if not isinstance(ryrx, tuple):
            ryrx = (ryrx, ryrx)

        (y, x), (yr, xr) = yx, ryrx
        _assert_in_range(y, 0, self.data.dims["y"], name="center height")
        _assert_in_range(x, 0, self.data.dims["x"], name="center width")
        _assert_positive(yr, name="radius height")
        _assert_positive(xr, name="radius width")

        return self.crop_corner(  # type: ignore[no-any-return]
            yx=(x - xr, y - yr), dydx=(xr * 2 + 1, yr * 2 + 1), **kwargs
        )

    @d.dedent
    def generate_equal_crops(
        self,
        yx: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
        as_array: bool = False,
        **kwargs: Any,
    ) -> Iterator["ImageContainer"]:
        """
        Decompose an image into equally sized crops.

        Parameters
        ----------
        yx
            TODO - rename.
        as_array
            TODO - docrep.
        kwargs
            Keyword arguments for :meth:`squidpy.im.ImageContainer.crop_corner`.

        Returns
        -------
        Triple of the following:

            - crop of size ``(ys, xs)``.
            - x-position of the crop.
            - y-position of the crop.
        """
        self._assert_not_empty()

        ys, xs = yx if isinstance(yx, tuple) else (yx, yx)
        if ys is None:
            ys = self.data.dims["y"]
        if xs is None:
            xs = self.data.dims["x"]

        if ys == self.data.dims["y"] and xs == self.data.dims["x"]:
            res = self.copy(deep=True)
            res._data = self._post_process(data=res.data, **kwargs)
            yield res
        else:
            _assert_in_range(ys, 0, self.data.dims["y"], name="ys")
            _assert_in_range(xs, 0, self.data.dims["x"], name="xs")

            # selecting coords like this means that we will have a small border at the right and bottom
            # where pixels are not selected. Depending on the crops size, this can be a substantial amount
            # of the image. Should use stop=(self.data.dims["x"] // xs) * (xs + 1) - 1
            # need to implement partial assembly of crops in uncrop_img to make it work.
            unique_xcoord = np.arange(start=0, stop=(self.data.dims["x"] // xs) * xs, step=xs)
            unique_ycoord = np.arange(start=0, stop=(self.data.dims["y"] // ys) * ys, step=ys)

            ycoords = np.repeat(unique_ycoord, len(unique_xcoord))
            xcoords = np.tile(unique_xcoord, len(unique_ycoord))

            # TODO: go in C order, not F order
            for y, x in zip(ycoords, xcoords):
                crop = self.crop_corner(yx=(y, x), dydx=(ys, xs), **kwargs)
                if as_array:
                    crop = crop.data.to_array().values
                yield crop

    @d.dedent
    def generate_spot_crops(
        self,
        adata: AnnData,
        library_id: str,
        spatial_key: str = Key.obsm.spatial,
        size: float = 1.0,
        obs_names: Optional[Iterable[Any]] = None,
        as_array: bool = False,
        return_obs: bool = False,
        **kwargs: Any,
    ) -> Iterator[Tuple["ImageContainer", Union[int, str]]]:
        """
        Iterate over all ``obs_ids`` in :attr:`adata.obs_names` and extract crops from images.

        Implemented for 10x spatial datasets.

        Parameters
        ----------
        %(adata)s
        library_id
            Key in :attr:`anndata.AnnData.uns` ``[spatial_key]`` from where to get the spot diameter.
        %(spatial_key)s
        size
            Amount of context (1.0 means size of spot, larger -> more context).
        obs_names
            Observations from :attr:`adata.obs_names` for which to generate the crops.
        as_array
            TODO.
        return_obs
            TODO.
        kwargs
            Keyword arguments for :meth:`crop_center`.

        Yields
        ------
        Tuple of the following:

            - crop of size ``(ys, xs)``.
            - obs_id of spot from ``adata``.
        """
        self._assert_not_empty()
        _assert_spatial_basis(adata, spatial_key)

        if obs_names is None:
            obs_names = adata.obs_names
        obs_names = _assert_non_empty_sequence(obs_names)

        adata = adata[obs_names, :]
        spatial = adata.obsm[spatial_key][:, :2]

        diameter = adata.uns[Key.uns.spatial][library_id]["scalefactors"]["spot_diameter_fullres"]
        radius = int(round(diameter // 2 * size))

        for i in range(adata.n_obs):
            crop = self.crop_center(spatial[i], ryrx=radius, **kwargs)
            crop.data.attrs["obs"] = adata.obs_names[i]
            if as_array:
                crop = crop.data.to_array().values
            yield (crop, adata.obs_names[i]) if return_obs else crop

    @classmethod
    def from_array(cls, arr: Union[np.ndarray, xr.Dataset], channel_id: str = "channels") -> "ImageContainer":
        """TODO."""
        # TODO: should also copy
        res = cls()
        res._data = arr
        return res

    @classmethod
    @d.get_sections(base="uncrop_img", sections=["Parameters", "Returns"])
    def uncrop_img(
        cls,
        crops: List["ImageContainer"],
        shape: Optional[Tuple[int, int]] = None,
    ) -> "ImageContainer":
        """
        Re-assemble image from crops and their positions.

        Fills remaining positions with zeros. Positions are given as upper right corners.

        Parameters
        ----------
        crops
            List of image crops.
        shape
            Shape of full image ``(y, x)``.

        Returns
        -------
        Assembled image with shape ``(y, x)``.
        """
        if not len(crops):
            raise ValueError("TODO: no crops supplied.")

        keys = set(crops[0].data.keys())
        dy, dx = -1, -1

        for crop in crops:
            if set(crop.data.keys()) != keys:
                raise KeyError("TODO: invalid keys")
            if crop.data.attrs.get("coords", None) is None:
                raise ValueError()
            coord = crop.data.attrs["coords"]  # the unpadded coordinates
            dy, dx = max(dy, coord.y0 + coord.dy), max(dx, coord.x0 + coord.dx)

        if shape is None:
            shape = (dy, dx)
        if shape < (dy, dx):
            raise ValueError("TODO: insufficient shape...")
        shape = tuple(shape)  # type: ignore[assignment]
        if len(shape) != 2:
            raise ValueError()

        # create resulting dataset
        data = xr.Dataset()
        for key in keys:
            img = crops[0].data[key]
            # get shape for this DataArray
            data[key] = xr.DataArray(np.zeros(shape + tuple(img.shape[2:]), dtype=img.dtype), dims=img.dims)

        # fill data with crops
        for crop in tqdm(crops, unit="crop"):
            for key in keys:
                coord = crop.data.attrs["coords"]
                padding = crop.data.attrs["padding"]
                data[key][coord.slice] = crop[key][coord.to_image_coordinates(padding=padding).slice]

        return cls.from_array(data)

    def show(self, key: Optional[str] = None) -> None:
        """TODO."""
        self._assert_not_empty()

        if key is None:
            if len(self) > 1:
                raise ValueError()
            key = list(self.data.keys())[0]
        if key not in self.data.keys():
            raise KeyError()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.imshow(self.data[key])

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
        from squidpy.pl import Interactive  # type: ignore[attr-defined]

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

    # TODO: apply style fn

    @property
    def data(self) -> xr.Dataset:
        """Underlying :class:`xarray.Dataset`."""
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape `(y, x)`."""
        return self.data.dims["y"], self.data.dims["x"]  # TODO changed shape, need to catch all calls to img.shape

    def copy(self, deep: bool = False) -> "ImageContainer":
        """TODO."""
        return deepcopy(self) if deep else copy(self)

    def _assert_not_empty(self) -> None:
        if not len(self):
            raise ValueError("No image has been added.")

    def __iter__(self) -> Iterator[str]:
        # TODO: determine what we want (keys, values), DataArray returns keys
        yield from self.data.values()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> xr.DataArray:
        return self.data[key]

    def __copy__(self) -> "ImageContainer":
        return type(self).from_array(self.data.copy(deep=False))

    def __deepcopy__(self, memodict: Mapping[str, Any] = MappingProxyType({})) -> "ImageContainer":
        return type(self).from_array(self.data.copy(deep=True))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__} object with {len(self.data.keys())} layer(s)\n"
        for layer in self.data.keys():
            s += f"    {layer}: "
            s += ", ".join(f"{dim} ({shape})" for dim, shape in zip(self.data[layer].dims, self.data[layer].shape))
            s += "\n"
        return s

    def __str__(self) -> str:
        return repr(self)
