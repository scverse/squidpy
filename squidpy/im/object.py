from copy import copy, deepcopy
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Mapping,
    TypeVar,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    TYPE_CHECKING,
)
from itertools import chain
import re

from scipy.sparse import spmatrix

import matplotlib.pyplot as plt

from skimage.util import img_as_float
from skimage.transform import rescale

from squidpy._utils import singledispatchmethod
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
from squidpy.im._utils import (
    _num_pages,
    CropCoords,
    CropPadding,
    _NULL_COORDS,
    _NULL_PADDING,
    _open_rasterio,
    TupleSerializer,
)
from squidpy.im.feature_mixin import FeatureMixin
from squidpy._constants._pkg_constants import Key

Pathlike_t = Union[str, Path]
Arraylike_t = Union[np.ndarray, xr.DataArray]
Input_t = Union[Pathlike_t, Arraylike_t, "ImageContainer"]
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
        img: Optional[Input_t] = None,
        img_id: str = "image",
        **kwargs: Any,
    ):
        self._data: xr.Dataset = xr.Dataset()
        self._data.attrs["coords"] = _NULL_COORDS  # can't save None to NetCDF
        self._data.attrs["padding"] = _NULL_PADDING
        self._data.attrs["scale"] = 1

        chunks = kwargs.pop("chunks", None)
        if img is not None:
            if chunks is not None:
                chunks = {"x": chunks, "y": chunks}
            self.add_img(img, img_id=img_id, chunks=chunks, **kwargs)

    @classmethod
    def load(cls, fname: Pathlike_t, lazy: bool = True, chunks: Optional[int] = None) -> "ImageContainer":
        """
        Load an NetCDF file.

        Parameters
        ----------
        fname
            Path to NetCDF file.
        lazy
            Use :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`.
        """
        res = cls()
        res.add_img(fname, img_id="image", chunks=chunks, lazy=lazy)
        return res

    def save(self, fname: Pathlike_t, **kwargs: Any) -> None:
        """
        Save dataset as NetCDF file.

        Parameters
        ----------
        fname
            Path where to save the NetCDF file.

        Returns
        -------
        Nothing, just saves the data to ``fname``.
        """
        fname = str(fname)
        if not (fname.endswith(".nc") or fname.endswith(".cdf")):
            fname += ".nc"

        try:
            attrs = self.data.attrs
            # TODO: https://github.com/pydata/xarray/issues/4790
            # scipy engine won't work: ValueError: could not safely cast array from dtype uint8 to int8
            self.data.attrs = {
                k: (v.to_tuple() if isinstance(v, TupleSerializer) else v) for k, v in self.data.attrs.items()
            }
            self.data.to_netcdf(fname, mode="w", **kwargs)
        finally:
            self.data.attrs = attrs

    def _get_image_id(self, img_id: str) -> str:
        pat = re.compile(rf"^{img_id}_(\d+)$")
        iterator = chain.from_iterable(pat.finditer(k) for k in self.data.keys())
        return f"{img_id}_{(max(map(lambda m: int(m.groups()[0][0]), iterator), default=-1) + 1)}"

    @d.get_sections(base="add_img", sections=["Parameters", "Raises"])
    def add_img(
        self,
        img: Input_t,
        img_id: Optional[str] = None,
        channel_id: str = "channels",
        lazy: bool = True,
        chunks: Optional[int] = None,
        **kwargs: Any,
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
            An array or a path to JPEG or TIFF file.
        img_id
            Key (name) to be used for img.
        channel_id
            Name of the channel dimension.
        lazy
            Use :mod:`rasterio` or :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`, used in call to :func:`xarray.open_rasterio` for TIFF images.
        kwargs
            TODO.

        Returns
        -------
        Nothing, just adds loaded img to ``data`` with key ``img_id``.

        Raises
        ------
        ValueError
            TODO.
        NotImplementedError
            TODO.
        """
        img_id = self._get_image_id("image") if img_id is None else img_id
        img = self._load_img(img, chunks=chunks, img_id=img_id, **kwargs)

        if img is not None:  # not reading .nc file
            if TYPE_CHECKING:
                assert isinstance(img, xr.DataArray)
            img = img.rename({img.dims[-1]: channel_id})

            logg.info(f"Adding `{img_id}` into object")
            self.data[img_id] = img
        if not lazy:
            # load in memory
            self.data.load()

    @singledispatchmethod
    def _load_img(
        self, img: Union[Pathlike_t, Input_t, "ImageContainer"], img_id: str, **kwargs: Any
    ) -> Optional[xr.DataArray]:
        """
        Load an image.

        See :meth:`add_img` for more details.
        """
        if isinstance(img, ImageContainer):
            if img_id not in img:
                raise KeyError(f"Image id `{img_id}` not found in `{img}`.")
            return self._load_img(img[img_id], **kwargs)
        raise NotImplementedError(type(img))

    @_load_img.register(str)
    @_load_img.register(Path)
    def _(self, img: Pathlike_t, chunks: Optional[int] = None, **_: Any) -> Optional[xr.DataArray]:
        img = Path(img)
        if not img.is_file():
            raise OSError(f"Path `{img}` does not exist.")

        suffix = img.suffix.lower()

        if suffix in (".jpg", ".jpeg"):
            return self._load_img(imread(str(img)))
        if suffix in (".nc", ".cdf"):
            if len(self._data):
                raise ValueError("Loading data from NetCDF is disallow if the container is not empty.")

            self._data = xr.open_dataset(img, chunks=chunks)
            self.data.attrs["coords"] = CropCoords.from_tuple(self.data.attrs["coords"])
            self.data.attrs["padding"] = CropPadding.from_tuple(self.data.attrs["padding"])
            return None
        if suffix in (".tif", ".tiff"):
            # calling _load_img ensures we can safely do the transpose
            return self._load_img(
                xr.concat(
                    [
                        _open_rasterio(f"GTIFF_DIR:{i}:{img}", chunks=chunks, parse_coordinates=False)
                        for i in range(1, _num_pages(img) + 1)
                    ],
                    dim="band",
                ),
                copy=False,
            ).transpose("y", "x", ...)

        raise ValueError(f"Unknown suffix `{img.suffix}`.")

    @_load_img.register(spmatrix)  # type: ignore[no-redef]
    def _(self, img: spmatrix, **_: Any) -> xr.DataArray:
        return self._load_img(img.A)

    @_load_img.register(np.ndarray)  # type: ignore[no-redef]
    def _(self, img: np.ndarray, **_: Any) -> xr.DataArray:
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"Expected image to have `3` dimensions, found `{img.ndim}`.")

        return xr.DataArray(img, dims=["y", "x", "channels"])

    @_load_img.register(xr.DataArray)  # type: ignore[no-redef]
    def _(self, img: xr.DataArray, copy: bool = True, **_: Any) -> xr.DataArray:
        if img.ndim == 2:
            img = img.expand_dims("channels", -1)
        if img.ndim != 3:
            raise ValueError(f"Expected image to have `3` dimensions, found `{img.ndim}`.")

        mapping: Dict[Hashable, str] = {}
        if "y" not in img.dims:
            logg.warning(f"Dimension `y` not found in the data. Assuming it's `{img.dims[0]}`")
            mapping[img.dims[0]] = "y"
        if "x" not in img.dims:
            logg.warning(f"Dimension `x` not found in the data. Assuming it's `{img.dims[1]}`")
            mapping[img.dims[1]] = "x"

        img = img.rename(mapping)  # this will fail if trying to map to the same names
        return img.copy() if copy else img

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
            rename me?
        TODO
            rename me?
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

        return self._from_dataset(crop)

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

        if mask_circle:
            # TODO: ignore/raise/ellipse?
            if data.dims["y"] != data.dims["x"]:
                logg.warning("TODO")
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
            TODO - rename?
        ryrx
            TODO - rename?
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
    def _from_dataset(cls, data: xr.Dataset, deep: Optional[bool] = None) -> "ImageContainer":
        """TODO."""
        res = cls()
        res._data = data if deep is None else data.copy(deep=deep)
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
            raise ValueError("No crops were supplied.")

        keys = set(crops[0].data.keys())
        dy, dx = -1, -1

        for crop in crops:
            if set(crop.data.keys()) != keys:
                raise KeyError(f"Expected to find `{sorted(keys)}` keys, found `{sorted(crop.data.keys())}`.")
            if crop.data.attrs.get("coords", None) is None:
                raise ValueError("Crop does not have coordinate metadata.")
            coord = crop.data.attrs["coords"]  # the unpadded coordinates
            dy, dx = max(dy, coord.y0 + coord.dy), max(dx, coord.x0 + coord.dx)

        if shape is None:
            shape = (dy, dx)
        shape = tuple(shape)  # type: ignore[assignment]
        if len(shape) != 2:
            raise ValueError(f"Expected `shape` to be of length `2`, found `{len(shape)}`.")
        if shape < (dy, dx):
            raise ValueError(f"Requested image of shape `{shape}`, but minimal shape must be `({dy}, {dx})`.")

        # create resulting dataset
        data = xr.Dataset()
        for key in keys:
            img = crops[0].data[key]
            # get shape for this DataArray
            data[key] = xr.DataArray(np.zeros(shape + tuple(img.shape[2:]), dtype=img.dtype), dims=img.dims)

        # fill data with crops
        for crop in crops:
            for key in keys:
                coord = crop.data.attrs["coords"]
                padding = crop.data.attrs["padding"]
                data[key][coord.slice] = crop[key][coord.to_image_coordinates(padding=padding).slice]

        return cls._from_dataset(data)

    @d.dedent
    def show(
        self,
        img_id: Optional[str] = None,
        channel: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[Pathlike_t] = None,
        **kwargs: Any,
    ) -> None:
        """
        TODO.

        Parameters
        ----------
        img_id
            TODO.
        channel
            TODO.
        %(plotting)s
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        %(plotting_returns)s
        """
        from squidpy.pl._utils import save_fig

        self._assert_not_empty()

        if img_id is None:
            if len(self) > 1:
                raise ValueError(f"Please supply `img_id=...` from: `{sorted(self.data.keys())}`.")
            img_id = list(self.data.keys())[0]
        if img_id not in self.data.keys():
            raise KeyError(f"Image id not found in `{sorted(self.data.keys())}`.")

        arr = self.data[img_id]
        if channel is not None:
            arr = arr[{arr.dims[-1]: channel}]

        fig, ax = plt.subplots(figsize=(8, 8) if figsize is None else figsize, dpi=dpi)
        ax.set_axis_off()

        ax.imshow(img_as_float(arr.values, force_copy=False), **kwargs)

        if save:
            save_fig(fig, save)

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

    @property
    def data(self) -> xr.Dataset:
        """Underlying :class:`xarray.Dataset`."""
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape `(y, x)`."""
        return self.data.dims["y"], self.data.dims["x"]

    def copy(self, deep: bool = False) -> "ImageContainer":
        """TODO."""
        return deepcopy(self) if deep else copy(self)

    def _assert_not_empty(self) -> None:
        if not len(self):
            raise ValueError("No image has been added.")

    def __iter__(self) -> Iterator[str]:
        yield from self.data.keys()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> xr.DataArray:
        return self.data[key]

    def __copy__(self) -> "ImageContainer":
        return type(self)._from_dataset(self.data, deep=False)

    def __deepcopy__(self, memodict: Mapping[str, Any] = MappingProxyType({})) -> "ImageContainer":
        return type(self)._from_dataset(self.data, deep=True)

    def _repr_html_(self) -> str:
        s = f"{self.__class__.__name__} object with {len(self.data.keys())} layer(s):"
        for layer in self.data.keys():
            s += f"<p style='text-indent: 25px;'>{layer}: "
            s += ", ".join(f"{dim} ({shape})" for dim, shape in zip(self.data[layer].dims, self.data[layer].shape))
            s += "</p>"
        return s

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[size={len(self)}, ids={sorted(self.data.keys())}]"

    def __str__(self) -> str:
        return repr(self)
