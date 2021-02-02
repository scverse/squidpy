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
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    TYPE_CHECKING,
)
from itertools import chain
import re

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
    def load(cls, path: Pathlike_t, lazy: bool = True, chunks: Optional[int] = None) -> "ImageContainer":
        """
        Load data from a `Zarr` store.

        Parameters
        ----------
        path
            Path to `Zarr` store.
        lazy
            Use :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`.
        """
        res = cls()
        res.add_img(path, img_id="image", chunks=chunks, lazy=lazy)

        return res

    def save(self, path: Pathlike_t, **kwargs: Any) -> None:
        """
        Save self to `Zarr` store.

        Parameters
        ----------
        path
            Path to `Zarr` store.

        Returns
        -------
        Nothing, just saves the data to ``path``.
        """
        attrs = self.data.attrs
        try:
            self._data = self.data.load()  # TODO: explain why this has to be done
            self.data.attrs = {
                k: (v.to_tuple() if isinstance(v, TupleSerializer) else v) for k, v in self.data.attrs.items()
            }
            self.data.to_zarr(path, mode="w", **kwargs, **kwargs)
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
        **kwargs: Any,  # TODO: update docstring
    ) -> None:
        """
        Add layer from in memory `np.ndarray`/`xr.DataArray` or on-disk tiff/jpg image.

        For :mod:`numpy` arrays, we assume that dims are ``(y, x, channel_id)``.

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
            Keyword arguments

        Returns
        -------
        Nothing, just adds loaded img to ``data`` with key ``img_id``.

        Notes
        -----
        Lazy loading via :mod:`dask` is not supported for on-disk jpg files. They will be loaded in memory.
        Multi-page tiffs will be loaded in one :class:`xarray.DataArray`, with the concatenated channel dimensions.

        Raises
        ------
        ValueError
            If loading from a file and an extension was not understood.
        NotImplementedError
            If loading for a specific data type has not been implemented.
        """
        img_id = self._get_image_id("image") if img_id is None else img_id
        img = self._load_img(img, chunks=chunks, img_id=img_id, **kwargs)

        if img is not None:  # not reading a .nc file
            if TYPE_CHECKING:
                assert isinstance(img, xr.DataArray)
            img = img.rename({img.dims[-1]: channel_id})

            logg.info(f"Adding `{img_id}` into object")
            # TODO log if overwriting

            # TODO: ValueError: cannot reindex or align along dimension 'channels'
            #       because the index has duplicate values
            #       happens when e.g.: sq.im.process_img(im, img_id='image', copy=False, processing="smooth", yx=500)
            #       and the channels have the same name

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
        raise NotImplementedError(f"Loader for class `{type(img).__name__}` is not yet implemented.")

    @_load_img.register(str)
    @_load_img.register(Path)
    def _(self, img: Pathlike_t, chunks: Optional[int] = None, **_: Any) -> Optional[xr.DataArray]:
        img = Path(img)
        suffix = img.suffix.lower()

        if suffix in (".jpg", ".jpeg"):
            return self._load_img(imread(str(img)))

        if img.is_dir():
            if len(self._data):
                raise ValueError("Loading data from `Zarr` is disallowed if the container is not empty.")

            self._data = xr.open_zarr(str(img), chunks=chunks)
            self.data.attrs["coords"] = CropCoords.from_tuple(self.data.attrs.get("coords", _NULL_COORDS.to_tuple()))
            self.data.attrs["padding"] = CropPadding.from_tuple(
                self.data.attrs.get("padding", _NULL_PADDING.to_tuple())
            )
            return None

        if suffix in (".nc", ".cdf"):
            if len(self._data):
                raise ValueError("Loading data from `NetCDF` is disallowed if the container is not empty.")

            self._data = xr.open_dataset(img, chunks=chunks)
            self.data.attrs["coords"] = CropCoords.from_tuple(self.data.attrs.get("coords", _NULL_COORDS.to_tuple()))
            self.data.attrs["padding"] = CropPadding.from_tuple(
                self.data.attrs.get("padding", _NULL_PADDING.to_tuple())
            )
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

    @_load_img.register(np.ndarray)  # type: ignore[no-redef]
    def _(self, img: np.ndarray, **_: Any) -> xr.DataArray:
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"Expected image to have `3` dimensions, found `{img.ndim}`.")

        return xr.DataArray(img, dims=["y", "x", "channels"], coords={"channels": [1, 1, 1]})

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
        size: Union[int, Tuple[int, int]],  # TODO: defaults
        scale: float = 1.0,
        cval: Union[int, float] = 0,
        mask_circle: bool = False,
    ) -> "ImageContainer":
        """
        Extract a crop from upper-left corner coordinates ``yx``.

        Parameters
        ----------
        yx
            Coordinates of the upper-left corner as ``(height, width)``.
        size
            Size of the crop in ``(height, width)``. If a :class:`int`, the crop will be square.
        scale
            Resolution of the crop (smaller -> smaller image).
            Rescaling is done using :func:`skimage.transform.rescale`.
        cval
            Fill value to use if ``mask_circle = True`` or parts of the crop are out of the image boundary.
        mask_circle
            Whether to set values, which are not within a circle bounded by the ``size``, to ``cval``.
            Only available if ``size`` defines a square.

        Returns
        -------
        Cropped image.

        Raises
        ------
        ValueError
            If ``mask_circle = True`` and ``size`` does not define a square.
        """
        self._assert_not_empty()
        if not isinstance(size, Iterable):
            size = (size, size)

        (y, x), (ys, xs) = yx, size
        _assert_positive(ys, name="height")
        _assert_positive(xs, name="width")

        orig = CropCoords(x0=x, y0=y, x1=x + xs, y1=y + ys)
        orig_dtypes = {key: arr.dtype for key, arr in self.data.items()}

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
            for key in orig_dtypes.keys():
                crop[key] = crop[key].astype(orig_dtypes[key])
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
            if data.dims["y"] != data.dims["x"]:
                raise ValueError(
                    f"Masking out circle is only available for square crops, "
                    f"found crop of shape `{(data.dims['y'], data.dims['x'])}`."
                )
            c = data.x.shape[0] // 2
            data = data.where((data.x - c) ** 2 + (data.y - c) ** 2 <= c ** 2, other=cval)

            for key, arr in self.data.items():
                data[key] = data[key].astype(arr.dtype, copy=False)

        return data

    @d.dedent
    def crop_center(
        self,
        center: Tuple[int, int],
        radius: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ) -> "ImageContainer":
        """
        Extract a circular crop based on ``center``.

        The extracted crop will have shape ``(radius[0] * 2 + 1, radius[1] * 2 + 1)``.

        Parameters
        ----------
        center
            Center in the from of ``(height, width)``.
        radius
            Radius along the height and width, respectively. If a :class:`int`, the crop will be square.
        kwargs
            Keyword arguments for :meth:`crop_corner`.

        Returns
        -------
        %(crop_corner.returns)s
        """
        if not isinstance(radius, Iterable):
            radius = (radius, radius)

        (y, x), (yr, xr) = center, radius
        _assert_in_range(y, 0, self.data.dims["y"], name="center height")
        _assert_in_range(x, 0, self.data.dims["x"], name="center width")
        _assert_positive(yr, name="radius height")
        _assert_positive(xr, name="radius width")

        return self.crop_corner(  # type: ignore[no-any-return]
            yx=(x - xr, y - yr), size=(xr * 2 + 1, yr * 2 + 1), **kwargs
        )

    @d.dedent
    def generate_equal_crops(
        self,
        size: Union[int, Tuple[int, int]],  # TODO: default, e.g. ~10%
        as_array: bool = False,
        **kwargs: Any,
    ) -> Iterator["ImageContainer"]:
        """
        Decompose image into equally sized crops.

        Parameters
        ----------
        size
            Height and width of the crop, respectively. If a :class:`int`, the crops will be square.
        as_array
            Whether to yield :class:`numpy.ndarray` or :class:`squidpy.im.ImageContainer`.
        kwargs
            Keyword arguments for :meth:`crop_corner`.

        Yields
        ------
        If ``as_array = True``, yields :class:`numpy.ndarray` crops. Otherwise, yields
        :class:`squidpy.im.ImageContainer`.

        Notes
        -----
        Crops which would go out of the image boundary are padded with ``cval`` to fit the ``size``.
        """
        self._assert_not_empty()

        ys, xs = size if isinstance(size, Iterable) else (size, size)
        y, x = self.data.dims["y"], self.data.dims["x"]

        _assert_in_range(ys, 0, y, name="ys")
        _assert_in_range(xs, 0, x, name="xs")

        unique_ycoord = np.arange(start=0, stop=(y // ys + (y % ys != 0)) * ys, step=ys)
        unique_xcoord = np.arange(start=0, stop=(x // xs + (x % xs != 0)) * xs, step=xs)

        ycoords = np.repeat(unique_ycoord, len(unique_xcoord))
        xcoords = np.tile(unique_xcoord, len(unique_ycoord))

        for y, x in zip(ycoords, xcoords):
            crop = self.crop_corner(yx=(y, x), size=(ys, xs), **kwargs)
            if as_array:
                crop = crop.data.to_array().values

            yield crop

    @d.dedent
    def generate_spot_crops(
        self,
        adata: AnnData,
        library_id: Optional[str] = None,
        spatial_key: str = Key.obsm.spatial,
        spot_scale: float = 1.0,
        obs_names: Optional[Iterable[Any]] = None,
        as_array: bool = False,
        return_obs: bool = False,
        **kwargs: Any,
    ) -> Iterator[Tuple["ImageContainer", Union[int, str]]]:
        """
        Iterate over :attr:`adata.obs_names` and extract crops.

        Implemented for 10X spatial datasets.

        Parameters
        ----------
        %(adata)s
        library_id
            Key in :attr:`anndata.AnnData.uns` ['{spatial_key}'] used to get the spot diameter.
        %(spatial_key)s
        spot_scale
            Scaling factor for the spot diameter. Larger values mean more context.
        obs_names
            Observations from :attr:`adata.obs_names` for which to generate the crops. If `None`, all names are used.
        as_array
            Whether to yield the crop as a :class:`numpy.ndarray` or :class:`squidpy.im.ImageContainer`.
        return_obs
            Whether to also yield names from ``obs_names``.
        kwargs
            Keyword arguments for :meth:`crop_center`.

        Yields
        ------
        If ``return_obs = True``, yields a :class:`tuple` ``(crop, obs_name)``. Otherwise, yields just the ``crop``.
        If ``as_array = True``, crop will be a :class:`numpy.array`.
        """
        self._assert_not_empty()
        _assert_positive(spot_scale, name="scale")
        _assert_spatial_basis(adata, spatial_key)
        library_id = Key.uns.library_id(adata, spatial_key=spatial_key, library_id=library_id)

        if obs_names is None:
            obs_names = adata.obs_names
        obs_names = _assert_non_empty_sequence(obs_names)

        adata = adata[obs_names, :]
        spatial = adata.obsm[spatial_key][:, :2]

        diameter = adata.uns[spatial_key][library_id]["scalefactors"]["spot_diameter_fullres"]
        radius = int(round(diameter // 2 * spot_scale))

        for i in range(adata.n_obs):
            crop = self.crop_center(spatial[i], radius=radius, **kwargs)
            crop.data.attrs["obs"] = adata.obs_names[i]
            if as_array:
                crop = crop.data.to_array().values

            yield (crop, adata.obs_names[i]) if return_obs else crop

    @classmethod
    def _from_dataset(cls, data: xr.Dataset, deep: Optional[bool] = None) -> "ImageContainer":
        """
        Utility function used for initialization.

        Parameters
        ----------
        data
            The :class:`xarray.Dataset` to use.
        deep
            If `None`, don't copy the ``data``. If `True`, deep-copy the data, otherwise, make a shallow copy.

        Returns
        -------
        The newly created container.
        """  # noqa: D401
        res = cls()
        res._data = data if deep is None else data.copy(deep=deep)
        return res

    @classmethod
    @d.get_sections(base="uncrop", sections=["Parameters", "Returns"])
    def uncrop(
        cls,
        crops: List["ImageContainer"],
        shape: Optional[Tuple[int, int]] = None,
    ) -> "ImageContainer":
        """
        Re-assemble image from crops and their positions.

        Fills remaining positions with zeros. Positions are given as upper-right corners.

        Parameters
        ----------
        crops
            List of image crops.
        shape
            Requested image shape as ``(height, width)``. If `None`, it is determined automatically from ``crops``.

        Returns
        -------
        Re-assembled image from ``crops``.
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
            if coord == _NULL_COORDS:
                raise ValueError(f"Null coordinate detected: `{coord}`.")

            dy, dx = max(dy, coord.y0 + coord.dy), max(dx, coord.x0 + coord.dx)

        if shape is None:
            shape = (dy, dx)
        shape = tuple(shape)  # type: ignore[assignment]
        if len(shape) != 2:
            raise ValueError(f"Expected `shape` to be of length `2`, found `{len(shape)}`.")
        if shape < (dy, dx):
            raise ValueError(f"Requested image of shape `{shape}`, but minimal shape must be `({dy}, {dx})`.")

        # create resulting dataset
        dataset = xr.Dataset()
        for key in keys:
            img = crop.data[key]
            # get shape for this DataArray
            dataset[key] = xr.DataArray(
                np.zeros(shape + tuple(img.shape[2:]), dtype=img.dtype), dims=img.dims, coords=img.coords
            )
            # fill data with crops
            for crop in crops:
                coord = crop.data.attrs["coords"]
                padding = crop.data.attrs["padding"]
                dataset[key][coord.slice] = crop[key][coord.to_image_coordinates(padding=padding).slice]

        return cls._from_dataset(dataset)

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
        Show an image within this container.

        Parameters
        ----------
        %(img_id)s
            If `None` and only 1 image is present, use its key.
        channel
            Channel to plot. If `None`, use all channels for plotting.
        %(plotting)s
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        %(plotting_returns)s
        """
        from squidpy.pl._utils import save_fig

        arr = self.data[self._singleton_id(img_id)]
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

            See :mod:`napari`'s `tutorial <https://napari.org/tutorials/fundamentals/shapes.html>`_ for more
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

    def apply(
        self,
        func: Callable[..., np.ndarray],
        img_id: Optional[str] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> "ImageContainer":
        """
        Apply a function to an image within this container.

        Parameters
        ----------
        func
            Function which takes a :class:`numpy.ndarray` as input and produces an image-like output.
        %(img_id)
        channel
            Apply ``func`` only over a specific ``channel``. If `None`, apply over the full image.
        kwargs
            Keyword arguments for ``func``.

        Returns
        -------
        New container with `1` image saved under ``img_id``.
        """
        img_id = self._singleton_id(img_id)
        arr = self[img_id]
        channel_id = arr.dims[-1]

        if channel is not None:
            arr = arr[{channel_id: channel}]

        res = ImageContainer(func(arr.values, **kwargs), img_id=img_id, channel_id=channel_id)
        res.data.attrs = copy(self.data.attrs)

        return res

    @property
    def data(self) -> xr.Dataset:
        """Underlying :class:`xarray.Dataset`."""
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape ``(y, x)``."""
        return self.data.dims["y"], self.data.dims["x"]

    def copy(self, deep: bool = False) -> "ImageContainer":
        """
        Return a copy of self.

        Parameters
        ----------
        deep
            Whether to make a deep copy or not.

        Returns
        -------
        Copy of self.
        """
        return deepcopy(self) if deep else copy(self)

    def _singleton_id(self, img_id: Optional[str]) -> str:
        self._assert_not_empty()

        if img_id is None:
            if len(self) > 1:
                raise ValueError(f"Please supply `img_id=...` from: `{sorted(self.data.keys())}`.")
            img_id = list(self.data.keys())[0]
        if img_id not in self.data.keys():
            raise KeyError(f"Image id not found in `{sorted(self.data.keys())}`.")

        return img_id

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
            s += f"<p style='text-indent: 25px; margin-top: 0px;'><strong>{layer}</strong>: "
            s += ", ".join(
                f"<em>{dim}</em> ({shape})" for dim, shape in zip(self.data[layer].dims, self.data[layer].shape)
            )
            s += "</p>"
        return s

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[shape={self.shape}, ids={sorted(self.data.keys())}]"

    def __str__(self) -> str:
        return repr(self)
