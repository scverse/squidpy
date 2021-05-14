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
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TYPE_CHECKING,
)
from pathlib import Path
from functools import partial
from itertools import chain
from typing_extensions import Literal
import re
import collections.abc

from scanpy import logging as logg
from anndata import AnnData
from scanpy.plotting.palettes import default_102 as default_palette

from dask import delayed
import numpy as np
import xarray as xr
import dask.array as da

from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage.util import img_as_float
from skimage.transform import rescale

from squidpy._docs import d, inject_docs
from squidpy._utils import singledispatchmethod
from squidpy.im._io import _lazy_load_image, _infer_dimensions
from squidpy.gr._utils import (
    _assert_in_range,
    _assert_positive,
    _assert_non_negative,
    _assert_spatial_basis,
    _assert_non_empty_sequence,
)
from squidpy.im._coords import (
    CropCoords,
    CropPadding,
    _NULL_COORDS,
    _NULL_PADDING,
    TupleSerializer,
    _update_attrs_scale,
    _update_attrs_coords,
)
from squidpy.im._feature_mixin import FeatureMixin
from squidpy._constants._constants import InferDimensions
from squidpy._constants._pkg_constants import Key

FoI_t = Union[int, float]
Pathlike_t = Union[str, Path]
Arraylike_t = Union[np.ndarray, xr.DataArray]
Input_t = Union[Pathlike_t, Arraylike_t, "ImageContainer"]
Interactive = TypeVar("Interactive")  # cannot import because of cyclic dependencies


__all__ = ["ImageContainer"]


def test(var: Union[Iterable[str], None]) -> None:
    if var is None:
        var = ["a", "b", "c"]
    var = np.zeros((len(list(var)), 10))

    print(var.shape)


@d.dedent  # trick to overcome not top-down order
@d.dedent
class ImageContainer(FeatureMixin):
    """
    Container for in memory :class:`numpy.ndarray`/:class:`xarray.DataArray` or on-disk *TIFF*/*JPEG*/*PNG* images.

    Wraps :class:`xarray.Dataset` to store several image layers with the same x and y dimensions in one object.
    Dimensions of stored images are ``(y, x, channels)``. The channel dimension may vary between image layers.

    This class also allows for lazy loading and processing using :mod:`dask`, and is given to all image
    processing functions, along with :class:`anndata.AnnData` instance, if necessary.

    Parameters
    ----------
    %(add_img.parameters)s
    scale
        Scaling factor of the image with respect to the spatial coordinates
        saved in the accompanying :class:`anndata.AnnData`.

    Raises
    ------
    %(add_img.raises)s
    """

    def __init__(
        self,
        img: Optional[Input_t] = None,
        layer: str = "image",
        lazy: bool = True,
        scale: float = 1.0,
        **kwargs: Any,
    ):
        self._data: xr.Dataset = xr.Dataset()
        self._data.attrs[Key.img.coords] = _NULL_COORDS  # can't save None to NetCDF
        self._data.attrs[Key.img.padding] = _NULL_PADDING
        self._data.attrs[Key.img.scale] = scale
        self._data.attrs[Key.img.mask_circle] = False

        if img is not None:
            self.add_img(img, layer=layer, **kwargs)
            if not lazy:
                self.compute()

    @classmethod
    def load(cls, path: Pathlike_t, lazy: bool = True, chunks: Optional[int] = None) -> "ImageContainer":
        """
        Load data from a *Zarr* store.

        Parameters
        ----------
        path
            Path to *Zarr* store.
        lazy
            Whether to use :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`. Only used when ``lazy = True``.

        Returns
        -------
        The loaded container.
        """
        res = cls()
        res.add_img(path, layer="image", chunks=chunks, lazy=lazy)

        return res

    @classmethod
    def concat(
        cls, imgs: Iterable["ImageContainer"], library_ids: Optional[Iterable[Union[str, None]]] = None, **kwargs: Any
    ) -> "ImageContainer":
        """
        Concatenate ``imgs`` in z dimension, returning the concatenated :class:`squidpy.im.ImageContainer`.

        All ``imgs`` need to have the same shape and the same name to be concatenated
        By default, all ``imgs`` need to have the same scale and crop attributes.
        Use ``combine_attrs = "override"`` to relax this.
        Note that this might lead to a mismatch between :class:`ImageContainer`
        and :class:`anndata.AnnData` coordinates.

        Parameters
        ----------
        imgs
            Images that should be concatenated in z
        library_ids
            Name for each image that will be associated to each z dimension.
            Should match the ``library_id`` in the corresponding :class:`anndata.AnnData` object.
            If None, the existing name of the z dimension is used.
        kwargs
            Keyword arguments for :func:`xarray.concat`

        Returns
        -------
        Concatenated :class:`squidpy.img.ImageContainer` with ``imgs`` stackes in z dimension.
        """
        # check that imgs are not already 3d
        for img in imgs:
            if img.data.dims["z"] > 1:
                raise ValueError(
                    f"Found image with {img.data.dims['z']} z dimensions.\
                    Can only concatenate images with 1 z dimension"
                )

        # check library_ids
        if library_ids is None:
            library_ids = [None] * len(list(imgs))
        _library_ids = np.concatenate([img._library_id_list(library_id) for img, library_id in zip(imgs, library_ids)])
        if len(set(_library_ids)) < len(_library_ids):
            raise ValueError(f"Found non-unique library_ids for z dimensions: {_library_ids}.")

        # add library_id to z dim
        prep_imgs = []
        for lid, img in zip(_library_ids, imgs):
            prep_img = img.copy()
            prep_img._data = prep_img.data.assign_coords(z=[lid])
            prep_imgs.append(prep_img)

        # concatenate
        kwargs.setdefault("combine_attrs", "identical")
        res = cls()
        res._data = xr.concat([img.data for img in prep_imgs], dim="z", **kwargs)
        return res

    def save(self, path: Pathlike_t, **kwargs: Any) -> None:
        """
        Save the container into a *Zarr* store.

        Parameters
        ----------
        path
            Path to a *Zarr* store.

        Returns
        -------
        Nothing, just saves the container.
        """
        attrs = self.data.attrs
        try:
            self._data = self.data.load()  # if we're loading lazily and immediately saving
            self.data.attrs = {
                k: (v.to_tuple() if isinstance(v, TupleSerializer) else v) for k, v in self.data.attrs.items()
            }
            self.data.to_zarr(str(path), mode="w", **kwargs, **kwargs)
        finally:
            self.data.attrs = attrs

    def _get_next_image_id(self, layer: str) -> str:
        pat = re.compile(rf"^{layer}_(\d*)$")
        iterator = chain.from_iterable(pat.finditer(k) for k in self.data.keys())
        return f"{layer}_{(max(map(lambda m: int(m.groups()[0]), iterator), default=-1) + 1)}"

    def _get_next_channel_id(self, channel: str) -> str:
        pat = re.compile(rf"^{channel}_(\d*)$")
        iterator = chain.from_iterable(pat.finditer(v.dims[-1]) for v in self.data.values())
        return f"{channel}_{(max(map(lambda m: int(m.groups()[0]), iterator), default=-1) + 1)}"

    def _library_id_list(self, library_id: Optional[Union[Iterable[str], str]] = None) -> List[str]:
        """Get list of library_ids from input or z coords."""
        if library_id is None:
            library_id = self.data.coords["z"].values
        if isinstance(library_id, str) or not isinstance(library_id, collections.abc.Iterable):
            library_id = [library_id]
        return list(library_id)

    @d.get_sections(base="add_img", sections=["Parameters", "Raises"])
    @d.dedent
    @inject_docs(id=InferDimensions)
    def add_img(
        self,
        img: Input_t,
        layer: Optional[str] = None,
        channel_dim: Optional[str] = None,
        library_id: Optional[Union[str, Iterable[str]]] = None,
        lazy: bool = True,
        chunks: Optional[Union[str, Tuple[int, ...]]] = None,
        infer_dimensions: Union[  # TODO: rename(dims)
            Literal["default", "prefer_channels", "prefer_z"], Tuple[str, ...]
        ] = InferDimensions.DEFAULT.s,  # type: ignore[assignment]
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Add a new image to the container.

        Parameters
        ----------
        img
            In memory array or a path to an on-disk image.
        %(img_layer)s
        channel_dim
            Name of the channel dimension.
        library_id
            Name for each Z-dimension of the image. This should correspond to the ``library_id`` in
            :attr:`anndata.AnnData.uns`.
        lazy
            Whether to use :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`. Only used when ``lazy = True``.
        infer_dimensions
            Where to save channel dimension when reading from a file or loading an array. Valid options are:

                - `{id.PREFER_CHANNELS.s!r}` - load the last dimension as channels.
                - `{id.PREFER_Z.s!r}` - load the last dimension as Z-dim.
                - `{id.DEFAULT.s!r}` - same as `{id.PREFER_CHANNELS.s!r}`, but for 4-dim arrays,
                  tries to also load the first dimension as channels iff the last dimension is 1.
                - a tuple of dimension names matching the shape of img, e.g. ('y', 'x', 'channels', 'z').

        copy
            Whether to copy the underlying data if ``img`` is an array.

        Returns
        -------
        Nothing, just adds a new ``layer`` to :attr:`data`.

        Raises
        ------
        ValueError
            If loading from a file/store with an unknown format or if a supplied channel dimension cannot be aligned.
        NotImplementedError
            If loading a specific data type has not been implemented.
        """
        layer = self._get_next_image_id("image") if layer is None else layer
        infer_dimensions: Union[InferDimensions, Tuple[str]] = (  # type: ignore[no-redef]
            InferDimensions(infer_dimensions) if not isinstance(infer_dimensions, tuple) else infer_dimensions
        )
        res: Optional[xr.DataArray] = self._load_img(
            img, chunks=chunks, layer=layer, copy=copy, infer_dimensions=infer_dimensions, **kwargs
        )

        if res is not None:  # not reading a .nc file
            res = res.rename({res.dims[-1]: "channels" if channel_dim is None else str(channel_dim)})

            # add z dim label
            z = res.shape[2]
            default_library_id = [f"library_id_{i}" for i in range(z)]
            library_id = self._library_id_list(default_library_id if library_id is None else library_id)
            if z > 1 and len(library_id) != z:
                raise ValueError(
                    f"img has {z} z dimensions, \
                please specify a list of {z} names for library_id"
                )
            res = res.assign_coords({"z": library_id})

            logg.info(f"{'Overwriting' if layer in self else 'Adding'} image layer `{layer}`")
            overwrite = False
            if layer in self:
                overwrite = True
                prev_img = self.data[layer]
            try:
                self.data[layer] = res
            except ValueError as e:
                if "cannot be aligned" not in str(e) or channel_dim is not None:
                    raise
                channel_dim = self._get_next_channel_id("channels")
                logg.warning(f"Channel dimension cannot be aligned with an existing one, using `{channel_dim}`")

                if TYPE_CHECKING:
                    assert isinstance(res, xr.DataArray)
                self.data[layer] = res.rename({res.dims[-1]: channel_dim})

            # when overwriting, might want to fill nans with the previous values
            # TODO might want to make this optional
            if overwrite:
                if self.data[layer].dims == prev_img.dims:
                    # can only update if dims are the same
                    self.data[layer] = self.data[layer].fillna(prev_img)

            if not lazy:
                self.data[layer].load()

    @singledispatchmethod
    def _load_img(
        self, img: Union[Pathlike_t, Input_t, "ImageContainer"], layer: str, **kwargs: Any
    ) -> Optional[xr.DataArray]:
        if isinstance(img, ImageContainer):
            if layer not in img:
                raise KeyError(f"Image identifier `{layer}` not found in `{img}`.")

            _ = kwargs.pop("infer_dimensions", None)
            return self._load_img(img[layer], **kwargs)

        raise NotImplementedError(f"Loading `{type(img).__name__}` is not yet implemented.")

    @_load_img.register(str)
    @_load_img.register(Path)
    def _(
        self,
        img: Pathlike_t,
        chunks: Optional[int] = None,
        infer_dimensions: InferDimensions = InferDimensions.DEFAULT,  # TODO: type(infer_dims) + rename
        **_: Any,
    ) -> Optional[xr.DataArray]:
        def transform_metadata(data: xr.Dataset) -> xr.Dataset:
            for layer, img in data.items():
                for dim in ("y", "x", "z"):
                    if dim not in img.dims:
                        raise ValueError(f"Expected dimension `{dim}` in layer `{layer}`, found `{img.dims}`.")

            data.attrs[Key.img.coords] = CropCoords.from_tuple(data.attrs.get(Key.img.coords, _NULL_COORDS.to_tuple()))
            data.attrs[Key.img.padding] = CropPadding.from_tuple(
                data.attrs.get(Key.img.padding, _NULL_PADDING.to_tuple())
            )
            if Key.img.mask_circle not in data.attrs:
                data.attrs[Key.img.mask_circle] = False

            if Key.img.scale not in data.attrs:
                data.attrs[Key.img.scale] = 1

            return data

        img = Path(img)
        logg.debug(f"Loading data from `{img}`")

        if not img.exists():
            raise OSError(f"Path `{img}` does not exist.")

        suffix = img.suffix.lower()

        if suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            return _lazy_load_image(img, infer_dimensions=infer_dimensions, chunks=chunks)

        if img.is_dir():
            if len(self._data):
                raise ValueError("Loading data from `Zarr` store is disallowed when the container is not empty.")

            self._data = transform_metadata(xr.open_zarr(str(img), chunks=chunks))
            return

        if suffix in (".nc", ".cdf"):
            if len(self._data):
                raise ValueError("Loading data from `NetCDF` is disallowed when the container is not empty.")

            self._data = transform_metadata(xr.open_dataset(img, chunks=chunks))
            return

        raise ValueError(f"Unknown suffix `{img.suffix}`.")

    @_load_img.register(da.Array)  # type: ignore[no-redef]
    @_load_img.register(np.ndarray)
    def _(
        self, img: np.ndarray, copy: bool = True, infer_dimensions: InferDimensions = InferDimensions.DEFAULT, **_: Any
    ) -> xr.DataArray:  # TODO: type(infer_dims), rename
        logg.debug(f"Loading data `numpy.array` of shape `{img.shape}`")

        img = img.copy() if copy else img
        shape, dims, _, _ = _infer_dimensions(img, infer_dimensions=infer_dimensions)

        return xr.DataArray(img.reshape(shape), dims=dims).transpose("y", "x", "z", "channels")

    @_load_img.register(xr.DataArray)  # type: ignore[no-redef]
    def _(
        self,
        img: xr.DataArray,
        copy: bool = True,
        infer_dimensions: InferDimensions = InferDimensions.DEFAULT,  # TODO: type(infer_dims), rename
        **_: Any,
    ) -> xr.DataArray:
        logg.debug(f"Loading data `xarray.DataArray` of shape `{img.shape}`")

        img = img.copy() if copy else img
        if not ("y" in img.dims and "x" in img.dims and "z" in img.dims):
            _, dims, _, axes = _infer_dimensions(img, infer_dimensions=infer_dimensions)
            img = img.expand_dims([f"tmp_{a}" for a in axes], axis=axes)

            logg.warning(f"Unable to find `y` or `x` dimension in `{img.dims}`. Renaming to `{dims}`")
            img = img.rename(dict(zip(img.dims, dims)))

        return img.transpose("y", "x", "z", ...)

    @d.get_sections(base="crop_corner", sections=["Parameters", "Returns"])
    @d.dedent
    def crop_corner(
        self,
        y: FoI_t,
        x: FoI_t,
        size: Optional[Union[FoI_t, Tuple[FoI_t, FoI_t]]] = None,
        scale: float = 1.0,
        cval: Union[int, float] = 0,
        mask_circle: bool = False,
        preserve_dtypes: bool = True,
        library_id: Optional[str] = None,
    ) -> "ImageContainer":
        """
        Extract a crop from the upper-left corner.

        Parameters
        ----------
        %(yx)s
        %(size)s
        scale
            Rescale the crop using :func:`skimage.transform.rescale`.
        cval
            Fill value to use if ``mask_circle = True`` or if crop goes out of the image boundary.
        mask_circle
            Whether to mask out values that are not within a circle defined by this crop.
            Only available if ``size`` defines a square.
        preserve_dtypes
            Whether to preserver the data types of underlying :class:`xarray.DataArray`, even if ``cval``
            is of different type.
        library_id
            Name of the z dim that is cropped. If None, all z dims are cropped

        Returns
        -------
        The cropped image of size ``size * scale``.

        Raises
        ------
        ValueError
            If the crop would completely lie outside of the image or if ``mask_circle = True`` and
            ``size`` does not define a square.

        Notes
        -----
        If ``preserve_dtypes = True`` but ``cval`` cannot be safely cast, ``cval`` will be set to 0.
        """
        self._assert_not_empty()
        y, x = self._convert_to_pixel_space((y, x))

        size = self._get_size(size)
        size = self._convert_to_pixel_space(size)

        ys, xs = size
        _assert_positive(ys, name="height")
        _assert_positive(xs, name="width")
        _assert_positive(scale, name="scale")

        orig = CropCoords(x0=x, y0=y, x1=x + xs, y1=y + ys)

        ymin, xmin = self.shape
        coords = CropCoords(
            x0=min(max(x, 0), xmin), y0=min(max(y, 0), ymin), x1=min(x + xs, xmin), y1=min(y + ys, ymin)
        )

        if not coords.dy:
            raise ValueError("Height of the crop is empty.")
        if not coords.dx:
            raise ValueError("Width of the crop is empty.")

        crop = self.data.isel(x=slice(coords.x0, coords.x1), y=slice(coords.y0, coords.y1)).copy(deep=False)
        if len(crop.z) > 1:
            crop = crop.sel(z=self._library_id_list(library_id))
        crop.attrs = _update_attrs_coords(crop.attrs, coords)

        if orig != coords:
            padding = orig - coords

            # because padding does not change dtype by itself
            for key, arr in crop.items():
                if preserve_dtypes:
                    if not np.can_cast(cval, arr.dtype, casting="safe"):
                        cval = 0
                else:
                    crop[key] = crop[key].astype(np.dtype(type(cval)), copy=False)
            crop = crop.pad(
                y=(padding.y_pre, padding.y_post),
                x=(padding.x_pre, padding.x_post),
                mode="constant",
                constant_values=cval,
            )
            # TODO does padding need to be updated as well?
            crop.attrs[Key.img.padding] = padding
        else:
            crop.attrs[Key.img.padding] = _NULL_PADDING
        return self._from_dataset(
            self._post_process(
                data=crop, scale=scale, cval=cval, mask_circle=mask_circle, preserve_dtypes=preserve_dtypes
            )
        )

    def _post_process(
        self,
        data: xr.Dataset,
        scale: FoI_t = 1,
        cval: FoI_t = 0,
        mask_circle: bool = False,
        preserve_dtypes: bool = True,
        **_: Any,
    ) -> xr.Dataset:
        def _rescale(arr: xr.DataArray) -> xr.DataArray:
            # once skimage==0.19.0, multichannel is deprecated
            scaling_fn = partial(
                rescale, scale=[scale, scale, 1], preserve_range=True, order=1, multichannel=True, cval=cval
            )
            dtype = arr.dtype

            if isinstance(arr.data, da.Array):
                shape = np.maximum(np.round(scale * np.asarray(arr.shape)), 1)
                # TODO: Z-dim
                shape[-1] = arr.shape[-1]
                shape[-2] = arr.shape[-2]
                return xr.DataArray(
                    da.from_delayed(delayed(lambda arr: scaling_fn(arr).astype(dtype))(arr), shape=shape, dtype=dtype),
                    dims=arr.dims,
                )

            return xr.DataArray(scaling_fn(arr).astype(dtype), dims=arr.dims)

        if scale != 1:
            attrs = data.attrs
            data = data.map(_rescale)
            data.attrs = _update_attrs_scale(attrs, scale)

        if mask_circle:
            if data.dims["y"] != data.dims["x"]:
                raise ValueError(
                    f"Masking circle is only available for square crops, "
                    f"found crop of shape `{(data.dims['y'], data.dims['x'])}`."
                )
            c = data.x.shape[0] // 2
            data = data.where((data.x - c) ** 2 + (data.y - c) ** 2 <= c ** 2, other=cval)
            data.attrs[Key.img.mask_circle] = True

        if preserve_dtypes:
            for key, arr in self.data.items():
                data[key] = data[key].astype(arr.dtype, copy=False)

        return data

    @d.dedent
    def crop_center(
        self,
        y: FoI_t,
        x: FoI_t,
        radius: Union[FoI_t, Tuple[FoI_t, FoI_t]],
        **kwargs: Any,
    ) -> "ImageContainer":
        """
        Extract a circular crop.

        The extracted crop will have shape ``(radius[0] * 2 + 1, radius[1] * 2 + 1)``.

        Parameters
        ----------
        %(yx)s
        radius
            Radius along the ``height`` and ``width`` dimensions, respectively.
        kwargs
            Keyword arguments for :meth:`crop_corner`.

        Returns
        -------
        %(crop_corner.returns)s
        """
        y, x = self._convert_to_pixel_space((y, x))
        _assert_in_range(y, 0, self.shape[0], name="height")
        _assert_in_range(x, 0, self.shape[1], name="width")

        if not isinstance(radius, Iterable):
            radius = (radius, radius)

        (yr, xr) = self._convert_to_pixel_space(radius)
        _assert_non_negative(yr, name="radius height")
        _assert_non_negative(xr, name="radius width")

        return self.crop_corner(  # type: ignore[no-any-return]
            y=y - yr, x=x - xr, size=(yr * 2 + 1, xr * 2 + 1), **kwargs
        )

    @d.dedent
    def generate_equal_crops(
        self,
        size: Optional[Union[FoI_t, Tuple[FoI_t, FoI_t]]] = None,
        as_array: Union[str, bool] = False,
        **kwargs: Any,
    ) -> Union[Iterator["ImageContainer"], Iterator[Dict[str, np.ndarray]]]:
        """
        Decompose image into equally sized crops.

        Parameters
        ----------
        %(size)s
        %(as_array)s
        kwargs
            Keyword arguments for :meth:`crop_corner`.

        Yields
        ------
        The crops, whose type depends on ``as_array``.

        Notes
        -----
        Crops going outside out of the image boundary are padded with ``cval``.
        """
        self._assert_not_empty()

        size = self._get_size(size)
        size = self._convert_to_pixel_space(size)

        y, x = self.shape
        ys, xs = size
        _assert_in_range(ys, 0, y, name="height")
        _assert_in_range(xs, 0, x, name="width")

        unique_ycoord = np.arange(start=0, stop=(y // ys + (y % ys != 0)) * ys, step=ys)
        unique_xcoord = np.arange(start=0, stop=(x // xs + (x % xs != 0)) * xs, step=xs)

        ycoords = np.repeat(unique_ycoord, len(unique_xcoord))
        xcoords = np.tile(unique_xcoord, len(unique_ycoord))

        for y, x in zip(ycoords, xcoords):
            yield self.crop_corner(y=y, x=x, size=(ys, xs), **kwargs)._maybe_as_array(as_array)

    @d.dedent
    def generate_spot_crops(
        self,
        adata: AnnData,
        library_id: Optional[Union[Iterable[str], str]] = None,
        spatial_key: str = Key.obsm.spatial,
        spot_scale: float = 1.0,
        obs_names: Optional[Iterable[Any]] = None,
        as_array: Union[str, bool] = False,
        return_obs: bool = False,
        **kwargs: Any,
    ) -> Union[
        Iterator["ImageContainer"],
        Iterator[np.ndarray],
        Iterator[Tuple[np.ndarray, ...]],
        Iterator[Dict[str, np.ndarray]],
    ]:
        """
        Iterate over :attr:`adata.obs_names` and extract crops.

        Implemented for 10X spatial datasets.
        For z-stacks, the specified `library_id` or list of `library_id` s needs to match the name of the z dimension.
        Always extracts 2D crops from the specified z dimension.

        Parameters
        ----------
        %(adata)s
        library_id
            Key in :attr:`anndata.AnnData.uns` ``['{spatial_key}']`` used to get the spot diameter.
            If `None`, there should either only exist one entry in  :attr:`anndata.AnnData.uns` ``['{spatial_key}']``,
            or a column :attr:`anndata.AnnData.obs` ``['library_id']`` containing the mapping from obs to library ids.
        %(spatial_key)s
        spot_scale
            Scaling factor for the spot diameter. Larger values mean more context.
        obs_names
            Observations from :attr:`adata.obs_names` for which to generate the crops. If `None`, all names are used.
        %(as_array)s
        return_obs
            Whether to also yield names from ``obs_names``.
        kwargs
            Keyword arguments for :meth:`crop_center`.

        Yields
        ------
        If ``return_obs = True``, yields a :class:`tuple` ``(crop, obs_name)``. Otherwise, yields just the crops.
        The type of the crops depends on ``as_array``.
        """
        self._assert_not_empty()
        _assert_positive(spot_scale, name="scale")
        _assert_spatial_basis(adata, spatial_key)

        # limit to obs_names
        if obs_names is None:
            obs_names = adata.obs_names
        obs_names = _assert_non_empty_sequence(obs_names, name="observations")
        adata = adata[obs_names, :]

        scale = self.data.attrs.get(Key.img.scale, 1)
        spatial = adata.obsm[spatial_key][:, :2]

        # get library_id for each obs
        if len(self.data.z) > 1 and library_id is None:
            # assign from adata
            # TODO raise nicer error message if not exists
            if "library_id" not in adata.obs:
                raise ValueError(
                    "No 'library_id' column in adata.obs. Pass a list of `library_id`s mapping obs to z dimensions."
                )
            obs_library_ids = adata.obs["library_id"]
        else:
            if len(self.data.z) > 1:
                logg.warning(
                    f"ImageContainer has {len(self.data.z)} z dimensions, will use library_id {library_id} for all."
                )
            if isinstance(library_id, str) or library_id is None:
                library_id = Key.uns.library_id(adata, spatial_key=spatial_key, library_id=library_id)
                obs_library_ids = [library_id] * len(adata.obs)
            else:
                if len(list(library_id)) != len(adata.obs):
                    raise ValueError(
                        f"specified {len(list(library_id))} library_ids but have {len(adata.obs)} entries in adata.obs."
                    )
                obs_library_ids = list(library_id)

        for i, obs in enumerate(adata.obs_names):
            # get spot diameter of current obs (might be different library ids)
            diameter = Key.uns.spot_diameter(adata, spatial_key=spatial_key, library_id=obs_library_ids[i]) * scale
            radius = int(round(diameter // 2 * spot_scale))

            # get coords in image pixel space from original space
            y = int(spatial[i][1] * scale)
            x = int(spatial[i][0] * scale)

            # if CropCoords exist, need to offset y and x
            if self.data.attrs.get(Key.img.coords, _NULL_COORDS) != _NULL_COORDS:
                y = int(y - self.data.attrs[Key.img.coords].y0)
                x = int(x - self.data.attrs[Key.img.coords].x0)
            crop = self.crop_center(y=y, x=x, radius=radius, library_id=obs_library_ids[i], **kwargs)
            crop.data.attrs[Key.img.obs] = obs
            crop = crop._maybe_as_array(as_array)

            yield (crop, obs) if return_obs else crop

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
            Requested image shape as ``(height, width)``. If `None`, it is automatically determined from ``crops``.

        Returns
        -------
        Re-assembled image from ``crops``.

        Raises
        ------
        ValueError
            If crop metadata was not found or if the requested ``shape`` is smaller than required by ``crops``.
        """
        if not len(crops):
            raise ValueError("No crops were supplied.")

        keys = set(crops[0].data.keys())
        scales = set()
        dy, dx = -1, -1

        for crop in crops:
            if set(crop.data.keys()) != keys:
                raise KeyError(f"Expected to find `{sorted(keys)}` keys, found `{sorted(crop.data.keys())}`.")

            coord = crop.data.attrs.get(Key.img.coords, None)
            if coord is None:
                raise ValueError("Crop does not have coordinate metadata.")
            if coord == _NULL_COORDS:
                raise ValueError(f"Null coordinates detected `{coord}`.")

            scales.add(crop.data.attrs.get(Key.img.scale, None))
            dy, dx = max(dy, coord.y0 + coord.dy), max(dx, coord.x0 + coord.dx)

        scales.discard(None)
        if len(scales) != 1:
            raise ValueError(f"Unable to uncrop images of different scales: `{sorted((scales))}`.")
        scale, *_ = scales

        if shape is None:
            shape = (dy, dx)
        # can be float because coords can be scaled
        shape = tuple(map(int, shape))  # type: ignore[assignment]
        if len(shape) != 2:
            raise ValueError(f"Expected `shape` to be of length `2`, found `{len(shape)}`.")
        if shape < (dy, dx):
            raise ValueError(f"Requested final image shape `{shape}`, but minimal is `({dy}, {dx})`.")

        # create resulting dataset
        dataset = xr.Dataset()
        dataset.attrs[Key.img.scale] = scale

        for key in keys:
            img = crop.data[key]
            # get shape for this DataArray
            dataset[key] = xr.DataArray(
                np.zeros(shape + tuple(img.shape[2:]), dtype=img.dtype), dims=img.dims, coords=img.coords
            )
            # fill data with crops
            for crop in crops:
                coord = crop.data.attrs[Key.img.coords]
                padding = crop.data.attrs.get(Key.img.padding, _NULL_PADDING)  # maybe warn
                dataset[key][coord.slice] = crop[key][coord.to_image_coordinates(padding=padding).slice]

        return cls._from_dataset(dataset)

    @d.dedent
    def show(
        self,
        layer: Optional[str] = None,
        channel: Optional[int] = None,
        channelwise: bool = False,
        library_id: Optional[str] = None,
        segmentation_layer: Optional[str] = None,
        segmentation_alpha: float = 0.75,
        ax: Optional[mpl.axes.Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[Pathlike_t] = None,
        **kwargs: Any,
    ) -> None:
        """
        Show an image within this container.

        Parameters
        ----------
        %(img_layer)s
        channel
            Channel to plot. If `None`, use all channels for plotting.
        channelwise
            Whether to plot each channel separately or not.
        library_id
            Name of z dim to plot. In `None`, plot all z dims as separate images.
        segmentation_layer
            Segmentation layer to plot.
        segmentation_alpha
            Alpha value for ``segmentation_layer``.
        ax
            Optional :mod:`matplotlib` ax where to plot the image. If not `None`, ``save``, ``figsize`` and
            ``dpi`` have no effect.
        %(plotting)s
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        %(plotting_returns)s

        Raises
        ------
        ValueError
            If  ``as_mask = True`` and the image layer has more than 1 channel and ``channelwise = False`` or
            if number fo supplied axes is different than the number of channels if ``channelwise = True``.
        """
        from squidpy.pl._utils import save_fig

        layer = self._get_layer(layer)
        arr: xr.DataArray = self.data[layer]
        n_channels = arr.shape[-1]
        channelwise = channelwise and n_channels > 1
        library_ids = list(self.data.coords["z"].values) if library_id is None else [library_id]

        if channel is not None:
            arr = arr[{arr.dims[-1]: channel}]
        arr = arr.sel(z=library_ids)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(
                ncols=n_channels if channelwise else 1,
                nrows=len(library_ids),
                figsize=(8, 8) if figsize is None else figsize,
                dpi=dpi,
                tight_layout=True,
                squeeze=False,
            )
            ax = ax.T
        else:
            # TODO: check if already a sequence of axes + maybe reshape
            ax = np.asarray([[ax]])

        if channelwise and len(ax) != n_channels:
            raise ValueError(f"Expected `{n_channels}` axes, found `{len(ax)}`.")

        if segmentation_layer is not None:
            seg_arr = self.data[segmentation_layer]
            if not seg_arr.attrs.get("segmentation", False):
                raise TypeError(f"Expected layer `{segmentation_layer!r}` to be marked as segmentation layer.")
            if not np.issubdtype(seg_arr.dtype, np.integer):
                raise TypeError(
                    f"Expected segmentation layer `{segmentation_layer!r}` to be of integer type, "
                    f"found `{seg_arr.dtype}`."
                )

            # TODO: handle seg along Z-stack
            seg_arr = seg_arr.values[..., 0, 0]  # force dask computation here
            seg_cmap = np.array(default_palette, dtype=object)[np.arange(np.max(seg_arr)) % len(default_palette)]
            seg_cmap[0] = "#00000000"  # don't plot black background
            seg_cmap = ListedColormap(seg_cmap)
        else:
            seg_arr, seg_cmap = None, None

        for c, row in enumerate(ax):
            for z, ax_ in enumerate(row):
                if channelwise:
                    img, title = arr[..., z, c], f"{layer}:{c}, library_id:{library_ids[z]}"
                else:
                    img, title = arr[:, :, z, ...], f"{layer}, library_id:{library_ids[z]}"

                ax_.imshow(img_as_float(img.values, force_copy=False), **kwargs)
                if seg_arr is not None:
                    ax_.imshow(
                        seg_arr,
                        cmap=seg_cmap,
                        interpolation="nearest",  # avoid artifacts
                        alpha=segmentation_alpha,
                        **{k: v for k, v in kwargs.items() if k not in ("cmap", "interpolation")},
                    )

                ax_.set_title(title)
                ax_.set_axis_off()

            if save and fig is not None:
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
        blending: Literal["opaque", "translucent", "additive"] = "opaque",
        symbol: Literal["disc", "square"] = "disc",
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
        symbol
            Symbol to use for the spots. Valid options are:

                - `'disc'` - circle.
                - `'square'`  - square.

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
            symbol=symbol,
        ).show()

    @d.dedent
    def apply(
        self,
        func: Callable[..., np.ndarray],
        layer: Optional[str] = None,
        new_layer: Optional[str] = None,
        channel: Optional[int] = None,
        library_id: Optional[str] = None,
        lazy: bool = False,
        chunks: Optional[Union[str]] = None,
        copy: bool = True,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> Optional["ImageContainer"]:
        """
        Apply a function to a layer within this container.

        Parameters
        ----------
        func
            A function which takes a :class:`numpy.ndarray` as input and produces an image-like output.
        %(img_layer)s
        new_layer
            Name of the new layer. If `None` and ``copy = False``, overwrites the data in ``layer``.
        channel
            Apply ``func`` only over a specific ``channel``. If `None`, use all channels.
        chunks
            Chunk size for :mod:`dask`. If `None`, don't use :mod:`dask`.
        %(copy_cont)s
        fn_kwargs
            Keyword arguments for ``func``.
        kwargs
            Keyword arguments for :func:`dask.array.map_blocks` or :func:`dask.array.map_overlap`, depending whether
            ``'depth'`` is present in ``map_kwargs``. Only used when ``chunks != None``.

        Returns
        -------
        If ``copy = True``, returns a new container with ``layer``.

        Raises
        ------
        ValueError
            If the ``func`` returns 0 or 1 dimensional array.
        """
        layer = self._get_layer(layer)
        if new_layer is None:
            new_layer = layer
        arr = self[layer]
        channel_dim = arr.dims[-1]
        dims = arr.dims

        if channel is not None:
            arr = arr[{channel_dim: channel}]
        if library_id is not None:
            arr = arr.sel(z=library_id)

        if chunks is not None:
            arr = da.asarray(arr.data).rechunk(chunks)
            res = (
                da.map_overlap(func, arr, **fn_kwargs, **kwargs)
                if "depth" in kwargs
                else da.map_blocks(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
            )
        else:
            res = func(arr.data, **fn_kwargs)

        if res.ndim in (0, 1):
            # TODO: allow volumetric segmentation?
            raise ValueError(f"Expected 2 or 3 dimensional array, found `{res.ndim}`.")
        elif res.ndim == 2:
            res = res[..., np.newaxis]
        # TODO: handle z-dim

        if chunks is not None:
            arr = da.asarray(arr.data).rechunk(chunks)
            res = (
                da.map_overlap(func, arr, **fn_kwargs, **kwargs)
                if "depth" in kwargs
                else da.map_blocks(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
            )
        else:
            res = func(arr.data, **fn_kwargs)

        if res.ndim in (0, 1):
            # TODO: allow volumetric segmentation?
            raise ValueError(f"Expected 2 or 3 dimensional array, found `{res.ndim}`.")
        elif res.ndim == 2:
            res = res[..., np.newaxis, np.newaxis]
        elif res.ndim == 3:  # assume that 3 dims are y, x, channels
            # is the 3rd dim channel or z?
            if channel is not None:  # its z, because channel was removed
                res = res[..., np.newaxis]
            else:
                res = res[:, :, np.newaxis, :]

        # TODO: handle z-dim

        library_id = library_id if library_id is not None else self[layer].coords["z"].values
        if copy:
            cont = ImageContainer(
                res,
                layer=new_layer,
                channel_dim=channel_dim,
                copy=True,
                lazy=lazy,
                infer_dimensions=dims,
                library_id=library_id,
            )
            cont.data.attrs = self.data.attrs.copy()

            return cont

        self.add_img(
            res,
            layer=new_layer,
            lazy=lazy,
            copy=new_layer != layer,
            channel_dim=f"{channel_dim}:{res.shape[-1]}" if self[layer].shape[-1] != res.shape[-1] else channel_dim,
            infer_dimensions=dims,
            library_id=library_id,
        )

    @d.dedent
    def subset(self, adata: AnnData, spatial_key: str = "spatial") -> AnnData:
        """
        Subset :class:`anndata.AnnData` object based on this image.

        Parameters
        ----------
        %(adata)s
        %(spatial_key)s

        Returns
        -------
        The subsetted copy :class:`anndata.AnnData` object.
        """
        return self._subset(adata, spatial_key=spatial_key, adjust_interactive=False)

    def _subset(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        adjust_interactive: bool = False,
    ) -> AnnData:
        _assert_spatial_basis(adata, spatial_key)

        adata = adata.copy()
        s = self.data.attrs.get(Key.img.scale, 1)
        coordinates = adata.obsm[spatial_key]

        if s != 1:
            # update coordinates with image scale
            coordinates = coordinates * s

        c: CropCoords = self.data.attrs.get(Key.img.coords, _NULL_COORDS)
        p: CropPadding = self.data.attrs.get(Key.img.padding, _NULL_PADDING)
        if c != _NULL_COORDS:
            mask = (
                (coordinates[:, 0] >= c.x0)
                & (coordinates[:, 0] <= c.x1)
                & (coordinates[:, 1] >= c.y0)
                & (coordinates[:, 1] <= c.y1)
            )

            adata = adata[mask, :].copy()
            # shift and scale appropriately for interactive viewer
            if adjust_interactive:
                adata.obsm[spatial_key] = coordinates[mask, :]
                adata.obsm[spatial_key][:, 0] -= c.x0 - p.x_pre
                adata.obsm[spatial_key][:, 1] -= c.y0 - p.y_pre
        elif adjust_interactive and s != 1:
            adata.obsm[spatial_key] = coordinates

        return adata

    def rename(self, old: str, new: str) -> "ImageContainer":
        """
        Rename a layer.

        Parameters
        ----------
        old
            Name of the layer to rename.
        new
            New name.

        Returns
        -------
        Modifies and returns self.
        """
        self._data = self.data.rename_vars({old: new})
        return self

    def compute(self) -> "ImageContainer":
        """Trigger lazy computation inplace."""
        self.data.load()
        return self

    @property
    def data(self) -> xr.Dataset:
        """Underlying :class:`xarray.Dataset`."""
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape ``(y, x)``."""
        if not len(self):
            return 0, 0
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

    @classmethod
    def _from_dataset(cls, data: xr.Dataset, deep: Optional[bool] = None) -> "ImageContainer":
        """
        Utility function used for initialization.

        Parameters
        ----------
        data
            The :class:`xarray.Dataset` to use.
        deep
            If `None`, don't copy the ``data``. If `True`, make a deep copy of the data, otherwise, make a shallow copy.

        Returns
        -------
        The newly created container.
        """  # noqa: D401
        res = cls()
        res._data = data if deep is None else data.copy(deep=deep)
        res._data.attrs.setdefault(Key.img.coords, _NULL_COORDS)  # can't save None to NetCDF
        res._data.attrs.setdefault(Key.img.padding, _NULL_PADDING)
        res._data.attrs.setdefault(Key.img.scale, 1.0)
        res._data.attrs.setdefault(Key.img.mask_circle, False)
        return res

    def _maybe_as_array(
        self, as_array: Union[str, Sequence[str], bool] = False
    ) -> Union["ImageContainer", Dict[str, np.ndarray], np.ndarray, Tuple[np.ndarray, ...]]:
        res = self
        if as_array:
            if len(res.data.z) == 1:  # do not return z dim if have only one
                res = {key: res[key].isel(z=0).values for key in res}  # type: ignore[assignment]
            else:
                res = {key: res[key].values for key in res}  # type: ignore[assignment]
        # this is just for convenience for DL iterators
        if isinstance(as_array, str):
            res = res[as_array]
        elif isinstance(as_array, Sequence):
            res = tuple(res[key] for key in as_array)  # type: ignore[assignment]

        return res

    def _get_layer(self, layer: Optional[str]) -> str:
        self._assert_not_empty()

        if layer is None:
            if len(self) > 1:
                raise ValueError(
                    f"Unable to determine which `layer` to use. "
                    f"Please supply one from `{sorted(self.data.keys())}`."
                )
            layer = list(self)[0]
        if layer not in self:
            raise KeyError(f"Image layer `{layer}` not found in `{sorted(self)}`")

        return layer

    def _assert_not_empty(self) -> None:
        if not len(self):
            raise ValueError("The container is empty.")

    def _get_size(self, size: Optional[Union[FoI_t, Tuple[Optional[FoI_t], Optional[FoI_t]]]]) -> Tuple[FoI_t, FoI_t]:
        if size is None:
            size = (None, None)
        if not isinstance(size, Iterable):
            size = (size, size)

        res = list(size)
        if size[0] is None:
            res[0] = self.shape[0]

        if size[1] is None:
            res[1] = self.shape[1]

        return tuple(res)  # type: ignore[return-value]

    def _convert_to_pixel_space(self, size: Tuple[FoI_t, FoI_t]) -> Tuple[int, int]:
        y, x = size
        if isinstance(y, float):
            _assert_in_range(y, 0, 1, name="y")
            y = int(self.shape[0] * y)
        if isinstance(x, float):
            _assert_in_range(x, 0, 1, name="x")
            x = int(self.shape[1] * x)

        return y, x

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.data.keys()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> xr.DataArray:
        return self.data[key]

    def __setitem__(self, key: str, value: Union[np.ndarray, xr.DataArray, da.Array]) -> None:
        if not isinstance(value, (np.ndarray, xr.DataArray, da.Array)):
            raise NotImplementedError(f"Adding `{type(value).__name__}` is not yet implemented.")
        self.add_img(value, layer=key, channel_dim=None, copy=True)

    def _ipython_key_completions_(self) -> Iterable[str]:
        return sorted(map(str, self.data.keys()))

    def __copy__(self) -> "ImageContainer":
        return type(self)._from_dataset(self.data, deep=False)

    def __deepcopy__(self, memodict: Mapping[str, Any] = MappingProxyType({})) -> "ImageContainer":
        return type(self)._from_dataset(self.data, deep=True)

    def _repr_html_(self) -> str:
        if not len(self):
            return f"{self.__class__.__name__} object with 0 layers"

        inflection = "" if len(self) <= 1 else "s"
        s = f"{self.__class__.__name__} object with {len(self.data.keys())} layer{inflection}:"
        style = "text-indent: 25px; margin-top: 0px; margin-bottom: 0px;"

        for i, layer in enumerate(self.data.keys()):
            s += f"<p style={style!r}><strong>{layer}</strong>: "
            s += ", ".join(
                f"<em>{dim}</em> ({shape})" for dim, shape in zip(self.data[layer].dims, self.data[layer].shape)
            )
            s += "</p>"
            if i == 9 and i < len(self) - 1:  # show only first 10 layers
                s += f"<p style={style!r}>and {len(self) - i  - 1} more...</p>"
                break

        return s

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[shape={self.shape}, layers={sorted(self.data.keys())}]"

    def __str__(self) -> str:
        return repr(self)
