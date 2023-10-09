from __future__ import annotations

import re
from copy import copy, deepcopy
from functools import partial
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Union,
)

import dask.array as da
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import validators
import xarray as xr
from anndata import AnnData
from dask import delayed
from matplotlib.colors import ListedColormap
from scanpy import logging as logg
from scanpy.plotting.palettes import default_102 as default_palette
from skimage.transform import rescale
from skimage.util import img_as_float

from squidpy._constants._constants import InferDimensions
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, singledispatchmethod
from squidpy.gr._utils import (
    _assert_in_range,
    _assert_non_empty_sequence,
    _assert_non_negative,
    _assert_positive,
    _assert_spatial_basis,
)
from squidpy.im._coords import (
    _NULL_COORDS,
    _NULL_PADDING,
    CropCoords,
    CropPadding,
    TupleSerializer,
    _update_attrs_coords,
    _update_attrs_scale,
)
from squidpy.im._feature_mixin import FeatureMixin
from squidpy.im._io import _assert_dims_present, _infer_dimensions, _lazy_load_image

FoI_t = Union[int, float]
Pathlike_t = Union[str, Path]
Arraylike_t = Union[NDArrayA, xr.DataArray]
InferDims_t = Union[Literal["default", "prefer_channels", "prefer_z"], Sequence[str]]
Input_t = Union[Pathlike_t, Arraylike_t, "ImageContainer"]
Interactive = TypeVar("Interactive")  # cannot import because of cyclic dependencies
_ERROR_NOTIMPLEMENTED_LIBID = f"It seems there are multiple `library_id` in `adata.uns[{Key.uns.spatial!r}]`.\n \
                                Loading multiple images is not implemented (yet), please specify a `library_id`."

__all__ = ["ImageContainer"]


@d.dedent  # trick to overcome not top-down order
@d.dedent
class ImageContainer(FeatureMixin):
    """
    Container for in memory arrays or on-disk images.

    Wraps :class:`xarray.Dataset` to store several image layers with the same `x`, `y` and `z` dimensions in one object.
    Dimensions of stored images are ``(y, x, z, channels)``. The channel dimension may vary between image layers.

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
        img: Input_t | None = None,
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
    def concat(
        cls,
        imgs: Iterable[ImageContainer],
        library_ids: Sequence[str | None] | None = None,
        combine_attrs: str = "identical",
        **kwargs: Any,
    ) -> ImageContainer:
        """
        Concatenate ``imgs`` in Z-dimension.

        All ``imgs`` need to have the same shape and the same name to be concatenated.

        Parameters
        ----------
        imgs
            Images that should be concatenated in Z-dimension.
        library_ids
            Name for each image that will be associated to each Z-dimension. This should match the ``library_id``
            in the corresponding :class:`anndata.AnnData` object.
            If `None`, the existing name of the Z-dimension is used for each image.
        combine_attrs
            How to combine attributes of ``imgs``. By default, all ``imgs`` need to have the same scale
            and crop attributes. Use ``combine_attrs = 'override'`` to relax this requirement.
            This might lead to a mismatch between :class:`ImageContainer` and :class:`anndata.AnnData` coordinates.
        kwargs
            Keyword arguments for :func:`xarray.concat`.

        Returns
        -------
        Concatenated :class:`squidpy.img.ImageContainer` with ``imgs`` stacks in Z-dimension.

        Raises
        ------
        ValueError
            If any of the ``imgs`` have more than 1 Z-dimension or if ``library_ids`` are not unique.
        """
        # check that imgs are not already 3d
        imgs = list(imgs)
        for img in imgs:
            if img.data.dims["z"] > 1:
                raise ValueError(
                    f"Currently, can concatenate only images with 1 Z-dimension, found `{img.data.dims['z']}`."
                )

        # check library_ids
        if library_ids is None:
            library_ids = [None] * len(imgs)
        if len(library_ids) != len(imgs):
            raise ValueError(f"Expected library ids to be of length `{len(imgs)}`, found `{len(library_ids)}`.")

        _library_ids = np.concatenate(
            [img._get_library_ids(library_id, allow_new=True) for img, library_id in zip(imgs, library_ids)]
        )
        if len(set(_library_ids)) != len(_library_ids):
            raise ValueError(f"Found non-unique library ids `{list(_library_ids)}`.")

        # add library_id to z dim
        prep_imgs = []
        for lid, img in zip(_library_ids, imgs):
            prep_img = img.copy()
            prep_img._data = prep_img.data.assign_coords(z=[lid])
            prep_imgs.append(prep_img)

        return cls._from_dataset(
            xr.concat([img.data for img in prep_imgs], dim="z", combine_attrs=combine_attrs, **kwargs)
        )

    @classmethod
    def load(cls, path: Pathlike_t, lazy: bool = True, chunks: int | None = None) -> ImageContainer:
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
        res.add_img(path, layer="image", chunks=chunks, lazy=True)

        return res if lazy else res.compute()

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

    @d.get_sections(base="add_img", sections=["Parameters", "Raises"])
    @d.dedent
    @inject_docs(id=InferDimensions)
    def add_img(
        self,
        img: Input_t,
        layer: str | None = None,
        dims: InferDims_t = InferDimensions.DEFAULT.s,
        library_id: str | Sequence[str] | None = None,
        lazy: bool = True,
        chunks: str | tuple[int, ...] | None = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Add a new image to the container.

        Parameters
        ----------
        img
            In-memory 2, 3 or 4-dimensional array, a URL to a *Zarr* store (ending in *.zarr*),
            or a path to an on-disk image.
        %(img_layer)s
        dims
            Where to save channel dimension when reading from a file or loading an array. Valid options are:

                - `{id.CHANNELS_LAST.s!r}` - load the last non-spatial dimension as channels.
                - `{id.Z_LAST.s!r}` - load the last non-spatial dimension as Z-dimension.
                - `{id.DEFAULT.s!r}` - same as `{id.CHANNELS_LAST.s!r}`, but for 4-dimensional arrays,
                  tries to also load the first dimension as channels if the last non-spatial dimension is 1.
                - a sequence of dimension names matching the shape of ``img``, e.g. ``('y', 'x', 'z', 'channels')``.
                  `'y'`, `'x'` and `'z'` must always be present.
        library_id
            Name for each Z-dimension of the image. This should correspond to the ``library_id``
            in :attr:`anndata.AnnData.uns`.
        lazy
            Whether to use :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`. Only used when ``lazy = True``.
        copy
            Whether to copy the underlying data if ``img`` is an in-memory array.

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
        dims: InferDimensions | Sequence[str] = (  # type: ignore[no-redef]
            InferDimensions(dims) if isinstance(dims, str) else dims
        )
        res: xr.DataArray | None = self._load_img(img, chunks=chunks, layer=layer, copy=copy, dims=dims, **kwargs)

        if res is not None:
            library_id = self._get_library_ids(library_id, res, allow_new=not len(self))
            try:
                res = res.assign_coords({"z": library_id})
            except ValueError as e:
                if "conflicting sizes for dimension 'z'" not in str(e):
                    raise
                # at this point, we know the container is not empty
                raise ValueError(
                    f"Expected image to have `{len(self.library_ids)}` Z-dimension(s), found `{res.sizes['z']}`."
                ) from None

            if TYPE_CHECKING:
                assert isinstance(res, xr.DataArray)
            logg.info(f"{'Overwriting' if layer in self else 'Adding'} image layer `{layer}`")
            try:
                self.data[layer] = res
            except ValueError as e:
                c_dim = res.dims[-1]
                if f"cannot reindex or align along dimension {str(c_dim)!r}" not in str(e):
                    raise
                channel_dim = self._get_next_channel_id(res)
                logg.warning(f"Channel dimension cannot be aligned with an existing one, using `{channel_dim}`")

                self.data[layer] = res.rename({res.dims[-1]: channel_dim})

            if not lazy:
                self.compute(layer)

    @singledispatchmethod
    def _load_img(self, img: Pathlike_t | Input_t | ImageContainer, layer: str, **kwargs: Any) -> xr.DataArray | None:
        if isinstance(img, ImageContainer):
            if layer not in img:
                raise KeyError(f"Image identifier `{layer}` not found in `{img}`.")

            _ = kwargs.pop("dims", None)
            return self._load_img(img[layer], **kwargs)

        raise NotImplementedError(f"Loading `{type(img).__name__}` is not yet implemented.")

    @_load_img.register(str)
    @_load_img.register(Path)
    def _(
        self,
        img_path: Pathlike_t,
        chunks: int | None = None,
        dims: InferDimensions | tuple[str, ...] = InferDimensions.DEFAULT,
        **_: Any,
    ) -> xr.DataArray | None:
        def transform_metadata(data: xr.Dataset) -> xr.Dataset:
            for key, img in data.items():
                if len(img.dims) != 4:
                    data[key] = img = img.expand_dims({"z": 1}, axis=-2)  # assume only channel dim is present
                _assert_dims_present(img.dims, include_z=True)

            data.attrs[Key.img.coords] = CropCoords.from_tuple(data.attrs.get(Key.img.coords, _NULL_COORDS.to_tuple()))
            data.attrs[Key.img.padding] = CropPadding.from_tuple(
                data.attrs.get(Key.img.padding, _NULL_PADDING.to_tuple())
            )
            data.attrs.setdefault(Key.img.mask_circle, False)
            data.attrs.setdefault(Key.img.scale, 1)

            return data

        img_path = str(img_path)
        is_url, suffix = validators.url(img_path), Path(img_path).suffix.lower()
        logg.debug(f"Loading data from `{img_path}`")

        if not is_url and not Path(img_path).exists():
            raise OSError(f"Path `{img_path}` does not exist.")

        if suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            return _lazy_load_image(img_path, dims=dims, chunks=chunks)

        if suffix == ".zarr" or Path(img_path).is_dir():  # can also be a URL
            if len(self._data):
                raise ValueError("Loading data from `Zarr` store is disallowed when the container is not empty.")
            self._data = transform_metadata(xr.open_zarr(img_path, chunks=chunks))
        elif suffix in (".nc", ".cdf"):
            if len(self._data):
                raise ValueError("Loading data from `NetCDF` is disallowed when the container is not empty.")

            self._data = transform_metadata(xr.open_dataset(img_path, chunks=chunks))
        else:
            raise ValueError(f"Unable to handle path `{img_path}`.")

    @_load_img.register(da.Array)
    @_load_img.register(np.ndarray)
    def _(
        self,
        img: NDArrayA,
        copy: bool = True,
        dims: InferDimensions | tuple[str, ...] = InferDimensions.DEFAULT,
        **_: Any,
    ) -> xr.DataArray:
        logg.debug(f"Loading `numpy.array` of shape `{img.shape}`")

        return self._load_img(xr.DataArray(img), copy=copy, dims=dims, warn=False)

    @_load_img.register(xr.DataArray)
    def _(
        self,
        img: xr.DataArray,
        copy: bool = True,
        warn: bool = True,
        dims: InferDimensions | tuple[str, ...] = InferDimensions.DEFAULT,
        **_: Any,
    ) -> xr.DataArray:
        logg.debug(f"Loading `xarray.DataArray` of shape `{img.shape}`")

        img = img.copy() if copy else img
        if not ("y" in img.dims and "x" in img.dims and "z" in img.dims):
            _, dims, _, expand_axes = _infer_dimensions(img, infer_dimensions=dims)
            if TYPE_CHECKING:
                assert isinstance(dims, Iterable)
            if warn:
                logg.warning(f"Unable to find `y`, `x` or `z` dimension in `{img.dims}`. Renaming to `{dims}`")
            # `axes` is always of length 0, 1 or 2
            if len(expand_axes):
                dimnames = ("z", "channels") if len(expand_axes) == 2 else (("channels",) if "z" in dims else ("z",))
                img = img.expand_dims([d for _, d in zip(expand_axes, dimnames)], axis=expand_axes)
            img = img.rename(dict(zip(img.dims, dims)))

        return img.transpose("y", "x", "z", ...)

    @classmethod
    @d.dedent
    def from_adata(
        cls,
        adata: AnnData,
        img_key: str | None = None,
        library_id: Sequence[str] | str | None = None,
        spatial_key: str = Key.uns.spatial,
        **kwargs: Any,
    ) -> ImageContainer:
        """
        Load an image from :mod:`anndata` object.

        Parameters
        ----------
        %(adata)s
        img_key
            Key in :attr:`anndata.AnnData.uns` ``['{spatial_key}']['{library_id}']['images']``.
            If `None`, the first key found is used.
        library_id
            Key in :attr:`anndata.AnnData.uns` ``['{spatial_key}']`` specifying which library to access.
        spatial_key
            Key in :attr:`anndata.AnnData.uns` where spatial metadata is stored.
        kwargs
            Keyword arguments for :class:`squidpy.im.ImageContainer`.

        Returns
        -------
        The image container.
        """
        library_id = Key.uns.library_id(adata, spatial_key, library_id)
        if not isinstance(library_id, str):
            raise NotImplementedError(_ERROR_NOTIMPLEMENTED_LIBID)
        spatial_data = adata.uns[spatial_key][library_id]
        if img_key is None:
            try:
                img_key = next(k for k in spatial_data.get("images", []))
            except StopIteration:
                raise KeyError(f"No images found in `adata.uns[{spatial_key!r}][{library_id!r}]['images']`") from None

        img: NDArrayA | None = spatial_data.get("images", {}).get(img_key, None)
        if img is None:
            raise KeyError(
                f"Unable to find the image in `adata.uns[{spatial_key!r}][{library_id!r}]['images'][{img_key!r}]`."
            )

        scale = spatial_data.get("scalefactors", {}).get(f"tissue_{img_key}_scalef", None)
        if scale is None and "scale" not in kwargs:
            logg.warning(
                f"Unable to determine the scale factor from "
                f"`adata.uns[{spatial_key!r}][{library_id!r}]['scalefactors']['tissue_{img_key}_scalef']`, "
                f"using `1.0`. Consider specifying it manually as `scale=...`"
            )
            scale = 1.0
        kwargs.setdefault("scale", scale)

        return cls(img, layer=img_key, library_id=library_id, **kwargs)

    @d.get_sections(base="crop_corner", sections=["Parameters", "Returns"])
    @d.dedent
    def crop_corner(
        self,
        y: FoI_t,
        x: FoI_t,
        size: FoI_t | tuple[FoI_t, FoI_t] | None = None,
        library_id: str | None = None,
        scale: float = 1.0,
        cval: int | float = 0,
        mask_circle: bool = False,
        preserve_dtypes: bool = True,
    ) -> ImageContainer:
        """
        Extract a crop from the upper-left corner.

        Parameters
        ----------
        %(yx)s
        %(size)s
        library_id
            Name of the Z-dimension to be cropped. If `None`, all Z-dimensions are cropped.
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
            crop = crop.sel(z=self._get_library_ids(library_id))
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
            scaling_fn = partial(
                rescale,
                preserve_range=True,
                scale=[scale, scale, 1],
                order=1,
                channel_axis=-1,
                cval=cval,
            )
            dtype = arr.dtype

            if isinstance(arr.data, da.Array):
                shape = np.maximum(np.round(scale * np.asarray(arr.shape)), 1)
                shape[-1] = arr.shape[-1]
                shape[-2] = arr.shape[-2]
                return xr.DataArray(
                    da.from_delayed(delayed(lambda arr: scaling_fn(arr).astype(dtype))(arr), shape=shape, dtype=dtype),
                    dims=arr.dims,
                )

            return xr.DataArray(scaling_fn(arr).astype(dtype), dims=arr.dims)

        if scale != 1:
            attrs = data.attrs
            library_ids = data.coords["z"]
            data = data.map(_rescale).assign_coords({"z": library_ids})
            data.attrs = _update_attrs_scale(attrs, scale)

        if mask_circle:
            if data.dims["y"] != data.dims["x"]:
                raise ValueError(
                    f"Masking circle is only available for square crops, "
                    f"found crop of shape `{(data.dims['y'], data.dims['x'])}`."
                )
            c = data.x.shape[0] // 2
            # manually reassign coordinates
            library_ids = data.coords["z"]
            data = data.where((data.x - c) ** 2 + (data.y - c) ** 2 <= c**2, other=cval).assign_coords(
                {"z": library_ids}
            )
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
        radius: FoI_t | tuple[FoI_t, FoI_t],
        **kwargs: Any,
    ) -> ImageContainer:
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
        size: FoI_t | tuple[FoI_t, FoI_t] | None = None,
        as_array: str | bool = False,
        squeeze: bool = True,
        **kwargs: Any,
    ) -> Iterator[ImageContainer] | Iterator[dict[str, NDArrayA]]:
        """
        Decompose image into equally sized crops.

        Parameters
        ----------
        %(size)s
        %(as_array)s
        squeeze
            Remove singleton dimensions from the results if ``as_array = True``.
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
            yield self.crop_corner(y=y, x=x, size=(ys, xs), **kwargs)._maybe_as_array(
                as_array, squeeze=squeeze, lazy=True
            )

    @d.dedent
    def generate_spot_crops(
        self,
        adata: AnnData,
        spatial_key: str = Key.obsm.spatial,
        library_id: Sequence[str] | str | None = None,
        spot_diameter_key: str = "spot_diameter_fullres",
        spot_scale: float = 1.0,
        obs_names: Iterable[Any] | None = None,
        as_array: str | bool = False,
        squeeze: bool = True,
        return_obs: bool = False,
        **kwargs: Any,
    ) -> Iterator[ImageContainer] | Iterator[NDArrayA] | Iterator[tuple[NDArrayA, ...]] | Iterator[dict[str, NDArrayA]]:
        """
        Iterate over :attr:`anndata.AnnData.obs_names` and extract crops.

        Implemented for 10X spatial datasets.
        For Z-stacks, the specified ``library_id`` or list of ``library_id`` need to match the name of the Z-dimension.
        Always extracts 2D crops from the specified Z-dimension.

        Parameters
        ----------
        %(adata)s
        %(spatial_key)s
        %(img_library_id)s
        spot_diameter_key
            Key in :attr:`anndata.AnnData.uns` ``['{spatial_key}']['{library_id}']['scalefactors']``
            where the spot diameter is stored.
        spot_scale
            Scaling factor for the spot diameter. Larger values mean more context.
        obs_names
            Observations from :attr:`anndata.AnnData.obs_names` for which to generate the crops.
            If `None`, all observations are used.
        %(as_array)s
        squeeze
            Remove singleton dimensions from the results if ``as_array = True``.
        return_obs
            Whether to also yield names from ``obs_names``.
        kwargs
            Keyword arguments for :meth:`crop_center`.

        Yields
        ------
        If ``return_obs = True``, yields a :class:`tuple` ``(crop, obs_name)``. Otherwise, yields just the crops.
        The type of the crops depends on ``as_array`` and the number of dimensions on ``squeeze``.
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

        if library_id is None:
            try:
                library_id = Key.uns.library_id(adata, spatial_key=spatial_key, library_id=None)
                if not isinstance(library_id, str):
                    raise NotImplementedError(_ERROR_NOTIMPLEMENTED_LIBID)
                obs_library_ids = [library_id] * adata.n_obs
            except ValueError as e:
                if "Unable to determine which library id to use" in str(e):
                    raise ValueError(
                        str(e)
                        + " Or specify a key in `adata.obs` containing a mapping from observations to library ids."
                    ) from e
                else:
                    raise e
        else:
            try:
                obs_library_ids = adata.obs[library_id]
            except KeyError:
                logg.debug(
                    f"Unable to find library ids in `adata.obs[{library_id!r}]`. "
                    f"Trying in `adata.uns[{spatial_key!r}]`"
                )
                library_id = Key.uns.library_id(adata, spatial_key=spatial_key, library_id=library_id)
                if not isinstance(library_id, str):
                    raise NotImplementedError(_ERROR_NOTIMPLEMENTED_LIBID) from None
                obs_library_ids = [library_id] * adata.n_obs

        lids = set(obs_library_ids)
        if len(self.data.z) > 1 and len(lids) == 1:
            logg.warning(
                f"ImageContainer has `{len(self.data.z)}` Z-dimensions, using library id `{next(iter(lids))}` for all"
            )

        if adata.n_obs != len(obs_library_ids):
            raise ValueError(f"Expected library ids to be of length `{adata.n_obs}`, found `{len(obs_library_ids)}`.")

        for i, (obs, lid) in enumerate(zip(adata.obs_names, obs_library_ids)):
            # get spot diameter of current obs (might be different library ids)
            diameter = (
                Key.uns.spot_diameter(
                    adata, spatial_key=spatial_key, library_id=lid, spot_diameter_key=spot_diameter_key
                )
                * scale
            )
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
            crop = crop._maybe_as_array(as_array, squeeze=squeeze, lazy=False)

            yield (crop, obs) if return_obs else crop

    @classmethod
    @d.get_sections(base="uncrop", sections=["Parameters", "Returns"])
    def uncrop(
        cls,
        crops: list[ImageContainer],
        shape: tuple[int, int] | None = None,
    ) -> ImageContainer:
        """
        Re-assemble image from crops and their positions.

        Fills remaining positions with zeros.

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
            raise ValueError(f"Unable to uncrop images of different scales `{sorted(scales)}`.")
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
        layer: str | None = None,
        library_id: str | Sequence[str] | None = None,
        channel: int | Sequence[int] | None = None,
        channelwise: bool = False,
        segmentation_layer: str | None = None,
        segmentation_alpha: float = 0.75,
        transpose: bool | None = None,
        ax: mpl.axes.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        save: Pathlike_t | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Show an image within this container.

        Parameters
        ----------
        %(img_layer)s
        library_id
            Name of Z-dimension to plot. In `None`, plot all Z-dimensions as separate images.
        channel
            Channels to plot. If `None`, use all channels.
        channelwise
            Whether to plot each channel separately or not.
        segmentation_layer
            Segmentation layer to plot over each ax.
        segmentation_alpha
            Alpha value for ``segmentation_layer``.
        transpose
            Whether to plot Z-dimensions in columns or in rows. If `None`, it will be set to ``not channelwise``.
        ax
            Optional :mod:`matplotlib` axes where to plot the image.
            If not `None`, ``save``, ``figsize`` and ``dpi`` have no effect.
        %(plotting)s
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        %(plotting_returns)s

        Raises
        ------
        ValueError
            If number of supplied axes is different than the number of requested Z-dimensions or channels.
        """
        from squidpy.pl._utils import save_fig

        layer = self._get_layer(layer)
        arr: xr.DataArray = self[layer]

        library_ids = self._get_library_ids(library_id)
        arr = arr.sel(z=library_ids)

        if channel is not None:
            channel = np.asarray([channel]).ravel()  # type: ignore[assignment]
            if not len(channel):  # type: ignore[arg-type]
                raise ValueError("No channels have been selected.")
            arr = arr[{arr.dims[-1]: channel}]
        else:
            channel = np.arange(arr.shape[-1])  # type: ignore[assignment]
        if TYPE_CHECKING:
            assert isinstance(channel, Sequence)

        n_channels = arr.shape[-1]
        if n_channels not in (1, 3, 4) and not channelwise:
            logg.warning(f"Unable to plot image with `{n_channels}`. Setting `channelwise=True`")
            channelwise = True

        if transpose is None:
            transpose = not channelwise

        fig = None
        nrows, ncols = len(library_ids), (n_channels if channelwise else 1)
        if transpose:
            nrows, ncols = ncols, nrows
        if ax is None:
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(8, 8) if figsize is None else figsize,
                dpi=dpi,
                tight_layout=True,
                squeeze=False,
            )
        elif isinstance(ax, mpl.axes.Axes):
            ax = np.array([ax])

        ax = np.asarray(ax)
        try:
            ax = ax.reshape(nrows, ncols)
        except ValueError:
            raise ValueError(f"Expected `ax` to be of shape `{(nrows, ncols)}`, found `{ax.shape}`.") from None

        if segmentation_layer is not None:
            seg_arr = self[segmentation_layer].sel(z=library_ids)
            if not seg_arr.attrs.get("segmentation", False):
                raise TypeError(f"Expected layer `{segmentation_layer!r}` to be marked as segmentation layer.")
            if not np.issubdtype(seg_arr.dtype, np.integer):
                raise TypeError(
                    f"Expected segmentation layer `{segmentation_layer!r}` to be of integer type, "
                    f"found `{seg_arr.dtype}`."
                )

            seg_arr = seg_arr.values
            seg_cmap = np.array(default_palette, dtype=object)[np.arange(np.max(seg_arr)) % len(default_palette)]
            seg_cmap[0] = "#00000000"  # transparent background
            seg_cmap = ListedColormap(seg_cmap)
        else:
            seg_arr, seg_cmap = None, None

        for z, row in enumerate(ax):
            for c, ax_ in enumerate(row):
                if transpose:
                    z, c = c, z

                title = layer
                if channelwise:
                    img = arr[..., z, c]
                    title += f":{channel[c]}"
                else:
                    img = arr[..., z, :]
                if len(self.data.coords["z"]) > 1:
                    title += f", library_id:{library_ids[z]}"

                ax_.imshow(img_as_float(img.values, force_copy=False), **kwargs)
                if seg_arr is not None:
                    ax_.imshow(
                        seg_arr[:, :, z, ...],
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
        library_key: str | None = None,
        library_id: str | Sequence[str] | None = None,
        cmap: str = "viridis",
        palette: str | None = None,
        blending: Literal["opaque", "translucent", "additive"] = "opaque",
        symbol: Literal["disc", "square"] = "disc",
        key_added: str = "shapes",
    ) -> Interactive:  # type: ignore[type-var]
        """
        Launch :mod:`napari` viewer.

        Parameters
        ----------
        %(adata)s
        %(spatial_key)s
        library_key
            Key in :attr:`adata.AnnData.obs` specifying mapping between observations and library ids.
            Required if the container has more than 1 Z-dimension.
        library_id
            Subset of library ids to visualize. If `None`, visualize all library ids.
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
            Key where to store :class:`napari.layers.Shapes`, which can be exported by pressing `SHIFT-E`:

                - :attr:`anndata.AnnData.obs` ``['{layer_name}_{key_added}']`` - boolean mask containing the selected
                  cells.
                - :attr:`anndata.AnnData.uns` ``['{layer_name}_{key_added}']['meshes']`` - list of :class:`numpy.array`,
                  defining a mesh in the spatial coordinates.

            See :mod:`napari`'s `tutorial <https://napari.org/howtos/layers/shapes.html>`_ for more
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
            library_key=library_key,
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
        func: Callable[..., NDArrayA] | Mapping[str, Callable[..., NDArrayA]],
        layer: str | None = None,
        new_layer: str | None = None,
        channel: int | None = None,
        lazy: bool = False,
        chunks: str | tuple[int, int] | None = None,
        copy: bool = True,
        drop: bool = True,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> ImageContainer | None:
        """
        Apply a function to a layer within this container.

        For each Z-dimension a different function can be defined, using its ``library_id`` name.
        For not mentioned ``library_id``'s the identity function is applied.

        Parameters
        ----------
        func
            A function or a mapping of ``{'{library_id}': function}`` which takes a :class:`numpy.ndarray` as input
            and produces an image-like output.
        %(img_layer)s
        new_layer
            Name of the new layer. If `None` and ``copy = False``, overwrites the data in ``layer``.
        channel
            Apply ``func`` only over a specific ``channel``. If `None`, use all channels.
        chunks
            Chunk size for :mod:`dask`. If `None`, don't use :mod:`dask`.
        %(copy_cont)s
        drop
            Whether to drop Z-dimensions that were not selected by ``func``. Only used when ``copy = True``.
        fn_kwargs
            Keyword arguments for ``func``.
        kwargs
            Keyword arguments for :func:`dask.array.map_overlap` or :func:`dask.array.map_blocks`, depending whether
            ``depth`` is present in ``fn_kwargs``. Only used when ``chunks != None``.
            Use ``depth`` to control boundary artifacts if ``func`` requires data from neighboring chunks,
            by default, ``boundary = 'reflect`` is used.

        Returns
        -------
        If ``copy = True``, returns a new container with ``layer``.

        Raises
        ------
        ValueError
            If the ``func`` returns 0 or 1 dimensional array.
        """

        def apply_func(func: Callable[..., NDArrayA], arr: xr.DataArray) -> NDArrayA | da.Array:
            if chunks is None:
                return func(arr.data, **fn_kwargs)
            arr = da.asarray(arr.data).rechunk(chunks)
            return (
                da.map_overlap(func, arr, **fn_kwargs, **kwargs)
                if "depth" in kwargs
                else da.map_blocks(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
            )

        if "depth" in kwargs:
            kwargs.setdefault("boundary", "reflect")

        layer = self._get_layer(layer)
        if new_layer is None:
            new_layer = layer

        arr = self[layer]
        library_ids = list(arr.coords["z"].values)
        dims, channel_dim = arr.dims, arr.dims[-1]

        if channel is not None:
            arr = arr[{channel_dim: channel}]

        if callable(func):
            res = apply_func(func, arr)
            new_library_ids = library_ids
        else:
            res = {}
            noop_library_ids = [] if copy and drop else list(set(library_ids) - set(func.keys()))
            for key, fn in func.items():
                res[key] = apply_func(fn, arr.sel(z=key))
            for key in noop_library_ids:
                res[key] = arr.sel(z=key).data

            new_library_ids = [lid for lid in library_ids if lid in res]
            try:
                res = da.stack([res[lid] for lid in new_library_ids], axis=2)
            except ValueError as e:
                if not len(noop_library_ids) or "must have the same shape" not in str(e):
                    # processing functions returned wrong shape
                    raise ValueError(
                        "Unable to stack an array because functions returned arrays of different shapes."
                    ) from e

                # funcs might have changed channel dims, replace noops with 0
                logg.warning(
                    f"Function changed the number of channels, cannot use identity "
                    f"for library ids `{noop_library_ids}`. Replacing with 0"
                )
                # TODO(michalk8): once (or if) Z-dim is not fixed, always drop ids
                tmp = next(iter(res.values()))
                for lid in noop_library_ids:
                    res[lid] = (np.zeros_like if chunks is None else da.zeros_like)(tmp)

                res = da.stack([res[lid] for lid in new_library_ids], axis=2)

        if res.ndim == 2:  # assume that dims are y, x
            res = res[..., np.newaxis]
        if res.ndim == 3:  # assume dims are y, x, z (changing of z dim is not supported)
            res = res[..., np.newaxis]
        if res.ndim != 4:
            raise ValueError(f"Expected `2`, `3` or `4` dimensional array, found `{res.ndim}`.")

        if copy:
            cont = ImageContainer(
                res,
                layer=new_layer,
                copy=True,
                lazy=lazy,
                dims=dims,
                library_id=new_library_ids,
            )
            cont.data.attrs = self.data.attrs.copy()
            return cont

        self.add_img(
            res,
            layer=new_layer,
            lazy=lazy,
            copy=new_layer != layer,
            dims=dims,
            library_id=new_library_ids,
        )

    @d.dedent
    def subset(self, adata: AnnData, spatial_key: str = Key.obsm.spatial, copy: bool = False) -> AnnData:
        """
        Subset :class:`anndata.AnnData` using this container.

        Useful when this container is a crop of the original image.

        Parameters
        ----------
        %(adata)s
        %(spatial_key)s
        copy
            Whether to return a copy of ``adata``.

        Returns
        -------
        Subset of :class:`anndata.AnnData`.
        """
        c: CropCoords = self.data.attrs.get(Key.img.coords, _NULL_COORDS)
        if c == _NULL_COORDS:  # not a crop
            return adata.copy() if copy else adata

        _assert_spatial_basis(adata, spatial_key)
        coordinates = adata.obsm[spatial_key]
        coordinates = coordinates * self.data.attrs.get(Key.img.scale, 1)

        mask = (
            (coordinates[:, 0] >= c.x0)
            & (coordinates[:, 0] <= c.x1)
            & (coordinates[:, 1] >= c.y0)
            & (coordinates[:, 1] <= c.y1)
        )

        return adata[mask, :].copy() if copy else adata[mask, :]

    def rename(self, old: str, new: str) -> ImageContainer:
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

    def compute(self, layer: str | None = None) -> ImageContainer:
        """
        Trigger lazy computation in-place.

        Parameters
        ----------
        layer
            Layer which to compute. If `None`, compute all layers.

        Returns
        -------
        Modifies and returns self.
        """
        if layer is None:
            self.data.load()
        else:
            self[layer].load()
        return self

    @property
    def library_ids(self) -> list[str]:
        """Library ids."""
        try:
            return list(map(str, self.data.coords["z"].values))
        except KeyError:
            return []

    @library_ids.setter
    def library_ids(self, library_ids: str | Sequence[str] | Mapping[str, str]) -> None:
        """Set library ids."""
        if isinstance(library_ids, Mapping):
            library_ids = [str(library_ids.get(lid, lid)) for lid in self.library_ids]
        elif isinstance(library_ids, str):
            library_ids = (library_ids,)

        library_ids = list(map(str, library_ids))
        if len(set(library_ids)) != len(library_ids):
            raise ValueError(f"Remapped library ids must be unique, found `{library_ids}`.")
        self._data = self.data.assign_coords({"z": library_ids})

    @property
    def data(self) -> xr.Dataset:
        """Underlying :class:`xarray.Dataset`."""
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape ``(y, x)``."""
        if not len(self):
            return 0, 0
        return self.data.dims["y"], self.data.dims["x"]

    def copy(self, deep: bool = False) -> ImageContainer:
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
    def _from_dataset(cls, data: xr.Dataset, deep: bool | None = None) -> ImageContainer:
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
        self,
        as_array: str | Sequence[str] | bool = False,
        squeeze: bool = True,
        lazy: bool = True,
    ) -> ImageContainer | dict[str, NDArrayA] | NDArrayA | tuple[NDArrayA, ...]:
        res = self
        if as_array:
            # do not trigger dask computation
            res = {key: (res[key].data if lazy else res[key].values) for key in res}  # type: ignore[assignment]
            if squeeze:
                axis = (2,) if len(self.data.z) == 1 else ()
                res = {
                    k: v.squeeze(axis=axis + ((3,) if v.shape[-1] == 1 else ()))
                    for k, v in res.items()  # type: ignore[assignment,attr-defined]
                }
        # this is just for convenience for DL iterators
        if isinstance(as_array, str):
            res = res[as_array]
        elif isinstance(as_array, Sequence):
            res = tuple(res[key] for key in as_array)  # type: ignore[assignment]

        if lazy:
            return res
        return res.compute() if isinstance(res, ImageContainer) else res

    def _get_next_image_id(self, layer: str) -> str:
        pat = re.compile(rf"^{layer}_(\d*)$")
        iterator = chain.from_iterable(pat.finditer(k) for k in self.data.keys())
        return f"{layer}_{(max((int(m.groups()[0]) for m in iterator), default=-1) + 1)}"

    def _get_next_channel_id(self, channel: str | xr.DataArray) -> str:
        if isinstance(channel, xr.DataArray):
            channel, *_ = (str(dim) for dim in channel.dims if dim not in ("y", "x", "z"))

        pat = re.compile(rf"^{channel}_(\d*)$")
        iterator = chain.from_iterable(pat.finditer(v.dims[-1]) for v in self.data.values())
        return f"{channel}_{(max((int(m.groups()[0]) for m in iterator), default=-1) + 1)}"

    def _get_library_id(self, library_id: str | None = None) -> str:
        self._assert_not_empty()

        if library_id is None:
            if len(self.library_ids) > 1:
                raise ValueError(
                    f"Unable to determine which library id to use. Please supply one from `{self.library_ids}`."
                )
            library_id = self.library_ids[0]

        if library_id not in self.library_ids:
            raise KeyError(f"Library id `{library_id}` not found in `{self.library_ids}`.")

        return library_id

    def _get_library_ids(
        self,
        library_id: str | Sequence[str] | None = None,
        arr: xr.DataArray | None = None,
        allow_new: bool = False,
    ) -> list[str]:
        """
        Get library ids.

        Parameters
        ----------
        library_id
            Requested library ids.
        arr
            If the current container is empty, try getting the library ids from the ``arr``.
        allow_new
            If `True`, don't check if the returned library ids are present in the non-empty container.
            This is set to `True` only in :meth:`concat` to allow for remapping.

        Returns
        -------
        The library ids.
        """
        if library_id is None:
            if len(self):
                library_id = self.library_ids
            elif isinstance(arr, xr.DataArray):
                try:
                    library_id = list(arr.coords["z"].values)
                except (KeyError, AttributeError) as e:
                    logg.warning(f"Unable to retrieve library ids, reason `{e}`. Using default names")
                    # at this point, it should have Z-dim
                    library_id = [str(i) for i in range(arr.sizes["z"])]
            else:
                raise ValueError("Please specify the number of library ids if the container is empty.")

        if isinstance(library_id, str):
            library_id = [library_id]
        if not isinstance(library_id, Iterable):
            raise TypeError(f"Expected library ids to be `iterable`, found `{type(library_id).__name__!r}`.")

        res = list(map(str, library_id))
        if not len(res):
            raise ValueError("No library ids have been selected.")

        if not allow_new and len(self) and not (set(res) & set(self.library_ids)):
            raise ValueError(f"Invalid library ids have been selected `{res}`. Valid options are `{self.library_ids}`.")

        return res

    def _get_layer(self, layer: str | None) -> str:
        self._assert_not_empty()

        if layer is None:
            if len(self) > 1:
                raise ValueError(
                    f"Unable to determine which layer to use. Please supply one from `{sorted(self.data.keys())}`."
                )
            layer = list(self)[0]
        if layer not in self:
            raise KeyError(f"Image layer `{layer}` not found in `{sorted(self)}`.")

        return layer

    def _assert_not_empty(self) -> None:
        if not len(self):
            raise ValueError("The container is empty.")

    def _get_size(self, size: FoI_t | tuple[FoI_t | None, FoI_t | None] | None) -> tuple[FoI_t, FoI_t]:
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

    def _convert_to_pixel_space(self, size: tuple[FoI_t, FoI_t]) -> tuple[int, int]:
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

    def __setitem__(self, key: str, value: NDArrayA | xr.DataArray | da.Array) -> None:
        if not isinstance(value, (np.ndarray, xr.DataArray, da.Array)):
            raise NotImplementedError(f"Adding `{type(value).__name__}` is not yet implemented.")
        self.add_img(value, layer=key, copy=True)

    def _ipython_key_completions_(self) -> Iterable[str]:
        return sorted(map(str, self.data.keys()))

    def __copy__(self) -> ImageContainer:
        return type(self)._from_dataset(self.data, deep=False)

    def __deepcopy__(self, memodict: Mapping[str, Any] = MappingProxyType({})) -> ImageContainer:
        return type(self)._from_dataset(self.data, deep=True)

    def _repr_html_(self) -> str:
        import html

        if not len(self):
            return f"{self.__class__.__name__} object with 0 layers"

        inflection = "" if len(self) <= 1 else "s"
        s = f"{self.__class__.__name__} object with {len(self.data.keys())} layer{inflection}:"
        style = "text-indent: 25px; margin-top: 0px; margin-bottom: 0px;"

        for i, layer in enumerate(self.data.keys()):
            s += f"<p style={style!r}><strong>{html.escape(str(layer))}</strong>: "
            s += ", ".join(
                f"<em>{html.escape(str(dim))}</em> ({shape})"
                for dim, shape in zip(self.data[layer].dims, self.data[layer].shape)
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
