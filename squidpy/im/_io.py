from __future__ import annotations

from pathlib import Path
from typing import (
    Mapping,
    Union,  # noqa: F401
)

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from PIL import Image
from scanpy import logging as logg
from skimage.io import imread
from tifffile import TiffFile

from squidpy._constants._constants import InferDimensions
from squidpy._docs import inject_docs
from squidpy._utils import NDArrayA


def _assert_dims_present(dims: tuple[str, ...], include_z: bool = True) -> None:
    missing_dims = ({"y", "x", "z"} if include_z else {"y", "x"}) - set(dims)
    if missing_dims:
        raise ValueError(f"Expected to find `{sorted(missing_dims)}` dimension(s) in `{dims}`.")


# modification of `skimage`'s `pil_to_ndarray`:
# https://github.com/scikit-image/scikit-image/blob/main/skimage/io/_plugins/pil_plugin.py#L55
def _infer_shape_dtype(fname: str) -> tuple[tuple[int, ...], np.dtype]:  # type: ignore[type-arg]
    def _palette_is_grayscale(pil_image: Image.Image) -> bool:
        # get palette as an array with R, G, B columns
        palette = np.asarray(pil_image.getpalette()).reshape((256, 3))
        # Not all palette colors are used; unused colors have junk values.
        start, stop = pil_image.getextrema()
        valid_palette = palette[start : stop + 1]
        # Image is grayscale if channel differences (R - G and G - B)
        # are all zero.
        return np.allclose(np.diff(valid_palette), 0)

    if fname.endswith(".tif") or fname.endswith(".tiff"):
        image = TiffFile(fname)
        try:
            return tuple(image.shaped_metadata[0]["shape"]), np.dtype(image.pages[0].dtype)
        except (IndexError, KeyError, TypeError):
            return (len(image.pages),) + image.pages[0].shape, np.dtype(image.pages[0].dtype)

    image = Image.open(fname)
    n_frames = getattr(image, "n_frames", 1)
    shape: tuple[int, ...] = (n_frames,) + image.size[::-1]

    if image.mode == "P":
        if _palette_is_grayscale(image):
            return shape, np.dtype("uint8")

        if image.format == "PNG" and "transparency" in image.info:
            return shape + (4,), np.dtype("uint8")  # converted to RGBA

        return shape + (3,), np.dtype("uint8")  # RGB
    if image.mode in ("1", "L"):
        return shape, np.dtype("uint8")  # L

    if "A" in image.mode:
        return shape + (4,), np.dtype("uint8")  # RGBA

    if image.mode == "CMYK":
        return shape + (3,), np.dtype("uint8")  # RGB

    if image.mode.startswith("I;16"):
        dtype = ">u2" if image.mode.endswith("B") else "<u2"
        if "S" in image.mode:
            dtype = dtype.replace("u", "i")

        return shape, np.dtype(dtype)

    if image.mode in ("RGB", "HSV", "LAB"):
        return shape + (3,), np.dtype("uint8")
    if image.mode == "F":
        return shape, np.dtype("float32")
    if image.mode == "I":
        return shape, np.dtype("int32")

    raise ValueError(f"Unable to infer image dtype for image mode `{image.mode}`.")


def _get_image_shape_dtype(fname: str) -> tuple[tuple[int, ...], np.dtype]:  # type: ignore[type-arg]
    try:
        return _infer_shape_dtype(fname)
    except Image.UnidentifiedImageError as e:
        logg.warning(e)
        return _infer_shape_dtype(fname)
    except Image.DecompressionBombError as e:
        logg.warning(e)
        old_max_image_pixels = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            return _infer_shape_dtype(fname)
        finally:
            Image.MAX_IMAGE_PIXELS = old_max_image_pixels


@inject_docs(id=InferDimensions)
def _infer_dimensions(
    obj: NDArrayA | xr.DataArray | str,
    infer_dimensions: InferDimensions | tuple[str, ...] = InferDimensions.DEFAULT,
) -> tuple[tuple[int, ...], tuple[str, ...], np.dtype, tuple[int, ...]]:  # type: ignore[type-arg]
    """
    Infer dimension names of an array.

    Parameters
    ----------
    obj
        Path to an image or an array.
    infer_dimensions
        Policy that determines how to name the dimensions. Valid options are:

            - `{id.CHANNELS_LAST.s!r}` - load `channels` dimension as `channels`.
            - `{id.Z_LAST.s!r}` - load `z` dimension as `channels`.
            - `{id.DEFAULT.s!r}` - only matters if the number of dimensions is `3` or `4`.
              If `z` dimension is `1`, load it as `z`.
              Otherwise, if `channels` dimension is `1`, load `z` dimension (now larger than `1`) as `channels`.
              Otherwise, load `z` dimension as `z` and `channels` as `channels`.

        If specified as :class:`tuple`, its length must be the same as the shape of ``obj``.

        The following assumptions are made when determining the dimension names:

            - two largest dimensions are assumed to be `y` and `x`, in this order.
            - `z` dimension comes always before `channel` dimension.
            - for `3` dimensional arrays, the last dimension is `channels`,
              in other cases, it will dealt with as if it were `z` dimension.

    Returns
    -------
    Tuple of the following:

        - :class:`tuple` of 4 :class:`int` describing the shape.
        - :class:`tuple` of 4 :class:`str` describing the dimensions.
        - the array :class:`numpy.dtype`.
        - :class:`tuple` of maximally 2 :class:`ints` which dimensions to expand.

    Raises
    ------
    ValueError
        If the array is not `2`, `3,` or `4` dimensional.
    """

    def dims(order: list[int]) -> tuple[str, ...]:
        return tuple(np.array(["z", "y", "x", "channels"], dtype=object)[np.argsort(order)])

    def infer(y: int, x: int, z: int, c: int) -> tuple[str, ...]:
        if infer_dimensions == InferDimensions.DEFAULT:
            if shape[z] == 1:
                return dims([z, y, x, c])
            if shape[c] == 1:
                return dims([c, y, x, z])

            return dims([z, y, x, c])

        if infer_dimensions == InferDimensions.Z_LAST:
            return dims([c, y, x, z])

        return dims([z, y, x, c])

    if isinstance(obj, str):
        shape, dtype = _get_image_shape_dtype(obj)
    else:
        shape, dtype = obj.shape, obj.dtype
    logg.debug(f"Found image of shape `{shape}` and dtype `{dtype}`")

    ndim = len(shape)

    if not isinstance(infer_dimensions, InferDimensions):
        if ndim not in (2, 3, 4):
            raise ValueError(f"Expected the image to be either `2`, `3` or `4` dimensional, found `{ndim}`.")
        # explicitly passed dims as tuple
        if len(infer_dimensions) != ndim:
            raise ValueError(f"Image is `{ndim}` dimensional, cannot assign to dims `{infer_dimensions}`.")
        _assert_dims_present(infer_dimensions, include_z=ndim == 4)

        infer_dimensions = tuple(infer_dimensions)
        add_shape = (1,) * (4 - ndim)
        dimnames = ("z", "channels") if ndim == 2 else (("channels",) if "z" in infer_dimensions else ("z",))

        return shape + add_shape, infer_dimensions + dimnames, dtype, tuple(ndim + i for i in range(len(add_shape)))

    if ndim == 2:
        # assume only spatial dims are present
        return shape + (1, 1), ("y", "x", "z", "channels"), dtype, (2, 3)

    x, y, *_ = np.argsort(shape, kind="stable")[::-1]
    if y > x:
        # assume 1st 2 dimensions are spatial and are y, x
        y, x = x, y

    if ndim == 3:
        c, *_ = {0, 1, 2} - {x, y}
        if c == 2:
            z, c, expand_dims = 0, 3, 0
            x += 1
            y += 1
            shape = (1,) + shape
        else:
            z, expand_dims = 3, 3
            shape = shape + (1,)

        return shape, infer(y, x, z, c), dtype, (expand_dims,)

    if ndim == 4:
        # assume z before c
        z, c = (i for i in range(ndim) if i not in (y, x))
        return shape, infer(y, x, z, c), dtype, ()

    raise ValueError(f"Expected the image to be either `2`, `3` or `4` dimensional, found `{ndim}`.")


def _lazy_load_image(
    fname: str | Path,
    dims: InferDimensions | tuple[str, ...] = InferDimensions.DEFAULT,
    chunks: int | str | tuple[int, ...] | Mapping[str, int | str] | None = None,
) -> xr.DataArray:
    # TODO(michalk8): switch to pims for tiffs? or switch completely once lazy-loading of JPEGs is done
    def read_unprotected(fname: str) -> NDArrayA:
        # not setting MAX_IMAGE_PIXELS causes problems when with processes and dask.distributed
        old_max_pixels = Image.MAX_IMAGE_PIXELS
        try:
            if fname.endswith(".tif") or fname.endswith(".tiff"):
                # do not use imread since it changes the shape to `y, x, ?, z`,
                # whereas we use `z, y, x, ?` in `_infer_shape_dtype`
                # return np.reshape(imread(fname, plugin="tifffile"), shape)
                return np.reshape(TiffFile(fname).asarray(), shape)

            Image.MAX_IMAGE_PIXELS = None
            return np.reshape(imread(fname, plugin="pil"), shape)
        except Image.UnidentifiedImageError as e:  # should not happen
            logg.error(str(e))
            return np.reshape(imread(fname), shape)
        finally:
            Image.MAX_IMAGE_PIXELS = old_max_pixels

    fname = str(fname)
    shape, dims, dtype, _ = _infer_dimensions(fname, dims)
    logg.debug(f"Reshaping to `{shape}` with dimensions `{dims}`")

    if isinstance(chunks, dict):
        chunks = tuple(chunks.get(d, "auto") for d in dims)  # type: ignore[union-attr]

    darr = da.from_delayed(delayed(read_unprotected)(fname), shape=shape, dtype=dtype)
    if chunks is not None:
        darr = darr.rechunk(chunks)

    return xr.DataArray(darr, dims=dims).transpose("y", "x", "z", ...)
