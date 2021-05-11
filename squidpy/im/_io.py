from PIL import Image
from typing import Tuple, Union, Mapping, Optional
from pathlib import Path

from scanpy import logging as logg

from dask import delayed
import numpy as np
import xarray as xr
import dask.array as da

from tifffile import TiffFile
from skimage.io import imread

# modification of `skimage`'s `pil_to_ndarray`:
# https://github.com/scikit-image/scikit-image/blob/main/skimage/io/_plugins/pil_plugin.py#L55
from squidpy._constants._constants import InferDimensions


def _infer_shape_dtype(fname: str) -> Tuple[Tuple[int, ...], np.dtype]:  # type: ignore[type-arg]
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
        return (len(image.pages),) + image.pages[0].shape, np.dtype(image.pages[0].dtype)

    image = Image.open(fname)
    n_frames = getattr(image, "n_frames", 1)
    shape: Tuple[int, ...] = (n_frames,) + image.size[::-1]

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


def _get_image_shape_dtype(fname: str) -> Tuple[Tuple[int, ...], np.dtype]:  # type: ignore[type-arg]
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


def _infer_dimensions(
    obj: Union[np.ndarray, xr.DataArray, str],
    infer_dimensions: InferDimensions = InferDimensions.DEFAULT,
) -> Tuple[Tuple[int, ...], Tuple[str, ...], np.dtype]:  # type: ignore[type-arg]
    if isinstance(obj, str):
        shape, dtype = _get_image_shape_dtype(obj)
    else:
        shape, dtype = obj.shape, obj.dtype

    ndim = len(shape)

    if ndim == 2:
        return shape + (1, 1), ("y", "x", "z", "channels"), dtype

    if ndim == 3:
        if shape[0] < shape[1] and shape[0] < shape[2]:
            # tiff with pages, but no channels
            if infer_dimensions == InferDimensions.PREFER_Z:
                return shape + (1,), ("z", "y", "x", "channels"), dtype
            return shape + (1,), ("channels", "y", "x", "z"), dtype

        if infer_dimensions == InferDimensions.PREFER_Z:
            return shape + (1,), ("y", "x", "z", "channels"), dtype

        return shape + (1,), ("y", "x", "channels", "z"), dtype

    if ndim == 4:
        if infer_dimensions == InferDimensions.DEFAULT:
            if shape[0] == 1:
                return shape, ("z", "y", "x", "channels"), dtype
            if shape[-1] == 1:
                return shape, ("channels", "y", "x", "z"), dtype

            return shape, ("z", "y", "x", "channels"), dtype

        if infer_dimensions == InferDimensions.PREFER_Z:
            return shape, ("channels", "y", "x", "z"), dtype

        return shape, ("z", "y", "x", "channels"), dtype

    raise ValueError(f"Expected the image to be either `2`, `3` or `4` dimensional, found `{ndim}`.")


def _lazy_load_image(
    fname: Union[str, Path],
    infer_dimensions: InferDimensions = InferDimensions.DEFAULT,
    chunks: Optional[Union[int, str, Tuple[int, ...], Mapping[str, Union[int, str]]]] = None,
) -> xr.DataArray:
    def read_unprotected(fname: str) -> np.ndarray:
        # not setting MAX_IMAGE_PIXELS caues problems when with processes and dask.distributed
        old_max_pixels = Image.MAX_IMAGE_PIXELS
        try:
            if fname.endswith(".tif") or fname.endswith(".tiff"):
                return np.reshape(imread(fname, plugin="tifffile"), shape)

            Image.MAX_IMAGE_PIXELS = None
            return np.reshape(imread(fname, plugin="pil"), shape)
        except Image.UnidentifiedImageError as e:  # should not happen
            logg.warning(e)
            return np.reshape(imread(fname), shape)
        finally:
            Image.MAX_IMAGE_PIXELS = old_max_pixels

    fname = str(fname)
    shape, dims, dtype = _infer_dimensions(fname, infer_dimensions)

    if isinstance(chunks, dict):
        chunks = tuple(chunks.get(d, "auto") for d in dims)

    darr = da.from_delayed(delayed(read_unprotected)(fname), shape=shape, dtype=dtype)
    if chunks is not None:
        darr = darr.rechunk(chunks)

    return xr.DataArray(darr, dims=dims).transpose("y", "x", "z", "channels")
