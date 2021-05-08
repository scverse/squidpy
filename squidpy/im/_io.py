from PIL import Image
from typing import Tuple, Union, Mapping, Optional
from pathlib import Path

from scanpy import logging as logg

import numpy as np
import xarray as xr

from tifffile import TiffFile
from skimage.io import imread


# modification of `skimage`'s `pil_to_ndarray`:
# https://github.com/scikit-image/scikit-image/blob/main/skimage/io/_plugins/pil_plugin.py#L55
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


def _read_metadata(fname: str) -> Tuple[Tuple[int, ...], np.dtype]:  # type: ignore[type-arg]
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


# TODO: remove me?
def _get_shape_pages(
    fname: str,
) -> Tuple[Tuple[int, ...], Optional[int], Optional[int], np.dtype]:  # type: ignore[type-arg]
    shape, dtype = _read_metadata(fname)
    ndim = len(shape)

    if ndim == 2:
        z_dim = c_dim = None
    elif ndim == 3:
        c_dim = 0
        z_dim = None
    elif ndim == 4:
        fst, lst = shape[0], shape[-1]
        if fst == 1:
            # channels last, can be 1
            c_dim, z_dim = 3, 0
        elif lst == 1:
            # channels first, cannot be 1
            c_dim, z_dim = 0, 3
        else:
            # assume z-dim is saved as e.g. TIFF stacks
            c_dim, z_dim = 3, 0
            logg.warning(
                f"Setting channel dimension to `{c_dim}` "
                f"and z-dimension to `{z_dim}` for an image of shape `{shape}`"
            )
    else:
        raise ValueError(f"Expected number of dimensions to be either `2`, `3`, or `4`, found `{ndim}`.")

    return shape, c_dim, z_dim, dtype


# TODO: simplify me/remove me
def _determine_dimensions(
    shape: Tuple[int, ...], c_dim: Optional[int], z_dim: Optional[int] = None
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    ndim = len(shape)
    if ndim == 2:
        return ("y", "x", "z", "channels"), shape + (1, 1)

    dims = [""] * 4

    if ndim == 3:
        if c_dim not in (0, 2):
            raise ValueError(f"Expected channel dimension to be either `0` or `2`, found `{c_dim}`.")
        delta = c_dim == 0

        dims[delta] = "y"
        dims[delta + 1] = "x"
        dims[c_dim] = "channels"
        dims[-1] = "z"

        return tuple(dims), shape + (1,)

    if ndim == 4:
        if c_dim not in (0, 3):
            raise ValueError(f"Expected channel dimension to be either `0` or `3`, found `{c_dim}`.")
        if z_dim not in (0, 3):
            raise ValueError(f"Expected z-dimension to be either `0` or `3`, found `{z_dim}`.")
        if c_dim == z_dim:
            raise ValueError(f"Expected z-dimension and channel dimension to be different, found `{c_dim}`.")

        dims[1] = "y"
        dims[2] = "x"
        dims[c_dim] = "channels"
        dims[z_dim] = "z"

        return tuple(dims), shape

    raise ValueError(f"Expected the image to be either 2, 3 or 4 dimensional, found `{ndim}`.")


def _lazy_load_image(
    fname: Union[str, Path], chunks: Optional[Union[int, str, Tuple[int, ...], Mapping[str, Union[int, str]]]] = None
) -> xr.DataArray:
    def read_unprotected(fname: str) -> np.ndarray:
        # causes a lot of problem when with processes and dask.distributed
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

    from dask import delayed
    import dask.array as da

    fname = str(fname)
    shape, c_dim, z_dim, dtype = _get_shape_pages(fname)

    dims, shape = _determine_dimensions(shape, c_dim, z_dim)
    if isinstance(chunks, dict):
        chunks = tuple(chunks.get(d, "auto") for d in dims)

    darr = da.from_delayed(delayed(read_unprotected)(fname), shape=shape, dtype=dtype)
    if chunks is not None:
        darr = darr.rechunk(chunks)

    # subsetting for bwd compatibility, will be removed once Z-dim is implemented
    return xr.DataArray(darr, dims=dims).transpose("y", "x", "z", "channels")[:, :, 0, :]
