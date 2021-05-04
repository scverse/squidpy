from typing import List, Tuple, Union, Mapping, Optional
from pathlib import Path
import warnings

from scanpy import logging as logg

import numpy as np
import xarray as xr

from skimage.io import imread


def _read_metadata(fname: str) -> Tuple[List[int], np.dtype]:  # type: ignore[type-arg]
    import pims

    def read() -> Tuple[List[int], np.dtype]:  # type: ignore[type-arg]
        with warnings.catch_warnings():
            # the error message is also contained in the exception
            warnings.filterwarnings("ignore", message=r".*errored: .*", category=UserWarning)
            with pims.open(fname) as img:
                shape = [len(img)] + list(img.frame_shape)
                dtype = np.dtype(img.pixel_type)

                return shape, dtype

    try:
        return read()
    except pims.UnknownFormatError as e:
        if "exceeds limit" not in str(e):
            raise
        from PIL import Image

        old_max_image_pixels = Image.MAX_IMAGE_PIXELS
        try:
            limit = (2 ** 14) ** 2
            Image.MAX_IMAGE_PIXELS = limit
            return read()
        except pims.UnknownFormatError as e:
            if "exceeds limit" in str(e):
                raise RuntimeError(
                    f"Unable to open `{fname}` because it exceeds the default limit of `{limit}` pixels. "
                    f"To increase the limit, run `import PIL; PIL.Image.MAX_IMAGE_PIXELS = ...`."
                ) from e
            raise
        finally:
            Image.MAX_IMAGE_PIXELS = old_max_image_pixels


def _get_shape_pages(fname: str) -> Tuple[List[int], Optional[int], Optional[int], np.dtype]:  # type: ignore[type-arg]
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


def _determine_dimensions(
    shape: List[int], c_dim: Optional[int], z_dim: Optional[int] = None
) -> Tuple[List[str], List[int]]:
    ndim = len(shape)
    if ndim == 2:
        return ["y", "x", "z", "channels"], shape + [1, 1]

    dims = [""] * 4

    if ndim == 3:
        if c_dim not in (0, 2):
            raise ValueError(f"Expected channel dimension to be either `0` or `2`, found `{c_dim}`.")
        delta = c_dim == 0
        dims[c_dim] = "channels"

        dims[delta] = "y"
        dims[delta + 1] = "x"
        dims[c_dim] = "channels"
        dims[-1] = "z"

        return dims, shape + [1]

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

        return dims, shape

    raise ValueError()


def _lazy_load_image(
    fname: Union[str, Path], chunks: Optional[Union[int, str, Tuple[int, ...], Mapping[str, int]]] = None
) -> xr.DataArray:
    from dask import delayed
    import dask.array as da

    if isinstance(chunks, dict):
        chunks = tuple(chunks.values())

    fname = str(fname)
    shape, c_dim, z_dim, dtype = _get_shape_pages(fname)
    dims, shape = _determine_dimensions(shape, c_dim, z_dim)

    darr = da.from_delayed(delayed(imread)(fname).reshape(shape), shape=shape, dtype=dtype)
    if chunks is not None:
        darr = darr.rechunk(chunks)

    # subsetting for bwd compatibility, will be removed once Z-dim is implemented
    return xr.DataArray(darr, dims=dims).transpose("y", "x", "z", "channels")[:, :, 0, :]
