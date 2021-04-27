from typing import List, Tuple, Union, Optional
from pathlib import Path

import numpy as np
import xarray as xr

from skimage.io import imread


def _get_shape_pages(fname: str) -> Tuple[List[int], Optional[int], Optional[int], np.dtype]:  # type: ignore[type-arg]
    import pims

    with pims.open(fname) as img:
        shape = [len(img)] + list(img.frame_shape)
        dtype = np.dtype(img.pixel_type)

    ndim = len(shape)
    if ndim == 2:
        z_dim = c_dim = None
    elif ndim == 3:
        c_dim = 0
        z_dim = None
    elif ndim == 4:
        fst, lst = shape[0], shape[-1]
        if fst == 1:
            c_dim, z_dim = 3, 0
        elif lst == 1:
            c_dim, z_dim = 0, 3
        else:
            # TODO: warn + channels first?
            raise RuntimeError("TODO.")
    # TODO: 5D, iff shape[0] == 1
    else:
        raise ValueError("TODO.")

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
            raise ValueError("TODO.")
        delta = c_dim == 0
        dims[c_dim] = "channels"

        dims[delta] = "y"
        dims[delta + 1] = "x"
        dims[c_dim] = "channels"
        dims[-1] = "z"

        return dims, shape + [1]

    if ndim == 4:
        if c_dim not in (0, 3):
            raise ValueError("TODO.")
        if z_dim not in (0, 3):
            raise ValueError("TODO.")
        if c_dim == z_dim:
            raise ValueError("TODO.")

        dims[1] = "y"
        dims[2] = "x"
        dims[c_dim] = "channels"
        dims[z_dim] = "z"

        return dims, shape

    raise ValueError()


def _lazy_load_image(fname: Union[str, Path], chunks: Optional[str] = None) -> xr.DataArray:
    from dask import delayed
    import dask.array as da

    fname = str(fname)
    shape, c_dim, z_dim, dtype = _get_shape_pages(fname)
    dims, shape = _determine_dimensions(shape, c_dim, z_dim)

    darr = da.from_delayed(delayed(imread)(fname).reshape(shape), shape=shape, dtype=dtype)
    if chunks is not None:
        darr = darr.rechunk(chunks)

    # TODO: subsetting for bwd compat
    return xr.DataArray(darr, dims=dims).transpose("y", "x", "z", "channels")[:, :, 0, :]
