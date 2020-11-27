from typing import List, Tuple, Optional

import numpy as np
import xarray as xr

import skimage.util
from skimage.draw import disk
from skimage.transform import rescale

from squidpy._docs import d


@d.get_sections(base="uncrop_img", sections=["Parameters", "Returns"])
def uncrop_img(
    crops: List[xr.DataArray],
    x: np.ndarray,
    y: np.ndarray,
    shape: Tuple[int, int],
    channel_id: str = "channels",
) -> xr.DataArray:
    """
    Re-assemble image from crops and their positions.

    Fills remaining positions with zeros. Positions are given as upper right corners.

    Parameters
    ----------
    crops
        List of image crops.
    x
        X coord of crop in pixel space. TODO: nice to have - relative space.
    y
        Y coord of crop in pixel space. TODO: nice to have - relative space.
    shape
        Shape of full image.
    channel_id
        Name of channel dim in :class:`xarray.DataArray`.

    Returns
    -------
    :class:`xarray.DataArray`:
        Array of shape ``(channels, y, x)``.
    """
    # TODO: maybe more descriptive names (y==height, x==width)? + extract to constants...
    # TODO: rewrite asserts
    # TODO: expose remaining positions default value
    assert np.max(y) < shape[0], f"y ({y}) is outsize of image range ({shape[0]})"
    assert np.max(x) < shape[1], f"x ({x}) is outsize of image range ({shape[1]})"

    dims = [channel_id, "y", "x"]
    img = xr.DataArray(np.zeros((crops[0].coords[channel_id].shape[0], shape[1], shape[0])), dims=dims)
    if len(crops) > 1:
        for c, x, y in zip(crops, x, y):
            x0 = x
            x1 = x + c.x.shape[0]
            y0 = y
            y1 = y + c.y.shape[0]
            # TODO: rewrite asserts
            assert x0 >= 0, f"x ({x0}) is outsize of image range ({0})"
            assert y0 >= 0, f"x ({y0}) is outsize of image range ({0})"
            assert x1 <= shape[0], f"x ({x1}) is outsize of image range ({shape[0]})"
            assert y1 <= shape[1], f"y ({y1}) is outsize of image range ({shape[1]})"
            img[:, y0:y1, x0:x1] = c
        return img
    else:
        # TODO: is this the expected behavior? maybe raise?
        img = crops[0]
    return img


@d.get_sections(base="crop_img", sections=["Parameters"])
@d.dedent
def crop_img(
    img: xr.DataArray,
    x: int,
    y: int,
    channel_id: str = "channels",
    xs: int = 100,  # TODO: are these default reasonable or should no defaults be specified?
    ys: int = 100,
    scale: float = 1.0,
    mask_circle: bool = False,
    cval: float = 0.0,
    dtype: Optional[str] = None,
) -> xr.DataArray:
    """
    Extract a crop right and down from ``x`` and ``y``.

    Params
    ------
    %(uncrop_img.parameters)s
    %(width_height)s
    %(crop_extra)s

    Returns
    -------
    %(uncrop_img.returns)s
    """
    # get conversion function
    if dtype is not None:
        if dtype == "uint8":
            convert = skimage.util.img_as_ubyte
        else:
            raise NotImplementedError(dtype)

    # TODO crop_extra should be created by `@d.keep_kwargs in `crop.py`` after
    # https://github.com/Chilipp/docrep/issues/21 is dealth with
    # TODO: rewrite assertions to "normal" errors so they can be more easily tested against
    assert y < img.y.shape[0], f"y ({y}) is outsize of image range ({img.y.shape[0]})"
    assert x < img.x.shape[0], f"x ({x}) is outsize of image range ({img.x.shape[0]})"

    assert xs > 0, "image size cannot be 0"
    assert ys > 0, "image size cannot be 0"

    if channel_id in img.dims:
        crop = (np.zeros((img.coords[channel_id].shape[0], ys, xs)) + cval).astype(img.dtype)
    else:
        crop = (np.zeros((1, ys, xs)) + cval).astype(img.dtype)
    crop = xr.DataArray(crop, dims=[channel_id, "y", "x"])

    # get crop coords
    x0 = x
    x1 = x + xs
    y0 = y
    y1 = y + ys

    # crop image and put in already prepared `crop`
    crop_x0 = min(x0, 0) * -1
    crop_y0 = min(y0, 0) * -1
    crop_x1 = xs - max(x1 - img.x.shape[0], 0)
    crop_y1 = ys - max(y1 - img.y.shape[0], 0)

    crop[
        {
            channel_id: slice(0, img.channels.shape[0]),
            "y": slice(crop_y0, crop_y1),
            "x": slice(crop_x0, crop_x1),
        }
    ] = img[
        {
            channel_id: slice(0, img.channels.shape[0]),
            "y": slice(max(y0, 0), y1),
            "x": slice(max(x0, 0), x1),
        }
    ]

    # scale crop
    if scale != 1:
        crop = rescale(crop, [1, scale, scale], preserve_range=True)
        crop = crop.astype(img.dtype)
        crop = xr.DataArray(crop, dims=[channel_id, "y", "x"])  # need to redefine after rescale

    # mask crop
    if mask_circle:
        assert xs == ys, "Crop has to be square to use mask_circle."
        # get coords inside circle
        rr, cc = disk(
            center=(crop.shape[1] // 2, crop.shape[2] // 2),
            radius=crop.shape[1] // 2,
            shape=(crop.shape[1], crop.shape[2]),
        )
        circle = np.zeros_like(crop)
        circle[:, rr, cc] = 1
        # set everything outside circle to cval
        crop.data[circle == 0] = cval

    # convert to dtype
    if dtype is not None:
        crop.data = convert(crop.data)
    return crop
