"""Functions exposed: process_img()."""

from types import MappingProxyType
from typing import Union, Mapping, Optional

import xarray as xr

import skimage
import skimage.filters

from .crop import uncrop_img
from .object import ImageContainer


def process_img(
    img: ImageContainer,
    img_id: str,
    processing: Union[str],
    processing_kwargs: Mapping = MappingProxyType({}),
    xs: Optional[int] = None,
    ys: Optional[int] = None,
    key_added: Optional[str] = None,
    inplace: bool = False,
) -> None:
    """
    Process image.

    Note that crop-wise processing can save memory but may change behaviour of cropping if global statistics are used.
    Leave xs and ys as None to process full image in one go.

    Parameters
    ----------
    img
        High-resolution image.
    img_id
        Key of image object to process.
    processing
        Name of processing method to use. Available are:

            - `'smooth'`: see :func:`skimage.filters.gaussian`.

    processing_kwargs
        Key word arguments to processing method. Available are:

            - for ``processing = 'smooth'``: see :func:`skimage.filters.gaussian`.
    xs
        Width of the crops in pixels.
    ys
        Height of the crops in pixels.
    key_added
        Key of new image sized array to add into img object. Defaults to "${img_id}_${processing}"
    inplace
        Whether to replace original image by processed one. Use this to save memory.

    Return
    ------
    TODO
    """
    crops, xcoord, ycoord = img.crop_equally(xs=xs, ys=ys, img_id=img_id)
    if processing == "smooth":
        crops = [skimage.filters.gaussian(x, **processing_kwargs) for x in crops]
    else:
        raise ValueError(f"did not recognize processing {processing}")
    channel_id = img.data[img_id].dims[0]  # channels are named the same as in source image
    # Make sure crops are xarrays:
    if not isinstance(crops[0], xr.DataArray):
        dims = [channel_id, "y", "x"]
        crops = [xr.DataArray(x, dims=dims) for x in crops]
    # Reassemble image:
    img_proc = uncrop_img(crops=crops, x=xcoord, y=ycoord, shape=img.shape, channel_id=channel_id)
    if inplace:
        img_id_new = img_id
    else:
        img_id_new = img_id + "_" + processing if key_added is None else key_added
    img.add_img(img=img_proc, img_id=img_id_new, channel_id=channel_id)
