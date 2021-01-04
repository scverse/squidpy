"""Functions exposed: process_img()."""

from types import MappingProxyType
from typing import Any, Union, Mapping, Optional

import xarray as xr

import skimage
import skimage.filters

from squidpy._docs import d, inject_docs
from squidpy.im.crop import uncrop_img
from squidpy.im.object import ImageContainer
from squidpy.constants._constants import Processing


@d.dedent
@inject_docs(p=Processing)
def process_img(
    img: ImageContainer,
    img_id: str,
    processing: Union[str, Processing],
    processing_kwargs: Mapping[str, Any] = MappingProxyType({}),
    xs: Optional[int] = None,
    ys: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = True,
) -> None:
    """
    Process an image.

    Note that crop-wise processing can save memory but may change behaviour of cropping if global statistics are used.
    Leave ``xs`` and ``ys`` as `None` in order to process the full image in one go.

    Parameters
    ----------
    %(img_container)s
    img_id
        Key of im object to process.
    processing
        Name of processing method to use. Available are:

            - `{p.SMOOTH.s!r}`: :func:`skimage.filters.gaussian`.

    processing_kwargs
        Key word arguments to processing method specified by ``processing``.
    %(width_height)s
    key_added
        Key of new image sized array to add into img object. Defaults to ``{{img_id}}_{{processing}}``.
    %(copy_cont)s

    Returns
    -------
    Nothing, just updates ``img``.
    """
    processing = Processing(processing)
    crops, xcoord, ycoord = img.crop_equally(xs=xs, ys=ys, img_id=img_id)

    if processing == Processing.SMOOTH:
        crops = [skimage.filters.gaussian(x, **processing_kwargs) for x in crops]
    else:
        raise NotImplementedError(processing)

    channel_id = img.data[img_id].dims[0]  # channels are named the same as in source image
    # Make sure crops are xarrays:
    if not isinstance(crops[0], xr.DataArray):
        dims = [channel_id, "y", "x"]
        crops = [xr.DataArray(x, dims=dims) for x in crops]
    # Reassemble im:
    img_proc = uncrop_img(crops=crops, x=xcoord, y=ycoord, shape=img.shape, channel_id=channel_id)
    img_id_new = (img_id + "_" + processing.v if key_added is None else key_added) if copy else img_id

    img.add_img(img=img_proc, img_id=img_id_new, channel_id=channel_id)
