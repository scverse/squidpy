from typing import Any, Tuple, Union, Callable, Optional
from functools import partial

from scanpy import logging as logg

import numpy as np

import skimage
import skimage.filters

from squidpy._docs import d, inject_docs
from squidpy.im._container import ImageContainer
from squidpy._constants._constants import Processing
from squidpy._constants._pkg_constants import Key

__all__ = ["process"]


@d.dedent
@inject_docs(p=Processing)
def process(
    img: ImageContainer,
    img_id: Optional[str] = None,
    method: Union[str, Callable[..., np.ndarray]] = "smooth",
    size: Optional[Tuple[int, int]] = None,
    key_added: Optional[str] = None,
    channel_dim: Optional[str] = None,
    copy: bool = False,
    **kwargs: Any,
) -> Optional[ImageContainer]:
    """
    Process an image by applying a transformation.

    Note that crop-wise processing can save memory but may change behaviour of cropping if global statistics are used.
    Leave ``size = None`` in order to process the full image in one go.

    Parameters
    ----------
    %(img_container)s
    method
        Processing method to use. Valid options are:

            - `{p.SMOOTH.s!r}` - :func:`skimage.filters.gaussian`.
            - `{p.GRAY.s!r}` - :func:`skimage.color.rgb2gray`.

        %(custom_fn)s
    %(img_id)s
    %(size)s
    key_added
        Key of new image layer to add into ``img`` object. If `None`, use ``'{{img_id}}_{{method}}'``.
    channel_dim
        Name of the channel dimension of the new image layer.

        Default is the same as the input image's, if the processing function does not change the number
        of channels, and ``'{{channel}}_{{processing}}'`` otherwise.
    %(copy_cont)s
    kwargs
        Keyword arguments for ``method``.

    Returns
    -------
    If ``copy = True``, returns the processed image with a new key `'{{key_added}}'`.

    Otherwise, it modifies the ``img`` with the following key:

        - :class:`squidpy.im.ImageContainer` ``['{{key_added}}']`` - the processed image.

    Raises
    ------
    NotImplementedError
        If ``method`` has not been implemented.
    """
    img_id = img._singleton_id(img_id)
    method = Processing(method) if isinstance(method, (str, Processing)) else method  # type: ignore[assignment]

    if channel_dim is None:
        channel_dim = img[img_id].dims[-1]
    img_id_new = Key.img.process(method, img_id, key_added=key_added)

    if callable(method):
        callback = method
    elif method == Processing.SMOOTH:  # type: ignore[comparison-overlap]
        callback = partial(skimage.filters.gaussian, multichannel=True)
    elif method == Processing.GRAY:  # type: ignore[comparison-overlap]
        if img[img_id].shape[-1] != 3:
            raise ValueError(f"Expected channel dimension to be `3`, found `{img[img_id].shape[-1]}`.")
        callback = skimage.color.rgb2gray
    else:
        raise NotImplementedError(f"Method `{method}` is not yet implemented.")

    start = logg.info(f"Processing image using `{method}` method")

    crops = [crop.apply(callback, img_id=img_id, copy=True, **kwargs) for crop in img.generate_equal_crops(size=size)]
    res: ImageContainer = ImageContainer.uncrop(crops=crops, shape=img.shape)

    # if the method changes the number of channels
    if res[img_id].shape[-1] != img[img_id].shape[-1]:
        modifier = "_".join(img_id_new.split("_")[1:]) if key_added is None else key_added
        channel_dim = f"{channel_dim}_{modifier}"

    res._data = res.data.rename({res[img_id].dims[-1]: channel_dim}).rename_vars({img_id: img_id_new})

    logg.info("Finish", time=start)

    if copy:
        return res

    img.add_img(img=res, img_id=img_id_new, channel_dim=channel_dim)
