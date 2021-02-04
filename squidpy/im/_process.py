from typing import Any, Tuple, Union, Callable, Optional
from functools import partial

import numpy as np

import skimage
import skimage.filters

from squidpy._docs import d, inject_docs
from squidpy.im._container import ImageContainer
from squidpy._constants._constants import Processing

__all__ = ["process"]


@d.dedent
@inject_docs(p=Processing)
def process(
    img: ImageContainer,
    img_id: Optional[str] = None,
    processing: Union[str, Callable[..., np.ndarray]] = "smooth",
    size: Optional[Tuple[int, int]] = None,
    key_added: Optional[str] = None,
    channel_id: Optional[str] = None,
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
    processing
        Name of processing method to use. Valid options are:

            - `{p.SMOOTH.s!r}` - :func:`skimage.filters.gaussian`.
            - `{p.GRAY.s!r}` - :func:`skimage.color.rgb2gray`.

        %(custom_fn)s
    %(img_id)s
    size
        Size of the crop as ``(height, width)``. If `None`, use the full size.
    key_added
        Key of new image layer to add into ``img`` object. If `None`, use ``'{{img_id}}_{{processing}}'``.
    channel_id
        Name of the channel dimension of the new image layer.

        Default is the same as the input image's, if the processing function does not change the number
        of channels, and ``'{{channel}}_{{processing}}'``, if it does.
    %(copy_cont)s
    kwargs
        Keyword arguments for ``processing`` function.

    Returns
    -------
    If ``copy = True``, returns the processed image with a new key `'{{key_added}}'`.

    Otherwise, it modifies the ``img`` with the following key:

        - :class:`squidpy.im.ImageContainer` ``['{{key_added}}']`` - the processed image.

    Raises
    ------
    NotImplementedError
        If a ``processing`` has not been implemented.
    """
    img_id = img._singleton_id(img_id)
    processing = (
        Processing(processing) if isinstance(processing, (str, Processing)) else processing  # type: ignore[assignment]
    )

    if channel_id is None:
        channel_id = img[img_id].dims[-1]
    img_id_new = f"{img_id}_{processing}"

    if callable(processing):
        callback = processing
        img_id_new = f"{img_id}_{getattr(callback, '__name__', 'custom')}"  # get the function name
    elif processing == Processing.SMOOTH:  # type: ignore[comparison-overlap]
        callback = partial(skimage.filters.gaussian, multichannel=True)
    elif processing == Processing.GRAY:  # type: ignore[comparison-overlap]
        if img[img_id].shape[-1] != 3:
            raise ValueError(f"Expected channel dimension to be `3`, found `{img[img_id].shape[-1]}`.")
        callback = skimage.color.rgb2gray
        channel_id = f"{channel_id}_{processing}"
    else:
        raise NotImplementedError(f"Processing `{processing}` is not yet implemented.")

    if key_added is not None:
        img_id_new = key_added

    # process crops
    crops = [crop.apply(callback, img_id=img_id, **kwargs) for crop in img.generate_equal_crops(size=size)]
    # reassemble image
    img_proc: ImageContainer = ImageContainer.uncrop(crops=crops, shape=img.shape)
    img_proc._data = img_proc.data.rename({img_proc[img_id].dims[-1]: channel_id}).rename_vars({img_id: img_id_new})

    if copy:
        return img_proc

    img.add_img(img=img_proc, img_id=img_id_new, channel_dim=channel_id)