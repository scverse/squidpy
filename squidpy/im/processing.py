"""Functions exposed: process_img()."""

from typing import Any, Tuple, Union, Optional

import skimage
import skimage.filters

from squidpy._docs import d, inject_docs
from squidpy.im.object import ImageContainer
from squidpy._constants._constants import Processing


@d.dedent
@inject_docs(p=Processing)
def process_img(
    img: ImageContainer,
    img_id: str,
    processing: Union[str, Processing],
    yx: Optional[Tuple[int, int]] = None,
    key_added: Optional[str] = None,
    channel_id: Optional[str] = None,
    copy: bool = False,
    **kwargs: Any,
) -> Optional[ImageContainer]:
    """
    Process an image.

    Note that crop-wise processing can save memory but may change behaviour of cropping if global statistics are used.
    Leave ``xs`` and ``ys`` as `None` in order to process the full image in one go.

    Parameters
    ----------
    %(img_container)s
    %(img_id)s
    processing
        Name of processing method to use. Valid options are:

            - `{p.SMOOTH.s!r}` - :func:`skimage.filters.gaussian`.
            - `{p.GRAY.s!r}` - :func:`skimage.color.rgb2gray`.

    yx
        TODO.
    key_added
        Key of new image layer to add into ``img`` object. If `None`, use ``{{img_id}}_{{processing}}``.
    channel_id
        Name of the channel dimension of the new image layer.

        Default is the same as the input image if the processing function does not change the number
        of channels, and ``{{channel}}_{{processing}}`` if it does.
    %(copy_cont)s
    kwargs
        Keyword arguments to processing method specified by ``processing``.

    Returns
    -------
    Nothing, just updates ``img`` with the processed image in layer ``key_added``.
    If ``copy = True``, returns the processed image.
    """
    processing = Processing(processing)
    if channel_id is None:
        channel_id = img[img_id].dims[-1]
        if processing == Processing.GRAY:
            channel_id = f"{channel_id}_{processing}"
    img_id_new = img_id + "_" + str(processing) if key_added is None else key_added

    # process crops
    crops = []
    for crop in img.generate_equal_crops(yx=yx):
        # TODO: custom processing
        if processing == Processing.SMOOTH:
            crop = ImageContainer(
                skimage.filters.gaussian(crop[img_id], **kwargs),
                img_id=img_id_new,
                channel_id=channel_id,
            )
        elif processing == Processing.GRAY:
            crop = ImageContainer(
                skimage.color.rgb2gray(crop[img_id], **kwargs), img_id=img_id_new, channel_id=channel_id
            )
        else:
            raise NotImplementedError(processing)

        crops.append(crop)

    # Reassemble image:
    img_proc = ImageContainer.uncrop_img(crops=crops)

    if copy:
        return img_proc  # type: ignore[no-any-return]

    # TODO: function to add ImageContainer
    img.add_img(img=img_proc[img_id_new], img_id=img_id_new)
