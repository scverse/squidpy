from types import MappingProxyType
from typing import Any, Union, Mapping, Callable, Optional

from scanpy import logging as logg

from dask import delayed
from dask_image.ndfilters import gaussian_filter as dask_gf
from scipy.ndimage.filters import gaussian_filter as scipy_gf
import numpy as np
import dask.array as da

from skimage.color import rgb2gray
from skimage.util.dtype import img_as_float32

from squidpy._docs import d, inject_docs
from squidpy.im._container import ImageContainer
from squidpy._constants._constants import Processing
from squidpy._constants._pkg_constants import Key

__all__ = ["process"]


def to_grayscale(img: Union[np.ndarray, da.Array]) -> Union[np.ndarray, da.Array]:
    if img.shape[-1] != 3:
        raise ValueError(f"Expected channel dimension to be `3`, found `{img.shape[-1]}`.")

    if isinstance(img, da.Array):
        img = da.from_delayed(delayed(img_as_float32)(img), shape=img.shape, dtype=np.float32)
        coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=img.dtype)

        return img @ coeffs

    return rgb2gray(img)


@d.dedent
@inject_docs(p=Processing)
def process(
    img: ImageContainer,
    layer: Optional[str] = None,
    method: Union[str, Callable[..., np.ndarray]] = "smooth",
    chunks: Optional[int] = None,
    lazy: bool = False,
    layer_added: Optional[str] = None,
    channel_dim: Optional[str] = None,
    copy: bool = False,
    apply_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Optional[ImageContainer]:
    """
    Process an image by applying a transformation.

    Note that crop-wise processing can save memory but may change behavior of cropping if global statistics are used.
    Leave ``size = None`` in order to process the full image in one go.

    Parameters
    ----------
    %(img_container)s
    %(img_layer)s
    method
        Processing method to use. Valid options are:

            - `{p.SMOOTH.s!r}` - :func:`skimage.filters.gaussian`.
            - `{p.GRAY.s!r}` - :func:`skimage.color.rgb2gray`.

        %(custom_fn)s
    %(chunks_lazy)s
    %(layer_added)s If `None`, use ``'{{layer}}_{{method}}'``.
    channel_dim
        Name of the channel dimension of the new image layer. Default is the same as the original, if the
        processing function does not change the number of channels, and ``'{{channel}}_{{processing}}'`` otherwise.
    %(copy_cont)s
    apply_kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.apply`.
    kwargs
        Keyword arguments for ``method``.

    Returns
    -------
    If ``copy = True``, returns a new container with the processed image in ``'{{layer_added}}'``.

    Otherwise, modifies the ``img`` with the following key:

        - :class:`squidpy.im.ImageContainer` ``['{{layer_added}}']`` - the processed image.

    Raises
    ------
    NotImplementedError
        If ``method`` has not been implemented.
    """
    layer = img._get_layer(layer)
    method = Processing(method) if isinstance(method, (str, Processing)) else method  # type: ignore[assignment]
    apply_kwargs = dict(apply_kwargs)
    apply_kwargs["lazy"] = lazy

    if channel_dim is None:
        channel_dim = img[layer].dims[-1]
    layer_new = Key.img.process(method, layer, layer_added=layer_added)

    if callable(method):
        callback = method
    elif method == Processing.SMOOTH:  # type: ignore[comparison-overlap]
        # TODO: handle Z-dim
        kwargs.setdefault("sigma", [1, 1, 0])  # TODO: Z-dim, allow for ints, replicate over spatial dims
        if chunks is not None:
            # dask_image already handles map_overlap
            chunks_, chunks = chunks, None
            callback = lambda arr, **kwargs: dask_gf(da.asarray(arr).rechunk(chunks_), **kwargs)  # noqa: E731
        else:
            callback = scipy_gf
    elif method == Processing.GRAY:  # type: ignore[comparison-overlap]
        apply_kwargs["drop_axis"] = 2  # TODO: Z-dim
        callback = to_grayscale
    else:
        raise NotImplementedError(f"Method `{method}` is not yet implemented.")

    start = logg.info(f"Processing image using `{method}` method")
    res: ImageContainer = img.apply(callback, layer=layer, copy=True, chunks=chunks, fn_kwargs=kwargs, **apply_kwargs)

    # if the method changes the number of channels
    if res[layer].shape[-1] != img[layer].shape[-1]:
        modifier = "_".join(layer_new.split("_")[1:]) if layer_added is None else layer_added
        channel_dim = f"{channel_dim}_{modifier}"

    res._data = res.data.rename({res[layer].dims[-1]: channel_dim}).rename_vars({layer: layer_new})

    logg.info("Finish", time=start)

    if copy:
        return res

    img.add_img(img=res, layer=layer_new, channel_dim=channel_dim, copy=False, lazy=lazy)
