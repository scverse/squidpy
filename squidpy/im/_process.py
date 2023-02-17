from __future__ import annotations

from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
    Union,  # noqa: F401
)

import dask.array as da
from dask_image.ndfilters import gaussian_filter as dask_gf
from scanpy import logging as logg
from scipy.ndimage import gaussian_filter as scipy_gf

from squidpy._constants._constants import Processing
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA
from squidpy.im._container import ImageContainer

__all__ = ["process"]


@d.dedent
@inject_docs(p=Processing)
def process(
    img: ImageContainer,
    layer: str | None = None,
    library_id: str | Sequence[str] | None = None,
    method: str | Callable[..., NDArrayA] = "smooth",
    chunks: int | None = None,
    lazy: bool = False,
    layer_added: str | None = None,
    channel_dim: str | None = None,
    copy: bool = False,
    apply_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> ImageContainer | None:
    """
    Process an image by applying a transformation.

    Parameters
    ----------
    %(img_container)s
    %(img_layer)s
    %(library_id)s
        If `None`, all Z-dimensions are processed at once, treating the image as a 3D volume.
    method
        Processing method to use. Valid options are:

            - `{p.SMOOTH.s!r}` - :func:`skimage.filters.gaussian`.
            - `{p.GRAY.s!r}` - :func:`skimage.color.rgb2gray`.

        %(custom_fn)s
    %(chunks_lazy)s
    %(layer_added)s
        If `None`, use ``'{{layer}}_{{method}}'``.
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
    from squidpy.pl._utils import _to_grayscale

    layer = img._get_layer(layer)
    method = Processing(method) if isinstance(method, (str, Processing)) else method  # type: ignore[assignment]
    apply_kwargs = dict(apply_kwargs)
    apply_kwargs["lazy"] = lazy

    if channel_dim is None:
        channel_dim = str(img[layer].dims[-1])
    layer_new = Key.img.process(method, layer, layer_added=layer_added)

    if callable(method):
        callback = method
    elif method == Processing.SMOOTH:  # type: ignore[comparison-overlap]
        if library_id is None:
            expected_ndim = 4
            kwargs.setdefault("sigma", [1, 1, 0, 0])  # y, x, z, c
        else:
            expected_ndim = 3
            kwargs.setdefault("sigma", [1, 1, 0])  # y, x, c

        sigma = kwargs["sigma"]
        if isinstance(sigma, int):
            kwargs["sigma"] = sigma = [sigma, sigma] + [0] * (expected_ndim - 2)
        if len(sigma) != expected_ndim:
            raise ValueError(f"Expected `sigma` to be of length `{expected_ndim}`, found `{len(sigma)}`.")

        if chunks is not None:
            # dask_image already handles map_overlap
            chunks_, chunks = chunks, None
            callback = lambda arr, **kwargs: dask_gf(da.asarray(arr).rechunk(chunks_), **kwargs)  # noqa: E731
        else:
            callback = scipy_gf
    elif method == Processing.GRAY:  # type: ignore[comparison-overlap]
        apply_kwargs["drop_axis"] = 3
        callback = _to_grayscale
    else:
        raise NotImplementedError(f"Method `{method}` is not yet implemented.")

    # to which library_ids should this function be applied?
    if library_id is not None:
        callback = {lid: callback for lid in img._get_library_ids(library_id)}  # type: ignore[assignment]

    start = logg.info(f"Processing image using `{method}` method")
    res: ImageContainer = img.apply(
        callback, layer=layer, copy=True, drop=copy, chunks=chunks, fn_kwargs=kwargs, **apply_kwargs
    )

    # if the method changes the number of channels
    if res[layer].shape[-1] != img[layer].shape[-1]:
        modifier = "_".join(layer_new.split("_")[1:]) if layer_added is None else layer_added
        channel_dim = f"{channel_dim}_{modifier}"

    res._data = res.data.rename({res[layer].dims[-1]: channel_dim})
    logg.info("Finish", time=start)

    if copy:
        return res.rename(layer, layer_new)

    img.add_img(
        img=res[layer],
        layer=layer_new,
        copy=False,
        lazy=lazy,
        dims=res[layer].dims,
        library_id=img[layer].coords["z"].values,
    )
