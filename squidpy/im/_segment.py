from __future__ import annotations

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Sequence,
    Union,  # noqa: F401
)

import dask.array as da
import numpy as np
from scanpy import logging as logg
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

from squidpy._constants._constants import SegmentationBackend
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, singledispatchmethod
from squidpy.im._container import ImageContainer

__all__ = ["SegmentationModel", "SegmentationWatershed", "SegmentationCustom"]
_SEG_DTYPE = np.uint32
_SEG_DTYPE_N_BITS = _SEG_DTYPE(0).nbytes * 8


class SegmentationModel(ABC):
    """
    Base class for all segmentation models.

    Contains core shared functions related to cell and nuclei segmentation.
    Specific segmentation models can be implemented by inheriting from this class.

    Parameters
    ----------
    model
        Underlying segmentation model.
    """

    def __init__(
        self,
        model: Any,
    ):
        self._model = model

    @singledispatchmethod
    @d.get_full_description(base="segment")
    @d.get_sections(base="segment", sections=["Parameters", "Returns"])
    @d.dedent
    def segment(self, img: NDArrayA | ImageContainer, **kwargs: Any) -> NDArrayA | ImageContainer:
        """
        Segment an image.

        Parameters
        ----------
        %(img_container)s
        %(img_layer)s
            Only used when ``img`` is :class:`squidpy.im.ImageContainer`.
        kwargs
            Keyword arguments for the underlying ``model``.

        Returns
        -------
        Segmentation mask for the high-resolution image of shape ``(height, width, z, 1)``.

        Raises
        ------
        ValueError
            If the number of dimensions is neither 2 nor 3.
        NotImplementedError
            If trying to segment a type for which the segmentation has not been registered.
        """
        raise NotImplementedError(f"Segmentation of `{type(img).__name__}` is not yet implemented.")

    @staticmethod
    def _precondition(img: NDArrayA | da.Array) -> NDArrayA | da.Array:
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"Expected `2` or `3` dimensions, found `{img.ndim}`.")

        return img

    @staticmethod
    def _postcondition(img: NDArrayA | da.Array) -> NDArrayA | da.Array:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"Expected segmentation to return `2` or `3` dimensional array, found `{img.ndim}`.")
        if not np.issubdtype(img.dtype, np.integer):
            raise TypeError(f"Expected segmentation to be of integer type, found `{img.dtype}`.")

        return img.astype(_SEG_DTYPE)

    @segment.register(np.ndarray)
    def _(self, img: NDArrayA, **kwargs: Any) -> NDArrayA:
        chunks = kwargs.pop("chunks", None)
        if chunks is not None:
            return self.segment(da.asarray(img).rechunk(chunks), **kwargs)  # type: ignore[no-any-return]

        img = SegmentationModel._precondition(img)
        img = self._segment(img, **kwargs)
        return SegmentationModel._postcondition(img)

    @segment.register(da.Array)
    def _(self, img: da.Array, chunks: str | int | tuple[int, ...] | None = None, **kwargs: Any) -> NDArrayA:
        img = SegmentationModel._precondition(img)
        if chunks is not None:
            img = img.rechunk(chunks)

        shift = int(np.prod(img.numblocks) - 1).bit_length()
        kwargs.setdefault("depth", {0: 30, 1: 30})
        kwargs.setdefault("boundary", "reflect")

        img = da.map_overlap(
            self._segment_chunk,
            img,
            dtype=_SEG_DTYPE,
            num_blocks=img.numblocks,
            shift=shift,
            drop_axis=img.ndim - 1,  # y, x, z, c; -1 seems to be bugged
            **kwargs,
        )
        from dask_image.ndmeasure._utils._label import (
            connected_components_delayed,
            label_adjacency_graph,
            relabel_blocks,
        )

        # max because labels are not continuous (and won't be continuous)
        label_groups = label_adjacency_graph(img, None, img.max())
        new_labeling = connected_components_delayed(label_groups)
        relabeled = relabel_blocks(img, new_labeling)

        return SegmentationModel._postcondition(relabeled)

    @segment.register(ImageContainer)
    def _(
        self,
        img: ImageContainer,
        layer: str,
        library_id: str | Sequence[str],
        channel: int | None = None,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> ImageContainer:
        channel_dim = img[layer].dims[-1]
        if img[layer].shape[-1] == 1:
            new_channel_dim = channel_dim
        else:
            new_channel_dim = f"{channel_dim}:{'all' if channel is None else channel}"

        _ = kwargs.pop("copy", None)
        # TODO(michalk8): allow volumetric segmentation? (precondition/postcondition needs change)
        if isinstance(library_id, str):
            func = {library_id: self.segment}
        elif isinstance(library_id, Sequence):
            func = {lid: self.segment for lid in library_id}
        else:
            raise TypeError(
                f"Expected library id to be `None` or of type `str` or `sequence`, found `{type(library_id).__name__}`."
            )

        res: ImageContainer = img.apply(func, layer=layer, channel=channel, fn_kwargs=fn_kwargs, copy=True, **kwargs)
        res._data = res.data.rename({channel_dim: new_channel_dim})

        for k in res:
            res[k].attrs["segmentation"] = True

        return res

    @abstractmethod
    def _segment(self, arr: NDArrayA, **kwargs: Any) -> NDArrayA:
        pass

    def _segment_chunk(
        self,
        block: NDArrayA,
        block_id: tuple[int, ...],
        num_blocks: tuple[int, ...],
        shift: int,
        **kwargs: Any,
    ) -> NDArrayA:
        if len(num_blocks) == 2:
            block_num = block_id[0] * num_blocks[1] + block_id[1]
        elif len(num_blocks) == 3:
            block_num = block_id[0] * (num_blocks[1] * num_blocks[2]) + block_id[1] * num_blocks[2]
        elif len(num_blocks) == 4:
            if num_blocks[-1] != 1:
                raise ValueError(
                    f"Expected the number of blocks in the Z-dimension to be `1`, found `{num_blocks[-1]}`."
                )
            block_num = block_id[0] * (num_blocks[1] * num_blocks[2]) + block_id[1] * num_blocks[2]
        else:
            raise ValueError(f"Expected either `2`, `3` or `4` dimensional chunks, found `{len(num_blocks)}`.")

        labels = self._segment(block, **kwargs).astype(_SEG_DTYPE)
        mask: NDArrayA = labels > 0
        labels[mask] = (labels[mask] << shift) | block_num

        return labels

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return repr(self)


class SegmentationWatershed(SegmentationModel):
    """Segmentation model based on :mod:`skimage` watershed segmentation."""

    def __init__(self) -> None:
        super().__init__(model=None)

    def _segment(
        self,
        arr: NDArrayA,
        thresh: float | None = None,
        geq: bool = True,
        **kwargs: Any,
    ) -> NDArrayA | da.Array:
        arr = arr.squeeze(-1)  # we always pass a 3D image
        if thresh is None:
            thresh = threshold_otsu(arr)
        mask: NDArrayA = (arr >= thresh) if geq else (arr < thresh)
        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=mask)
        local_maxi = np.zeros(distance.shape, dtype=np.bool_)
        local_maxi[tuple(coords.T)] = True

        markers, _ = ndi.label(local_maxi)

        return np.asarray(watershed(-distance, markers, mask=mask))


class SegmentationCustom(SegmentationModel):
    """
    Segmentation model based on a user-defined function.

    Parameters
    ----------
    func
        Segmentation function to use. Can be any :func:`callable`, as long as it has the following signature:
        :class:`numpy.ndarray` ``(height, width, channels)`` **->** :class:`numpy.ndarray` ``(height, width[, 1])``.
        The segmentation must be of :class:`numpy.uint32` type, where 0 marks background.
    """

    def __init__(self, func: Callable[..., NDArrayA]):
        if not callable(func):
            raise TypeError()
        super().__init__(model=func)

    def _segment(self, arr: NDArrayA, **kwargs: Any) -> NDArrayA:
        return np.asarray(self._model(arr, **kwargs))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[function={getattr(self._model, '__name__', None)}]"

    def __str__(self) -> str:
        return repr(self)


@d.dedent
@inject_docs(m=SegmentationBackend)
def segment(
    img: ImageContainer,
    layer: str | None = None,
    library_id: str | Sequence[str] | None = None,
    method: str | SegmentationModel | Callable[..., NDArrayA] = "watershed",
    channel: int | None = 0,
    chunks: str | int | tuple[int, int] | None = None,
    lazy: bool = False,
    layer_added: str | None = None,
    copy: bool = False,
    **kwargs: Any,
) -> ImageContainer | None:
    """
    Segment an image.

    Parameters
    ----------
    %(img_container)s
    %(img_layer)s
    %(library_id)s
        If `None`, all Z-dimensions are segmented separately.
    method
        Segmentation method to use. Valid options are:

            - `{m.WATERSHED.s!r}` - :func:`skimage.segmentation.watershed`.

        %(custom_fn)s
    channel
        Channel index to use for segmentation. If `None`, use all channels.
    %(chunks_lazy)s
    %(layer_added)s If `None`, use ``'segmented_{{model}}'``.
    thresh
        Threshold for creation of masked image. The areas to segment should be contained in this mask.
        If `None`, it is determined by `Otsu's method <https://en.wikipedia.org/wiki/Otsu%27s_method>`_.
        Only used if ``method = {m.WATERSHED.s!r}``.
    geq
        Treat ``thresh`` as upper or lower bound for defining areas to segment. If ``geq = True``, mask is defined
        as ``mask = arr >= thresh``, meaning high values in ``arr`` denote areas to segment.
        Only used if ``method = {m.WATERSHED.s!r}``.
    %(copy_cont)s
    %(segment_kwargs)s

    Returns
    -------
    If ``copy = True``, returns a new container with the segmented image in ``'{{layer_added}}'``.

    Otherwise, modifies the ``img`` with the following key:

        - :class:`squidpy.im.ImageContainer` ``['{{layer_added}}']`` - the segmented image.
    """
    layer = img._get_layer(layer)
    kind = SegmentationBackend.CUSTOM if callable(method) else SegmentationBackend(method)
    layer_new = Key.img.segment(kind, layer_added=layer_added)
    kwargs["chunks"] = chunks
    library_id = img._get_library_ids(library_id)

    if not isinstance(method, SegmentationModel):
        if kind == SegmentationBackend.WATERSHED:
            if channel is None and img[layer].shape[-1] > 1:
                raise ValueError("Watershed segmentation does not work with multiple channels.")
            method: SegmentationModel = SegmentationWatershed()  # type: ignore[no-redef]
        elif kind == SegmentationBackend.CUSTOM:
            if not callable(method):
                raise TypeError(f"Expected `method` to be a callable, found `{type(method)}`.")
            method = SegmentationCustom(func=method)
        else:
            raise NotImplementedError(f"Model `{kind}` is not yet implemented.")

    if TYPE_CHECKING:
        assert isinstance(method, SegmentationModel)

    start = logg.info(f"Segmenting an image of shape `{img[layer].shape}` using `{method}`")
    res: ImageContainer = method.segment(
        img,
        layer=layer,
        channel=channel,
        library_id=library_id,
        chunks=None,
        fn_kwargs=kwargs,
        copy=True,
        drop=copy,
        lazy=lazy,
    )
    logg.info("Finish", time=start)

    if copy:
        return res.rename(layer, layer_new)

    img.add_img(
        res[layer],
        layer=layer_new,
        copy=False,
        lazy=lazy,
        dims=res[layer].dims,
        library_id=res[layer].coords["z"].values,
    )
