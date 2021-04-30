from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (
    Any,
    List,
    Tuple,
    Union,
    Mapping,
    Callable,
    Optional,
    Sequence,
    TYPE_CHECKING,
)
from dask_image.ndmeasure import label

from dask import delayed
from scipy import ndimage as ndi
import numpy as np
import dask.array as da

from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, singledispatchmethod
from squidpy.im._container import ImageContainer
from squidpy._constants._constants import SegmentationBackend

__all__ = ["SegmentationModel", "SegmentationWatershed", "SegmentationCustom"]


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
    def segment(self, img: Union[np.ndarray, ImageContainer], **kwargs: Any) -> Union[np.ndarray, ImageContainer]:
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
        Segmentation mask for the high-resolution image of shape ``(height, width, 1)``.

        Raises
        ------
        ValueError
            If the number of dimensions is neither 2 nor 3 or if there are more than 1 channels.
        NotImplementedError
            If trying to segment a type for which the segmentation has not been registered.
        """
        raise NotImplementedError(f"Segmentation of `{type(img).__name__}` is not yet implemented.")

    @staticmethod
    def _precondition(img: Union[np.ndarray, da.Array]) -> Union[np.ndarray, da.Array]:
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"Expected `3` dimensions, found `{img.ndim}`.")
        return img

    @staticmethod
    def _postcondition(img: Union[np.ndarray, da.Array]) -> Union[np.ndarray, da.Array]:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"Expected segmentation to return `3` dimensional array, found `{img.ndim}`.")

        return img

    @segment.register(da.Array)
    @segment.register(np.ndarray)
    def _(self, img: np.ndarray, **kwargs: Any) -> np.ndarray:
        img = SegmentationModel._precondition(img)
        img = self._segment(img, **kwargs)
        return SegmentationModel._postcondition(img)

    @segment.register(ImageContainer)  # type: ignore[no-redef]
    def _(
        self,
        img: ImageContainer,
        layer: str,
        channel: Optional[int] = None,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> ImageContainer:
        # simple inversion of control, we rename the channel dim later
        return img.apply(self.segment, layer=layer, channel=channel, fn_kwargs=fn_kwargs, **kwargs)

    @abstractmethod
    def _segment(self, arr: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return repr(self)


class SegmentationWatershed(SegmentationModel):
    """Segmentation model based on :mod:`skimage` watershed segmentation."""

    def __init__(self) -> None:
        super().__init__(model=None)

    def _segment_dask(
        self,
        arr: da.Array,
        thresh: Optional[float] = None,
        geq: bool = True,
        chunks: Union[int, str, Tuple[int, ...]] = "auto",
        **_: Any,
    ) -> da.Array:
        def _local_maxi(mask: np.ndarray, distance: np.ndarray) -> np.ndarray:
            coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=mask)
            local_maxi = np.zeros(distance.shape, dtype=np.bool_)
            local_maxi[tuple(coords.T)] = True

            return local_maxi

        def _distance(mask: np.ndarray) -> np.ndarray:
            return ndi.distance_transform_edt(mask)  # type: ignore[no-any-return]

        def _watershed(img: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
            return watershed(img, markers, mask=mask)  # type: ignore[no-any-return]

        arr = arr.rechunk(chunks).squeeze(-1)  # we always pass 3D image
        if thresh is None:
            thresh = da.from_delayed(delayed(threshold_otsu)(arr), shape=(), dtype=arr.dtype)
        mask = (arr >= thresh) if geq else (arr < thresh)

        distance = mask.map_blocks(_distance)
        local_maxi = da.map_blocks(_local_maxi, mask, distance, dtype=bool)
        markers, _ = label(local_maxi)

        return da.map_overlap(_watershed, -distance, markers, mask, depth=10, dtype=int)

    def _segment(
        self,
        arr: np.ndarray,
        thresh: Optional[float] = None,
        geq: bool = True,
        chunks: Optional[Union[int, str, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, da.Array]:
        # TODO: better approach?
        if chunks is not None:
            return self._segment_dask(arr, thresh=thresh, geq=geq, chunks=chunks, **kwargs)

        arr = arr.squeeze(-1)  # we always pass 3D image
        if thresh is None:
            thresh = threshold_otsu(arr)
        mask = (arr >= thresh) if geq else (arr < thresh)

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
        :class:`numpy.ndarray` ``(height, width, channels)`` **->** :class:`numpy.ndarray` ``(height, width[, channels])``.
    """  # noqa: E501

    def __init__(self, func: Callable[..., np.ndarray]):
        if not callable(func):
            raise TypeError()
        super().__init__(model=func)

    def _segment(self, arr: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model(arr, **kwargs))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[function={getattr(self._model, '__name__', None)}]"

    def __str__(self) -> str:
        return repr(self)


@d.dedent
@inject_docs(m=SegmentationBackend)
def segment(
    img: ImageContainer,
    layer: Optional[str] = None,
    method: Union[str, SegmentationModel, Callable[..., np.ndarray]] = "watershed",
    channel: Optional[int] = 0,
    size: Optional[Union[str, int, Tuple[int, int]]] = None,
    layer_added: Optional[str] = None,
    copy: bool = False,
    show_progress_bar: bool = True,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    **kwargs: Any,
) -> Optional[ImageContainer]:
    """
    Segment an image.

    If ``size`` is defined, iterate over crops of that size and segment those. Recommended for large images.

    Parameters
    ----------
    %(img_container)s
    %(img_layer)s
    %(seg_blob.parameters)s
            - `{m.WATERSHED.s!r}` - :func:`skimage.segmentation.watershed`.

        %(custom_fn)s
    channel
        Channel index to use for segmentation. If `None`, pass all channels.
    %(size)s
    %(layer_added)s
        If `None`, use ``'segmented_{{model}}'``.
    thresh
        Threshold for creation of masked image. The areas to segment should be contained in this mask.
        If `None`, it is determined by `Otsu's method <https://en.wikipedia.org/wiki/Otsu%27s_method>`_.
        Only used if ``method = {m.WATERSHED.s!r}``.
    geq
        Treat ``thresh`` as upper or lower bound for defining areas to segment. If ``geq = True``, mask is defined
        as ``mask = arr >= thresh``, meaning high values in ``arr`` denote areas to segment.
    invert
        Whether to segment an inverted array. Only used if ``method`` is one of :mod:`skimage` blob methods.
    %(copy_cont)s
    %(segment_kwargs)s
    %(parallelize)s
    kwargs
        Keyword arguments for ``method``.

    Returns
    -------
    If ``copy = True``, returns a new container with the segmented image in ``'{{layer_added}}'``.

    Otherwise, modifies the ``img`` with the following key:

        - :class:`squidpy.im.ImageContainer` ``['{{layer_added}}']`` - the segmented image.
    """
    layer = img._get_layer(layer)
    img[layer].dims[-1]

    # layer_new = Key.img.segment(kind, layer_added=layer_added)

    if not isinstance(method, SegmentationModel):
        kind = SegmentationBackend.CUSTOM if callable(method) else SegmentationBackend(method)
        if kind == SegmentationBackend.WATERSHED:
            method: SegmentationModel = SegmentationWatershed()  # type: ignore[no-redef]
        elif kind == SegmentationBackend.CUSTOM:
            if not callable(method):
                raise TypeError(f"Expected `method` to be a callable, found `{type(method)}`.")
            method = SegmentationCustom(func=method)
        else:
            raise NotImplementedError(f"Model `{kind}` is not yet implemented.")

    if TYPE_CHECKING:
        assert isinstance(method, SegmentationModel)

    # TODO: see TODOs in ImageContainer.apply (_is_delayed)
    res: ImageContainer = method.segment(img, layer=layer, channel=channel, fn_kwargs=kwargs, _is_delayed=True)
    for k in res:
        res[k].attrs["segmentation"] = True

    return res
    # TODO: handle the stuff below (most likely remove the par. part, not needed)

    """
    n_jobs = _get_n_cores(n_jobs)
    crops: List[ImageContainer] = list(img.generate_equal_crops(size=size, as_array=False))
    start = logg.info(f"Segmenting `{len(crops)}` crops using `{segmentation_model}` and `{n_jobs}` core(s)")

    crops: List[ImageContainer] = parallelize(  # type: ignore[no-redef]
        _segment,
        collection=crops,
        unit="crop",
        extractor=lambda res: list(chain.from_iterable(res)),
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar and len(crops) > 1,
    )(model=segmentation_model, layer=layer, layer_new=layer_new, channel=channel, **kwargs)

    res: ImageContainer = ImageContainer.uncrop(crops, shape=img.shape)
    res._data = res.data.rename({channel_dim: f"{channel_dim}:{channel}"})
    for k in res:
        res[k].attrs["segmentation"] = True

    logg.info("Finish", time=start)

    if copy:
        return res

    img.add_img(res, layer=layer_new, copy=False, channel_dim=res[layer_new].dims[-1])
    """


def _segment(
    crops: Sequence[ImageContainer],
    model: SegmentationModel,
    layer: str,
    layer_new: str,
    channel: int,
    queue: Optional[SigQueue] = None,
    **kwargs: Any,
) -> List[ImageContainer]:
    segmented_crops = []
    for crop in crops:
        crop = model.segment(crop, layer=layer, channel=channel, **kwargs)
        crop._data = crop.data.rename({layer: layer_new})
        segmented_crops.append(crop)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return segmented_crops
