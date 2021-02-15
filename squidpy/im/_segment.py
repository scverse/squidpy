from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Callable, Optional, Sequence, TYPE_CHECKING
from itertools import chain

from scanpy import logging as logg

from scipy import ndimage as ndi
import numpy as np

from skimage.util import invert as invert_arr, img_as_float
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
import skimage

from squidpy._docs import d, inject_docs
from squidpy._utils import (
    Signal,
    SigQueue,
    parallelize,
    _get_n_cores,
    singledispatchmethod,
)
from squidpy.gr._utils import _assert_in_range
from squidpy.im._utils import _circular_mask
from squidpy.im._container import ImageContainer
from squidpy._constants._constants import SegmentationBackend
from squidpy._constants._pkg_constants import Key

__all__ = ["SegmentationModel", "SegmentationWatershed", "SegmentationBlob", "SegmentationCustom"]


class SegmentationModel(ABC):
    """
    Base class for all segmentation models.

    Contains core shared functions related contained to cell and nuclei segmentation.
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

    @segment.register(np.ndarray)
    def _(self, img: np.ndarray, **kwargs: Any) -> np.ndarray:
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"Expected `3` dimensions, found `{img.ndim}`.")
        if img.shape[-1] != 1:
            raise ValueError(f"Expected only `1` channel, found `{img.shape[-1]}`.")

        arr = self._segment(img, **kwargs)

        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"Expected segmentation to return `3` dimensional array, found `{arr.ndim}`.")

        return arr

    @segment.register(ImageContainer)  # type: ignore[no-redef]
    def _(self, img: ImageContainer, layer: str, channel: int = 0, **kwargs: Any) -> ImageContainer:
        # simple inversion of control, we rename the channel dim later
        return img.apply(self.segment, layer=layer, channel=channel, **kwargs)

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

    def _segment(self, arr: np.ndarray, thresh: Optional[float] = None, geq: bool = True, **kwargs: Any) -> np.ndarray:
        arr = arr.squeeze(-1)  # we always pass 3D image

        if not np.issubdtype(arr.dtype, np.floating):
            arr = img_as_float(arr, force_copy=False)

        if thresh is None:
            thresh = threshold_otsu(arr)
        else:
            _assert_in_range(thresh, 0, 1, name="thresh")

        # get binarized image
        if geq:
            mask = arr >= thresh
            arr = invert_arr(arr)
        else:
            mask = arr < thresh

        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=mask)
        local_maxi = np.zeros(distance.shape, dtype=np.bool_)
        local_maxi[tuple(coords.T)] = True

        markers, _ = ndi.label(local_maxi)

        return np.asarray(watershed(arr, markers, mask=mask))


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


@d.get_sections(base="seg_blob", sections=["Parameters"])
@inject_docs(m=SegmentationBackend)
class SegmentationBlob(SegmentationCustom):
    """
    Segmentation model based on :mod:`skimage` blob detection.

    Parameters
    ----------
    model
        Segmentation method to use. Valid options are:

            - `{m.LOG.s!r}` - :func:`skimage.feature.blob_log`. Blobs are assumed to be light on dark.
            - `{m.DOG.s!r}` - :mod:`skimage.feature.blob_dog`. Blobs are assumed to be light on dark.
            - `{m.DOH.s!r}` - :mod:`skimage.feature.blob_doh`. Blobs can be light on dark or vice versa.
    """

    def __init__(self, model: SegmentationBackend):
        model = SegmentationBackend(model)
        if model == SegmentationBackend.LOG:
            func = skimage.feature.blob_log
        elif model == SegmentationBackend.DOG:
            func = skimage.feature.blob_dog
        elif model == SegmentationBackend.DOH:
            func = skimage.feature.blob_doh
        else:
            raise NotImplementedError(f"Unknown blob model `{model}`.")

        super().__init__(func=func)

    def _segment(self, arr: np.ndarray, invert: bool = False, **kwargs: Any) -> np.ndarray:
        arr = arr.squeeze(-1)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = img_as_float(arr, force_copy=False)
        if invert:
            arr = invert_arr(arr)

        blob_mask = np.zeros_like(arr, dtype=np.bool_)
        # invalid value encountered in double_scalar, invalid value encountered in subtract
        with np.errstate(divide="ignore", invalid="ignore"):
            blobs = self._model(arr, **kwargs)

        for blob in blobs:
            blob_mask[_circular_mask(blob_mask, *blob)] = True

        return blob_mask


@d.dedent
@inject_docs(m=SegmentationBackend)
def segment(
    img: ImageContainer,
    layer: Optional[str] = None,
    method: Union[str, Callable[..., np.ndarray]] = "watershed",
    channel: int = 0,
    size: Optional[Union[int, Tuple[int, int]]] = None,
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
        Channel index to use for segmentation.
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
    channel_dim = img[layer].dims[-1]

    kind = SegmentationBackend.CUSTOM if callable(method) else SegmentationBackend(method)
    layer_new = Key.img.segment(kind, layer_added=layer_added)

    if kind in (SegmentationBackend.LOG, SegmentationBackend.DOG, SegmentationBackend.DOH):
        segmentation_model: SegmentationModel = SegmentationBlob(model=kind)
    elif kind == SegmentationBackend.WATERSHED:
        segmentation_model = SegmentationWatershed()
    elif kind == SegmentationBackend.CUSTOM:
        if TYPE_CHECKING:
            assert callable(method)
        segmentation_model = SegmentationCustom(func=method)
    else:
        raise NotImplementedError(f"Model `{kind}` is not yet implemented.")

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

    if isinstance(segmentation_model, SegmentationWatershed):
        # By convention, segments are numbered from 1..number of segments within each crop.
        # Next, we have to account for that before merging the crops so that segments are not confused.
        # TODO use overlapping crops to not create confusion at boundaries
        counter = 0
        for crop in crops:
            data = crop[layer_new].data
            data[data > 0] += counter
            counter += np.max(crop[layer_new].data)

    res: ImageContainer = ImageContainer.uncrop(crops, shape=img.shape)
    res._data = res.data.rename({channel_dim: f"{channel_dim}:{channel}"})

    logg.info("Finish", time=start)

    if copy:
        return res

    img.add_img(res, layer=layer_new, copy=False, channel_dim=res[layer_new].dims[-1])


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
