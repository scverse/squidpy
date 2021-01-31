"""Functions exposed: segment(), evaluate_nuclei_segmentation()."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Callable, Optional, Sequence, TYPE_CHECKING
from itertools import chain
from multiprocessing import Manager

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
from squidpy.gr._utils import _assert_in_range, _assert_non_negative
from squidpy.im.object import ImageContainer
from squidpy._constants._constants import SegmentationBackend
from squidpy._constants._pkg_constants import Key


class Counter:
    """Atomic counter to work with :mod:`joblib`."""

    def __init__(self, value: int = 0):
        manager = Manager()
        self._value = manager.Value("i", value)
        self._lock = manager.Lock()

    def __iadd__(self, other: int) -> "Counter":
        if not isinstance(other, int):
            return NotImplemented  # type: ignore[unreachable]
        with self._lock:
            self._value.value += other

        return self

    @property
    def value(self) -> int:
        """Return the value."""
        with self._lock:
            return self._value.value

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return repr(self)


class SegmentationModel(ABC):
    """
    Base class for segmentation models.

    Contains core shared functions related contained to cell and nuclei segmentation.
    Specific segmentation models can be implemented by inheriting from this class.

    This class is not instantiated by user but used in the background by the functional API.

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
    def segment(self, img: Union[np.ndarray, ImageContainer], **_kwargs: Any) -> Union[np.ndarray, ImageContainer]:
        """
        Segment an image.

        Parameters
        ----------
        %(img_hr)s
        increment
            By default, segments are numbered from `1`. TODO better name.
        img_id
            Only used when ``image`` is :class:`squidpy.im.ImageContainer`.

        Returns
        -------
        Segmentation mask for the high-resolution image of shape `(x, y, 1)`.
        """
        raise NotImplementedError(type(img))

    @segment.register(np.ndarray)
    def _(self, img: np.ndarray, increment: int = 0, **kwargs: Any) -> np.ndarray:
        _assert_non_negative(increment, name="start")
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"Expected `3` dimensions, found `{img.ndim}`.")
        if img.shape[-1] != 1:
            raise ValueError(f"Expected only `1` channel, found `{img.shape[-1]}`.")

        arr = self._segment(img, **kwargs)

        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]

        # TODO: only for watershed?
        if increment > 0:
            arr[arr > 0] = arr[arr > 0] + increment

        return arr

    @segment.register(ImageContainer)  # type: ignore[no-redef]
    def _(self, img: ImageContainer, img_id: str, channel: int = 0, **kwargs: Any) -> ImageContainer:
        arr = img[img_id]
        channel_id = arr.dims[-1]

        arr = arr[{channel_id: channel}].values  # channel name is last dimension of img
        arr = self.segment(arr, **kwargs)

        return ImageContainer(arr, img_id=img_id, channel_id=f"segmented_{channel_id}_{channel}")

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

    @d.dedent
    def _segment(self, arr: np.ndarray, thresh: Optional[float] = None, geq: bool = True, **kwargs: Any) -> np.ndarray:
        """
        %(segment.full_desc)s

        Parameters
        ----------
        %(segment.parameters)s
        thresh
             Threshold for creation of masked image. The areas to segment should be contained in this mask.
        geq
            Treat ``thresh`` as upper or lower (greater-equal = geq) bound for defining areas to segment.
            If ``geq = True``, mask is defined as ``mask = arr >= thresh``, meaning high values in ``arr``
            denote areas to segment.
        %(segment_kwargs)s

        Returns
        -------
        %(segment.returns)s
        """  # noqa: D400
        arr = arr.squeeze(-1)
        # TODO: should we do this?
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
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((5, 5)), labels=mask)
        markers, _ = ndi.label(local_maxi)

        return watershed(arr, markers, mask=mask)


class SegmentationGeneric(SegmentationModel):
    """
    TODO.

    Parameters
    ----------
    func
        Function which takes a :class:`numpy.ndarray` of shape TODO.
    """

    def __init__(self, func: Callable[..., np.ndarray]):
        if not callable(func):
            raise TypeError()
        super().__init__(model=func)

    @d.dedent
    def _segment(self, arr: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        %(segment.full_desc)s

        Parameters
        -----------
        %(segment.parameters)s
        %(segment_kwargs)s

        Returns
        -------
        %(segment.returns)s
        """  # noqa: D400
        return self._model(arr, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[function={getattr(self._model, '__name__', None)}]"

    def __str__(self) -> str:
        return repr(self)


class SegmentationBlob(SegmentationGeneric):
    """
    Segmentation model based on :mod:`skimage` blob detection.

    Parameters
    ----------
    kind
        TODO.
    """

    def __init__(self, kind: SegmentationBackend):
        kind = SegmentationBackend(kind)
        if kind == SegmentationBackend.LOG:
            func = skimage.feature.blob_log
        elif kind == SegmentationBackend.DOG:
            func = skimage.feature.blob_dog
        elif kind == SegmentationBackend.DOH:
            func = skimage.feature.blob_doh
        else:
            raise NotImplementedError(f"Unknown blob model `{kind}`.")

        super().__init__(func=func)

    def _segment(self, arr: np.ndarray, invert: bool = False, **kwargs: Any) -> np.ndarray:
        def circular_mask(arr: np.ndarray, y: int, x: int, radius: float) -> np.ndarray:
            Y, X = np.ogrid[: arr.shape[0], : arr.shape[1]]

            return ((Y - y) ** 2 + (X - x) ** 2) <= radius ** 2

        arr = arr.squeeze(-1)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = img_as_float(arr, force_copy=False)
        if invert:
            arr = invert_arr(arr)

        blob_mask = np.zeros_like(arr, dtype=np.bool_)
        # invalid value encountered in double_scalar
        # invalid value encountered in subtract
        with np.errstate(divide="ignore", invalid="ignore"):
            blobs = self._model(arr, **kwargs)

        for blob in blobs:
            blob_mask[circular_mask(blob_mask, *blob)] = True

        return blob_mask


@d.dedent
@inject_docs(m=SegmentationBackend)
def segment_img(
    img: ImageContainer,
    img_id: str,
    model: Union[str, Callable[..., np.ndarray]] = "watershed",
    channel: int = 0,
    yx: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    show_progress_bar: bool = True,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    **kwargs: Any,
) -> Optional[ImageContainer]:
    """
    %(segment.full_desc)s

    If ``xs`` and ``ys`` are defined, iterate over crops of size ``(xs, ys)`` and segment those. TODO.
    Recommended for large images.

    Parameters
    ----------
    %(img_container)s
    %(img_id)s
    model
        TODO.
        Segmentation method to use. Available are:

            - `{m.LOG.s!r}` - blob extraction with :mod:`skimage`. Blobs are assumed to be light on dark.
            - `{m.DOG.s!r}` - blob extraction with :mod:`skimage`. Blobs are assumed to be light on dark.
            - `{m.DOG.s!r}` - blob extraction with :mod:`skimage`. Blobs can be light on dark or vice versa.
            - `{m.WATERSHED.s!r}` - :func:`skimage.segmentation.watershed`.

        Alternatively, any :func:`callable` object can be passed as long as TODO.
    channel
        Channel index to use for segmentation.
    yx
        TODO.
    key_added
        Key of new image sized array to add into img object. If `None`, use ``'segmented_{{model}}'``.
    %(copy_cont)s
    %(segment_kwargs)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns segmented image as :class:`squidpy.im.ImageContainer` with a key based on ``key_added``.

    Otherwise, it modifies the ``img`` with the following key:

        - :class:`squidpy.im.ImageContainer` ``['{{key_added}}']`` - the segmented image.
    """  # noqa: D400
    kind = SegmentationBackend.GENERIC if callable(model) else SegmentationBackend(model)
    img_id_new = Key.img.segment(kind, key_added=key_added)

    if kind in (SegmentationBackend.LOG, SegmentationBackend.DOG, SegmentationBackend.DOH):
        segmentation_model: SegmentationModel = SegmentationBlob(kind=kind)
    elif kind == SegmentationBackend.WATERSHED:
        segmentation_model = SegmentationWatershed()
    elif kind == SegmentationBackend.GENERIC:
        if TYPE_CHECKING:
            assert callable(model)
        segmentation_model = SegmentationGeneric(func=model)
    else:
        raise NotImplementedError(f"Model `{kind}` is not yet implemented.")

    counter, n_jobs = Counter(), _get_n_cores(n_jobs)
    crops = list(img.generate_equal_crops(yx=yx, as_array=False))

    start = logg.info(f"Segmenting `{len(crops)}` crops using `{segmentation_model}` and `{n_jobs}` core(s)")
    res: ImageContainer = ImageContainer.uncrop_img(
        parallelize(
            _segment,
            collection=crops,
            unit="crop",
            extractor=lambda res: list(chain.from_iterable(res)),
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )(model=segmentation_model, img_id=img_id, img_id_new=img_id_new, counter=counter, channel=channel, **kwargs),
        shape=img.shape,
    )
    logg.info("Finish", time=start)

    if copy:
        return res

    img.add_img(res, img_id=img_id_new, copy=False)


def _segment(
    crops: Sequence[ImageContainer],
    model: SegmentationModel,
    img_id: str,
    img_id_new: str,
    channel: int,
    counter: Counter,
    queue: Optional[SigQueue] = None,
    **kwargs: Any,
) -> List[ImageContainer]:
    segmented_crops = []

    # By convention, segments are numbered from 1..number of segments within each crop.
    # Next, we have to account for that before merging the crops so that segments are not confused.
    # TODO use overlapping crops to not create confusion at boundaries
    for crop in crops:
        crop = model.segment(crop, img_id=img_id, channel=channel, increment=counter.value, **kwargs)
        crop._data = crop.data.rename({img_id: img_id_new})

        counter += int(np.max(crop[img_id_new].values))
        segmented_crops.append(crop)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return segmented_crops
