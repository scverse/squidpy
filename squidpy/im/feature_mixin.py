from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Callable, Iterable, Optional, Sequence
from typing_extensions import Protocol

import numpy as np
import xarray as xr

from skimage.util import img_as_ubyte
from skimage.feature import greycoprops, greycomatrix
import skimage.measure

from squidpy._docs import d

Feature_t = Dict[str, Any]
Channel_t = Union[int, Sequence[int]]


def _get_channels(xr_img: xr.DataArray, channels: Optional[Channel_t]) -> List[int]:
    """Get correct channel ranges for feature calculation."""
    # if channels is None, compute features for all channels
    if channels is None:
        return list(range(xr_img.shape[-1]))
    if isinstance(channels, int):
        return [channels]

    return list(channels)


# define protocol to get rid of mypy indexing errors
class HasGetItemProtocol(Protocol):
    """Protocol for FeatureMixin to have correct definition of ImageContainer."""

    def __getitem__(self, key: str) -> xr.DataArray:
        ...


class FeatureMixin:
    """Mixin class for ImageContainer implementing feature extraction functions."""

    @d.dedent
    def get_summary_features(
        self: HasGetItemProtocol,
        img_id: str,
        feature_name: str = "summary",
        channels: Optional[Channel_t] = None,
        quantiles: Sequence[float] = (0.9, 0.5, 0.1),
        # TODO: mean/std argument not really necessary?
        mean: bool = False,
        std: bool = False,
    ) -> Feature_t:
        """
        Calculate summary statistics of image channels.

        Parameters
        ----------
        %(img_id)s
        %(feature_name)s
        %(channels)s
        quantiles
            Quantiles that are computed.
        mean
            Compute mean.
        std
            Compute std deviation.

        Returns
        -------
        Returns features with the following keys for each channel `c` in ``channels``:

            - `'{feature_name}_ch-{c}_quantile-{q}'` - the quantile features for each quantile `q` in ``quantiles``.
            - `'{feature_name}_ch-{c}_mean'` - the mean.
            - `'{feature_name}_ch-{c}_std'` - the standard deviation.
        """
        channels = _get_channels(self[img_id], channels)

        features = {}
        for c in channels:
            for q in quantiles:
                features[f"{feature_name}_ch-{c}_quantile-{q}"] = np.quantile(self[img_id][:, :, c], q)
            if mean:
                features[f"{feature_name}_ch-{c}_mean"] = np.mean(self[img_id][:, :, c].values)
            if std:
                features[f"{feature_name}_ch-{c}_std"] = np.std(self[img_id][:, :, c].values)

        return features

    @d.dedent
    def get_histogram_features(
        self: HasGetItemProtocol,
        img_id: str,
        feature_name: str = "histogram",
        channels: Optional[Channel_t] = None,
        bins: int = 10,
        v_range: Optional[Tuple[int, int]] = None,
    ) -> Feature_t:
        """
        Compute histogram counts of color channel values.

        Returns one feature per bin and channel.

        Parameters
        ----------
        %(img_id)s
        %(feature_name)s
        %(channels)s
        bins
            Number of binned value intervals.
        v_range
            Range on which values are binned. If `None`, use the whole image range.

        Returns
        -------
        Returns features with the following keys for each channel `c` in ``channels``:

            - `'{feature_name}_ch-{c}_bin-{i}'` - the histogram counts for each bin `i` in ``bins``.
        """
        channels = _get_channels(self[img_id], channels)
        # if v_range is None, use whole-image range
        if v_range is None:
            v_range = np.min(self[img_id].values), np.max(self[img_id].values)

        features = {}
        for c in channels:
            hist, _ = np.histogram(self[img_id][:, :, c], bins=bins, range=v_range, weights=None, density=False)
            for i, count in enumerate(hist):
                features[f"{feature_name}_ch-{c}_bin-{i}"] = count

        return features

    @d.dedent
    def get_texture_features(
        self: HasGetItemProtocol,
        img_id: str,
        feature_name: str = "texture",
        channels: Optional[Channel_t] = None,
        props: Iterable[str] = ("contrast", "dissimilarity", "homogeneity", "correlation", "ASM"),
        distances: Iterable[int] = (1,),
        angles: Iterable[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    ) -> Feature_t:
        """
        Calculate texture features.

        A grey level co-occurrence matrix (`GLCM <https://en.wikipedia.org/wiki/Co-occurrence_matrix>`_) is computed
        for different combinations of distance and angle.

        The distance defines the pixel difference of co-occurrence. The angle define the direction along which
        we check for co-occurrence. The GLCM includes the number of times that grey-level :math:`j` occurs at a distance
        :math:`d` and at an angle theta from grey-level :math:`i`.

        Parameters
        ----------
        %(img_id)s
        %(feature_name)s
        %(channels)s
        props
            Texture features that are calculated, see the `prop` argument in :func:`skimage.feature.greycoprops`.
        distances
            The `distances` argument in :func:`skimage.feature.greycomatrix`.
        angles
            The `angles` argument in :func:`skimage.feature.greycomatrix`.

        Returns
        -------
        Returns features with the following keys for each channel `c` in ``channels``:

            - `'{feature_name}_ch-{c}_{p}_dist-{dist}_angle-{a}'` - the GLCM properties, for each `p` in ``props``,
              `d` in ``distances`` and `a` in ``angles``.

        Notes
        -----
        If the image is not of type :class:`numpy.uint8`, it will be converted.
        """
        channels = _get_channels(self[img_id], channels)
        arr = self[img_id][..., channels].values

        # check that image has values < 256
        if not np.issubdtype(arr.dtype, np.uint8):
            arr = img_as_ubyte(arr, force_copy=False)

        features = {}
        for c in channels:
            comatrix = greycomatrix(arr[..., c], distances=distances, angles=angles, levels=256)
            for p in props:
                tmp_features = greycoprops(comatrix, prop=p)
                for d_idx, dist in enumerate(distances):
                    for a_idx, a in enumerate(angles):
                        features[f"{feature_name}_ch-{c}_{p}_dist-{dist}_angle-{a:.2f}"] = tmp_features[d_idx, a_idx]
        return features

    @d.dedent
    def get_segmentation_features(
        self: HasGetItemProtocol,
        img_id: str,
        label_img_id: str,
        feature_name: str = "segmentation",
        channels: Optional[Channel_t] = None,
        props: Iterable[str] = ("label", "area", "mean_intensity"),
        # TODO: mean/std argument not really necessary?
        mean: bool = True,
        std: bool = False,
    ) -> Feature_t:
        """
        Calculate segmentation features using :func:`skimage.measure.regionprops`.

        Features are calculated using ``label_img_id``, a cell segmentation of ``img_id``
        (e.g. resulting from calling :func:`squidpy.im.segment_img`).

        Depending on the specified parameters, mean and std of the requested props are returned.
        For the `'label'` feature, the number of labels is returned, i.e. the number of cells in this img.

        Parameters
        ----------
        %(img_container)s
        %(img_id)s
        %(feature_name)s
        %(channels)s
            Only relevant for features that use the intensity image ``img_id``.
        props
            Segmentation features that are calculated. See `properties` in :func:`skimage.measure.regionprops_table`.
            Valid options are:

                - `'area'`
                - `'bbox_area'`
                - `'convex_area'`
                - `'eccentricity'`
                - `'equivalent_diameter'`
                - `'euler_number'`
                - `'extent'`
                - `'feret_diameter_max'`
                - `'filled_area'`
                - `'label'`
                - `'major_axis_length'`
                - `'max_intensity'` - uses intensity image ``img_id``.
                - `'mean_intensity'` - uses intensity image ``img_id``.
                - `'min_intensity'` - uses intensity image ``img_id``.
                - `'minor_axis_length'`
                - `'orientation'`
                - `'perimeter'`
                - `'perimeter_crofton'`
                - `'solidity'`

        mean
            Return mean feature values.
        std
            Return std feature values.

        Returns
        -------
        Returns features with the following keys:

            - `'{feature_name}_label'` - if `'label`` was in ``props``.
            - `'{feature_name}_{p}_mean'` - mean for each non-intensity property `p` in ``props``.
            - `'{feature_name}_{p}_std'` - standard deviation for each non-intensity property `p` in ``props``.
            - `'{feature_name}_ch-{c}_{p}_mean'` - mean for each intensity property `p` in ``props`` and channel `c` in
              ``channels``.
            - `'{feature_name}_ch-{c}_{p}_std'` - standard deviation for each intensity property `p` in ``props`` and
              channel `c` in ``channels``.
        """
        # TODO check that passed a valid prop
        channels = _get_channels(self[img_id], channels)

        features = {}
        # calculate features that do not depend on the intensity image
        no_intensity_props = [p for p in props if "intensity" not in p]
        tmp_features = skimage.measure.regionprops_table(
            self[label_img_id].values[:, :, 0], properties=no_intensity_props
        )
        for p in no_intensity_props:
            if p == "label":
                features[f"{feature_name}_{p}"] = len(tmp_features["label"])
            else:
                if mean:
                    features[f"{feature_name}_{p}_mean"] = np.mean(tmp_features[p])
                if std:
                    features[f"{feature_name}_{p}_std"] = np.std(tmp_features[p])

        # calculate features that depend on the intensity image
        intensity_props = [p for p in props if "intensity" in p]
        for c in channels:
            tmp_features = skimage.measure.regionprops_table(
                self[label_img_id].values[:, :, 0], intensity_image=self[img_id].values[:, :, c], properties=props
            )
            for p in intensity_props:
                if mean:
                    features[f"{feature_name}_ch-{c}_{p}_mean"] = np.mean(tmp_features[p])
                if std:
                    features[f"{feature_name}_ch-{c}_{p}_std"] = np.std(tmp_features[p])

        return features

    @d.dedent
    def get_custom_features(
        self: HasGetItemProtocol,
        img_id: str,
        func: Callable[[np.ndarray], Any],
        channels: Optional[Channel_t] = None,
        feature_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Feature_t:
        """
        Calculate custom features using ``func``.

        The feature extractor ``func`` takes as an input :class:`numpy.ndarray` of shape ``(y, x, channels)`` and
        optional ``kwargs`` and needs to return one or more of :class:`float`.

        Parameters
        ----------
        %(img_id)s
        func
            Feature extraction function.
        %(channels)s
        %(feature_name)s
        kwargs
            Keyword arguments for ``func``.

        Returns
        -------
        Returns features with the following keys:

            - `'{feature_name}_{i}'` - i-th feature value.

        Examples
        --------
        Simple example would be to calculate the mean of a specified channel::

            img = squidpy.im.ImageContainer(...)
            img.get_custom_features(imd_id=..., func=numpy.mean, channels=0)

        This can also be done by passing ``mean = True`` to :meth:`squidpy.im.ImageContainer.get_summary_features`.
        """
        channels = _get_channels(self[img_id], channels)
        feature_name = getattr(func, "__name__", "custom") if feature_name is None else feature_name

        # calculate features by calling feature_fn
        res = func(self[img_id].values[:, :, channels], **kwargs)  # type: ignore[call-arg]
        if not isinstance(res, Iterable):
            res = [res]
        features = {f"{feature_name}_{i}": f for i, f in enumerate(res)}

        return features
