from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Protocol,
    Sequence,
    Union,
)

import numpy as np
import skimage.measure
import xarray as xr
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

from squidpy._constants._pkg_constants import Key
from squidpy._docs import d
from squidpy._utils import NDArrayA
from squidpy.gr._utils import _assert_non_empty_sequence
from squidpy.im._coords import _NULL_PADDING, CropCoords

Feature_t = dict[str, Any]
Channel_t = Union[int, Sequence[int]]


def _get_channels(xr_img: NDArrayA | xr.DataArray, channels: Channel_t | None) -> list[int]:
    """Get correct channel ranges for feature calculation."""
    # if channels is None, compute features for all channels
    all_channels = list(range(xr_img.shape[-1]))

    if channels is None:
        channels = all_channels
    if isinstance(channels, int):
        channels = [channels]

    for c in channels:
        if c not in all_channels:
            raise ValueError(f"Channel `{c}` is not in `{all_channels}`.")

    return list(channels)


_valid_seg_prop = sorted(
    {
        "area",
        "bbox_area",
        "centroid",
        "convex_area",
        "eccentricity",
        "equivalent_diameter",
        "euler_number",
        "extent",
        "feret_diameter_max",
        "filled_area",
        "label",
        "major_axis_length",
        "max_intensity",
        "mean_intensity",
        "min_intensity",
        "minor_axis_length",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        "solidity",
    }
)


# define protocol to get rid of mypy indexing errors
class HasGetItemProtocol(Protocol):
    """Protocol for FeatureMixin to have correct definition of ImageContainer."""

    def __getitem__(self, key: str) -> xr.DataArray: ...

    @property
    def data(self) -> xr.Dataset:  # noqa: D102
        ...

    def _get_layer(self, layer: str | None) -> str: ...

    def _get_library_id(self, library_id: str | None) -> str: ...


class FeatureMixin:
    """Mixin class for ImageContainer implementing feature extraction functions."""

    @d.dedent
    def features_summary(
        self: HasGetItemProtocol,
        layer: str,
        library_id: str | None = None,
        feature_name: str = "summary",
        channels: Channel_t | None = None,
        quantiles: Sequence[float] = (0.9, 0.5, 0.1),
    ) -> Feature_t:
        """
        Calculate summary statistics of image channels.

        Parameters
        ----------
        %(img_layer)s
        %(library_id_features)s
        %(feature_name)s
        %(channels)s
        quantiles
            Quantiles that are computed.

        Returns
        -------
        Returns features with the following keys for each channel `c` in ``channels``:

            - ``'{feature_name}_ch-{c}_quantile-{q}'`` - the quantile features for each quantile `q` in ``quantiles``.
            - ``'{feature_name}_ch-{c}_mean'`` - the mean.
            - ``'{feature_name}_ch-{c}_std'`` - the standard deviation.
        """
        layer = self._get_layer(layer)
        library_id = self._get_library_id(library_id)
        arr = self[layer].sel(z=library_id)

        quantiles = _assert_non_empty_sequence(quantiles, name="quantiles")
        channels = _get_channels(arr, channels)
        channels = _assert_non_empty_sequence(channels, name="channels")

        features = {}
        for c in channels:
            tmp_arr = arr[..., c].values
            for q in quantiles:
                features[f"{feature_name}_ch-{c}_quantile-{q}"] = np.quantile(tmp_arr, q)
            features[f"{feature_name}_ch-{c}_mean"] = np.mean(tmp_arr)
            features[f"{feature_name}_ch-{c}_std"] = np.std(tmp_arr)

        return features

    @d.dedent
    def features_histogram(
        self: HasGetItemProtocol,
        layer: str,
        library_id: str | None = None,
        feature_name: str = "histogram",
        channels: Channel_t | None = None,
        bins: int = 10,
        v_range: tuple[int, int] | None = None,
    ) -> Feature_t:
        """
        Compute histogram counts of color channel values.

        Returns one feature per bin and channel.

        Parameters
        ----------
        %(img_layer)s
        %(library_id_features)s
        %(feature_name)s
        %(channels)s
        bins
            Number of binned value intervals.
        v_range
            Range on which values are binned. If `None`, use the whole image range.

        Returns
        -------
        Returns features with the following keys for each channel `c` in ``channels``:

            - ``'{feature_name}_ch-{c}_bin-{i}'`` - the histogram counts for each bin `i` in ``bins``.
        """
        layer = self._get_layer(layer)
        library_id = self._get_library_id(library_id)
        arr = self[layer].sel(z=library_id)

        channels = _get_channels(arr, channels)
        channels = _assert_non_empty_sequence(channels, name="channels")

        # if v_range is None, use whole-image range
        if v_range is None:
            v_range = np.min(arr.values), np.max(arr.values)

        features = {}
        for c in channels:
            hist, _ = np.histogram(arr[..., c].values, bins=bins, range=v_range, weights=None, density=False)
            for i, count in enumerate(hist):
                features[f"{feature_name}_ch-{c}_bin-{i}"] = count

        return features

    @d.dedent
    def features_texture(
        self: HasGetItemProtocol,
        layer: str,
        library_id: str | None = None,
        feature_name: str = "texture",
        channels: Channel_t | None = None,
        props: Sequence[str] = ("contrast", "dissimilarity", "homogeneity", "correlation", "ASM"),
        distances: Sequence[int] = (1,),
        angles: Sequence[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    ) -> Feature_t:
        """
        Calculate texture features.

        A gray level co-occurrence matrix (`GLCM <https://en.wikipedia.org/wiki/Co-occurrence_matrix>`_) is computed
        for different combinations of distance and angle.

        The distance defines the pixel difference of co-occurrence. The angle define the direction along which
        we check for co-occurrence. The GLCM includes the number of times that gray-level :math:`j` occurs at a distance
        :math:`d` and at an angle theta from gray-level :math:`i`.

        Parameters
        ----------
        %(img_layer)s
        %(library_id_features)s
        %(feature_name)s
        %(channels)s
        props
            Texture features that are calculated, see the `prop` argument in :func:`skimage.feature.graycoprops`.
        distances
            The `distances` argument in :func:`skimage.feature.graycomatrix`.
        angles
            The `angles` argument in :func:`skimage.feature.graycomatrix`.

        Returns
        -------
        Returns features with the following keys for each channel `c` in ``channels``:

            - ``'{feature_name}_ch-{c}_{p}_dist-{dist}_angle-{a}'`` - the GLCM properties, for each `p` in ``props``,
              `d` in ``distances`` and `a` in ``angles``.

        Notes
        -----
        If the image is not of type :class:`numpy.uint8`, it will be converted.
        """
        layer = self._get_layer(layer)
        library_id = self._get_library_id(library_id)

        props = _assert_non_empty_sequence(props, name="properties")
        angles = _assert_non_empty_sequence(angles, name="angles")
        distances = _assert_non_empty_sequence(distances, name="distances")

        channels = _get_channels(self[layer], channels)
        channels = _assert_non_empty_sequence(channels, name="channels")

        arr = self[layer].sel(z=library_id)[..., channels].values

        if not np.issubdtype(arr.dtype, np.uint8):
            arr = img_as_ubyte(arr, force_copy=False)  # values must be in [0, 255]

        features = {}
        for c in channels:
            comatrix = graycomatrix(arr[..., c], distances=distances, angles=angles, levels=256)
            for p in props:
                tmp_features = graycoprops(comatrix, prop=p)
                for d_idx, dist in enumerate(distances):
                    for a_idx, a in enumerate(angles):
                        features[f"{feature_name}_ch-{c}_{p}_dist-{dist}_angle-{a:.2f}"] = tmp_features[d_idx, a_idx]
        return features

    @d.dedent
    def features_segmentation(
        self: HasGetItemProtocol,
        label_layer: str,
        intensity_layer: str | None = None,
        library_id: str | None = None,
        feature_name: str = "segmentation",
        channels: Channel_t | None = None,
        props: Sequence[str] = ("label", "area", "mean_intensity"),
    ) -> Feature_t:
        """
        Calculate segmentation features using :func:`skimage.measure.regionprops`.

        Features are calculated using ``label_layer``, a cell segmentation of ``intensity_layer``, resulting from
        from calling e.g. :func:`squidpy.im.segment`.

        Depending on the specified parameters, mean and std of the requested props are returned.
        For the `'label'` feature, the number of labels is returned, i.e. the number of cells in this image.

        Parameters
        ----------
        label_layer
            Name of the image layer used to calculate the non-intensity properties.
        intensity_layer
            Name of the image layer used to calculate the intensity properties.
        %(library_id_features)s
        %(feature_name)s
        %(channels)s
            Only relevant for features that use the ``intensity_layer``.
        props
            Segmentation features that are calculated. See `properties` in :func:`skimage.measure.regionprops_table`.
            Each feature is calculated for each segment (e.g., nucleus) and mean and std values are returned, except
            for `'centroid'` and `'label'`. Valid options are:

                - `'area'` - number of pixels of segment.
                - `'bbox_area'` - number of pixels of bounding box area of segment.
                - `'centroid'` - centroid coordinates of segment.
                - `'convex_area'` - number of pixels in convex hull of segment.
                - `'eccentricity'` - eccentricity of ellipse with same second moments as segment.
                - `'equivalent_diameter'` - diameter of circles with same area as segment.
                - `'euler_number'` - Euler characteristic of segment.
                - `'extent'` - ratio of pixels in segment to its bounding box.
                - `'feret_diameter_max'` - longest distance between points around convex hull of segment.
                - `'filled_area'` - number of pixels of segment with all holes filled in.
                - `'label'` - number of segments.
                - `'major_axis_length'` - length of major axis of ellipse with same second moments as segment.
                - `'max_intensity'` - maximum intensity of ``intensity_layer`` in segment.
                - `'mean_intensity'` - mean intensity of ``intensity_layer`` in segment.
                - `'min_intensity'` - min intensity of ``intensity_layer`` in segment.
                - `'minor_axis_length'` - length of minor axis of ellipse with same second moments as segment.
                - `'orientation'` - angle of major axis of ellipse with same second moments as segment.
                - `'perimeter'` - perimeter of segment using 4-connectivity.
                - `'perimeter_crofton'` - perimeter of segment approximated by the Crofton formula.
                - `'solidity'` - ratio of pixels in the segment to the convex hull of the segment.

        Returns
        -------
        Returns features with the following keys:

            - ``'{feature_name}_label'`` - if `'label`` is in ``props``.
            - ``'{feature_name}_centroid'`` - if `'centroid`` is in ``props``.
            - ``'{feature_name}_{p}_mean'`` - mean for each non-intensity property `p` in ``props``.
            - ``'{feature_name}_{p}_std'`` - standard deviation for each non-intensity property `p` in ``props``.
            - ``'{feature_name}_ch-{c}_{p}_mean'`` - mean for each intensity property `p` in ``props`` and
              channel `c` in ``channels``.
            - ``'{feature_name}_ch-{c}_{p}_std'`` - standard deviation for each intensity property `p` in ``props`` and
              channel `c` in ``channels``.
        """

        def convert_to_full_image_coordinates(x: NDArrayA, y: NDArrayA) -> NDArrayA:
            if not len(y):
                return np.array([[]], dtype=np.float64)

            if self.data.attrs.get("mask_circle", False):
                if self.data.dims["y"] != self.data.dims["x"]:
                    raise ValueError(f"Crop is not a square: `{self.data.dims}`.")
                c = self.data.dims["x"] // 2  # center
                mask = (x - c) ** 2 + (y - c) ** 2 <= c**2
                y = y[mask]
                x = x[mask]

            if not len(y):
                return np.array([[]], dtype=np.float64)  # because of masking, should not happen

            coord = self.data.attrs.get(
                Key.img.coords, CropCoords(x0=0, y0=0, x1=self.data.dims["x"], y1=self.data.dims["y"])
            )  # fall back to default (i.e no crop) coordinates
            padding = self.data.attrs.get(Key.img.padding, _NULL_PADDING)  # fallback to no padding
            y_slc, x_slc = coord.to_image_coordinates(padding).slice

            # relative coordinates
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            x = (x - np.min(x)) / (np.max(x) - np.min(x))

            # coordinates in the uncropped image
            y = coord.slice[0].start + (y_slc.stop - y_slc.start) * y
            x = coord.slice[1].start + (x_slc.stop - x_slc.start) * x

            return np.c_[x, y]  # type: ignore[no-any-return]

        label_layer = self._get_layer(label_layer)
        library_id = self._get_library_id(library_id)

        props = _assert_non_empty_sequence(props, name="properties")
        for prop in props:
            if prop not in _valid_seg_prop:
                raise ValueError(f"Invalid property `{prop}`. Valid properties are `{_valid_seg_prop}`.")

        no_intensity_props = [p for p in props if "intensity" not in p]
        intensity_props = [p for p in props if "intensity" in p]

        if len(intensity_props):
            if intensity_layer is None:
                raise ValueError("Please specify `intensity_layer` if using intensity properties.")
            channels = _get_channels(self[intensity_layer], channels)
            channels = _assert_non_empty_sequence(channels, name="channels")
        else:
            channels = ()

        features: dict[str, Any] = {}
        label_arr = self[label_layer].sel(z=library_id)
        label_arr_0 = label_arr[..., 0].values
        # calculate features that do not depend on the intensity image
        tmp_features = skimage.measure.regionprops_table(label_arr_0, properties=no_intensity_props)
        for p in no_intensity_props:
            if p == "label":
                features[f"{feature_name}_{p}"] = len(tmp_features["label"])
            elif p == "centroid":
                features[f"{feature_name}_centroid"] = convert_to_full_image_coordinates(
                    tmp_features["centroid-0"], tmp_features["centroid-1"]
                )
            else:
                features[f"{feature_name}_{p}_mean"] = np.mean(tmp_features[p])
                features[f"{feature_name}_{p}_std"] = np.std(tmp_features[p])

        # calculate features that depend on the intensity image
        for c in channels:
            if TYPE_CHECKING:
                assert isinstance(intensity_layer, str)
            tmp_features = skimage.measure.regionprops_table(
                label_arr_0,
                intensity_image=self[intensity_layer].sel(z=library_id)[..., c].values,
                properties=props,
            )
            for p in intensity_props:
                features[f"{feature_name}_ch-{c}_{p}_mean"] = np.mean(tmp_features[p])
                features[f"{feature_name}_ch-{c}_{p}_std"] = np.std(tmp_features[p])

        return features

    @d.dedent
    def features_custom(
        self: HasGetItemProtocol,
        func: Callable[[NDArrayA], Any],
        layer: str | None,
        channels: Channel_t | None = None,
        feature_name: str | None = None,
        **kwargs: Any,
    ) -> Feature_t:
        """
        Calculate features using a custom function.

        The feature extractor ``func`` can be any :func:`callable`, as long as it has the following signature:
        :class:`numpy.ndarray` ``(height, width, z, channels)`` **->** :class:`float`/:class:`Sequence`.

        Parameters
        ----------
        func
            Feature extraction function.
        %(img_layer)s
        %(channels)s
        %(feature_name)s
        kwargs
            Keyword arguments for ``func``.
            If `additional_layers` is present, all layers with the specified names will be passed to ``func``.
            using ``func(..., <layer-name>=<layer-values>)`` for each layer name in `additional_layers`.

        Returns
        -------
        Returns features with the following keys:

            - ``'{feature_name}_{i}'`` - i-th feature value.

        Examples
        --------
        Simple example would be to calculate the mean of a specified channel, as already done in
        :meth:`squidpy.im.ImageContainer.features_summary`::

            img = squidpy.im.ImageContainer(...)
            img.features_custom(imd_id=..., func=numpy.mean, channels=0)
        """
        layer = self._get_layer(layer)
        channels = _get_channels(self[layer], channels)
        feature_name = getattr(func, "__name__", "custom") if feature_name is None else feature_name

        additional_layers = kwargs.pop("additional_layers", None)
        if additional_layers is not None:
            for additional_layer in additional_layers:
                additional_layer = self._get_layer(additional_layer)
                kwargs[additional_layer] = self[additional_layer].values
        # calculate features by calling feature_fn
        res = func(self[layer][..., channels].values, **kwargs)
        if not isinstance(res, Iterable):
            res = [res]
        features = {f"{feature_name}_{i}": f for i, f in enumerate(res)}

        return features
