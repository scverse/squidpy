# TODO: disable data-science-types because below does not generate types in shpinx + create an issue
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Union, Mapping, Iterable, Optional

from scanpy import logging as logg
from anndata import AnnData

import numpy as np
import pandas as pd

from skimage.feature import greycoprops, greycomatrix
import skimage.measure

from squidpy._docs import d, inject_docs
from squidpy.gr._utils import Signal, parallelize, _get_n_cores
from squidpy.im.object import ImageContainer
from squidpy.constants._constants import ImageFeature

Feature_t = Dict[str, Any]


@d.dedent
@inject_docs(f=ImageFeature)
def calculate_image_features(
    adata: AnnData,
    img: ImageContainer,
    img_id: Optional[str] = None,
    features: Union[str, Iterable[str]] = ImageFeature.SUMMARY.s,
    features_kwargs: Mapping[str, Any] = MappingProxyType({}),
    key: str = "img_features",
    copy: bool = False,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """
    Calculate image features for all obs_ids in adata, using the high-resolution tissue image contained in ``img``.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    %(img_id)s
    features
        Features to be calculated. Available features:

        - `{f.TEXTURE.s!r}`: summary stats based on repeating patterns \
          :func:`squidpy.im.get_texture_features()`.
        - `{f.SUMMARY.s!r}`: summary stats of each image channel :func:`squidpy.im.get_summary_features()`.
        - `{f.COLOR_HIST.s!r}`: counts in bins of image channel's histogram \
          :func:`squidpy.im.get_histogram_features()`.
        - `{f.SEGMENTATION.s!r}`: stats of a cell segmentation mask :func:`squidpy.im.get_segmentation_features()`.

    features_kwargs
        Keyword arguments for the different features that should be generated.
    key
        Key to use for saving calculated table in :attr:`anndata.AnnData.obsm`.
        TODO: should this be called key_added?
    %(copy)s
    %(parallelize)s
    kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.crop_spot_generator`.

    Returns
    -------
    :class:`pandas.DataFrame` or None
        TODO: rephrase
        `None` if ``copy = False``, otherwise the :class:`pandas.DataFrame`.

    Raises
    ------
    :class:`NotImplementedError`
        If a feature string is not known.
    """
    if isinstance(features, str):
        features = [features]
    features = [ImageFeature(f) for f in features]

    n_jobs = _get_n_cores(n_jobs)
    logg.info(f"Calculating features `{list(features)}` using `{n_jobs}` core(s)")

    res = parallelize(
        _calculate_image_features_helper,
        collection=adata.obs.index,
        extractor=pd.concat,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(adata, img, img_id=img_id, features=features, features_kwargs=features_kwargs, **kwargs)

    if copy:
        return res

    adata.obsm[key] = res


def _calculate_image_features_helper(
    obs_ids: Iterable[Any],
    adata: AnnData,
    img: ImageContainer,
    img_id: Optional[str],
    features: List[ImageFeature],
    features_kwargs: Mapping[str, Any],
    queue=None,
    **kwargs,
) -> pd.DataFrame:
    features_list = []

    if img_id is None:
        img_id = list(img.data.keys())[0]

    for crop, _ in img.generate_spot_crops(adata, obs_ids=obs_ids, **kwargs):

        # get features for this crop
        # TODO this could be solved in a more elegant manner
        # TODO could the values ImageFeature.TEXTURE etc be functions?
        features_dict = {}
        for feature in features:
            feature = ImageFeature(feature)
            feature_kwargs = features_kwargs.get(feature.s, {})

            if feature == ImageFeature.TEXTURE:
                get_feature_fn = get_texture_features
            elif feature == ImageFeature.COLOR_HIST:
                get_feature_fn = get_histogram_features
            elif feature == ImageFeature.SUMMARY:
                get_feature_fn = get_summary_features
            elif feature == ImageFeature.SEGMENTATION:
                get_feature_fn = get_segmentation_features
            else:
                raise NotImplementedError(feature)

            features_dict.update(get_feature_fn(crop, img_id=img_id, **feature_kwargs))
        features_list.append(features_dict)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame(features_list, index=list(obs_ids))


def _get_channels(xr_img, channels):
    """Get correct channel ranges for feature calculation."""
    # if channels is None, compute features for all channels
    if channels is None:
        channels = range(xr_img.shape[-1])
    if isinstance(channels, int):
        channels = [channels]
    return channels


@d.dedent
def get_summary_features(
    img: ImageContainer,
    img_id: str,
    feature_name: str = "summary",
    channels: Optional[Union[Tuple[int], int]] = None,
    quantiles: Tuple[float] = (0.9, 0.5, 0.1),
    mean: bool = False,
    std: bool = False,
):
    """
    Calculate summary statistics of image channels.

    Parameters
    ----------
    %(img_container)s
    %(img_id)s
    %(feature_name)s
    channels
        Channels for which summary features are computed. Default is all channels
    quantiles
        Quantiles that are computed.
    mean
        Compute mean.
    std
        Compute std deviation.

    Returns
    -------
    %(feature_ret)s
    """
    channels = _get_channels(img[img_id], channels)

    features = {}
    for c in channels:
        for q in quantiles:
            features[f"{feature_name}_quantile_{q}_ch_{c}"] = np.quantile(img[img_id][:, :, c], q)
        if mean:
            features[f"{feature_name}_mean_ch_{c}"] = np.mean(img[img_id][:, :, c])
        if std:
            features[f"{feature_name}_std_ch_{c}"] = np.std(img[img_id][:, :, c])

    return features


@d.dedent
def get_histogram_features(
    img: ImageContainer,
    img_id: str,
    feature_name: str = "histogram",
    channels: Optional[Tuple[int]] = None,
    bins: int = 10,
    v_range: Optional[Tuple[int, int]] = None,
) -> Feature_t:
    """
    Compute histogram counts of color channel values.

    Returns one feature per bin and channel.

    Parameters
    ----------
    %(img_container)s
    %(img_id)s
    %(feature_name)s
    channels
        Channels for which histograms are computed. Default is all channels
    bins
        Number of binned value intervals.

    v_range
        Range on which values are binned. Default is whole image range

    Returns
    -------
    %(feature_ret)s
    """
    channels = _get_channels(img[img_id], channels)
    # if v_range is None, use whole-image range
    v_range = np.min(img[img_id].values), np.max(img[img_id].values)

    features = {}
    for c in channels:
        hist = np.histogram(img[img_id][:, :, c], bins=bins, range=v_range, weights=None, density=False)
        for i, count in enumerate(hist[0]):
            features[f"{feature_name}_ch_{c}_bin_{i}"] = count

    return features


@d.dedent
def get_texture_features(
    img: ImageContainer,
    img_id: str,
    feature_name: str = "texture",
    channels: Optional[Tuple[int]] = None,
    props: Iterable[str] = ("contrast", "dissimilarity", "homogeneity", "correlation", "ASM"),
    distances: Iterable[int] = (1,),
    angles: Iterable[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> Feature_t:
    """
    Calculate texture features.

    A grey level co-occurence matrix (GLCM) is computed for different combinations of distance and angle.
    The distance defines the pixel difference of co occurence. The angle define the direction along which
    we check for co-occurence. The GLCM includes the number of times that grey-level j occurs at a distance
    d and at an angle theta from grey-level i.

    From a given GLCM texture features are inferred.
    TODO: add reference to GLCM.

    Parameters
    ----------
    %(img_container)s
    %(img_id)s
    %(feature_name)s
    channels
        Channels for which histograms are computed. Default is all channels.
    props
        Texture features that are calculated. See `prop` in :func:`skimage.feature.greycoprops`.
    distances
        See `distances` in :func:`skimage.feature.greycomatrix`.
    angles
        See `angles` in :func:`skimage.feature.greycomatrix`.
    Returns
    -------
    %(feature_ret)s
    """
    channels = _get_channels(img[img_id], channels)

    features = {}
    for c in channels:
        comatrix = greycomatrix(img[img_id][:, :, c], distances=distances, angles=angles, levels=256)
        for p in props:
            tmp_features = greycoprops(comatrix, prop=p)
            for d_idx, dist in enumerate(distances):
                for a_idx, a in enumerate(angles):
                    features[f"{feature_name}_{p}_ch_{c}_dist_{dist}_angle_{a:.2f}"] = tmp_features[d_idx, a_idx]
    return features


@d.dedent
def get_segmentation_features(
    img: ImageContainer,
    img_id: str,
    feature_name: str = "segmentation",
    channels: Optional[Tuple[int]] = None,
    label_img_id: Optional[ImageContainer] = None,
    props: Iterable[str] = ("label", "area", "mean_intensity"),
    mean: bool = True,
    std: bool = False,
):
    """
    Calculate segmentation features using :func:`skimage.measure.regionprops`.

    Features are calculated using ``label_img_id``, a cell segmentation of ``img``
    (e.g. resulting from calling :func:`squidpy.im.segment_img`).
    Depending on the specified parameters, mean and std of the requested props is returned.
    For the 'label' feature, the number of labels is returned, i.e. the number of cells in this img.

    Parameters
    ----------
    %(img_container)s
    %(img_id)s
    %(feature_name)s
    channels
        Channels for which segmentation features are computed. Default is all channels.
        Only relevant for features that use the intensity image ``img``.
    props
        Segmentation features that are calculated. See `properties` in :func:`skimage.measure.regionprops_table`.
        Supported props:

        - area
        - bbox_area
        - convex_area
        - eccentricity
        - equivalent_diameter
        - euler_number
        - extent
        - feret_diameter_max
        - filled_area
        - label
        - major_axis_length
        - max_intensity (uses intensity image ``img``)
        - mean_intensity (uses intensity image ``img``)
        - min_intensity (uses intensity image ``img``)
        - minor_axis_length
        - orientation
        - perimeter
        - perimeter_crofton
        - solidity

    mean
        Return mean feature values.
    std
        Return std feature values.

    Returns
    -------
    %(feature_ret)s

    Raises
    ------
    ValueError
        if ``label_img_id`` is None
    """
    # TODO check that passed a valid prop
    channels = _get_channels(img[img_id], channels)
    if label_img_id is None:
        raise ValueError("Please pass a value for label_img_id to get_segmentation_features")

    features = {}
    # calculate features that do not depend on the intensity image
    no_intensity_props = [p for p in props if "intensity" not in p]
    tmp_features = skimage.measure.regionprops_table(img[label_img_id].values[:, :, 0], properties=no_intensity_props)
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
            img[label_img_id].values[:, :, 0], intensity_image=img[img_id].values[:, :, c], properties=props
        )
        for p in intensity_props:
            if mean:
                features[f"{feature_name}_{p}_ch{c}_mean"] = np.mean(tmp_features[p])
            if std:
                features[f"{feature_name}_{p}_ch{c}_std"] = np.std(tmp_features[p])

    return features
