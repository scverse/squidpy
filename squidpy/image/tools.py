from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Mapping, Iterable, Optional

from anndata import AnnData

import numpy as np
import pandas as pd

import skimage.feature as sk_image
from skimage.feature import greycoprops, greycomatrix

from squidpy.image.object import ImageContainer

Img_t = np.ndarray  # TODO: not sure why this fails even with data-science-types: np.ndarray[np.float64]
Feature_t = Dict[str, Any]


def calculate_image_features(
    adata: AnnData,
    img: ImageContainer,
    features: Optional[List[str]] = "summary",
    features_kwargs: Optional[Mapping] = MappingProxyType({}),
    key: Optional[str] = "img_features",
    copy: Optional[bool] = False,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """
    Get image features for spot ids from image file.

    Parameters
    ----------
    adata
        Spatial adata object
    img
        High resolution image from which feature should be calculated
    features
        Features to be calculated. Available features:

            - `'hog'`: histogram of oriented gradients :func:`squidpy.image.get_hog_features()`.
            - `'texture'`: summary stats based on repeating patterns :func:`squidpy.image.get_grey_texture_features()`.
            - `'summary'`: summary stats of each color channel :func:`squidpy.image.get_summary_stats()`.
            - `'color_hist'`: counts in bins of each color channel's histogram :func:`squidpy.image.get_color_hist()`.

    features_kwargs
        keyword arguments for the different features that should be generated.
    key
        Key to use for saving calculated table in :attr:`anndata.AnnData.obsm`.
    copy
        If True, return :class:`pandas.DataFrame` with calculated features.
    kwargs
        Keyword arguments passed to :meth:`squidpy.image.ImageContainer.crop_spot_generator` function.

    Returns
    -------
    :class:`pandas.DataFrame` or None
        `None` if ``copy = False``, otherwise the :class:`pandasDataFrame`.
    """
    available_features = ["hog", "texture", "summary", "color_hist"]
    if isinstance(features, str):
        features = [features]

    for feature in features:
        assert (
            feature in available_features
        ), f"feature: {feature} not a valid feature, select on of {available_features} "

    features_list = []
    obs_ids = []
    for obs_id, crop in img.crop_spot_generator(adata, **kwargs):
        # get np.array from crop and restructure to dimensions: y,x,channels
        crop = crop.transpose("y", "x", ...).data
        # if crop has no color channel, reshape
        if len(crop.shape) == 2:
            crop = crop[:, :, np.newaxis]

        # get features for this crop
        features_dict = get_features_statistics(crop, features=features, features_kwargs=features_kwargs)
        features_list.append(features_dict)

        obs_ids.append(obs_id)

    features_log = pd.DataFrame(features_list)
    features_log["obs_id"] = obs_ids
    features_log.set_index(["obs_id"], inplace=True)

    # modify adata in place or return features_log
    if copy:
        return features_log

    adata.obsm[key] = features_log


def get_features_statistics(
    img: Img_t, features: Iterable[str], features_kwargs: Optional[Mapping[str, Any]] = MappingProxyType({})
) -> Feature_t:
    """
    Calculate feature statistics.

    Parameters
    ----------
    img
        RGB image in uint8 format.
    features
        Feature names of features to be extracted.
    features_kwargs
        keyword arguments for the different features that should be generated.

    Returns
    -------
    :class:`dict`
        Dictionary of feature values.

    Raises
    ------
    :class:`NotImplementedError`
        If a feature string is not known.
    """
    # TODO: valuedispatch would be cleaner
    stat_dict = {}
    for feature in features:
        feature_kwargs = features_kwargs.get(feature, {})
        if feature == "hog":
            get_feature_fn = get_hog_features
        elif feature == "texture":
            get_feature_fn = get_grey_texture_features
        elif feature == "color_hist":
            get_feature_fn = get_color_hist
        elif feature == "summary":
            get_feature_fn = get_summary_stats
        else:
            raise NotImplementedError(f"feature {feature} is not implemented")
        stat_dict.update(get_feature_fn(img, **feature_kwargs))

    return stat_dict


def get_hog_features(img: Img_t, feature_name: str = "hog") -> Feature_t:
    """
    Calculate histogram of oriented gradients (hog) features.

    Parameters
    ----------
    img
        RGB image in uint8 format, in format H, W[, C].
    feature_name
        base name of feature in resulting feature values :class:`dict`.

    Returns
    -------
    :class:`dict`
        Dictionary of feature values.
    """
    hog_dict = {}
    hog_features = sk_image.hog(img)
    for k, hog_feature in enumerate(hog_features):
        hog_dict[f"{feature_name}_{k}"] = hog_feature
    return hog_dict


def get_summary_stats(
    img: Img_t,
    feature_name: str = "summary",
    quantiles: Tuple[int, int, int] = (0.9, 0.5, 0.1),
    mean: bool = False,
    std: bool = False,
    channels: Tuple[int, int, int] = (0, 1, 2),
):
    """
    Calculate summary statistics of color channels.

    Parameters
    ----------
    img
        RGB image in uint8 format, in format H, W[, C].
    feature_name
        base name of feature in resulting feature values :class:`dict`.
    quantiles
        Quantiles that are computed.
    mean
        Compute mean.
    std
        Compute std.
    channels
        Define for which channels histograms are computed.

    Returns
    -------
    :class:`dict`
        Dictionary of feature values.
    """
    # if channels is None, compute features for all channels
    if channels is None:
        channels = range(img.shape[-1])
    if isinstance(channels, int):
        channels = [channels]

    stats = {}
    for c in channels:
        for q in quantiles:
            stats[f"{feature_name}_quantile_{q}_ch_{c}"] = np.quantile(img[:, :, c], q)
        if mean:
            stats[f"{feature_name}_mean_ch_{c}"] = np.mean(img[:, :, c])
        if std:
            stats[f"{feature_name}_std_ch_{c}"] = np.std(img[:, :, c])

    return stats


def get_color_hist(
    img: Img_t,
    feature_name: str = "color_hist",
    bins: int = 10,
    channels: Tuple[int, int, int] = (0, 1, 2),
    v_range: Tuple[int, int] = (0, 255),
) -> Feature_t:
    """
    Compute histogram counts of color channel values.

    Parameters
    ----------
    img
        RGB image in uint8 format.
    feature_name
        Base name of feature in resulting feature values dict.
    bins
        Number of binned value intervals.
    channels
        Define for which channels histograms are computed.
    v_range
        Range on which values are binned.

    Returns
    -------
    :class:`dict`
        Dictionary of feature values.
    """
    # if channels is None, compute features for all channels
    if channels is None:
        channels = range(img.shape[-1])

    features = {}
    for c in channels:
        hist = np.histogram(img[:, :, c], bins=bins, range=v_range, weights=None, density=False)
        for i, count in enumerate(hist[0]):
            features[f"{feature_name}_ch_{c}_bin_{i}"] = count

    return features


def get_grey_texture_features(
    img: Img_t,
    feature_name: str = "texture",
    props: Iterable[str] = ("contrast", "dissimilarity", "homogeneity", "correlation", "ASM"),
    distances: Iterable[int] = (1,),
    angles: Iterable[int] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> Feature_t:
    """
    Calculate texture features.

    A grey level co-occurence matrix (GLCM) is computed for different combinations of distance and angle.
    The distance defines the pixel difference of co occurence. The angle define the direction along which
    we check for co-occurence. The GLCM includes the number of times that grey-level j occurs at a distance
    d and at an angle theta from grey-level i.

    From a given GLCM texture features are inferred.

    Parameters
    ----------
    img
        RGB image in uint8 format.
    feature_name
        Base name of feature in resulting feature values dict.
    props
        Texture features that are calculated. See `prop` in skimage.feature.greycoprops.
    distances
        See `distances` in :func:`skimage.feature.greycomatrix`.
    angles
        See `angles` in :func:`skimage.feature.greycomatrix`.

    Returns
    -------
    :class:`dict`
        Dictionary of feature values.
    """
    features = {}
    # if img has only one channel, do not need to convert to grey
    if img.shape[-1] == 1:
        grey_img = img[:, :, 0]
    else:
        # get grey scale image
        multiplier = [0.299, 0.587, 0.114]
        grey_img = np.dot(img, multiplier).astype(np.uint8)

    comatrix = greycomatrix(grey_img, distances=distances, angles=angles, levels=256)
    for p in props:
        tmp_features = greycoprops(comatrix, prop=p)
        for d_idx, d in enumerate(distances):
            for a_idx, a in enumerate(angles):
                features[f"{feature_name}_{p}_dist_{d}_angle_{a:.2f}"] = tmp_features[d_idx, a_idx]
    return features
