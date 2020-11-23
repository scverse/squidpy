from types import MappingProxyType
from typing import List, Mapping, Optional

from anndata import AnnData

import numpy as np
import pandas as pd

import skimage.feature as sk_image
from skimage.feature import greycoprops, greycomatrix

from spatial_tools.image.object import ImageContainer


def calculate_image_features(
    adata: AnnData,
    img: ImageContainer,
    features: Optional[List[str]] = "summary",
    features_kwargs: Optional[Mapping] = MappingProxyType({}),
    key: Optional[str] = "img_features",
    copy: Optional[bool] = False,
    **kwargs,
):
    """
    Get image features for spot ids from image file.

    Params
    ------
    adata: AnnData
        Spatial scanpy adata object
    img: ImageContainer
        High resolution image from which feature should be calculated
    features: Optional[List[str]]
        Features to be calculated. Available features:
        (for detailed descriptions see docstrings of each feature's function)
        "hog": histogram of oriented gradients (`get_hog_features()`)
        "texture": summary stats based on repeating patterns (`get_grey_texture_feature()`)
        "summary": summary stats of each color channel (`get_summary_stats()`)
        "color_hist": counts in bins of each color channel's histogram (`get_color_hist()`)
        Features to be calculated. Available features:
        ["hog", "texture", "summary", "color_hist"]
    features_kwargs: Optional[dict[str, dict]]
        keyword arguments for the different features that should be generated.
    key: Optional[str]
        Key to use for saving calculated table in adata.obsm.
        Default is "img_features"
    copy: Optional[bool]
        If True, return pd.DataFrame with calculated features.
        Default is False
    kwargs: keyword arguments passed to ImageContainer.crop_spot_generator function.
        Contain dataset_name, img_id, sizef, scale, mask_circle, cval, dtype


    Returns
    -------
    None if copy is False
    pd.DataFrame if copy is True
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
    else:
        adata.obsm[key] = features_log


def get_features_statistics(
    img,
    features,
    features_kwargs: Optional[Mapping] = MappingProxyType({}),
):
    """
    Calculate feature statistics.

    Params
    ------
    img: np.array
        rgb image in uint8 format.
    features: list of strings
        feature names of features to be extracted.
    features_kwargs: Optional[dict[str, dict]]
        keyword arguments for the different features that should be generated.

    Raises
    ------
    NotImplementedError:
        if a feature string is not known

    Returns
    -------
    dict of feature values.
    """
    stat_dict = {}
    for feature in features:
        get_feature_fn = None
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


def get_hog_features(img, feature_name="hog"):
    """
    Calculate histogram of oriented gradients (hog) features.

    Params
    ------
    img: M, N[, C] np.array
        rgb image in uint8 format.
    feature_name: str
        base name of feature in resulting feature values dict.

    Returns
    -------
    dict of feature values.
    """
    hog_dict = {}
    hog_features = sk_image.hog(img)
    for k, hog_feature in enumerate(hog_features):
        hog_dict[f"{feature_name}_{k}"] = hog_feature
    return hog_dict


def get_summary_stats(img, feature_name="summary", quantiles=(0.9, 0.5, 0.1), mean=False, std=False, channels=None):
    """
    Calculate summary statistics of color channels.

    Params
    ------
    img: np.array
        rgb image in uint8 format, in format H, W[, C]
    feature_name: str
        base name of feature in resulting feature values dict.
    qunatiles: list of floats
        Quantiles that are computed
    mean: bool
        Compute mean
    std: bool
        Compute std
    channels: list of ints
        define for which channels histograms are computed

    Returns
    -------
    dict of feature values.
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


def get_color_hist(img, feature_name="color_hist", bins=10, channels=None, v_range=(0, 255)):
    """
    Compute histogram counts of color channel values.

    Params
    ------
    img: np.array
        rgb image in uint8 format.
    feature_name: str
        base name of feature in resulting feature values dict.
    bins: int
        number of binned value intervals.
    channels: list of ints
        define for which channels histograms are computed.
    v_range: tuple of two ints
        Range on which values are binned..

    Returns
    -------
    dict of feature values
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
    img,
    feature_name="texture",
    props=("contrast", "dissimilarity", "homogeneity", "correlation", "ASM"),
    distances=(1,),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
):
    """
    Calculate texture features.

    A grey level co-occurence matrix (GLCM) is computed for different combinations of distance and angle.
    The distance defines the pixel difference of co occurence. The angle define the direction along which
    we check for co-occurence. The GLCM includes the number of times that grey-level j occurs at a distance
    d and at an angle theta from grey-level i.
    From a given GLCM texture features are infered.

    Params
    ------
    img: np.array
        rgb image in uint8 format.
    feature_name: str
        base name of feature in resulting feature values dict.
    props: list of strs
        texture features that are calculated. See `prop` in skimage.feature.greycoprops.
    distances: list of ints
        See `distances` in skimage.feature.greycomatrix.
    angles: list of floats
        See `angles` in skimage.feature.greycomatrix.

    Returns
    -------
    dict of feature values
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
