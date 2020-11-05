# flake8: noqa
from typing import List, Optional

from anndata import AnnData

import numpy as np
import pandas as pd

import skimage.feature as sk_image
from skimage.feature import greycoprops, greycomatrix

from spatial_tools.image.object import ImageContainer


def calculate_image_features(
    adata: AnnData,
    img: ImageContainer,
    features: Optional[List[str]] = [
        "summary",
    ],
    key: Optional = None,
    **kwargs,
):
    """\
    Get image features for spot ids from image file.

    Params
    ------
    adata: AnnData
        Spatial scanpy adata object
    img: ImageContainer
        High resolution image from which feature should be calculated
    features: Optional[List[str]]
        Features to be calculated. Available features:
        ["hog", "texture", "summary", "color_hist"]
    key: Optional[str]
        key to use for saving calculated table in adata.obsm.
        If None, function returns features table
    kwargs: keyword arguments passed to ImageContainer.crop_spot_generator function.
        Contain dataset_name, sizef, scale, mask_circle, cval, dtype


    Returns
    -------
    None if key is specified
    pd.DataFrame if key is None
    """
    available_features = ["hog", "texture", "summary", "color_hist"]

    for feature in features:
        assert (
            feature in available_features
        ), f"feature: {feature} not a valid feature, select on of {available_features} "

    features_list = []
    obs_ids = []
    for obs_id, crop in img.crop_spot_generator(adata, **kwargs):
        # get np.array from crop and restructure to dimensions: y,x,channels
        crop = crop.transpose("y", "x", ...).data

        # get features for this crop
        features_dict = get_features_statistics(crop, features=features)
        features_list.append(features_dict)

        obs_ids.append(obs_id)

    features_log = pd.DataFrame(features_list)
    features_log["obs_id"] = obs_ids
    features_log.set_index(["obs_id"], inplace=True)

    # modify adata in place or return features_log
    if key is None:
        return features_log
    else:
        adata.obsm[key] = features_log
        return


def get_features_statistics(im, features):
    """
    Calculate histogram of oriented gradients (hog) features.

    Params
    ---------
    img: np.array
        rgb image in uint8 format.
    features: list of strings
        feature names of features to be extracted.

    Returns
    -------
    dict of feature values.
    """
    stat_dict = {}
    for feature in features:
        if feature == "hog":
            stat_dict.update(get_hog_features(im, feature))
        if feature == "texture":
            stat_dict.update(get_grey_texture_features(im, feature))
        if feature == "color_hist":
            stat_dict.update(get_color_hist(im, feature))
        if feature == "summary":
            stat_dict.update(get_summary_stats(im, feature))
    return stat_dict


def get_hog_features(img, feature_name="hog"):
    """
    Calculate histogram of oriented gradients (hog) features.

    Params
    ---------
    img: M, N[, C] np.array
        rgb image in uint8 format.
    feature_name: str
        name of feature for string id.

    Returns
    -------
    dict of feature values.
    """
    hog_dict = {}
    hog_features = sk_image.hog(img)
    for k, hog_feature in enumerate(hog_features):
        hog_dict[f"{feature_name}_{k}"] = hog_feature
    return hog_dict


def get_summary_stats(img, feature, quantiles=(0.9, 0.5, 0.1), mean=False, std=False, channels=(0, 1, 2)):
    """
    Calculate summary statistics of color channels.

    Params
    ------
    img: np.array
        rgb image in uint8 format.
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
    # if img has no color channel, reshape
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    # if final axis is not color channel, 3 or 1, then roll axis
    if (img.shape[-1] != 3) and (img.shape[-1] != 1):
        img = np.rollaxis(img, 3, 1)

    stats = {}
    for c in channels:
        for q in quantiles:
            stats[f"{feature}_quantile_{q}_ch_{c}"] = np.quantile(img[:, :, c], q)
        if mean:
            stats[f"{feature}_mean_ch_{c}"] = np.mean(img[:, :, c])
        if std:
            stats[f"{feature}_std_ch_{c}"] = np.std(img[:, :, c])
    return stats


def get_color_hist(img, feature, bins=10, channels=(0, 1, 2), v_range=(0, 255)):
    """
    Compute histogram counts of color channel values.

    Params
    ------
    img: np.array
        rgb image in uint8 format.
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
    features = {}
    for c in channels:
        hist = np.histogram(img[:, :, c], bins=bins, range=v_range, weights=None, density=False)
        for i, count in enumerate(hist[0]):
            features[f"{feature}_ch_{c}_bin_{i}"] = count
    return features


def get_grey_texture_features(
    img,
    feature,
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
    # get grey scale image
    multiplier = [0.299, 0.587, 0.114]
    grey_img = np.dot(img, multiplier).astype(np.uint8)

    comatrix = greycomatrix(grey_img, distances=distances, angles=angles, levels=256)
    for p in props:
        tmp_features = greycoprops(comatrix, prop=p)
        for d_idx, d in enumerate(distances):
            for a_idx, a in enumerate(angles):
                features[f"{feature}_{p}_dist_{d}_angle_{a:.2f}"] = tmp_features[d_idx, a_idx]
    return features
