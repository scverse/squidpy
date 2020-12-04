# TODO: disable data-science-types because below does not generate types in shpinx + create an issue
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Union, Mapping, Iterable, Optional

from anndata import AnnData

import numpy as np
import pandas as pd

import skimage.feature as sk_image
from skimage.feature import greycoprops, greycomatrix

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
    Get im features for spot ids from im file.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    features
        Features to be calculated. Available features:

            - `{f.HOG.s!r}`: histogram of oriented gradients :func:`squidpy.im.get_hog_features()`.
            - `{f.TEXTURE.s!r}`: summary stats based on repeating patterns \
            :func:`squidpy.im.get_grey_texture_features()`.
            - `{f.SUMMARY.s!r}`: summary stats of each color channel :func:`squidpy.im.get_summary_stats()`.
            - `{f.COLOR_HIST.s!r}`: counts in bins of each color channel's histogram \
            :func:`squidpy.im.get_color_hist()`.

    features_kwargs
        Keyword arguments for the different features that should be generated.
    key
        Key to use for saving calculated table in :attr:`anndata.AnnData.obsm`.
    %(copy)s
    %(parallelize)s
    kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.crop_spot_generator`.

    Returns
    -------
    :class:`pandas.DataFrame` or None
        TODO: rephrase
        `None` if ``copy = False``, otherwise the :class:`pandas.DataFrame`.
    """
    if isinstance(features, str):
        features = [features]
    features = [ImageFeature(f) for f in features]

    n_jobs = _get_n_cores(kwargs.pop("n_jobs", None))
    res = parallelize(
        _calculate_image_features_helper,
        collection=adata.obs.index,
        extractor=pd.concat,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(adata, img, features=features, features_kwargs=features_kwargs, **kwargs)

    if copy:
        return res

    adata.obsm[key] = res


def _calculate_image_features_helper(
    obs_ids: Iterable[Any],
    adata: AnnData,
    img: ImageContainer,
    features: List[ImageFeature],
    features_kwargs: Mapping[str, Any],
    queue=None,
    **kwargs,
) -> pd.DataFrame:
    features_list = []

    for _, crop in img.crop_spot_generator(adata, obs_ids=obs_ids, **kwargs):
        # get np.array from crop and restructure to dimensions: y,x,channels
        crop = crop.transpose("y", "x", ...).data
        # if crop has no color channel, reshape
        if len(crop.shape) == 2:
            crop = crop[:, :, np.newaxis]

        # get features for this crop
        features_dict = get_features_statistics(crop, features=features, features_kwargs=features_kwargs)
        features_list.append(features_dict)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame(features_list, index=list(obs_ids))


# TODO: refactor all of below into ImageContainer?
@d.dedent
def get_features_statistics(
    img: np.ndarray[np.float64],
    features: Iterable[Union[str, ImageFeature]],
    features_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Feature_t:
    """
    Calculate feature statistics.

    Parameters
    ----------
    %(img_uint8)s
    %(feature_name)s
    features_kwargs
        keyword arguments for the different features that should be generated.

    Returns
    -------
    %(feature_ret)s

    Raises
    ------
    :class:`NotImplementedError`
        If a feature string is not known.
    """
    # TODO: valuedispatch would be cleaner
    stat_dict = {}
    for feature in features:
        feature = ImageFeature(feature)
        feature_kwargs = features_kwargs.get(feature.s, {})

        if feature == ImageFeature.HOG:
            get_feature_fn = get_hog_features
        elif feature == ImageFeature.TEXTURE:
            get_feature_fn = get_grey_texture_features
        elif feature == ImageFeature.COLOR_HIST:
            get_feature_fn = get_color_hist
        elif feature == ImageFeature.SUMMARY:
            get_feature_fn = get_summary_stats
        else:
            raise NotImplementedError(feature)

        # TODO: refactor
        stat_dict.update(get_feature_fn(img, **feature_kwargs))

    return stat_dict


@d.dedent
def get_hog_features(img: np.ndarray[np.float64], feature_name: str = "hog") -> Feature_t:
    """
    Calculate histogram of oriented gradients (hog) features.

    Parameters
    ----------
    %(img_uint8)s
    %(feature_name)s

    Returns
    -------
    %(feature_ret)s
    """
    hog_dict = {}
    hog_features = sk_image.hog(img)
    for k, hog_feature in enumerate(hog_features):
        hog_dict[f"{feature_name}_{k}"] = hog_feature
    return hog_dict


@d.dedent
def get_summary_stats(
    img: np.ndarray[np.float64],
    feature_name: str = "summary",
    quantiles: Tuple[float, float, float] = (0.9, 0.5, 0.1),
    mean: bool = False,
    std: bool = False,
    channels: Tuple[int, int, int] = (0, 1, 2),
):
    """
    Calculate summary statistics of color channels.

    Parameters
    ----------
    %(img_uint8)s
    %(feature_name)s
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
    %(feature_ret)s
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


@d.dedent
def get_color_hist(
    img: np.ndarray[np.float64],
    feature_name: str = "color_hist",
    bins: int = 10,
    channels: Tuple[int, int, int] = (0, 1, 2),
    v_range: Tuple[int, int] = (0, 255),
) -> Feature_t:
    """
    Compute histogram counts of color channel values.

    Parameters
    ----------
    %(img_uint8)s
    %(feature_name)s
    bins
        Number of binned value intervals.
    channels
        Define for which channels histograms are computed.
    v_range
        Range on which values are binned.

    Returns
    -------
    %(feature_ret)s
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


@d.dedent
def get_grey_texture_features(
    img: np.ndarray[np.float64],
    feature_name: str = "texture",
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
    %(img_uint8)s
    %(feature_name)s
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
    features = {}
    # if img has only one channel, do not need to convert to grey
    if img.shape[-1] == 1:
        grey_img = img[:, :, 0]
    else:
        # get grey scale im
        multiplier = [0.299, 0.587, 0.114]
        grey_img = np.dot(img, multiplier).astype(np.uint8)

    comatrix = greycomatrix(grey_img, distances=distances, angles=angles, levels=256)
    for p in props:
        tmp_features = greycoprops(comatrix, prop=p)
        for d_idx, dist in enumerate(distances):
            for a_idx, a in enumerate(angles):
                features[f"{feature_name}_{p}_dist_{dist}_angle_{a:.2f}"] = tmp_features[d_idx, a_idx]
    return features
