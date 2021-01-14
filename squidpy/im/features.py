# TODO: disable data-science-types because below does not generate types in shpinx + create an issue
from __future__ import annotations

from types import MappingProxyType
from typing import Any, List, Union, Mapping, Iterable, Optional

from scanpy import logging as logg
from anndata import AnnData

import pandas as pd

from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, parallelize, _get_n_cores
from squidpy.im.object import ImageContainer
from squidpy.constants._constants import ImageFeature


@d.dedent
@inject_docs(f=ImageFeature)
def calculate_image_features(
    adata: AnnData,
    img: ImageContainer,
    img_id: Optional[str] = None,
    features: Union[str, Iterable[str]] = ImageFeature.SUMMARY.s,
    features_kwargs: Mapping[str, Any] = MappingProxyType({}),
    key_added: str = "img_features",
    copy: bool = False,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = False,
    **kwargs: Any,
) -> Optional[pd.DataFrame]:
    """
    Calculate image features for all observations in ``adata``.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    %(img_id)s
    features
        Features to be calculated. Available features:

        - `{f.TEXTURE.s!r}`: summary stats based on repeating patterns \
          :meth:`squidpy.im.ImageContainer.get_texture_features()`.
        - `{f.SUMMARY.s!r}`: summary stats of each image channel \
          :meth:`squidpy.im.ImageContainer.get_summary_features()`.
        - `{f.COLOR_HIST.s!r}`: counts in bins of image channel's histogram \
          :meth:`squidpy.im.ImageContainer.get_histogram_features()`.
        - `{f.SEGMENTATION.s!r}`: stats of a cell segmentation mask \
          :meth:`squidpy.im.ImageContainer.get_segmentation_features()`.

    features_kwargs
        Keyword arguments for the different features that should be generated.
    key_added
        Key to use for saving calculated table in :attr:`anndata.AnnData.obsm`.
    %(copy)s
    %(parallelize)s
    kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.crop_spot_generator`.

    Returns
    -------
    `None` if ``copy = False``, otherwise the :class:`pandas.DataFrame`.

    Raises
    ------
    NotImplementedError
        If a feature is not known.
    """
    if isinstance(features, (str, ImageFeature)):
        features = [features]
    features = [ImageFeature(f) for f in features]  # type: ignore[no-redef,misc]

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

    adata.obsm[key_added] = res


def _calculate_image_features_helper(
    obs_ids: Iterable[Any],
    adata: AnnData,
    img: ImageContainer,
    img_id: Optional[str],
    features: List[ImageFeature],
    features_kwargs: Mapping[str, Any],
    queue: Optional[SigQueue] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    features_list = []

    if img_id is None:
        img_id = list(img.data.keys())[0]

    for crop, _ in img.generate_spot_crops(adata, obs_ids=obs_ids, **kwargs):
        # get features for this crop
        # TODO: valuedispatch would be cleaner
        # TODO could the values ImageFeature.TEXTURE etc be functions?
        features_dict = {}
        for feature in features:
            feature = ImageFeature(feature)
            feature_kwargs = features_kwargs.get(feature.s, {})

            if feature == ImageFeature.TEXTURE:
                res = crop.get_texture_features(img_id=img_id, **feature_kwargs)
            elif feature == ImageFeature.COLOR_HIST:
                res = crop.get_histogram_features(img_id=img_id, **feature_kwargs)
            elif feature == ImageFeature.SUMMARY:
                res = crop.get_summary_features(img_id=img_id, **feature_kwargs)
            elif feature == ImageFeature.SEGMENTATION:
                res = crop.get_segmentation_features(img_id=img_id, **feature_kwargs)
            else:
                raise NotImplementedError(feature)

            features_dict.update(res)
        features_list.append(features_dict)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame(features_list, index=list(obs_ids))
