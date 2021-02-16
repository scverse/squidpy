from types import MappingProxyType
from typing import Any, List, Union, Mapping, Optional, Sequence, TYPE_CHECKING

from scanpy import logging as logg
from anndata import AnnData

import pandas as pd

from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, parallelize, _get_n_cores
from squidpy.gr._utils import _save_data
from squidpy.im._container import ImageContainer
from squidpy._constants._constants import ImageFeature

__all__ = ["calculate_image_features"]


@d.dedent
@inject_docs(f=ImageFeature)
def calculate_image_features(
    adata: AnnData,
    img: ImageContainer,
    layer: Optional[str] = None,
    features: Union[str, Sequence[str]] = ImageFeature.SUMMARY.s,
    features_kwargs: Mapping[str, Mapping[str, Any]] = MappingProxyType({}),
    key_added: str = "img_features",
    copy: bool = False,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    **kwargs: Any,
) -> Optional[pd.DataFrame]:
    """
    Calculate image features for all observations in ``adata``.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    %(img_layer)s
    features
        Features to be calculated. Valid options are:

        - `{f.TEXTURE.s!r}` - summary stats based on repeating patterns
          :meth:`squidpy.im.ImageContainer.features_texture`.
        - `{f.SUMMARY.s!r}` - summary stats of each image channel
          :meth:`squidpy.im.ImageContainer.features_summary`.
        - `{f.COLOR_HIST.s!r}` - counts in bins of image channel's histogram
          :meth:`squidpy.im.ImageContainer.features_histogram`.
        - `{f.SEGMENTATION.s!r}` - stats of a cell segmentation mask
          :meth:`squidpy.im.ImageContainer.features_segmentation`.
        - `{f.CUSTOM.s!r}` - extract features using a custom function
          :meth:`squidpy.im.ImageContainer.features_custom`.

    features_kwargs
        Keyword arguments for the different features that should be generated, such as
        ``{{ {f.TEXTURE.s!r}: {{ ... }}, ... }}``.
    key_added
        Key in :attr:`anndata.AnnData.obsm` where to store the calculated features.
    %(copy)s
    %(parallelize)s
    kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.generate_spot_crops`.

    Returns
    -------
    If ``copy = True``, returns a :class:`panda.DataFrame` where columns correspond to the calculated features.

    Otherwise, modifies the ``adata`` object with the following key:

        - :attr:`anndata.AnnData.uns` ``['{{key_added}}']`` - the above mentioned dataframe.

    Raises
    ------
    ValueError
        If a feature is not known.
    """
    layer = img._get_layer(layer)
    if isinstance(features, (str, ImageFeature)):
        features = [features]
    features = sorted({ImageFeature(f).s for f in features})

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating features `{list(features)}` using `{n_jobs}` core(s)")

    res = parallelize(
        _calculate_image_features_helper,
        collection=adata.obs_names,
        extractor=pd.concat,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(adata, img, layer=layer, features=features, features_kwargs=features_kwargs, **kwargs)

    if copy:
        logg.info("Finish", time=start)
        return res

    _save_data(adata, attr="obsm", key=key_added, data=res, time=start)


def _calculate_image_features_helper(
    obs_ids: Sequence[str],
    adata: AnnData,
    img: ImageContainer,
    layer: str,
    features: List[ImageFeature],
    features_kwargs: Mapping[str, Any],
    queue: Optional[SigQueue] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    features_list = []
    for crop in img.generate_spot_crops(adata, obs_names=obs_ids, return_obs=False, as_array=False, **kwargs):
        if TYPE_CHECKING:
            assert isinstance(crop, ImageContainer)
        # load crop in memory to enable faster processing
        crop._data = crop.data.load()

        features_dict = {}
        for feature in features:
            feature = ImageFeature(feature)
            feature_kwargs = features_kwargs.get(feature.s, {})

            if feature == ImageFeature.TEXTURE:
                res = crop.features_texture(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.COLOR_HIST:
                res = crop.features_histogram(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.SUMMARY:
                res = crop.features_summary(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.SEGMENTATION:
                res = crop.features_segmentation(intensity_layer=layer, **feature_kwargs)
            elif feature == ImageFeature.CUSTOM:
                res = crop.features_custom(layer=layer, **feature_kwargs)
            else:
                # should never get here
                raise NotImplementedError(f"Feature `{feature}` is not yet implemented.")

            features_dict.update(res)
        features_list.append(features_dict)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame(features_list, index=list(obs_ids))
