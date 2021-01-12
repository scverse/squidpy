import pytest

from anndata import AnnData

import numpy as np
import pandas as pd

from squidpy.im.object import ImageContainer
from squidpy.im.features import calculate_image_features


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_calculate_image_features(adata: AnnData, cont: ImageContainer, n_jobs: int):
    features = ["texture", "summary", "histogram"]
    res = calculate_image_features(adata, cont, features=features, copy=True, n_jobs=n_jobs)

    assert isinstance(res, pd.DataFrame)
    np.testing.assert_array_equal(res.index, adata.obs_names)
    assert [key for key in res.keys() if "texture" in key] != [], "feature name texture not in dict keys"
    assert [key for key in res.keys() if "summary" in key] != [], "feature name summary not in dict keys"
    assert [key for key in res.keys() if "histogram" in key] != [], "feature name histogram not in dict keys"


def test_get_summary_features():
    img = ImageContainer(np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8), img_id="image")
    feature = "test_summary_stats"
    stats = img.get_summary_features(
        img_id="image", feature_name=feature, quantiles=[0.9, 0.5, 0.1], mean=True, std=True, channels=[0, 1, 2]
    )

    assert isinstance(stats, dict)
    assert [key for key in stats.keys() if feature not in key] == [], "feature name not in dict keys"
    assert [key for key in stats.keys() if "mean" in key] != [], "mean not in dict keys"
    assert [key for key in stats.keys() if "std" in key] != [], "std not in dict keys"
    assert [key for key in stats.keys() if "quantile" in key] != [], "quantile not in dict keys"


def test_get_segmentation_features():
    # create image and label mask
    img = ImageContainer(np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8), img_id="image")
    mask = np.zeros((100, 100), dtype="uint8")
    mask[20:30, 10:20] = 1
    mask[50:60, 30:40] = 2
    img.add_img(mask, img_id="segmented", channel_id="mask")

    props = ["area", "label", "mean_intensity"]
    feature_name = "segmentation"
    stats = img.get_segmentation_features(
        img_id="image", feature_name=feature_name, props=props, label_img_id="segmented", mean=True, std=True
    )

    assert isinstance(stats, dict)
    assert [key for key in stats.keys() if feature_name not in key] == [], "feature name not in dict keys"
    for p in props:
        assert [key for key in stats.keys() if p in key] != [], f"{p} not in dict keys"

    # counted correct number of segments?
    assert stats["segmentation_label"] == 2
