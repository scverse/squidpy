from anndata import AnnData

import numpy as np

from squidpy.im.tools import (
    get_hog_features,
    get_summary_stats,
    get_features_statistics,
    calculate_image_features,
)
from squidpy.im.object import ImageContainer


# cont is now in conftest.py
def test_calculate_image_features(adata: AnnData, cont: ImageContainer):
    features = ["hog", "texture", "summary", "color_hist"]
    _ = calculate_image_features(adata, cont, features=features)


def test_get_features_statistics():
    img = np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8)
    features = ["hog", "texture", "summary", "color_hist"]
    stats = get_features_statistics(img, features)

    assert isinstance(stats, dict)
    assert [key for key in stats.keys() if "hog" in key] != [], "feature name hog not in dict keys"
    assert [key for key in stats.keys() if "texture" in key] != [], "feature name texture not in dict keys"
    assert [key for key in stats.keys() if "summary" in key] != [], "feature name summary not in dict keys"
    assert [key for key in stats.keys() if "color_hist" in key] != [], "feature name color_hist not in dict keys"


def test_get_hog_features():
    img = np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8)
    feature = "test_summary_hog"

    stats = get_hog_features(img, feature)

    assert type(stats) == dict, "stats output not dict"
    assert [key for key in stats.keys() if feature not in key] == [], "feature name not in dict keys"


def test_get_summary_stats():
    img = np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8)
    feature = "test_summary_stats"

    stats = get_summary_stats(img, feature, quantiles=[0.9, 0.5, 0.1], mean=True, std=True, channels=[0, 1, 2])

    assert isinstance(stats, dict)
    assert [key for key in stats.keys() if feature not in key] == [], "feature name not in dict keys"
    assert [key for key in stats.keys() if "mean" in key] != [], "mean not in dict keys"
    assert [key for key in stats.keys() if "std" in key] != [], "std not in dict keys"
    assert [key for key in stats.keys() if "quantile" in key] != [], "quantile not in dict keys"
