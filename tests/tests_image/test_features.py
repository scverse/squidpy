import pytest

from anndata import AnnData

import numpy as np
import pandas as pd

from squidpy.im.tools import (
    scale_features,
    get_hog_features,
    get_summary_stats,
    get_features_statistics,
    calculate_image_features,
)
from squidpy.im.object import ImageContainer


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_calculate_image_features(adata: AnnData, cont: ImageContainer, n_jobs: int):
    features = ["hog", "texture", "summary", "color_hist"]
    res = calculate_image_features(adata, cont, features=features, copy=True, n_jobs=n_jobs)

    assert isinstance(res, pd.DataFrame)
    np.testing.assert_array_equal(res.index, adata.obs_names)


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


def test_scale_features(feature_df: pd.DataFrame):
    key = "f_key"
    for inplace in [True, False]:
        adata = AnnData(obs=pd.DataFrame(index=feature_df.index), obsm={key: feature_df})
        options = ["scale", (0, 1), "abs", "robust", "uniform", "normal", lambda x: x * x]

        for o in options:
            scale_features(feature_df, features="all", scaling=o, inplace=inplace)
            scale_features(adata, key=key, features="all", scaling=o, inplace=inplace)

        options_dict = {
            c: options[i] for i, c in enumerate(feature_df.columns[: min(len(feature_df.columns), len(options))])
        }
        dicts_features = list(options_dict)

        scale_features(feature_df, features=dicts_features, scaling=options_dict, inplace=inplace)
        scale_features(adata, key=key, features=dicts_features, scaling=options_dict, inplace=inplace)
