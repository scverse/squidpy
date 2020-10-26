import pytest
import numpy as np
import scanpy
import os
import tifffile
from anndata._core import anndata

from spatial_tools.image.tools import get_summary_stats, get_hog_features, get_features_statistics, get_image_features


def get_dummy_data():
    r = np.random.RandomState(100)
    adata = anndata.AnnData(r.rand(200, 100), obs={"cluster": r.randint(0, 3, 200)})

    adata.obsm["spatial"] = np.stack(
    [r.randint(0, 500, 200), r.randint(0, 500, 200)], axis=1)
    return adata


def test_get_image_features(tmpdir):
    features = ["hog", "texture", "summary", "color_hist"]
    img = np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8)

    adata = get_dummy_data()

    dataset_folder = os.path.join(tmpdir, "_data")
    dataset_name = "test"
    img_name = "test_img.tiff"

    features_pd = get_image_features(adata, dataset_folder, dataset_name, img_name, features = features)

    # remove tmp dir when done
    os.remove(dataset_folder)


def test_get_features_statistics():
    img = np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8)
    features = ["hog", "texture", "summary", "color_hist"]
    stats = get_features_statistics(img, features)

    assert type(stats) == dict, "stats output not dict"
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

    assert type(stats) == dict, "stats output not dict"
    assert [key for key in stats.keys() if feature not in key] == [], "feature name not in dict keys"
    assert [key for key in stats.keys() if "mean" in key] != [], "mean not in dict keys"
    assert [key for key in stats.keys() if "std" in key] != [], "std not in dict keys"
    assert [key for key in stats.keys() if "quantile" in key] != [], "quantile not in dict keys"


test_get_image_features(tmpdir="./")