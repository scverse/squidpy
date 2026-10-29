from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData
from spatialdata.models import PointsModel
from spatialdata.transformations import get_transformation

from squidpy.experimental.im import rasterize_points


def _sdata(xy: np.ndarray) -> SpatialData:
    return SpatialData(points={"cells": PointsModel.parse(pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}))})


@pytest.fixture
def xy() -> np.ndarray:
    return np.random.default_rng(0).normal(loc=(300.0, 150.0), scale=40.0, size=(500, 2))


def test_each_channel_holds_one_unit_of_mass_per_point(xy: np.ndarray) -> None:
    image = rasterize_points(_sdata(xy), points_key="cells", dx=10.0, blur=[2.0, 0.5])
    assert image.dims == ("c", "y", "x")
    np.testing.assert_allclose(np.asarray(image).sum(axis=(1, 2)), [len(xy), len(xy)], rtol=1e-9)


def test_the_registered_image_sits_where_the_points_are(xy: np.ndarray) -> None:
    image = rasterize_points(_sdata(xy), points_key="cells", dx=10.0, blur=1.0)
    density = np.asarray(image)[0]
    rows, cols = np.indices(density.shape)
    centre_px = np.array([(density * cols).sum() / density.sum(), (density * rows).sum() / density.sum(), 1.0])

    matrix = get_transformation(image, "global").to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    np.testing.assert_allclose((matrix @ centre_px)[:2], xy.mean(axis=0), atol=10.0 / 2)


def test_key_added_stores_the_image_and_none_leaves_sdata_alone(xy: np.ndarray) -> None:
    sdata = _sdata(xy)
    rasterize_points(sdata, points_key="cells")
    assert not sdata.images

    rasterize_points(sdata, points_key="cells", key_added="density")
    assert list(sdata.images) == ["density"]


def test_unknown_points_key_lists_what_is_available(xy: np.ndarray) -> None:
    with pytest.raises(KeyError, match=r"Available: \['cells'\]"):
        rasterize_points(_sdata(xy), points_key="nope")
