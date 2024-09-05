from __future__ import annotations

import time

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import skimage.measure
import spatialdata as sd
from spatialdata import SpatialData
from spatialdata.datasets import blobs, raccoon

import squidpy as sq
from squidpy.im import _measurements

# noinspection PyProtectedMember
from squidpy.im._feature import _get_region_props

# noinspection PyProtectedMember
from squidpy.im._measurements import border_occupied_factor


@pytest.fixture
def sdata_racoon() -> SpatialData:
    return raccoon()


@pytest.fixture
def sdata_blobs() -> SpatialData:
    return blobs()


@pytest.fixture
def morphology_methods() -> list[str]:
    return ["label", "area", "eccentricity", "circularity", "granularity", "border_occupied_factor"]


@pytest.mark.xfail(
    raises=AssertionError, reason="calling skimage.measure.label on segmentation produces additional labels"
)
class TestSkimageBackend:
    def test_label_bug_present(self, sdata_blobs):
        npt.assert_array_equal(
            np.unique(skimage.measure.label(sdata_blobs["blobs_labels"])),
            np.unique(sdata_blobs["blobs_labels"]),
        )


class TestMorphology:
    def test_quantify_morphology_granularity(self, sdata_blobs):
        granular_spectrum_length = 16
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            methods=["label", "granularity"],
            split_by_channels=True,
            granularity_granular_spectrum_length=granular_spectrum_length,
        )

        columns = sdata_blobs["table"].obsm["morphology"].columns
        for channel in range(3):
            for granular_spectrum in range(granular_spectrum_length):
                assert f"granularity_ch{channel}_{granular_spectrum}" in columns

    def test_quantify_morphology_border_occupied_factor(self, sdata_blobs):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            methods=["label", "border_occupied_factor"],
            split_by_channels=True,
        )
        assert "border_occupied_factor" in sdata_blobs["table"].obsm["morphology"].columns

    def test__get_region_props(self, sdata_blobs):
        region_props = _get_region_props(
            label_element=sdata_blobs["blobs_labels"],
            image_element=sdata_blobs["blobs_image"],
            props=["label", "area", "eccentricity"],
        )
        assert len(region_props) == len(sdata_blobs["table"].obs)

    def test_quantify_morphology_callables(self, sdata_blobs, morphology_methods):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            methods=morphology_methods,
            split_by_channels=True,
        )

        assert "morphology" in sdata_blobs["table"].obsm.keys()
        assert isinstance(sdata_blobs["table"].obsm["morphology"], pd.DataFrame)
        assert "circularity" in sdata_blobs["table"].obsm["morphology"].columns
        assert "border_occupied_factor" in sdata_blobs["table"].obsm["morphology"].columns

    def test_quantify_morphology_all(self, sdata_blobs):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            split_by_channels=True,
        )

        for name in _measurements._all_regionprops_names():
            assert any([column.startswith(name) for column in sdata_blobs["table"].obsm["morphology"].columns])
        # for column in sdata_blobs["table"].obsm["morphology"].columns:
        #     assert any(column.startswith(name) for name in _measurements._all_regionprops_names())

    @pytest.mark.xfail(
        raises=ValueError,
        reason="For the moment, there is no association " "between the multiscale labels and a table in blobs.",
    )
    def test_quantify_morphology_multiscale(self, sdata_blobs, morphology_methods):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_multiscale_labels",
            image="blobs_multiscale_image",
            methods=morphology_methods,
            split_by_channels=True,
        )


# TODO Remove for release
@pytest.fixture
def sdata_merfish() -> SpatialData:
    zarr_path = "./data/merfish.zarr"
    return sd.read_zarr(zarr_path)


# TODO Remove for release
@pytest.fixture
def sdata_mibitof() -> SpatialData:
    zarr_path = "./data/mibitof.zarr"
    return sd.read_zarr(zarr_path)


class TestMorphologyPerformance:
    def test_performance(self, sdata_mibitof, morphology_methods):
        start_time = time.perf_counter()
        sq.im.quantify_morphology(
            sdata=sdata_mibitof,
            label="point8_labels",
            image="point8_image",
            methods=morphology_methods,
            split_by_channels=True,
        )
        end_time = time.perf_counter()
        assert end_time - start_time < 60


class TestMeasurements:
    def test_border_occupied_factor(self):
        label_image = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 2, 2, 2, 0],
                [0, 1, 1, 1, 2, 2, 2, 0],
                [0, 0, 3, 3, 2, 2, 2, 0],
                [0, 0, 3, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        expected = {1: 3 / 8, 2: 3 / 8, 3: 2 / 4}
        actual = border_occupied_factor(label_image)
        assert actual == expected
