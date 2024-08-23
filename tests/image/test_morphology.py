from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import skimage.measure
import squidpy as sq
from spatialdata import SpatialData
from spatialdata.datasets import blobs, raccoon

# noinspection PyProtectedMember
from squidpy.im._feature import _get_region_props


@pytest.fixture
def sdata_racoon() -> SpatialData:
    return raccoon()


@pytest.fixture
def sdata_blobs() -> SpatialData:
    return blobs()


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
    def test__get_region_props(self, sdata_blobs):
        region_props = _get_region_props(
            label_element=sdata_blobs["blobs_labels"],
            image_element=sdata_blobs["blobs_image"],
            props=["label", "area", "eccentricity"],
        )
        assert len(region_props) == len(sdata_blobs["table"].obs)

    def test_quantify_morphology_callables(self, sdata_blobs):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            methods=["label", "area", "eccentricity"],
            split_by_channels=True,
        )

        assert "morphology" in sdata_blobs["table"].obsm.keys()
        assert isinstance(sdata_blobs["table"].obsm["morphology"], pd.DataFrame)
