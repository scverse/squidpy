""" "
Tests to make sure the Visium example datasets load.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import spatialdata as sd
from anndata.tests.helpers import assert_adata_equal
from scanpy import settings

from squidpy.datasets import visium, visium_hne_sdata


@pytest.mark.timeout(120)
@pytest.mark.internet()
@pytest.mark.parametrize(
    "sample",
    [
        "V1_Mouse_Kidney",
        "Targeted_Visium_Human_SpinalCord_Neuroscience",
        "Visium_FFPE_Human_Breast_Cancer",
    ],
)
def test_visium_datasets(sample):
    # Tests that reading / downloading datasets works
    # and it does not have any global effects
    sample_dataset = visium(sample)
    sample_dataset_again = visium(sample)
    assert_adata_equal(sample_dataset, sample_dataset_again)

    # Test that downloading dataset again returns the same data
    # (uses cache)
    sample_dataset_again = visium(sample)
    assert_adata_equal(sample_dataset, sample_dataset_again)

    # Test that downloading tissue image works
    sample_dataset = visium(sample, include_hires_tiff=True)
    expected_image_path = (Path(settings.datasetdir) / "visium" / sample / "image.tif").resolve()
    spatial_metadata = sample_dataset.uns["spatial"][sample]["metadata"]
    image_path = Path(spatial_metadata["source_image_path"]).resolve()
    assert image_path == expected_image_path

    # Test that tissue image exists and is a valid image file
    assert image_path.exists()

    # Test that tissue image is a tif image file (using `file`)
    process = subprocess.run(["file", "--mime-type", image_path], stdout=subprocess.PIPE)
    output = process.stdout.strip().decode()
    assert output == str(image_path) + ": image/tiff"


# since this is 400mb's
# and need to unpack it,
# if downloading on other integration tests, it will timeout
@pytest.mark.timeout(240)
@pytest.mark.internet()
def test_visium_sdata_dataset():
    # Not passing path uses scanpy.settings.datasetdir
    sdata = visium_hne_sdata()
    assert isinstance(sdata, sd.SpatialData)
    assert list(sdata.shapes.keys()) == ["spots"]
    assert list(sdata.images.keys()) == ["hne"]
