""""
Tests to make sure the Visium example datasets load.
"""

import subprocess
from pathlib import Path

import pytest
from anndata.tests.helpers import assert_adata_equal
from scanpy._settings import settings
from squidpy.datasets import visium


@pytest.mark.timeout(120)
@pytest.mark.internet()
@pytest.mark.parametrize(
    "sample", ["V1_Mouse_Kidney", "Targeted_Visium_Human_SpinalCord_Neuroscience", "Visium_FFPE_Human_Breast_Cancer"]
)
def test_visium_datasets(tmpdir, sample):
    # Tests that reading / downloading datasets works and it does not have any global effects
    sample_dataset = visium(sample)
    sample_dataset_again = visium(sample)
    assert_adata_equal(sample_dataset, sample_dataset_again)

    # Test that changing the dataset directory doesn't break reading
    settings.datasetdir = Path(tmpdir)
    sample_dataset_again = visium(sample)
    assert_adata_equal(sample_dataset, sample_dataset_again)

    # Test that downloading tissue image works
    sample_dataset = visium(sample, include_hires_tiff=True)
    expected_image_path = settings.datasetdir / sample / "image.tif"
    image_path = Path(sample_dataset.uns["spatial"][sample]["metadata"]["source_image_path"])
    assert image_path == expected_image_path

    # Test that tissue image exists and is a valid image file
    assert image_path.exists()

    # Test that tissue image is a tif image file (using `file`)
    process = subprocess.run(["file", "--mime-type", image_path], stdout=subprocess.PIPE)
    output = process.stdout.strip().decode()  # make process output string
    assert output == str(image_path) + ": image/tiff"
