""" "
Tests to make sure the Visium example datasets load.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import spatialdata as sd
from anndata.tests.helpers import assert_adata_equal

from squidpy.datasets import visium, visium_hne_sdata
from squidpy.datasets._downloader import DEFAULT_CACHE_DIR


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
    # Use DEFAULT_CACHE_DIR to match what download_data.py uses
    base_dir = DEFAULT_CACHE_DIR / "visium"

    # Tests that reading / downloading datasets works
    # and it does not have any global effects
    sample_dataset = visium(sample, base_dir=base_dir)
    sample_dataset_again = visium(sample, base_dir=base_dir)
    assert_adata_equal(sample_dataset, sample_dataset_again)

    # Test that downloading dataset again returns the same data
    # (uses cache)
    sample_dataset_again = visium(sample, base_dir=base_dir)
    assert_adata_equal(sample_dataset, sample_dataset_again)

    # Test that downloading tissue image works
    sample_dataset = visium(sample, base_dir=base_dir, include_hires_tiff=True)
    expected_image_path = base_dir / sample / "image.tif"
    spatial_metadata = sample_dataset.uns["spatial"][sample]["metadata"]
    image_path = Path(spatial_metadata["source_image_path"])
    assert image_path == expected_image_path

    # Test that tissue image exists and is a valid image file
    assert image_path.exists()

    # Test that tissue image is a tif image file (using `file`)
    process = subprocess.run(["file", "--mime-type", image_path], stdout=subprocess.PIPE)
    output = process.stdout.strip().decode()
    assert output == str(image_path) + ": image/tiff"


@pytest.mark.timeout(120)
@pytest.mark.internet()
def test_visium_sdata_dataset():
    # Not passing path uses DEFAULT_CACHE_DIR (~/.cache/squidpy)
    sdata = visium_hne_sdata()
    assert isinstance(sdata, sd.SpatialData)
    assert list(sdata.shapes.keys()) == ["spots"]
    assert list(sdata.images.keys()) == ["hne"]
