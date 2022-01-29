"""
Tests to make sure the visium example datasets load.
"""

from pathlib import Path
import pytest
import subprocess

from scanpy._settings import settings
from anndata.tests.helpers import assert_adata_equal
from scanpy.datasets._ebi_expression_atlas import ebi_expression_atlas

import squidpy as sq


@pytest.mark.internet()
def test_visium_datasets(tmp_dataset_dir, tmpdir):
    # Tests that reading/ downloading works and is does not have global effects
    hheart = sq.datasets.visium_sge("V1_Human_Heart")
    mbrain = sq.datasets.visium_sge("V1_Adult_Mouse_Brain")
    hheart_again = sq.datasets.visium_sge("V1_Human_Heart")
    assert_adata_equal(hheart, hheart_again)

    # Test that changing the dataset dir doesn't break reading
    settings.datasetdir = Path(tmpdir)
    mbrain_again = sq.datasets.visium_sge("V1_Adult_Mouse_Brain")
    assert_adata_equal(mbrain, mbrain_again)

    # Test that downloading tissue image works
    mbrain = sq.datasets.visium_sge("V1_Adult_Mouse_Brain", include_hires_tiff=True)
    expected_image_path = settings.datasetdir / "V1_Adult_Mouse_Brain" / "image.tif"
    image_path = Path(mbrain.uns["spatial"]["V1_Adult_Mouse_Brain"]["metadata"]["source_image_path"])
    assert image_path == expected_image_path

    # Test that tissue image exists and is a valid image file
    assert image_path.exists()

    # Test that tissue image is a tif image file (using `file`)
    process = subprocess.run(["file", "--mime-type", image_path], stdout=subprocess.PIPE)
    output = process.stdout.strip().decode()  # make process output string
    assert output == str(image_path) + ": image/tiff"


def test_download_failure():
    from urllib.error import HTTPError

    with pytest.raises(HTTPError):
        ebi_expression_atlas("not_a_real_accession")
