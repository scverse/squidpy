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

tmpdir = None


@pytest.mark.internet()
@pytest.mark.parametrize(
    # Loading samples from spaceranger versions 1.1.0, 1.2.0 and 1.3.0
    "sample1",
    "sample2",
    [
        ("V1_Human_Heart", "V1_Adult_Mouse_Brain"),
        ("Targeted_Visium_Human_Glioblastoma_Pan_Cancer", "Parent_Visium_Human_BreastCancer"),
        ("Visium_FFPE_Mouse_Kidney", "Visium_FFPE_Human_Prostate_IF"),
    ],
)
def testsamples(sample1, sample2):
    # Tests that reading / downloading works and it does not have global effects
    testsample1 = sq.datasets.visium_sge(sample1)
    testsample2 = sq.datasets.visium_sge(sample2)
    testsample1_again = sq.datasets.visium_sge(sample1)
    assert_adata_equal(testsample1, testsample1_again)

    # Test that changing the dataset dir doesn't break reading
    settings.datasetdir = Path(tmpdir)
    testsample2_again = sq.datasets.visium_sge(sample2)
    assert_adata_equal(testsample2, testsample2_again)

    # Test that downloading tissue image works
    testsample2 = sq.datasets.visium_sge(sample2, include_hires_tiff=True)
    expected_image_path = settings.datasetdir / sample2 / "image.tif"
    image_path = Path(testsample2.uns["spatial"][sample2]["metadata"]["source_image_path"])
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
