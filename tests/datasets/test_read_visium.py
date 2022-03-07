""""
Test to read the example datasets.
"""
from pathlib import Path
from posixpath import dirname

from anndata.tests.helpers import assert_adata_equal

from squidpy.read import read_visium

ROOT = Path(dirname(dirname(__file__)))


def test_read_visium():
    # Test that reading .h5 file works and does not have any global effects.
    h5_pth = ROOT / "_data/"
    spec_genome_v3 = read_visium(h5_pth, genome="GRCh38")
    nospec_genome_v3 = read_visium(h5_pth)
    assert_adata_equal(spec_genome_v3, nospec_genome_v3)


text_file_path = str(ROOT / "_data/spatial")
text_file = str(ROOT / "_data/spatial/tissue_positions_list.csv")


def test_read_text():
    # Test that reading .txt file works and does not have any global effects.
    anndata1 = read_visium(path=text_file_path, count_file=text_file, text_kwargs={"delimiter": ","})
    anndata2 = read_visium(path=text_file_path, count_file=text_file, text_kwargs={"delimiter": ","})
    assert_adata_equal(anndata1, anndata2)
