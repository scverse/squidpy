from anndata.tests.helpers import assert_adata_equal

from squidpy.read import visium


def test_read_visium():
    # Test that reading .h5 file works and does not have any global effects.
    h5_file_path = "tests/_data"
    spec_genome_v3 = visium(h5_file_path, genome="GRCh38", load_images=True)
    nospec_genome_v3 = visium(h5_file_path)
    assert_adata_equal(spec_genome_v3, nospec_genome_v3)


def test_read_text():
    text_file_path = "tests/_data/"
    text_file = "spatial/tissue_positions_list.csv"
    adata1 = visium(path=text_file_path, counts_file=text_file, library_id="foo")
    adata2 = visium(path=text_file_path, counts_file=text_file, library_id="foo")
    assert_adata_equal(adata1, adata2)
