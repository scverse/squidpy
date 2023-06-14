from anndata.tests.helpers import assert_adata_equal
from squidpy._constants._pkg_constants import Key
from squidpy.read import visium


def test_read_visium():
    # Test that reading .h5 file works and does not have any global effects.
    h5_file_path = "tests/_data"
    spec_genome_v3 = visium(h5_file_path, genome="GRCh38", load_images=True)
    nospec_genome_v3 = visium(h5_file_path)
    assert_adata_equal(spec_genome_v3, nospec_genome_v3)
    adata = spec_genome_v3
    assert "spatial" in adata.uns.keys()
    lib_id = list(adata.uns[Key.uns.spatial].keys())[0]
    assert Key.uns.image_key in adata.uns[Key.uns.spatial][lib_id].keys()
    assert Key.uns.scalefactor_key in adata.uns[Key.uns.spatial][lib_id].keys()
    assert Key.uns.image_res_key in adata.uns[Key.uns.spatial][lib_id][Key.uns.image_key]
    assert Key.uns.size_key in adata.uns[Key.uns.spatial][lib_id][Key.uns.scalefactor_key]


def test_read_text():
    text_file_path = "tests/_data/"
    text_file = "spatial/tissue_positions_list.csv"
    adata1 = visium(path=text_file_path, counts_file=text_file, library_id="foo")
    adata2 = visium(path=text_file_path, counts_file=text_file, library_id="foo")
    assert_adata_equal(adata1, adata2)
