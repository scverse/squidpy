import numpy as np
from anndata import AnnData
from pandas.testing import assert_frame_equal
from squidpy.gr import sepal, spatial_neighbors

UNS_KEY = "sepal_score"


def test_sepal_seq_par(adata: AnnData):
    """Check whether sepal results are the same for seq. and parallel computation."""
    spatial_neighbors(adata, coord_type="grid")
    rng = np.random.default_rng(42)
    adata.var["highly_variable"] = rng.choice([True, False], size=adata.var_names.shape, p=[0.005, 0.995])

    sepal(adata, max_neighs=6)
    df = sepal(adata, max_neighs=6, copy=True, n_jobs=1)
    df_parallel = sepal(adata, max_neighs=6, copy=True, n_jobs=2)

    idx_df = df.index.values
    idx_adata = adata[:, adata.var.highly_variable.values].var_names.values

    assert UNS_KEY in adata.uns.keys()
    assert df.columns.shape == (1,)
    # test highly variable
    assert adata.uns[UNS_KEY].shape == df.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    assert_frame_equal(df, df_parallel)


def test_sepal_square_seq_par(adata_squaregrid: AnnData):
    """Test sepal for square grid."""
    adata = adata_squaregrid
    spatial_neighbors(adata, radius=1.0)
    rng = np.random.default_rng(42)
    adata.var["highly_variable"] = rng.choice([True, False], size=adata.var_names.shape)

    sepal(adata, max_neighs=4)
    df_parallel = sepal(adata, copy=True, n_jobs=2, max_neighs=4)

    idx_df = df_parallel.index.values
    idx_adata = adata[:, adata.var.highly_variable.values].var_names.values

    assert UNS_KEY in adata.uns.keys()
    assert df_parallel.columns.shape == (1,)
    # test highly variable
    assert adata.uns[UNS_KEY].shape == df_parallel.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    assert_frame_equal(adata.uns[UNS_KEY], df_parallel)
