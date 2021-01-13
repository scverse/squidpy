from anndata import AnnData

import numpy as np

from squidpy.gr import moran, ripley_k, co_occurrence

MORAN_K = "moranI"


def test_ripley_k(adata: AnnData):
    """
    check ripley score and shape
    """
    ripley_k(adata, cluster_key="leiden")

    # assert ripley in adata.uns
    assert "ripley_k_leiden" in adata.uns.keys()
    # assert clusters intersection
    cat_ripley = set(adata.uns["ripley_k_leiden"]["leiden"].unique())
    cat_adata = set(adata.obs["leiden"].cat.categories)
    assert cat_ripley.isdisjoint(cat_adata) is False


def test_moran(dummy_adata: AnnData):
    """
    check moran results
    """
    moran(dummy_adata)
    dummy_adata.var["highly_variable"] = np.random.choice([True, False], size=dummy_adata.var_names.shape)
    df = moran(dummy_adata, copy=True)

    idx_df = df.index.values
    idx_adata = dummy_adata[:, dummy_adata.var.highly_variable.values].var_names.values

    assert MORAN_K in dummy_adata.uns.keys()
    # assert fdr correction in adata.uns
    assert "pval_sim_fdr_bh" in dummy_adata.uns[MORAN_K]
    assert dummy_adata.uns[MORAN_K].columns.shape == (4,)
    # test highly variable
    assert dummy_adata.uns[MORAN_K].shape != df.shape
    # assert idx are sorted
    assert ~np.array_equal(idx_df, idx_adata)
    # assert elements are same
    assert np.all(np.in1d(idx_df, idx_adata))


def test_co_occurrence(adata: AnnData):
    """
    check ripley score and shape
    """
    co_occurrence(adata, cluster_key="leiden")

    # assert occurrence in adata.uns
    assert "leiden_co_occurrence" in adata.uns.keys()
    assert "occ" in adata.uns["leiden_co_occurrence"].keys()
    assert "interval" in adata.uns["leiden_co_occurrence"].keys()

    # assert shapes
    arr = adata.uns["leiden_co_occurrence"]["occ"]
    assert arr.ndim == 3
    assert arr.shape[2] == 49
    assert arr.shape[1] == arr.shape[0] == adata.obs["leiden"].unique().shape[0]
