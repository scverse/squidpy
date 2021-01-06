from anndata import AnnData

from squidpy.gr import moran, ripley_k, co_occurrence


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
    check ripley score and shape
    """
    # spatial_connectivity is missing
    moran(dummy_adata)

    # assert fdr correction in adata.uns
    assert "pval_sim_fdr_bh" in dummy_adata.var.columns


def test_co_occurrence(adata: AnnData):
    """
    check ripley score and shape
    """
    co_occurrence(adata, cluster_key="leiden")

    # assert occurrence in adata.uns
    assert "cluster_co_occurrence" in adata.uns.keys()
    assert "occ" in adata.uns["cluster_co_occurrence"].keys()
    assert "interval" in adata.uns["cluster_co_occurrence"].keys()

    # assert shapes
    arr = adata.uns["cluster_co_occurrence"]["occ"]
    assert arr.ndim == 3
    assert arr.shape[2] == 49
    assert arr.shape[1] == arr.shape[0] == adata.obs["cluster"].unique().shape[0]
