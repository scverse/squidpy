from anndata import AnnData

import numpy as np

from squidpy.gr import moran, ripley_k, co_occurrence


def test_ripley_k(adata: AnnData):
    """
    check ripley score and shape
    """
    ripley_k(adata, cluster_key="cluster")

    # assert ripley in adata.uns
    assert "ripley_k_cluster" in adata.uns.keys()
    # assert unique clusters in both
    assert np.array_equal(adata.obs["cluster"].unique(), adata.uns["ripley_k_cluster"]["cluster"].unique())

    # TO-DO assess length of distances


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
    co_occurrence(adata, cluster_key="cluster")

    # assert occurrence in adata.uns
    assert "cluster_co_occurrence" in adata.uns.keys()
    assert "occ" in adata.uns["cluster_co_occurrence"].keys()
    assert "interval" in adata.uns["cluster_co_occurrence"].keys()

    # assert shapes
    arr = adata.uns["cluster_co_occurrence"]["occ"]
    assert arr.ndim == 3
    assert arr.shape[2] == 49
    assert arr.shape[1] == arr.shape[0] == adata.obs["cluster"].unique().shape[0]
