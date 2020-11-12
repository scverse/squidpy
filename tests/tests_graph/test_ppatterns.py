from anndata import AnnData

import numpy as np

from spatial_tools.graph import moran, ripley_k


# dummy_adata is now in conftest.py
def test_ripley_k(dummy_adata: AnnData):
    """
    check ripley score and shape
    """
    ripley_k(dummy_adata, cluster_key="cluster")

    # assert ripley in adata.uns
    assert "ripley_k_cluster" in dummy_adata.uns.keys()
    # assert unique clusters in both
    assert np.array_equal(dummy_adata.obs["cluster"].unique(), dummy_adata.uns["ripley_k_cluster"]["cluster"].unique())

    # TO-DO assess length of distances


def test_moran(dummy_adata: AnnData):
    """
    check ripley score and shape
    """
    # spatial_connectivity is missing
    moran(dummy_adata)

    # assert fdr correction in adata.uns
    assert "pval_sim_fdr_bh" in dummy_adata.var.columns
