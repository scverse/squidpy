import pytest

import anndata

import numpy as np

from spatial_tools.graph import ripley_k


def get_dummy_data():
    r = np.random.RandomState(100)
    adata = anndata.AnnData(r.rand(200, 100), obs={"cluster": r.randint(0, 3, 200)})

    adata.obsm["spatial"] = np.stack([r.randint(0, 500, 200), r.randint(0, 500, 200)], axis=1)
    return adata


def test_ripley_k():
    """
    check ripley score and shape
    """
    adata = get_dummy_data()
    ripley_k(adata, cluster_key="cluster")

    # assert ripley in adata.uns
    assert "ripley_k_cluster" in adata.uns.keys()
    # assert unique clusters in both
    assert np.array_equal(adata.obs["cluster"].unique(), adata.uns["ripley_k_cluster"]["cluster"].unique())

    # TO-DO assess length of distances
