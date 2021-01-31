import pytest

from anndata import AnnData

import numpy as np
import pandas as pd

from squidpy.gr import (
    nhood_enrichment,
    centrality_scores,
    spatial_neighbors,
    interaction_matrix,
)


def test_nhood_enrichment(adata: AnnData):

    ckey = "leiden"
    spatial_neighbors(adata)
    nhood_enrichment(adata, cluster_key=ckey)

    assert adata.uns[f"{ckey}_nhood_enrichment"]["zscore"].dtype == np.dtype("float64")
    assert adata.uns[f"{ckey}_nhood_enrichment"]["count"].dtype == np.dtype("uint32")
    assert adata.uns[f"{ckey}_nhood_enrichment"]["zscore"].shape[0] == adata.obs.leiden.cat.categories.shape[0]
    assert adata.uns[f"{ckey}_nhood_enrichment"]["count"].shape[0] == adata.obs.leiden.cat.categories.shape[0]


def test_centrality_scores(nhood_data: AnnData):
    """
    check that scores fit the expected shape + content
    """
    adata = nhood_data
    centrality_scores(
        adata=adata,
        cluster_key="leiden",
        connectivity_key="spatial",
    )
    # assert saving in .uns
    key = "leiden_centrality_scores"
    assert key in adata.uns_keys()
    # assert centrality scores are computed for each cluster
    assert isinstance(adata.uns[key], pd.DataFrame)
    assert len(adata.obs["leiden"].unique()) == adata.uns[key].shape[0]
    assert adata.uns[key]["degree_centrality"].dtype == np.dtype("float64")
    assert adata.uns[key]["average_clustering"].dtype == np.dtype("float64")
    assert adata.uns[key]["closeness_centrality"].dtype == np.dtype("float64")


@pytest.mark.parametrize("copy", [True, False])
def test_interaction_matrix_copy(nhood_data: AnnData, copy: bool):
    """
    check that interaction matrix fits the expected shape
    """
    adata = nhood_data
    res = interaction_matrix(
        adata=adata,
        cluster_key="leiden",
        connectivity_key="spatial",
        copy=copy,
    )
    # assert saving in .uns
    key = "leiden_interactions"
    n_cls = adata.obs["leiden"].nunique()

    if not copy:
        assert res is None
        assert key in adata.uns_keys()
        res = adata.uns[key]
    else:
        assert key not in adata.uns_keys()

    assert isinstance(res, np.ndarray)
    assert res.shape == (n_cls, n_cls)


@pytest.mark.parametrize("normalized", [True, False])
def test_interaction_matrix_normalize(nhood_data: AnnData, normalized: bool):
    """
    check that interaction matrix fits the expected shape
    """
    adata = nhood_data
    res = interaction_matrix(
        adata=adata,
        cluster_key="leiden",
        connectivity_key="spatial",
        copy=True,
        normalized=normalized,
    )
    n_cls = adata.obs["leiden"].nunique()

    assert isinstance(res, np.ndarray)
    assert res.shape == (n_cls, n_cls)

    if normalized:
        np.testing.assert_allclose(res.sum(1), 1.0), res.sum(1)
    else:
        assert not np.allclose(res.sum(1), 1.0), res.sum(1)
