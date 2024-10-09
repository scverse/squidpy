from __future__ import annotations

import numpy as np
import pytest
import scipy
from anndata import AnnData
from pandas.testing import assert_frame_equal
from scipy.sparse import issparse

from squidpy.gr import calculate_niche, spatial_neighbors
from squidpy.gr._niche import _aggregate, _calculate_neighborhood_profile, _hop, _normalize, _setdiag, _utag

SPATIAL_CONNECTIVITIES_KEY = "spatial_connectivities"
N_NEIGHBORS = 20


def test_neighborhood_profile_calculation(adata_seqfish: AnnData):
    """Check whether niche calculation using neighborhood profile approach works as intended."""
    spatial_neighbors(adata_seqfish, coord_type="generic", delaunay=False, n_neighs=N_NEIGHBORS)
    calculate_niche(
        adata_seqfish,
        groups="celltype_mapped_refined",
        flavor="neighborhood",
        n_neighbors=N_NEIGHBORS,
        resolutions=[0.1],
        min_niche_size=100,
    )
    niches = adata_seqfish.obs["neighborhood_niche_res=0.1"]

    # assert no nans, more niche labels than non-niche labels, and at least 100 obs per niche
    assert niches.isna().sum() == 0
    assert len(niches[niches != "not_a_niche"]) > len(niches[niches == "not_a_niche"])
    for label in niches.unique():
        if label != "not_a_niche":
            assert len(niches[niches == label]) >= 100

    rel_nhood_profile, abs_nhood_profile = _calculate_neighborhood_profile(
        adata_seqfish,
        groups="celltype_mapped_refined",
        subset_groups=None,
        spatial_connectivities_key=SPATIAL_CONNECTIVITIES_KEY,
    )
    # assert shape obs x groups
    assert rel_nhood_profile.shape == (
        adata_seqfish.n_obs,
        len(adata_seqfish.obs["celltype_mapped_refined"].cat.categories),
    )
    assert abs_nhood_profile.shape == rel_nhood_profile.shape
    # normalization
    assert int(rel_nhood_profile.sum(axis=1).sum()) == adata_seqfish.n_obs
    assert rel_nhood_profile.sum(axis=1).max() == 1
    # maximum amount of categories equals n_neighbors
    assert abs_nhood_profile.sum(axis=1).max() == N_NEIGHBORS


def test_utag(adata_seqfish: AnnData):
    """Check whether niche calculation using UTAG approach works as intended."""
    spatial_neighbors(adata_seqfish, coord_type="generic", delaunay=False, n_neighs=N_NEIGHBORS)
    calculate_niche(adata_seqfish, flavor="utag", n_neighbors=N_NEIGHBORS, resolutions=[0.1, 1.0])

    niches = adata_seqfish.obs["utag_res=1.0"]
    niches_low_res = adata_seqfish.obs["utag_res=0.1"]

    assert niches.isna().sum() == 0
    assert niches.nunique() > niches_low_res.nunique()

    # assert shape obs x var and sparsity in new feature matrix
    new_feature_matrix = _utag(adata_seqfish, normalize_adj=True, spatial_connectivity_key=SPATIAL_CONNECTIVITIES_KEY)
    assert new_feature_matrix.shape == adata_seqfish.X.shape
    assert issparse(new_feature_matrix)

    spatial_neighbors(adata_seqfish, coord_type="generic", delaunay=False, n_neighs=40)
    new_feature_matrix_more_neighs = _utag(
        adata_seqfish, normalize_adj=True, spatial_connectivity_key=SPATIAL_CONNECTIVITIES_KEY
    )

    # matrix products should differ when using different amount of neighbors
    try:
        assert_frame_equal(new_feature_matrix, new_feature_matrix_more_neighs)
    except AssertionError:
        pass
    else:
        raise AssertionError


def test_cellcharter_approach(adata_seqfish: AnnData):
    """Check whether niche calculation using CellCharter approach works as intended."""

    spatial_neighbors(adata_seqfish, coord_type="generic", delaunay=False, n_neighs=N_NEIGHBORS)
    calculate_niche(
        adata_seqfish, groups="celltype_mapped_refined", flavor="cellcharter", adj_subsets=3, n_components=5
    )
    niches = adata_seqfish.obs["cellcharter_niche"]

    assert niches.nunique() == 5
    assert niches.isna().sum() == 0

    adj = adata_seqfish.obsp[SPATIAL_CONNECTIVITIES_KEY]
    adj_hop = _setdiag(adj, value=0)
    assert adj_hop.shape == adj.shape
    assert issparse(adj_hop)
    assert isinstance(adj_hop, scipy.sparse.csr_matrix)

    adj_visited = _setdiag(adj.copy(), 1)  # Track visited neighbors
    adj_hop, adj_visited = _hop(adj_hop, adj, adj_visited)
    assert adj_hop.shape == adj.shape
    assert adj_hop.shape == adj_visited.shape

    assert np.array(np.sum(adj, axis=1)).squeeze() == N_NEIGHBORS
    adj_hop_norm = _normalize(adj_hop)
    assert adj_hop_norm.shape == adj.shape

    mean_aggr_matrix = _aggregate(adata_seqfish, adj_hop_norm, "mean")
    assert mean_aggr_matrix.shape == adata_seqfish.X.shape
    var_aggr_matrix = _aggregate(adata_seqfish, adj_hop_norm, "variance")
    assert var_aggr_matrix.shape == adata_seqfish.X.shape

    # TODO: add test for GMM


# TODO: comppare results to previously calculated niches
