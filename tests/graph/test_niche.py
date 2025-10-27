from __future__ import annotations

import numpy as np
import pytest
import scipy
from anndata import AnnData
from pandas.testing import assert_frame_equal
from scipy.sparse import issparse

import squidpy as sq
from squidpy.gr import calculate_niche, spatial_neighbors
from squidpy.gr._niche import (
    _aggregate,
    _calculate_neighborhood_profile,
    _hop,
    _normalize,
    _setdiag,
    _utag,
)
from tests.conftest import PlotTester, PlotTesterMeta

SPATIAL_CONNECTIVITIES_KEY = "spatial_connectivities"
N_NEIGHBORS = 20
GROUPS = "celltype_mapped_refined"


def test_niche_calc_nhood(adata_seqfish: AnnData):
    """Check whether niche calculation using neighborhood profile approach works as intended."""
    spatial_neighbors(adata_seqfish, coord_type="generic", delaunay=False, n_neighs=N_NEIGHBORS)
    calculate_niche(
        adata_seqfish,
        groups=GROUPS,
        flavor="neighborhood",
        n_neighbors=N_NEIGHBORS,
        resolutions=[0.1],
        min_niche_size=100,
    )
    niches = adata_seqfish.obs["nhood_niche_res=0.1"]

    # assert no nans, more niche labels than non-niche labels, and at least 100 obs per niche
    assert niches.isna().sum() == 0
    assert len(niches[niches != "not_a_niche"]) > len(niches[niches == "not_a_niche"])
    for label in niches.unique():
        if label != "not_a_niche":
            assert len(niches[niches == label]) >= 100

    # get obs x neighbor matrix from sparse matrix
    matrix = adata_seqfish.obsp[SPATIAL_CONNECTIVITIES_KEY].tocoo()

    # get obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood
    rel_nhood_profile = _calculate_neighborhood_profile(adata_seqfish, groups=GROUPS, matrix=matrix, abs_nhood=False)
    abs_nhood_profile = _calculate_neighborhood_profile(adata_seqfish, groups=GROUPS, matrix=matrix, abs_nhood=True)
    # assert shape obs x groups
    assert rel_nhood_profile.shape == (
        adata_seqfish.n_obs,
        len(adata_seqfish.obs[GROUPS].cat.categories),
    )
    assert abs_nhood_profile.shape == rel_nhood_profile.shape
    # normalization
    assert int(rel_nhood_profile.sum(axis=1).sum()) == adata_seqfish.n_obs
    assert round(rel_nhood_profile.sum(axis=1).max(), 2) == 1
    # maximum amount of categories equals n_neighbors
    assert abs_nhood_profile.sum(axis=1).max() == N_NEIGHBORS


def test_niche_calc_utag(adata_seqfish: AnnData):
    """Check whether niche calculation using UTAG approach works as intended."""
    spatial_neighbors(adata_seqfish, coord_type="generic", delaunay=False, n_neighs=N_NEIGHBORS)
    calculate_niche(adata_seqfish, flavor="utag", n_neighbors=N_NEIGHBORS, resolutions=[0.1, 1.0])

    niches = adata_seqfish.obs["utag_niche_res=1.0"]
    niches_low_res = adata_seqfish.obs["utag_niche_res=0.1"]

    assert niches.isna().sum() == 0
    assert niches.nunique() > niches_low_res.nunique()

    # assert shape obs x var and sparsity in new feature matrix
    new_feature_matrix = _utag(
        adata_seqfish,
        normalize_adj=True,
        spatial_connectivity_key=SPATIAL_CONNECTIVITIES_KEY,
    )
    assert new_feature_matrix.shape == adata_seqfish.X.shape
    assert issparse(new_feature_matrix)

    spatial_neighbors(adata_seqfish, coord_type="generic", delaunay=False, n_neighs=40)
    new_feature_matrix_more_neighs = _utag(
        adata_seqfish,
        normalize_adj=True,
        spatial_connectivity_key=SPATIAL_CONNECTIVITIES_KEY,
    )

    # matrix products should differ when using different amount of neighbors
    try:
        assert_frame_equal(new_feature_matrix, new_feature_matrix_more_neighs)
    except AssertionError:
        pass
    else:
        raise AssertionError
