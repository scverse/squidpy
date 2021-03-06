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
from squidpy._constants._pkg_constants import Key

_CK = "leiden"


class TestNhoodEnrichment:
    def _assert_common(self, adata: AnnData):
        key = Key.uns.nhood_enrichment(_CK)
        assert adata.uns[key]["zscore"].dtype == np.dtype("float64")
        assert adata.uns[key]["count"].dtype == np.dtype("uint32")
        assert adata.uns[key]["zscore"].shape[0] == adata.obs.leiden.cat.categories.shape[0]
        assert adata.uns[key]["count"].shape[0] == adata.obs.leiden.cat.categories.shape[0]

    def test_nhood_enrichment(self, adata: AnnData):
        spatial_neighbors(adata)
        nhood_enrichment(adata, cluster_key=_CK)

        self._assert_common(adata)

    @pytest.mark.parametrize("backend", ["threading", "multiprocessing", "loky"])
    def test_parallel_works(self, adata: AnnData, backend: str):
        spatial_neighbors(adata)

        nhood_enrichment(adata, cluster_key=_CK, n_jobs=2, n_perms=20, backend=backend)

        self._assert_common(adata)

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_reproducibility(self, adata: AnnData, n_jobs: int):
        spatial_neighbors(adata)

        res1 = nhood_enrichment(adata, cluster_key=_CK, seed=42, n_jobs=n_jobs, n_perms=20, copy=True)
        res2 = nhood_enrichment(adata, cluster_key=_CK, seed=42, n_jobs=n_jobs, n_perms=20, copy=True)
        res3 = nhood_enrichment(adata, cluster_key=_CK, seed=43, n_jobs=n_jobs, n_perms=20, copy=True)

        assert len(res1) == len(res2)
        assert len(res2) == len(res3)

        for key in range(len(res1)):
            np.testing.assert_array_equal(res2[key], res1[key])
            if key == 0:  # z-score
                with pytest.raises(AssertionError):
                    np.testing.assert_array_equal(res3[key], res2[key])
            else:  # counts
                np.testing.assert_array_equal(res3[key], res2[key])


def test_centrality_scores(nhood_data: AnnData):
    adata = nhood_data
    centrality_scores(
        adata=adata,
        cluster_key=_CK,
        connectivity_key="spatial",
    )

    key = Key.uns.centrality_scores(_CK)

    assert key in adata.uns_keys()
    assert isinstance(adata.uns[key], pd.DataFrame)
    assert len(adata.obs[_CK].unique()) == adata.uns[key].shape[0]
    assert adata.uns[key]["degree_centrality"].dtype == np.dtype("float64")
    assert adata.uns[key]["average_clustering"].dtype == np.dtype("float64")
    assert adata.uns[key]["closeness_centrality"].dtype == np.dtype("float64")


@pytest.mark.parametrize("copy", [True, False])
def test_interaction_matrix_copy(nhood_data: AnnData, copy: bool):
    adata = nhood_data
    res = interaction_matrix(
        adata=adata,
        cluster_key=_CK,
        connectivity_key="spatial",
        copy=copy,
    )

    key = Key.uns.interaction_matrix(_CK)
    n_cls = adata.obs[_CK].nunique()

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
    adata = nhood_data
    res = interaction_matrix(
        adata=adata,
        cluster_key=_CK,
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
        assert len(adata.obsp["spatial_connectivities"].data) == res.sum()


def test_interaction_matrix_values(adata_intmat: AnnData):
    result_weighted = interaction_matrix(adata_intmat, "cat", weights=True, copy=True)
    result_unweighted = interaction_matrix(adata_intmat, "cat", weights=False, copy=True)

    expected_weighted = np.array(
        [
            [5, 1],
            [2, 3],
        ]
    )
    expected_unweighted = np.array(
        [
            [4, 1],
            [2, 2],
        ]
    )

    np.testing.assert_array_equal(expected_weighted, result_weighted)
    np.testing.assert_array_equal(expected_unweighted, result_unweighted)


def test_interaction_matrix_nan_values(adata_intmat: AnnData):
    adata_intmat.obs["cat"].iloc[0] = np.nan
    result_weighted = interaction_matrix(adata_intmat, "cat", weights=True, copy=True)
    result_unweighted = interaction_matrix(adata_intmat, "cat", weights=False, copy=True)

    expected_weighted = np.array(
        [
            [2, 1],
            [2, 3],
        ]
    )
    expected_unweighted = np.array(
        [
            [1, 1],
            [2, 2],
        ]
    )

    np.testing.assert_array_equal(expected_weighted, result_weighted)
    np.testing.assert_array_equal(expected_unweighted, result_unweighted)
