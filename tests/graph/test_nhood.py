from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from squidpy._constants._pkg_constants import Key
from squidpy.gr import (
    centrality_scores,
    interaction_matrix,
    nhood_enrichment,
    spatial_neighbors,
)

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

        # Test that the same seed produces the same results
        np.testing.assert_array_equal(res2.zscore, res1.zscore)
        np.testing.assert_array_equal(res2.counts, res1.counts)

        # Test that different seeds produce different z-scores but same counts
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(res3.zscore, res2.zscore)
        np.testing.assert_array_equal(res3.counts, res2.counts)


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

    expected_weighted = np.array([[5, 1], [2, 3]])
    expected_unweighted = np.array([[4, 1], [2, 2]])

    np.testing.assert_array_equal(expected_weighted, result_weighted)
    np.testing.assert_array_equal(expected_unweighted, result_unweighted)


def test_interaction_matrix_nan_values(adata_intmat: AnnData):
    adata_intmat.obs.loc["0", "cat"] = np.nan
    result_weighted = interaction_matrix(adata_intmat, "cat", weights=True, copy=True)
    result_unweighted = interaction_matrix(adata_intmat, "cat", weights=False, copy=True)

    expected_weighted = np.array([[2, 1], [2, 3]])
    expected_unweighted = np.array([[1, 1], [2, 2]])

    np.testing.assert_array_equal(expected_weighted, result_weighted)
    np.testing.assert_array_equal(expected_unweighted, result_unweighted)


@pytest.mark.parametrize("normalization", ["none", "total", "conditional"])
def test_nhood_enrichment_normalization_modes(adata: AnnData, normalization: str):
    spatial_neighbors(adata)
    result = nhood_enrichment(adata, cluster_key=_CK, normalization=normalization, n_jobs=1, n_perms=20, copy=True)

    if normalization == "conditional":
        z, count, ccr = result
        assert isinstance(ccr, np.ndarray)
    else:
        z, count = result

    assert isinstance(z, np.ndarray)
    assert isinstance(count, np.ndarray)
    assert z.shape == count.shape
    assert z.shape[0] == adata.obs[_CK].cat.categories.shape[0]


def test_conditional_normalization_zero_division(adata: AnnData):
    adata = adata.copy()
    min_cells = 10
    if _CK not in adata.obs:
        raise ValueError(f"Cluster key '{_CK}' not in adata.obs")
    if not pd.api.types.is_categorical_dtype(adata.obs[_CK]):
        adata.obs[_CK] = adata.obs[_CK].astype("category")
    adata.obs[_CK] = adata.obs[_CK].cat.add_categories("isolated")
    adata.obs.loc[adata.obs.index[0], _CK] = "isolated"
    spatial_neighbors(adata)
    valid_clusters = [c for c, count in adata.obs[_CK].value_counts().items() if count >= min_cells]
    valid_idx = [i for i, cat in enumerate(adata.obs[_CK].cat.categories) if cat in valid_clusters]

    result = nhood_enrichment(adata, cluster_key=_CK, normalization="conditional", copy=True)
    assert result is not None
    zscore, count_normalized, conditional_ratio = result
    assert not np.any(np.isinf(zscore))
    assert not np.any(np.isinf(count_normalized))
    assert not np.any(np.isinf(conditional_ratio))
    assert not np.isnan(zscore[np.ix_(valid_idx, valid_idx)]).any()
    assert not np.isnan(count_normalized[np.ix_(valid_idx, valid_idx)]).any()
    assert not np.isnan(conditional_ratio[np.ix_(valid_idx, valid_idx)]).any()


@pytest.mark.parametrize(
    "normalization, expected_dtype",
    [
        ("none", np.uint32),
        ("total", np.float64),
        ("conditional", np.float64),
    ],
)
def test_output_dtype(adata: AnnData, normalization: str, expected_dtype):
    spatial_neighbors(adata)
    result = nhood_enrichment(
        adata,
        cluster_key=_CK,
        normalization=normalization,
        n_jobs=1,
        n_perms=20,
        copy=True,
    )

    if normalization == "conditional":
        _, count, _ = result
    else:
        _, count = result

    assert count.dtype == expected_dtype


def test_invalid_normalization_raises(adata: AnnData):
    spatial_neighbors(adata)
    with pytest.raises(ValueError, match="Invalid normalization mode"):
        nhood_enrichment(adata, cluster_key=_CK, normalization="invalid_mode", copy=True)
