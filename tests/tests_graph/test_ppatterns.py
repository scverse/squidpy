import pytest

from anndata import AnnData

from pandas.testing import assert_frame_equal
import numpy as np

from squidpy.gr import moran, ripley_k, co_occurrence

MORAN_K = "moranI"


def test_ripley_k(adata: AnnData):
    """Check ripley score and shape."""
    ripley_k(adata, cluster_key="leiden")

    # assert ripley in adata.uns
    assert "ripley_k_leiden" in adata.uns.keys()
    # assert clusters intersection
    cat_ripley = set(adata.uns["ripley_k_leiden"]["leiden"].unique())
    cat_adata = set(adata.obs["leiden"].cat.categories)
    assert cat_ripley.isdisjoint(cat_adata) is False


def test_moran_seq_par(dummy_adata: AnnData):
    """Check whether moran results are the same for seq. and parallel computation."""
    moran(dummy_adata)
    dummy_adata.var["highly_variable"] = np.random.choice([True, False], size=dummy_adata.var_names.shape)
    df = moran(dummy_adata, copy=True, n_jobs=1, seed=42, n_perms=50)
    df_parallel = moran(dummy_adata, copy=True, n_jobs=2, seed=42, n_perms=50)

    idx_df = df.index.values
    idx_adata = dummy_adata[:, dummy_adata.var.highly_variable.values].var_names.values

    assert MORAN_K in dummy_adata.uns.keys()
    assert "pval_sim_fdr_bh" in dummy_adata.uns[MORAN_K]
    assert dummy_adata.uns[MORAN_K].columns.shape == (4,)
    # test highly variable
    assert dummy_adata.uns[MORAN_K].shape != df.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    with pytest.raises(AssertionError, match=r'.*\(column name="pval_sim"\) are different.*'):
        # because the seeds will be different, we don't expect the pval_sim values to be the same
        assert_frame_equal(df, df_parallel)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_moran_reproducibility(dummy_adata: AnnData, n_jobs: int):
    """Check moran reproducibility results."""
    moran(dummy_adata)
    dummy_adata.var["highly_variable"] = np.random.choice([True, False], size=dummy_adata.var_names.shape)
    # seed will work only when multiprocessing/loky
    df_1 = moran(dummy_adata, copy=True, n_jobs=n_jobs, seed=42, n_perms=50)
    df_2 = moran(dummy_adata, copy=True, n_jobs=n_jobs, seed=42, n_perms=50)

    idx_df = df_1.index.values
    idx_adata = dummy_adata[:, dummy_adata.var.highly_variable.values].var_names.values

    assert MORAN_K in dummy_adata.uns.keys()
    # assert fdr correction in adata.uns
    assert "pval_sim_fdr_bh" in dummy_adata.uns[MORAN_K]
    assert dummy_adata.uns[MORAN_K].columns.shape == (4,)
    # test highly variable
    assert dummy_adata.uns[MORAN_K].shape != df_1.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    assert_frame_equal(df_1, df_2)


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


# @pytest.mark.parametrize(("ys", "xs"), [(10, 10), (None, None), (10, 20)])
@pytest.mark.parametrize(("n_jobs", "n_splits"), [(1, 2), (2, 2)])
def test_co_occurrence_reproducibility(adata: AnnData, n_jobs: int, n_splits: int):
    """Check co_occurrence reproducibility results."""
    arr_1, interval_1 = co_occurrence(adata, cluster_key="leiden", copy=True, n_jobs=n_jobs, n_splits=n_splits)
    arr_2, interval_2 = co_occurrence(adata, cluster_key="leiden", copy=True, n_jobs=n_jobs, n_splits=n_splits)

    np.testing.assert_array_equal(sorted(interval_1), sorted(interval_2))
    np.testing.assert_allclose(arr_1, arr_2)
