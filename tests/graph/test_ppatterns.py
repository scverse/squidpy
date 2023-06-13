from typing import Any, Literal

import numpy as np
import pytest
from anndata import AnnData
from pandas.testing import assert_frame_equal
from squidpy._constants._pkg_constants import Key
from squidpy.gr import co_occurrence, spatial_autocorr
from squidpy.gr._ppatterns import _find_min_max

MORAN_K = "moranI"
GEARY_C = "gearyC"


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_seq_par(dummy_adata: AnnData, mode: str):
    """Check whether spatial autocorr results are the same for seq. and parallel computation."""
    spatial_autocorr(dummy_adata, mode=mode)
    dummy_adata.var["highly_variable"] = np.random.choice([True, False], size=dummy_adata.var_names.shape)
    df = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=1, seed=42, n_perms=50)
    df_parallel = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=2, seed=42, n_perms=50)

    idx_df = df.index.values
    idx_adata = dummy_adata[:, dummy_adata.var.highly_variable.values].var_names.values

    if mode == "moran":
        UNS_KEY = MORAN_K
    elif mode == "geary":
        UNS_KEY = GEARY_C
    assert UNS_KEY in dummy_adata.uns.keys()
    assert "pval_sim_fdr_bh" in df
    assert "pval_norm_fdr_bh" in dummy_adata.uns[UNS_KEY]
    assert dummy_adata.uns[UNS_KEY].columns.shape == (4,)
    assert df.columns.shape == (9,)
    # test pval_norm same
    np.testing.assert_array_equal(df["pval_norm"].values, df_parallel["pval_norm"].values)
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    with pytest.raises(AssertionError, match=r'.*\(column name="pval_z_sim"\) are different.*'):
        # because the seeds will be different, we don't expect the pval_sim values to be the same
        assert_frame_equal(df, df_parallel)


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_spatial_autocorr_reproducibility(dummy_adata: AnnData, n_jobs: int, mode: str):
    """Check spatial autocorr reproducibility results."""
    rng = np.random.RandomState(42)
    spatial_autocorr(dummy_adata, mode=mode)
    dummy_adata.var["highly_variable"] = rng.choice([True, False], size=dummy_adata.var_names.shape)
    # seed will work only when multiprocessing/loky
    df_1 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=n_jobs, seed=42, n_perms=50)
    df_2 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=n_jobs, seed=42, n_perms=50)

    idx_df = df_1.index.values
    idx_adata = dummy_adata[:, dummy_adata.var["highly_variable"].values].var_names.values

    if mode == "moran":
        UNS_KEY = MORAN_K
    elif mode == "geary":
        UNS_KEY = GEARY_C
    assert UNS_KEY in dummy_adata.uns.keys()
    # assert fdr correction in adata.uns
    assert "pval_sim_fdr_bh" in df_1
    assert "pval_norm_fdr_bh" in dummy_adata.uns[UNS_KEY]
    # test pval_norm same
    np.testing.assert_array_equal(df_1["pval_norm"].values, df_2["pval_norm"].values)
    np.testing.assert_array_equal(df_1["var_norm"].values, df_2["var_norm"].values)
    assert dummy_adata.uns[UNS_KEY].columns.shape == (4,)
    assert df_2.columns.shape == (9,)
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df_1.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    assert_frame_equal(df_1, df_2)


@pytest.mark.parametrize(
    "attr,layer,genes",
    [
        ("X", None, None),
        ("obs", None, None),
        ("obs", None, "foo"),
        ("obsm", "spatial", None),
        ("obsm", "spatial", [1, 0]),
    ],
)
def test_spatial_autocorr_attr(dummy_adata: AnnData, attr: Literal["X", "obs", "obsm"], layer: str, genes: Any):
    if attr == "obs":
        if isinstance(genes, str):
            dummy_adata.obs[genes] = np.random.RandomState(42).normal(size=(dummy_adata.n_obs,))
            index = [genes]
        else:
            index = dummy_adata.obs.select_dtypes(include=np.number).columns
    elif attr == "X":
        index = dummy_adata.var_names if genes is None else genes
    elif attr == "obsm":
        index = np.arange(dummy_adata.obsm[layer].shape[1]) if genes is None else genes

    spatial_autocorr(dummy_adata, attr=attr, mode="moran", layer=layer, genes=genes)

    df = dummy_adata.uns[MORAN_K]
    np.testing.assert_array_equal(np.isfinite(df), True)
    np.testing.assert_array_equal(sorted(df.index), sorted(index))


def test_co_occurrence(adata: AnnData):
    """
    check co_occurrence score and shape
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


@pytest.mark.parametrize("size", [1, 3])
def test_co_occurrence_explicit_interval(adata: AnnData, size: int):
    minn, maxx = _find_min_max(adata.obsm[Key.obsm.spatial])
    interval = np.linspace(minn, maxx, size)
    if size == 1:
        with pytest.raises(ValueError, match=r"Expected interval to be of length"):
            _ = co_occurrence(adata, cluster_key="leiden", copy=True, interval=interval)
    else:
        _, interval_1 = co_occurrence(adata, cluster_key="leiden", copy=True, interval=interval)

        assert interval is not interval_1
        np.testing.assert_allclose(interval, interval_1)  # allclose because in the func, we use f32


def test_use_raw(dummy_adata: AnnData):
    var_names = [str(i) for i in range(10)]
    raw = dummy_adata[:, dummy_adata.var_names[: len(var_names)]].copy()
    raw.var_names = var_names
    dummy_adata.raw = raw

    df = spatial_autocorr(dummy_adata, use_raw=True, copy=True)

    np.testing.assert_equal(sorted(df.index), sorted(var_names))
