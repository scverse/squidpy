from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
from anndata import AnnData
from pandas.testing import assert_frame_equal

from squidpy._constants._pkg_constants import Key
from squidpy.gr import co_occurrence, spatial_autocorr
from squidpy.gr._ppatterns import _autocorr_perms, _find_min_max, _score_perms

MORAN_K = "moranI"
GEARY_C = "gearyC"


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_seq_par(dummy_adata: AnnData, mode: str):
    """Check whether spatial autocorr results are the same for seq. and parallel computation."""
    spatial_autocorr(dummy_adata, mode=mode)
    dummy_adata.var["highly_variable"] = np.random.choice([True, False], size=dummy_adata.var_names.shape)
    df = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=1, rng=np.random.default_rng(42), n_perms=50)
    df_parallel = spatial_autocorr(
        dummy_adata, mode=mode, copy=True, n_jobs=2, rng=np.random.default_rng(42), n_perms=50
    )

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
    np.testing.assert_allclose(df["pval_norm"].values, df_parallel["pval_norm"].values, atol=1e-12)
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # each permutation draws from its own spawned generator, so the simulated columns do not
    # depend on how the features were split across workers
    df_parallel = df_parallel.loc[df.index]  # align in case the stat-based sort ties differently
    for col in ("pval_sim", "pval_z_sim", "var_sim"):
        np.testing.assert_allclose(df[col].values, df_parallel[col].values, atol=1e-12)


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_spatial_autocorr_reproducibility(dummy_adata: AnnData, n_jobs: int, mode: str):
    """Check spatial autocorr reproducibility results."""
    rng = np.random.RandomState(42)
    spatial_autocorr(dummy_adata, mode=mode)
    dummy_adata.var["highly_variable"] = rng.choice([True, False], size=dummy_adata.var_names.shape)
    # seed will work only when multiprocessing/loky
    df_1 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=n_jobs, rng=np.random.default_rng(42), n_perms=50)
    df_2 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_jobs=n_jobs, rng=np.random.default_rng(42), n_perms=50)

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
    assert dummy_adata.uns[UNS_KEY].columns.shape == (4,)
    assert df_2.columns.shape == (9,)
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df_1.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    assert_frame_equal(df_1, df_2)


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_degenerate_feature_is_nan(mode: str):
    """A gene whose permutations all score alike has no z-test, and must not poison the FDR column.

    Its variance is exactly zero, so the z-score is undefined. Filling `pval_z_sim` with
    `np.empty` left those entries at whatever memory held, which reads as a significant
    p-value, and a NaN handed to `multipletests` spreads over every other gene.
    """
    import scipy.sparse as sps

    from squidpy.gr import spatial_neighbors_knn

    rng = np.random.default_rng(0)
    n = 400
    X = np.zeros((n, 3), dtype=np.float32)
    X[5, 0] = 4.0  # expressed in a single cell: every permutation gives the same score
    X[:, 1] = rng.poisson(2, n)
    X[7, 2] = 1.0
    adata = AnnData(sps.csr_matrix(X))
    adata.var_names = ["solo", "normal", "solo2"]
    adata.obsm["spatial"] = rng.random((n, 2))
    spatial_neighbors_knn(adata, n_neighs=6)

    df = spatial_autocorr(adata, mode=mode, n_perms=13, rng=3, copy=True, show_progress_bar=False)
    degenerate, real = ["solo", "solo2"], "normal"

    assert (df.loc[degenerate, "var_sim"] == 0.0).all()
    assert df.loc[degenerate, "pval_z_sim"].isna().all()
    assert df.loc[degenerate, "pval_z_sim_fdr_bh"].isna().all()
    # the gene with a defined z-test keeps one, and its correction is unaffected
    assert np.isfinite(df.loc[real, "pval_z_sim"])
    assert np.isfinite(df.loc[real, "pval_z_sim_fdr_bh"])
    # the permutation p-value is a tally, so it stays defined for every gene
    assert df["pval_sim"].notna().all()


@pytest.mark.parametrize(("mode", "stat"), [("moran", "I"), ("geary", "C")])
def test_spatial_autocorr_offset_feature_precision(mode: str, stat: str):
    """Both statistics ignore a constant shift, so a large offset must not move the score.

    The kernel rebuilds the centred quantities by subtraction, e.g. `g @ x - x_bar * w_sum`,
    which cancels away roughly `(mean / sd) ** 2` digits. Dense features are centred before
    they reach it; without that, an offset of 1e6 moved Moran's I by several percent.
    """
    from squidpy.gr import spatial_neighbors_knn

    rng = np.random.default_rng(1)
    n = 400
    base = rng.standard_normal(n)
    adata = AnnData(np.stack([base, base + 1e6], axis=1).astype(np.float64))
    adata.var_names = ["plain", "offset"]
    adata.obsm["spatial"] = rng.random((n, 2))
    spatial_neighbors_knn(adata, n_neighs=6)

    # `n_perms` routes the observed score through the kernel rather than scanpy
    df = spatial_autocorr(adata, mode=mode, n_perms=1, rng=0, copy=True, show_progress_bar=False)
    # 1e-8 is the floor for recovering a unit-scale value from one offset by 1e6, not slack:
    # without the centring the difference was 5e-2
    np.testing.assert_allclose(df.loc["offset", stat], df.loc["plain", stat], rtol=1e-8)


def test_spatial_autocorr_ties_match_scanpy():
    """A gene with one non-zero count ties most permutations exactly; ties must count as in scanpy."""
    import scipy.sparse as sps
    from scanpy.metrics import gearys_c
    from sklearn.preprocessing import normalize

    from squidpy.gr import spatial_neighbors_knn

    rng = np.random.default_rng(0)
    X = np.zeros((300, 2), dtype=np.float32)
    X[0, 0] = 5
    X[:, 1] = rng.poisson(2, 300)
    adata = AnnData(sps.csr_matrix(X))
    adata.obsm["spatial"] = rng.random((300, 2))
    spatial_neighbors_knn(adata, n_neighs=6)
    df = spatial_autocorr(adata, mode="geary", n_perms=50, rng=0, copy=True).loc[adata.var_names]

    # the pre-numba algorithm: scanpy on the row-permuted graph, one spawned generator per permutation
    g = normalize(adata.obsp["spatial_connectivities"], norm="l1", axis=1)
    vals = adata.X.T
    obs = gearys_c(g, vals)
    sims = np.stack([gearys_c(g[gen.permutation(300), :], vals) for gen in np.random.default_rng(0).spawn(50)])
    large = (sims >= obs).sum(axis=0)
    large = np.minimum(large, 50 - large)
    assert large[0] > 0  # the scenario really has ties
    np.testing.assert_array_equal(df["pval_sim"].values, (large + 1) / 51)


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_perm_blocks(dummy_adata: AnnData, mode: str, monkeypatch):
    """Drawing the permutations block by block must not change the result."""
    import squidpy.gr._ppatterns as ppatterns

    kw = {"mode": mode, "copy": True, "rng": 42, "n_perms": 50}
    expected = spatial_autocorr(dummy_adata, **kw)
    monkeypatch.setattr(ppatterns, "_PERM_BLOCK_SIZE", 7 * dummy_adata.n_obs)  # 8 blocks, the last one short
    assert_frame_equal(spatial_autocorr(dummy_adata, **kw), expected)


def test_spatial_autocorr_full_gene_list_reordered(dummy_adata: AnnData):
    """A full-length but reordered `genes` must not take the identity fast path in `extract_X`.

    `extract_X` skips `adata[:, genes]` when the selection is every gene in `var_names` order.
    Dropping the order check from that guard leaves a list of the right length taking the fast
    path, which returns `X` in var order while labelling the rows in the caller's order.
    """
    genes = list(dummy_adata.var_names)
    kw = {"mode": "geary", "n_perms": 20, "rng": 0, "copy": True, "show_progress_bar": False}
    fwd = spatial_autocorr(dummy_adata, genes=genes, **kw)
    rev = spatial_autocorr(dummy_adata, genes=genes[::-1], **kw)

    # a gene's statistic cannot depend on the order the caller listed the genes in
    assert set(fwd.index) == set(rev.index)
    np.testing.assert_allclose(fwd["C"], rev.loc[fwd.index, "C"], rtol=1e-12)


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_csc_connectivities(dummy_adata: AnnData, mode: str):
    """A CSC graph must give the CSR result: `normalize(axis=1, copy=False)` leaves CSC untouched."""
    key = Key.obsp.spatial_conn()
    kw = {"mode": mode, "copy": True, "rng": 42, "n_perms": 50}
    csc = dummy_adata.copy()
    csc.obsp[key] = csc.obsp[key].tocsc()
    before = csc.obsp[key].copy()

    assert_frame_equal(spatial_autocorr(csc, **kw), spatial_autocorr(dummy_adata, **kw))
    # row-normalization must not leak back into the caller's graph
    np.testing.assert_array_equal(csc.obsp[key].toarray(), before.toarray())


def test_spatial_autocorr_v183_positional_backend(dummy_adata: AnnData):
    """A v1.8.3 positional call through ``backend`` binds every value and warns about ``backend``.

    Delete this together with the ``backend`` shim. The positional signature it pins only exists
    because `@deprecated_params` keeps accepting `backend` in its v1.8.3 slot; once that is dropped
    for 1.10.0 there is no positional form left to protect and the call below starts raising.
    """
    kw = {"mode": "moran", "n_perms": 20, "rng": 0, "copy": True, "n_jobs": 1, "show_progress_bar": False}
    expected = spatial_autocorr(dummy_adata, **kw)
    args = (
        "spatial_connectivities",
        None,
        "moran",
        True,
        20,
        False,
        "fdr_bh",
        "X",
        None,
        0,
        False,
        True,
        1,
        "loky",
        False,
    )
    with pytest.warns(FutureWarning) as record:
        got = spatial_autocorr(dummy_adata, *args)
    messages = [str(w.message) for w in record]
    assert any("`backend`" in m for m in messages)
    assert any("`seed`" in m for m in messages)
    assert_frame_equal(got, expected)


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_spatial_autocorr_var_norm_formula(dummy_adata: AnnData, mode: str):
    """Analytic ``var_norm`` must use the variance matching the chosen statistic.

    Regression test for #1183: Geary's C and Moran's I have different sampling
    variances under the normality assumption (Cliff & Ord 1981). Reusing Moran's
    variance for Geary's C produced a miscalibrated analytic p-value.
    """
    from sklearn.preprocessing import normalize

    from squidpy.gr._ppatterns import _g_moments

    uns_key = MORAN_K if mode == "moran" else GEARY_C
    spatial_autocorr(dummy_adata, mode=mode, transformation=True, n_perms=None, rng=np.random.default_rng(0))
    var_norm = float(dummy_adata.uns[uns_key]["var_norm"].iloc[0])

    # Reconstruct the exact (row-standardised) weight matrix the routine used.
    g = dummy_adata.obsp["spatial_connectivities"].copy()
    normalize(g, norm="l1", axis=1, copy=False)
    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
    moran_var = (n * n * s1 - n * s2 + 3 * s02) / ((n - 1) * (n + 1) * s02) - (1.0 / (n - 1)) ** 2
    geary_var = ((2 * s1 + s2) * (n - 1) - 4 * s02) / (2 * (n + 1) * s02)

    expected = moran_var if mode == "moran" else geary_var
    np.testing.assert_allclose(var_norm, expected, rtol=1e-10)
    if mode == "geary":
        # the two formulas differ here, so the test would fail if Moran's were reused
        assert not np.isclose(geary_var, moran_var, rtol=1e-3)


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


def test_co_occurrence_reproducibility(adata: AnnData):
    """Check co_occurrence reproducibility results."""
    arr_1, interval_1 = co_occurrence(adata, cluster_key="leiden", copy=True)
    arr_2, interval_2 = co_occurrence(adata, cluster_key="leiden", copy=True)

    np.testing.assert_array_equal(sorted(interval_1), sorted(interval_2))
    np.testing.assert_allclose(arr_1, arr_2)


def test_co_occurrence_missing_labels_raise(adata: AnnData):
    # a missing label has code -1, which the kernel would count as a phantom cluster
    adata.obs["leiden"] = adata.obs["leiden"].copy()
    adata.obs.loc[adata.obs_names[:3], "leiden"] = np.nan
    with pytest.raises(ValueError, match=r"contains missing values"):
        co_occurrence(adata, cluster_key="leiden", copy=True)


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


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_score_perms_matches_scanpy_per_permutation(mode: str):
    """The kernel reuses per-row sums across permutations; pin it to the naive scanpy reference.

    ``_score_perms`` accumulates each row of the graph once and reindexes those sums per
    permutation. The reference below is the definition it replaced: build ``g[perm, :]`` and hand it
    to :mod:`scanpy` for every permutation. Agreement has to hold to floating-point reordering only.
    """
    from scanpy.metrics import gearys_c, morans_i
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import normalize

    from squidpy._constants._constants import SpatialAutocorr

    n, n_genes, n_perms = 300, 4, 6
    rng = np.random.default_rng(0)
    g = csr_matrix(kneighbors_graph(rng.random((n, 2)), 5, mode="connectivity"))
    normalize(g, norm="l1", axis=1, copy=False)
    vals = rng.random((n_genes, n), dtype=np.float32)

    autocorr = SpatialAutocorr(mode)
    moran = autocorr == SpatialAutocorr.MORAN
    observed, got = _score_perms(g, vals, mode=autocorr, n_perms=n_perms, rng=0, n_jobs=1, show_progress_bar=False)

    func = morans_i if autocorr == SpatialAutocorr.MORAN else gearys_c
    expected = np.stack([func(g[gen.permutation(n), :], vals) for gen in np.random.default_rng(0).spawn(n_perms)])
    np.testing.assert_allclose(observed, func(g, vals), rtol=1e-9)

    # `_score_perms` only keeps reductions of the permutation scores, so the per-permutation
    # comparison happens one level down, against the kernel that still returns them.
    gg = g.astype(np.float64, copy=False)
    gt = gg.T.tocsr()
    perms = np.stack([gen.permutation(n).astype(np.int32) for gen in np.random.default_rng(0).spawn(n_perms)])
    sims = np.stack(
        [
            _autocorr_perms(
                gt.indptr,
                gt.indices,
                gt.data,
                np.asarray(gg.sum(axis=1)).ravel(),
                np.asarray(gg.sum(axis=0)).ravel(),
                np.arange(n, dtype=np.int32),
                np.ascontiguousarray(vals[m], np.float64),
                gg.data.sum(),
                perms,
                moran,
            )
            for m in range(n_genes)
        ]
    ).T
    assert sims.shape == (n_perms, n_genes)
    np.testing.assert_allclose(sims, expected, rtol=1e-9)

    # and the accumulators must be those same scores, reduced
    assert got["n_perms"] == n_perms
    np.testing.assert_array_equal(got["count_ge"], (sims >= observed).sum(axis=0))
    np.testing.assert_allclose(got["mean"], expected.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(got["var"], expected.var(axis=0), rtol=1e-9)


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_score_perms_thread_invariant(mode: str):
    """Permutation scores must not depend on `n_jobs`, which sizes the thread pool."""
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import normalize

    from squidpy._constants._constants import SpatialAutocorr

    n, n_perms = 300, 8
    rng = np.random.default_rng(0)
    g = csr_matrix(kneighbors_graph(rng.random((n, 2)), 5, mode="connectivity"))
    normalize(g, norm="l1", axis=1, copy=False)
    vals = rng.random((3, n), dtype=np.float32)

    autocorr = SpatialAutocorr(mode)
    serial = _score_perms(g, vals, mode=autocorr, n_perms=n_perms, rng=0, n_jobs=1, show_progress_bar=False)
    threaded = _score_perms(g, vals, mode=autocorr, n_perms=n_perms, rng=0, n_jobs=4, show_progress_bar=False)
    np.testing.assert_array_equal(serial[0], threaded[0])
    assert serial[1].keys() == threaded[1].keys()
    for key in serial[1]:
        np.testing.assert_array_equal(serial[1][key], threaded[1][key], err_msg=key)


def test_spatial_autocorr_backend_deprecated(dummy_adata: AnnData):
    """``backend`` no longer selects a process pool; it warns and is ignored."""
    with pytest.warns(FutureWarning, match=r"`backend`.*deprecated"):
        with_backend = spatial_autocorr(
            dummy_adata, copy=True, n_perms=10, rng=0, backend="loky", show_progress_bar=False
        )
    without = spatial_autocorr(dummy_adata, copy=True, n_perms=10, rng=0, show_progress_bar=False)
    assert_frame_equal(with_backend, without)
