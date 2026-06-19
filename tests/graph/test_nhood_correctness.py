"""Correctness tests for :func:`squidpy.gr.nhood_enrichment`.

These tests pin down the *numerical* behaviour of neighborhood enrichment, as
opposed to the shape/dtype smoke tests in ``test_nhood.py``.

The reference implementations below are written from scratch with plain Python
loops and share no code with the numba kernels in ``squidpy.gr._nhood``. Their
only purpose is to be an independent specification: when the reference and the
real implementation agree, the numba codegen *and* the normalization math are
validated. They are deliberately slow and simple.

They also act as the regression guard for refactors of the permutation /
parallelization machinery: as long as the numpy ``RandomState`` ordering is
preserved, :func:`_reference_nhood_enrichment` reproduces the exact z-score for
``n_jobs=1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from squidpy._constants._pkg_constants import Key
from squidpy.gr import nhood_enrichment, spatial_neighbors
from squidpy.gr._utils import _shuffle_group

_CK = "leiden"


# --------------------------------------------------------------------------- #
# Independent reference implementations (plain Python, no numba)
# --------------------------------------------------------------------------- #
def _ref_count(adj: csr_matrix, int_clust: np.ndarray, n_cls: int) -> np.ndarray:
    """``count[a, b]`` = number of directed edges from a cluster-``a`` cell to a cluster-``b`` neighbor."""
    adj = adj.tocsr()
    count = np.zeros((n_cls, n_cls), dtype=np.uint32)
    for i in range(adj.shape[0]):
        a = int_clust[i]
        for j in adj.indices[adj.indptr[i] : adj.indptr[i + 1]]:
            count[a, int_clust[j]] += 1
    return count


def _ref_total(count: np.ndarray) -> np.ndarray:
    row_sums = count.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1
    return count / row_sums


def _ref_conditional(adj: csr_matrix, int_clust: np.ndarray, n_cls: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(count_normalized, conditional_ratio)`` for the conditional mode."""
    adj = adj.tocsr()
    count = _ref_count(adj, int_clust, n_cls)

    # per-cell, per-cluster neighbor counts -> boolean "has at least one neighbor of type b"
    has_neighbor = np.zeros((len(int_clust), n_cls), dtype=bool)
    for i in range(len(int_clust)):
        for j in adj.indices[adj.indptr[i] : adj.indptr[i + 1]]:
            has_neighbor[i, int_clust[j]] = True

    cond_counts = np.zeros((n_cls, n_cls), dtype=np.float64)
    conditional_ratio = np.full((n_cls, n_cls), np.nan, dtype=np.float64)
    for a in range(n_cls):
        a_cells = int_clust == a
        n_a = int(a_cells.sum())
        if n_a == 0:
            continue
        for b in range(n_cls):
            cond_counts[a, b] = has_neighbor[a_cells, b].sum()
        conditional_ratio[a, :] = cond_counts[a, :] / n_a

    safe = cond_counts.copy()
    safe[safe == 0] = 1.0
    return count / safe, conditional_ratio


def _ref_normalize(adj: csr_matrix, int_clust: np.ndarray, n_cls: int, normalization: str) -> np.ndarray:
    """``count_normalized`` for an arbitrary normalization mode."""
    count = _ref_count(adj, int_clust, n_cls)
    if normalization == "none":
        return count.astype(np.float64)
    if normalization == "total":
        return _ref_total(count)
    if normalization == "conditional":
        return _ref_conditional(adj, int_clust, n_cls)[0]
    raise ValueError(normalization)


def _chunk_perms(n_perms: int, n_jobs: int) -> list[list[int]]:
    """Split ``range(n_perms)`` into contiguous chunks the way ``parallelize`` does.

    ``parallelize`` uses ``n_split = n_jobs`` and ``step = ceil(n_perms / n_split)``,
    then keeps the non-empty contiguous slices. Each chunk is seeded by its first
    permutation index, so the chunk boundaries determine the RNG streams.
    """
    step = int(np.ceil(n_perms / n_jobs))
    chunks = [list(range(k * step, min((k + 1) * step, n_perms))) for k in range(int(np.ceil(n_perms / step)))]
    return [c for c in chunks if c]


def _reference_nhood_enrichment(
    adj: csr_matrix,
    int_clust: np.ndarray,
    n_cls: int,
    *,
    n_perms: int,
    seed: int,
    normalization: str,
    libraries: pd.Series | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Full z-score reference replicating the production seeding scheme for any ``n_jobs``.

    The permutation indices are split into ``n_jobs`` contiguous chunks. Each chunk
    gets its own ``RandomState(seed + chunk_start)`` and shuffles a private copy of
    the labels in place, once per permutation. The chunks are concatenated in order,
    exactly mirroring ``parallelize(..., extractor=np.vstack)``.
    """
    observed = _ref_normalize(adj, int_clust, n_cls, normalization)

    perms = np.empty((n_perms, n_cls, n_cls), dtype=np.float64)
    pos = 0
    for chunk in _chunk_perms(n_perms, n_jobs):
        rs = np.random.RandomState(None if seed is None else seed + chunk[0])
        shuffled = int_clust.copy()
        for _ in chunk:
            if libraries is not None:
                shuffled = _shuffle_group(shuffled, libraries, rs)
            else:
                rs.shuffle(shuffled)
            perms[pos] = _ref_normalize(adj, shuffled, n_cls, normalization)
            pos += 1

    std = perms.std(axis=0)
    std[std == 0] = np.nan
    return (observed - perms.mean(axis=0)) / std


# --------------------------------------------------------------------------- #
# Tiny, fully deterministic graph
# --------------------------------------------------------------------------- #
@pytest.fixture()
def adata_tiny() -> AnnData:
    """A 6-cell, 3-cluster graph with a hand-chosen, symmetric adjacency.

    Layout (clusters in brackets)::

        0[0] - 1[0] - 2[1] - 3[1] - 4[2] - 5[2]
        |_______________________________________|   (0-5 edge closes the ring)
                       |_______|                      (2-4 cross edge)
    """
    n = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (2, 4)]
    rows, cols = [], []
    for i, j in edges:
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float64)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))

    clusters = pd.Categorical.from_codes([0, 0, 1, 1, 2, 2], categories=["A", "B", "C"])
    adata = AnnData(
        np.zeros((n, n)),
        obs={_CK: clusters},
        obsp={"spatial_connectivities": adj},
    )
    return adata


def _int_clust(adata: AnnData) -> np.ndarray:
    return adata.obs[_CK].cat.codes.to_numpy()


# --------------------------------------------------------------------------- #
# Deterministic output: counts
# --------------------------------------------------------------------------- #
def test_counts_match_reference_tiny(adata_tiny: AnnData):
    adj = adata_tiny.obsp["spatial_connectivities"]
    int_clust = _int_clust(adata_tiny)
    n_cls = 3

    result = nhood_enrichment(adata_tiny, cluster_key=_CK, n_perms=20, seed=0, copy=True)

    expected = _ref_count(adj, int_clust, n_cls)
    np.testing.assert_array_equal(result.counts, expected)
    # the counts matrix is independent of the normalization mode
    assert result.counts.dtype == np.uint32


def test_counts_hardcoded_tiny(adata_tiny: AnnData):
    """A fully hand-computable expectation, so the reference itself can't drift silently."""
    result = nhood_enrichment(adata_tiny, cluster_key=_CK, n_perms=20, seed=0, copy=True)
    # cluster A = {0,1}, B = {2,3}, C = {4,5}
    # directed edges by (src cluster -> dst cluster), counting both directions of each undirected edge:
    #   A-A: 0-1            -> 2
    #   A-B: 1-2            -> 1 each way
    #   A-C: 5-0            -> 1 each way
    #   B-B: 2-3            -> 2
    #   B-C: 3-4, 2-4       -> 2 each way
    #   C-C: 4-5            -> 2
    expected = np.array(
        [
            [2, 1, 1],
            [1, 2, 2],
            [1, 2, 2],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(result.counts, expected)


@pytest.mark.parametrize("normalization", ["none", "total", "conditional"])
def test_counts_invariant_to_normalization(adata_tiny: AnnData, normalization: str):
    """Raw ``counts`` must be the observed edge counts regardless of normalization."""
    adj = adata_tiny.obsp["spatial_connectivities"]
    expected = _ref_count(adj, _int_clust(adata_tiny), 3)
    result = nhood_enrichment(adata_tiny, cluster_key=_CK, normalization=normalization, n_perms=20, seed=0, copy=True)
    np.testing.assert_array_equal(result.counts, expected)


# --------------------------------------------------------------------------- #
# Deterministic output: conditional_ratio
# --------------------------------------------------------------------------- #
def test_conditional_ratio_matches_reference_tiny(adata_tiny: AnnData):
    adj = adata_tiny.obsp["spatial_connectivities"]
    int_clust = _int_clust(adata_tiny)
    _, expected_ratio = _ref_conditional(adj, int_clust, 3)

    result = nhood_enrichment(adata_tiny, cluster_key=_CK, normalization="conditional", n_perms=20, seed=0, copy=True)
    np.testing.assert_allclose(result.conditional_ratio, expected_ratio)


def test_conditional_ratio_is_a_fraction(adata_tiny: AnnData):
    """Conditional ratios are fractions of cells, so they live in ``[0, 1]``."""
    result = nhood_enrichment(adata_tiny, cluster_key=_CK, normalization="conditional", n_perms=20, seed=0, copy=True)
    ratio = result.conditional_ratio
    assert np.all((ratio >= 0) & (ratio <= 1))


def test_conditional_ratio_none_for_other_modes(adata_tiny: AnnData):
    for normalization in ("none", "total"):
        result = nhood_enrichment(
            adata_tiny, cluster_key=_CK, normalization=normalization, n_perms=20, seed=0, copy=True
        )
        assert result.conditional_ratio is None


# --------------------------------------------------------------------------- #
# Full z-score: independent reference for the n_jobs=1 path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_jobs", [1, 2, 3])
@pytest.mark.parametrize("normalization", ["none", "total", "conditional"])
def test_zscore_matches_reference_tiny(adata_tiny: AnnData, normalization: str, n_jobs: int):
    adj = adata_tiny.obsp["spatial_connectivities"]
    int_clust = _int_clust(adata_tiny)
    seed, n_perms = 0, 50

    result = nhood_enrichment(
        adata_tiny,
        cluster_key=_CK,
        normalization=normalization,
        n_perms=n_perms,
        seed=seed,
        n_jobs=n_jobs,
        copy=True,
    )
    expected = _reference_nhood_enrichment(
        adj, int_clust, 3, n_perms=n_perms, seed=seed, normalization=normalization, n_jobs=n_jobs
    )
    np.testing.assert_allclose(result.zscore, expected, equal_nan=True)


@pytest.mark.parametrize("n_jobs", [1, 2, 3])
def test_zscore_reference_holds_on_real_data(adata: AnnData, n_jobs: int):
    """The same independent reference must reproduce z-scores on a realistic graph, for every n_jobs.

    This locks the chunked ``seed + chunk_start`` seeding scheme: a refactor that
    preserves the exact permutation stream (the stated goal) keeps this green.
    """
    spatial_neighbors(adata)
    adj = adata.obsp["spatial_connectivities"]
    int_clust = adata.obs[_CK].cat.codes.to_numpy()
    n_cls = adata.obs[_CK].cat.categories.shape[0]
    seed, n_perms = 7, 30

    result = nhood_enrichment(adata, cluster_key=_CK, n_perms=n_perms, seed=seed, n_jobs=n_jobs, copy=True)
    expected = _reference_nhood_enrichment(
        adj, int_clust, n_cls, n_perms=n_perms, seed=seed, normalization="none", n_jobs=n_jobs
    )
    np.testing.assert_allclose(result.zscore, expected, equal_nan=True)


# --------------------------------------------------------------------------- #
# Statistical sanity: structure that survives RNG changes
# --------------------------------------------------------------------------- #
def test_self_enrichment_positive_for_clustered_graph():
    """Cells wired preferentially to their own cluster must show positive diagonal z-scores."""
    rng = np.random.RandomState(0)
    n_per = 40
    n_cls = 3
    n = n_per * n_cls
    labels = np.repeat(np.arange(n_cls), n_per)

    rows, cols = [], []
    for c in range(n_cls):
        members = np.where(labels == c)[0]
        # connect each cell to several same-cluster neighbors -> strong self-enrichment
        for i in members:
            for j in rng.choice(members[members != i], size=4, replace=False):
                rows += [i, j]
                cols += [j, i]
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    adata = AnnData(
        np.zeros((n, 1)),
        obs={_CK: pd.Categorical.from_codes(labels, categories=list("ABC"))},
        obsp={"spatial_connectivities": adj},
    )
    result = nhood_enrichment(adata, cluster_key=_CK, n_perms=200, seed=0, n_jobs=1, copy=True)

    diag = np.diag(result.zscore)
    off_diag = result.zscore[~np.eye(n_cls, dtype=bool)]
    assert np.all(diag > 0), diag
    assert np.all(diag > off_diag.max()), (diag, off_diag)
