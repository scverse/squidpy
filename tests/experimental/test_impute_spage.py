"""Tests for sq.experimental.tl.impute_spage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_array, csr_matrix
from sklearn.neighbors import NearestNeighbors
from spatialdata import SpatialData
from spatialdata.models import TableModel

import squidpy as sq
from squidpy.experimental.tl._impute_spage import _cosine_neighbors, _spage_weights

_GENES = [f"g{i}" for i in range(12)]
_N_MEASURED = 8  # the query measures g0-g7, g8-g11 are imputed
_UNMEASURED = _GENES[_N_MEASURED:]


def _data() -> tuple[AnnData, AnnData, np.ndarray]:
    """Reference and query from one rank-3 model, and the true query values of the unmeasured genes."""
    rng = np.random.default_rng(0)
    loadings = rng.normal(size=(3, len(_GENES)))

    def sample(n: int) -> np.ndarray:
        x = rng.normal(size=(n, 3)) @ loadings + 0.1 * rng.normal(size=(n, len(_GENES)))
        return np.maximum(x, 0)  # non-negative with zeros, like expression

    query_x = sample(150)
    ref = AnnData(sample(300), var=pd.DataFrame(index=_GENES))
    query = AnnData(query_x[:, :_N_MEASURED], var=pd.DataFrame(index=_GENES[:_N_MEASURED]))
    return ref, query, query_x[:, _N_MEASURED:]


def _impute(data_ref, data_query, **kwargs) -> pd.DataFrame | None:
    return sq.experimental.tl.impute_spage(
        data_ref, data_query, **({"genes": _UNMEASURED, "n_pv": 3, "inplace": False} | kwargs)
    )


@pytest.mark.parametrize("fmt", [csr_matrix, csr_array])
def test_sparse_matches_dense(fmt):
    ref, query, _ = _data()
    dense = _impute(ref, query)

    ref.X, query.X = fmt(ref.X), fmt(query.X)
    sparse = _impute(ref, query)

    pd.testing.assert_frame_equal(sparse, dense, rtol=1e-6)


def test_recovers_unmeasured_genes():
    # guards against a bug shared by the implementation and `_spage_reference`
    ref, query, truth = _data()

    res = _impute(ref, query, n_neighbors=20)

    # cosine distance ignores a cell's magnitude, so even published SpaGE only reaches 0.65-0.80 here;
    # misaligned genes or neighbors would correlate near 0
    for imputed, expected in zip(res.to_numpy().T, truth.T, strict=True):
        assert np.corrcoef(imputed, expected)[0, 1] > 0.6


def test_stores_in_obsm():
    ref, query, _ = _data()
    expected = _impute(ref, query)
    assert "imputed" not in query.obsm

    assert _impute(ref, query, key_added="imputed", inplace=True) is None

    pd.testing.assert_frame_equal(query.obsm["imputed"], expected)
    assert expected.index.equals(query.obs_names)
    assert expected.columns.tolist() == _UNMEASURED


@pytest.mark.parametrize(("ref_in_sdata", "query_in_sdata"), [(True, True), (False, True), (True, False)])
def test_spatialdata(ref_in_sdata, query_in_sdata):
    ref, query, _ = _data()
    expected = _impute(ref, query)
    data_ref = SpatialData(tables={"t": TableModel.parse(ref)}) if ref_in_sdata else ref
    data_query = SpatialData(tables={"t": TableModel.parse(query)}) if query_in_sdata else query

    table_key = ("t" if ref_in_sdata else None, "t" if query_in_sdata else None)
    _impute(data_ref, data_query, table_key=table_key, inplace=True)

    stored = (data_query.tables["t"] if query_in_sdata else query).obsm["spage"]
    pd.testing.assert_frame_equal(stored, expected)


def test_single_spatialdata():
    ref, query, _ = _data()
    expected = _impute(ref, query)
    sdata = SpatialData(tables={"ref": TableModel.parse(ref), "query": TableModel.parse(query)})

    _impute(sdata, None, table_key=("ref", "query"), inplace=True)

    pd.testing.assert_frame_equal(sdata.tables["query"].obsm["spage"], expected)


@pytest.mark.parametrize(("genes", "columns"), [(["g10", "g0"], ["g10", "g0"]), ("g8", ["g8"])])
def test_genes(genes, columns):
    ref, query, _ = _data()
    default = _impute(ref, query)

    res = _impute(ref, query, genes=genes)

    assert res.columns.tolist() == columns
    pd.testing.assert_series_equal(res[columns[0]], default[columns[0]])


def test_gene_order_does_not_matter():
    ref, query, _ = _data()

    res = _impute(ref, query[:, ::-1])

    pd.testing.assert_frame_equal(res, _impute(ref, query), rtol=1e-6)


@pytest.mark.parametrize("fmt", [np.asarray, csr_array])
def test_layer(fmt):
    ref, query, _ = _data()
    expected = _impute(ref, query)
    for adata in (ref, query):
        adata.layers["norm"] = fmt(adata.X)
        adata.X = np.ones_like(adata.X)

    res = _impute(ref, query, layer="norm")

    pd.testing.assert_frame_equal(res, expected, rtol=1e-6)


def test_layer_pair():
    ref, query, _ = _data()
    expected = _impute(ref, query)
    ref.layers["norm"] = ref.X
    ref.X = np.ones_like(ref.X)

    res = _impute(ref, query, layer=("norm", None))

    pd.testing.assert_frame_equal(res, expected)


@pytest.mark.parametrize("constant_in", ["ref", "query"])
def test_constant_genes_are_ignored(constant_in):
    ref, query, _ = _data()
    (ref if constant_in == "ref" else query).X[:, 0] = 1
    # without g0 in the query, it is not shared
    expected = _impute(ref, query[:, 1:])

    res = _impute(ref, query)

    pd.testing.assert_frame_equal(res, expected)


def test_float32():
    ref, query, _ = _data()
    expected = _impute(ref, query, n_neighbors=300)
    ref.X, query.X = ref.X.astype(np.float32), query.X.astype(np.float32)

    res = _impute(ref, query, n_neighbors=300)

    assert (res.dtypes == np.float32).all()
    pd.testing.assert_frame_equal(res, expected, check_dtype=False, rtol=1e-4)


def test_integer_reference_gives_float64():
    ref, query, _ = _data()
    ref.X = (10 * ref.X).astype(np.int64)

    res = _impute(ref, query)

    assert (res.dtypes == np.float64).all()
    ref.X = ref.X.astype(np.float64)
    pd.testing.assert_frame_equal(res, _impute(ref, query))


def test_n_jobs_invariance():
    ref, query, _ = _data()
    # drop reference cells with duplicate shared-gene expression, whose ties may be broken by the thread count
    _, first = np.unique(ref[:, :_N_MEASURED].X, axis=0, return_index=True)
    ref = ref[np.sort(first)]
    serial = _impute(ref, query, n_jobs=1)

    parallel = _impute(ref, query, n_jobs=-1)

    pd.testing.assert_frame_equal(parallel, serial)


def test_n_pv_bound_counts_varying_genes():
    ref, query, _ = _data()
    _impute(ref, query, n_pv=_N_MEASURED)

    query.X[:, 0] = 1
    with pytest.raises(ValueError, match="n_pv"):
        _impute(ref, query, n_pv=_N_MEASURED)


def test_cosine_neighbors_matches_sklearn_cosine():
    rng = np.random.default_rng(0)
    ref, query = rng.normal(size=(50, 3)), rng.normal(size=(20, 3))
    ref[7] = query[3] = 0
    knn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(ref)
    expected_distances, expected_indices = knn.kneighbors(query)

    distances, indices = _cosine_neighbors(ref, query, n_neighbors=10, n_jobs=1)

    np.testing.assert_allclose(distances, expected_distances, atol=1e-12)
    assert (distances[3] == 1).all()
    others = np.arange(len(query)) != 3  # the zero query row's neighbors are all at distance 1, in any order
    np.testing.assert_array_equal(np.sort(indices[others]), np.sort(expected_indices[others]))
    assert 7 not in indices[others]


@pytest.mark.parametrize(
    ("distances", "expected"),
    [
        pytest.param([[0.2, 0.3, 1.5]], [[0.6, 0.4, 0.0]], id="drops-distance-above-1"),
        pytest.param([[0.1, 0.2, 0.3]], [[5 / 12, 1 / 3, 1 / 4]], id="three-kept"),
        pytest.param([[0.5, 1.0]], [[1.0, 0.0]], id="single-kept"),
        pytest.param([[0.0, 0.0, 0.0]], [[1 / 3, 1 / 3, 1 / 3]], id="all-zero"),
        pytest.param([[1.0, 1.2]], [[0.0, 0.0]], id="none-kept"),
        pytest.param([[0.2, 0.3, 1.5], [0.5, 1.0, 1.0]], [[0.6, 0.4, 0.0], [1.0, 0.0, 0.0]], id="rows-independent"),
    ],
)
def test_spage_weights(distances, expected):
    np.testing.assert_allclose(_spage_weights(np.asarray(distances)), expected)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"genes": ["g8", "foo"]}, KeyError, r"\['foo'\]", id="unknown-gene"),
        pytest.param({"genes": []}, ValueError, "No genes", id="no-genes"),
        pytest.param({"n_pv": 1}, ValueError, "n_pv", id="n_pv-too-small"),
        pytest.param({"n_pv": _N_MEASURED + 1}, ValueError, "n_pv", id="n_pv-too-large"),
        pytest.param({"n_neighbors": 0}, ValueError, "n_neighbors", id="n_neighbors-zero"),
        pytest.param({"n_neighbors": 301}, ValueError, "n_neighbors", id="n_neighbors-too-large"),
        pytest.param({"cosine_threshold": 1.0}, ValueError, "cosine_threshold", id="cosine_threshold-range"),
        pytest.param({"cosine_threshold": 0.9999}, ValueError, "found `0`", id="no-principal-vector"),
        pytest.param({"cosine_threshold": 0.999}, ValueError, "found `1`", id="single-principal-vector"),
        pytest.param({"layer": "foo"}, KeyError, "Layer `foo` not found", id="unknown-layer"),
        pytest.param({"layer": (None, "foo")}, KeyError, "Layer `foo` not found", id="unknown-query-layer"),
        pytest.param({"layer": ("a", "b", "c")}, ValueError, "pair", id="layer-not-a-pair"),
    ],
)
def test_invalid_input(kwargs, error, match):
    ref, query, _ = _data()

    with pytest.raises(error, match=match):
        _impute(ref, query, **kwargs)


def test_no_shared_genes():
    ref, query, _ = _data()
    query.var_names = [f"other{i}" for i in range(query.n_vars)]

    with pytest.raises(ValueError, match="no genes in common"):
        _impute(ref, query)


def test_query_required_for_anndata():
    ref, _, _ = _data()

    with pytest.raises(ValueError, match="`data_query` is required"):
        _impute(ref, None)


def test_single_spatialdata_needs_table_key_pair():
    ref, query, _ = _data()
    sdata = SpatialData(tables={"ref": TableModel.parse(ref), "query": TableModel.parse(query)})

    with pytest.raises(ValueError, match=r"table_key=\(ref, query\)"):
        _impute(sdata, None, table_key="ref")


@pytest.mark.parametrize(
    ("table_key", "error", "match"),
    [
        pytest.param(None, ValueError, "pass `table_key`", id="sdata-without-table-key"),
        pytest.param("t", ValueError, "the reference is an AnnData", id="table-key-for-anndata"),
        pytest.param((None, "foo"), KeyError, "no such table in the query", id="unknown-table"),
    ],
)
def test_invalid_table(table_key, error, match):
    ref, query, _ = _data()
    data_query = SpatialData(tables={"t": TableModel.parse(query)})

    with pytest.raises(error, match=match):
        _impute(ref, data_query, table_key=table_key)


def test_reference_and_query_must_differ():
    ref, _, _ = _data()

    with pytest.raises(ValueError, match="same table"):
        _impute(ref, ref)
