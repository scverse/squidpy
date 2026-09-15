from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from fast_array_utils.conv import to_dense
from pandas import Series
from scanpy.pp import neighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from spatialdata import SpatialData
from spatialdata.models import TableModel

from squidpy.gr import (
    _niche,
    calculate_niche,
    calculate_niche_cellcharter,
    calculate_niche_neighborhood,
    calculate_niche_utag,
    spatial_neighbors_knn,
)
from squidpy.gr._nhood import _aggregate_over, nhood_aggregate
from squidpy.gr._niche import _fit_clusterers, _precomputed_embedding, compute_hop_adjacency_matrices

N_NEIGHBORS = 20

# Niche labels come from Leiden clustering, whose exact partition is not stable across
# igraph/leidenalg versions on tiny toy graphs (multiple equal-modularity optima). These
# tests therefore assert squidpy's behavioural contract - every cell is assigned, the
# postprocessors and library stratification behave, and a fixed seed is reproducible -
# rather than a specific (arbitrary) partition. See scverse/squidpy#1260.


def _assert_all_assigned(adata: AnnData, column: str) -> Series:
    """Every observation receives a niche label under ``column``."""
    assert column in adata.obs.columns
    niches = adata.obs[column]
    assert len(niches) == adata.n_obs
    assert niches.notna().all()
    return niches


def test_niche_calc_nhood_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using neighborhood profile approach works as intended for dummy_adata2."
    rerun = dummy_adata2.copy()
    calculate_niche(dummy_adata2, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.0, rng=0)
    niches = _assert_all_assigned(dummy_adata2, "nhood_niche_res=1.0")

    # a fixed rng gives reproducible niches
    calculate_niche(rerun, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.0, rng=0)
    assert (niches.to_numpy() == rerun.obs["nhood_niche_res=1.0"].to_numpy()).all()


def test_niche_calc_utag_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using utag approach works as intended for dummy_adata2."
    rerun = dummy_adata2.copy()
    calculate_niche(dummy_adata2, flavor="utag", n_neighbors=3, resolutions=1.0, rng=0)
    niches = _assert_all_assigned(dummy_adata2, "utag_niche_res=1.0")

    # a fixed rng gives reproducible niches
    calculate_niche(rerun, flavor="utag", n_neighbors=3, resolutions=1.0, rng=0)
    assert (niches.to_numpy() == rerun.obs["utag_niche_res=1.0"].to_numpy()).all()


def test_niche_calc_cellcharter_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using cellcharter approach works as intended for dummy_adata2."

    calculate_niche(dummy_adata2, flavor="cellcharter", distance=2, aggregation="mean", rng=np.random.default_rng(0))

    _assert_all_assigned(dummy_adata2, "cellcharter_niche")


def test_niche_calc_spatialleiden_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using spatialleiden approach works as intended for dummy_adata2."
    pytest.importorskip("spatialleiden")

    # need the latent_connectivities_key, meaning have to run the graph construction
    neighbors(dummy_adata2, n_neighbors=3, use_rep="X")

    calculate_niche(
        dummy_adata2,
        flavor="spatialleiden",
        latent_connectivities_key="connectivities",
        spatial_connectivities_key="spatial_connectivities",
        resolutions=1.0,
        rng=np.random.default_rng(0),
    )

    _assert_all_assigned(dummy_adata2, "spatialleiden_res=1.0")


# rng handling


def test_niche_cellcharter_rng_reproducible(dummy_adata2: AnnData):
    "The same `rng` must give the same niches, a different one must be free to differ."
    kwargs = {"distance": 2, "aggregation": "mean"}

    first = calculate_niche_cellcharter(dummy_adata2, rng=np.random.default_rng(0), copy=True, **kwargs)
    second = calculate_niche_cellcharter(dummy_adata2, rng=np.random.default_rng(0), copy=True, **kwargs)
    assert (first.obs["cellcharter_niche"] == second.obs["cellcharter_niche"]).all()

    # not a guarantee about the labels themselves, only that the seed is actually wired through
    other = calculate_niche_cellcharter(dummy_adata2, rng=np.random.default_rng(1), copy=True, **kwargs)
    assert list(other.obs["cellcharter_niche"]) != list(first.obs["cellcharter_niche"])


def test_niche_cellcharter_rng_none_runs(dummy_adata2: AnnData):
    "`rng=None` (the default) must work: it means 'draw from OS entropy', not 'missing argument'."
    calculate_niche_cellcharter(dummy_adata2, distance=2, aggregation="mean")
    assert "cellcharter_niche" in dummy_adata2.obs.columns


# more special test cases


def test_niche_calc_library_key_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation when library_key is supplied works as intended for dummy_adata2."

    # add library_key information in dummy_adata
    dummy_adata2.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5

    calculate_niche(
        dummy_adata2, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.5, library_key="batch"
    )

    niches = _assert_all_assigned(dummy_adata2, "nhood_niche_res=1.5")
    # niches are computed per library and prefixed with the originating library
    for cell, label in niches.items():
        assert label.startswith(f"lib={dummy_adata2.obs['batch'][cell]}_")


def test_niche_calc_spatialleiden_library_key_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation for spatialleiden works as intended for dummy_adata2 when library_key is supplied."
    pytest.importorskip("spatialleiden")

    # need the latent_connectivities_key, meaning have to run the graph construction
    neighbors(dummy_adata2, n_neighbors=3, use_rep="X")

    # add library_key information in dummy_adata
    dummy_adata2.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5

    calculate_niche(
        dummy_adata2,
        flavor="spatialleiden",
        latent_connectivities_key="connectivities",
        spatial_connectivities_key="spatial_connectivities",
        resolutions=1.0,
        library_key="batch",
        rng=np.random.default_rng(0),
    )

    niches = _assert_all_assigned(dummy_adata2, "spatialleiden_res=1.0")
    # niches are computed per library and prefixed with the originating library
    for cell, label in niches.items():
        assert label.startswith(f"lib={dummy_adata2.obs['batch'][cell]}_")


def test_niche_calc_nhood_multipostprocessor_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using neighborhood profile approach works as intended for dummy_adata2, when using both, mask and min_niche_size postprocessors"
    mask = Series(
        [False, False, True, True, True, True, True, True, True, True],
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )
    calculate_niche(
        dummy_adata2,
        flavor="neighborhood",
        groups="celltype",
        n_neighbors=3,
        resolutions=1.0,
        mask=mask,
        min_niche_size=3,
    )
    niches = _assert_all_assigned(dummy_adata2, "nhood_niche_res=1.0")
    # masked-out observations are never assigned to a real niche
    assert (niches[["a", "b"]] == "not_a_niche").all()
    # every real niche respects the requested minimum size
    real = niches[niches != "not_a_niche"]
    assert (real.astype(str).value_counts() >= 3).all()


def test_niche_calc_nhood_dummy_sdata(dummy_adata2: AnnData):
    "Check whether niche calculation works as intended for the spatialdata version of dummy_adata2."

    # make adata into sdata object
    adata_for_sdata = TableModel.parse(dummy_adata2)
    sdata = SpatialData(
        # images={"hne": img_for_sdata},
        # shapes={"spots": shapes_for_sdata},
        tables={"adata": adata_for_sdata},
    )

    calculate_niche(sdata, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.0, table_key="adata")

    _assert_all_assigned(sdata["adata"], "nhood_niche_res_1.0")


# test cases for _compute_hop_adjacency_matrices


def test_niche_calc_nhood(adata_seqfish: AnnData):
    """Check whether niche calculation using neighborhood profile approach works as intended."""
    spatial_neighbors_knn(adata_seqfish, n_neighs=N_NEIGHBORS)
    calculate_niche(
        adata_seqfish,
        groups="celltype_mapped_refined",
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


def test_niche_calc_utag(adata_seqfish: AnnData):
    """Check whether niche calculation using UTAG approach works as intended."""
    spatial_neighbors_knn(adata_seqfish, n_neighs=N_NEIGHBORS)
    calculate_niche(adata_seqfish, flavor="utag", n_neighbors=N_NEIGHBORS, resolutions=[0.1, 1.0])

    niches = adata_seqfish.obs["utag_niche_res=1.0"]
    niches_low_res = adata_seqfish.obs["utag_niche_res=0.1"]

    assert niches.isna().sum() == 0
    assert niches.nunique() > niches_low_res.nunique()


def test_niche_copy_semantics(dummy_adata2: AnnData):
    "copy=True returns an annotated copy and leaves the input untouched; copy=False mutates and returns None."
    key = "nhood_niche_res=1.0"
    kwargs = {"groups": "celltype", "n_neighbors": 3, "resolutions": 1.0, "rng": 0}

    out = calculate_niche_neighborhood(dummy_adata2, copy=True, **kwargs)
    assert key in out.obs.columns
    assert key not in dummy_adata2.obs.columns

    assert calculate_niche_neighborhood(dummy_adata2, **kwargs) is None
    assert (dummy_adata2.obs[key] == out.obs[key]).all()


def test_niche_neighborhood_rejects_too_few_hop_weights(dummy_adata2: AnnData):
    with pytest.raises(ValueError, match=r"'n_hop_weights' has 1 value"):
        calculate_niche_neighborhood(
            dummy_adata2, groups="celltype", resolutions=1.0, n_neighbors=3, distance=3, n_hop_weights=[1.0]
        )


def test_calculate_niche_deprecation_is_a_future_warning(dummy_adata2: AnnData):
    with pytest.warns(FutureWarning, match=r"`calculate_niche` is deprecated"):
        calculate_niche(dummy_adata2, flavor="utag", n_neighbors=3, resolutions=1.0, rng=0)


@pytest.mark.parametrize("distance", [0, -3])
def test_cellcharter_rejects_a_distance_below_one(dummy_adata2: AnnData, distance: int):
    spatial_neighbors_knn(dummy_adata2, n_neighs=3)
    with pytest.raises(ValueError, match=r"'distance' must be >= 1"):
        calculate_niche_cellcharter(dummy_adata2, distance=distance, n_components=2, rng=0)


def test_neighborhood_profile_weights_by_path_count(dummy_adata2: AnnData):
    """Through the public call, with `scale=False` so the raw profile survives."""
    spatial_neighbors_knn(dummy_adata2, n_neighs=3)
    adj = dummy_adata2.obsp["spatial_connectivities"]
    one_hot = pd.get_dummies(dummy_adata2.obs["celltype"], dtype=np.float64).to_numpy()

    out = calculate_niche_neighborhood(
        dummy_adata2,
        groups="celltype",
        resolutions=1.0,
        n_neighbors=3,
        distance=3,
        scale=False,
        copy=True,
        rng=0,
    )
    got = np.asarray(out.obsm["niche_embedding"])

    expected, power = np.zeros_like(got), adj
    for hop in range(3):
        if hop:
            power = power @ adj
        profile = power @ one_hot
        total = profile.sum(axis=1)[:, None]
        expected += np.divide(profile, total, out=np.zeros_like(profile), where=total != 0)
    np.testing.assert_allclose(got, expected / 3, rtol=1e-6, atol=1e-7)


def _toarray(mat):
    """Densify a sparse (or already-dense) matrix for easy comparison in assertions."""
    return np.asarray(mat.todense()) if hasattr(mat, "todense") else np.asarray(mat)


_PATH3 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
_PATH3_RING2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
_TRIANGLE = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
_CYCLE4 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=float)
_SELF_LOOPED = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
_HUB_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (4, 5)]


def _stored_zero_path3():
    """`_PATH3` carrying an explicit 0.0 at (0, 2); a user-supplied `obsp` may have those."""
    adj = csr_matrix(_PATH3)
    adj[0, 2] = adj[2, 0] = 0.0
    assert adj.nnz == 6, "the zeros must be stored for this case to test anything"
    return adj


def _hub_graph(weight=1.0):
    adjacency = np.zeros((6, 6))
    for i, j in _HUB_EDGES:
        adjacency[i, j] = adjacency[j, i] = weight
    return csr_matrix(adjacency)


HOP_RING_CASES = [
    ("path, second ring is the far pair", _PATH3, 2, [_PATH3, _PATH3_RING2]),
    ("path, nothing past the diameter", _PATH3, 3, [_PATH3, _PATH3_RING2, np.zeros((3, 3))]),
    ("stored zeros are not edges", _stored_zero_path3(), 2, [_PATH3, _PATH3_RING2]),
    ("triangle has no second ring", _TRIANGLE, 2, [_TRIANGLE, np.zeros((3, 3))]),
    ("hop 1 keeps a self-loop", _SELF_LOOPED, 1, [_SELF_LOOPED]),
    ("self-loops never propagate", np.eye(3), 3, [np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))]),
    ("an isolated node stays isolated", np.zeros((1, 1)), 3, [np.zeros((1, 1))] * 3),
    ("rings are binary, not path counts", _CYCLE4, 2, [_CYCLE4, np.roll(np.eye(4), 2, axis=1)]),
]


@pytest.mark.parametrize(
    ("adjacency", "max_hop", "expected"),
    [case[1:] for case in HOP_RING_CASES],
    ids=[case[0] for case in HOP_RING_CASES],
)
def test_hop_rings(adjacency, max_hop, expected):
    rings = compute_hop_adjacency_matrices(csr_matrix(adjacency), max_hop=max_hop)
    assert len(rings) == max_hop
    for hop, want in enumerate(expected):
        np.testing.assert_array_equal(
            _toarray(rings[hop]).astype(float), np.asarray(want, dtype=float), err_msg=f"hop {hop + 1}"
        )


@pytest.mark.parametrize("weight", [1.0, 0.5])
def test_hop_rings_are_boolean_and_disjoint_on_a_weighted_graph(weight):
    rings = compute_hop_adjacency_matrices(_hub_graph(weight), max_hop=2)
    assert rings[0].dtype == bool, "the graph is cast to bool, as CellCharter does"
    assert not rings[1][0, 1], "a one-hop neighbour reappeared in the two-hop ring"
    assert rings[1][0, 5], "a genuinely new two-hop pair went missing"


def test_hop_rings_are_not_matrix_powers():
    """Powers count every 2-walk with multiplicity; rings hold only newly reached pairs."""
    adjacency = _hub_graph()
    rings = compute_hop_adjacency_matrices(adjacency, max_hop=2)
    powers = [adjacency, adjacency @ adjacency]

    assert (rings[0] != powers[0]).nnz == 0, "hop 1 is the graph itself either way"
    assert powers[1][0, 1] == 3, "0 and 1 are joined by three 2-step paths"
    assert not rings[1][0, 1], "0 and 1 were already joined at one hop"


# ---------------------------------------------------------------- guards
# One test per guard added in response to review. A mutation run showed the suite caught none
# of them, so each of these fails if its guard is removed.


def _tiny(n: int = 40, libraries: list[str] | None = None) -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(X=csr_matrix(rng.random((n, 6)).astype(np.float32)))
    adata.obsm["spatial"] = rng.random((n, 2)) * 10
    adata.obs["ct"] = pd.Categorical([f"t{k}" for k in rng.integers(0, 3, n)])
    if libraries is not None:
        adata.obs["library"] = libraries
    spatial_neighbors_knn(adata, n_neighs=4)
    return adata


def test_hop_adjacency_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match=r"must be square"):
        compute_hop_adjacency_matrices(csr_matrix(np.ones((3, 5), dtype=float)), max_hop=2)


def test_hop_adjacency_accepts_an_array_like():
    "A plain nested list still works; the square check must not read `.shape` off it."
    rings = compute_hop_adjacency_matrices([[0, 1, 0], [1, 0, 1], [0, 1, 0]], max_hop=2)
    assert np.array_equal(_toarray(rings[1]).astype(float), _PATH3_RING2)


@pytest.mark.parametrize("hops", [(0,), (1,)])
def test_nhood_aggregate_rejects_an_unknown_aggregation(hops):
    "hop 0 returns the features unaggregated, so it never reaches the per-hop check."
    adata = _tiny()
    with pytest.raises(ValueError, match=r"'aggregation' must be"):
        nhood_aggregate(adata, hops=hops, aggregation="median")


def test_nhood_aggregate_rejects_a_weight_count_mismatch():
    adata = _tiny()
    with pytest.raises(ValueError, match=r"'hop_weights' has 1 value"):
        nhood_aggregate(adata, hops=(1, 2), hop_weights=[1.0])


def test_nhood_aggregate_rejects_weights_summing_to_zero():
    "Averaging over a zero total used to return an all-NaN matrix silently."
    adata = _tiny()
    with pytest.raises(ValueError, match=r"must not sum to zero"):
        nhood_aggregate(adata, hops=(1, 2), hop_weights=[1.0, -1.0], aggregation="mean")
    # counts stay counts, so `sum` never divides and the weights may cancel
    assert nhood_aggregate(adata, hops=(1, 2), hop_weights=[1.0, -1.0], aggregation="sum") is not None


@pytest.mark.parametrize("key", [None, "", 5])
def test_niche_rejects_an_unusable_embedding_key(key):
    "`obsm[None]` is accepted by AnnData and only fails later, at write_h5ad."
    with pytest.raises(ValueError, match=r"'embedding_key_added' must be a non-empty string"):
        calculate_niche_cellcharter(_tiny(), distance=2, n_components=2, rng=0, embedding_key_added=key)


def test_niche_library_key_with_a_skipped_first_library_still_writes_labels():
    "An empty first library must not silently drop every later library's labels."
    adata = _tiny(n=40, libraries=[None] * 8 + ["a"] * 16 + ["b"] * 16)
    calculate_niche_neighborhood(adata, groups="ct", resolutions=1.0, n_neighbors=4, rng=0, library_key="library")
    col = "nhood_niche_res=1.0"
    assert col in adata.obs, "the labels of the non-empty libraries were dropped"
    assert str(adata.obs[col].dtype) == "category"
    assert (adata.obs[col][8:] != "not_a_niche").any()


def test_niche_library_key_with_no_usable_library_raises():
    adata = _tiny(n=20, libraries=[None] * 20)
    with pytest.raises(ValueError, match=r"no observation has a 'library'"):
        calculate_niche_neighborhood(adata, groups="ct", resolutions=1.0, n_neighbors=4, rng=0, library_key="library")


def test_niche_library_key_rerun_overwrites_labels():
    "A second in-place call must not keep the first run's labels."
    adata = _tiny(n=40, libraries=["a"] * 20 + ["b"] * 20)
    calculate_niche_cellcharter(adata, distance=2, n_components=2, rng=0, library_key="library")
    first = np.asarray(adata.obs["cellcharter_niche"].astype(str)).copy()
    calculate_niche_cellcharter(adata, distance=2, n_components=4, rng=99, library_key="library")
    second = np.asarray(adata.obs["cellcharter_niche"].astype(str))
    assert not (first == second).all(), "the re-run silently kept the previous labels"


def test_weighted_graph_warning_points_at_the_caller():
    "`stacklevel` must skip squidpy's own frames, including the `functools.partial` hop."
    adata = _tiny()
    spatial_neighbors_knn(adata, n_neighs=4, transform="spectral")
    with pytest.warns(UserWarning, match="non-binary") as caught:
        calculate_niche_cellcharter(adata, distance=2, n_components=2, rng=0)
    assert caught[0].filename == __file__, f"attributed to {caught[0].filename}"


def test_clusterer_without_a_random_state_is_rejected():
    "A deterministic estimator satisfies the protocol and then rejects the seed the pipeline sets."
    from sklearn.cluster import DBSCAN

    adata = _tiny()
    with pytest.raises(TypeError, match=r"no 'random_state'"):
        _fit_clusterers(adata, np.asarray(to_dense(adata.X)), {"c": DBSCAN(eps=3.0)}, np.random.default_rng(0))


def test_library_key_embeds_each_library_on_its_own(monkeypatch):
    "Stratifying exists to fit the embedding per library, not once on the pooled object."
    rng = np.random.default_rng(0)
    half = 60
    adata = AnnData(X=csr_matrix(rng.random((2 * half, 12)).astype(np.float32)))
    adata.obsm["spatial"] = np.vstack([rng.random((half, 2)) * 10, rng.random((half, 2)) * 10 + 100])
    adata.obs["ct"] = pd.Categorical(["a"] * 50 + ["b"] * 10 + ["a"] * 10 + ["b"] * 50)
    adata.obs["section"] = pd.Categorical(["s1"] * half + ["s2"] * half)
    spatial_neighbors_knn(adata, n_neighs=6, library_key="section")

    seen: list[int] = []
    original = _niche._nhood_profile_embedding

    def spy(adata_arg, **kwargs):
        seen.append(adata_arg.n_obs)
        return original(adata_arg, **kwargs)

    monkeypatch.setattr(_niche, "_nhood_profile_embedding", spy)
    calculate_niche_neighborhood(
        adata, groups="ct", resolutions=1.0, n_neighbors=10, rng=0, library_key="section", scale=True
    )
    # one fit per library, on that library's own observations; a pooled fit is a single 2 * half
    assert seen == [half, half], f"embedder was handed {seen} observations"


def test_library_key_writes_no_pooled_embedding():
    "Fitted per library, the blocks are in different spaces, so there is no one array to store."
    rng = np.random.default_rng(1)
    adata = AnnData(X=csr_matrix(rng.random((70, 20)).astype(np.float32)))
    adata.obsm["spatial"] = np.vstack([rng.random((15, 2)) * 10, rng.random((55, 2)) * 10 + 100])
    adata.obs["ct"] = pd.Categorical([f"t{k}" for k in rng.integers(0, 3, 70)])
    adata.obs["section"] = pd.Categorical(["s1"] * 15 + ["s2"] * 55)
    spatial_neighbors_knn(adata, n_neighs=4, library_key="section")

    calculate_niche_utag(adata, resolutions=1.0, n_neighbors=8, rng=0, library_key="section")
    assert "niche_embedding" not in adata.obsm
    assert "utag_niche_res=1.0" in adata.obs, "the labels must still be written"


def _with_embedding(cols: int, n: int = 60) -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(X=csr_matrix(rng.random((n, 10)).astype(np.float32)))
    adata.obsm["spatial"] = rng.random((n, 2)) * 12
    adata.obsm["emb"] = rng.random((n, cols))
    spatial_neighbors_knn(adata, n_neighs=5)
    return adata


def test_use_rep_is_truncated_to_n_components():
    "v1.8.3 and main both clustered only the first `n_components` columns of `use_rep`."
    adata = _with_embedding(cols=20)
    assert _precomputed_embedding(adata, obsm_key="emb", n_components=10).shape[1] == 10


def test_use_rep_narrower_than_n_components_is_rejected():
    adata = _with_embedding(cols=5)
    with pytest.raises(ValueError, match=r"Embedding has 5 components, but n_components=10"):
        calculate_niche_cellcharter(adata, use_rep="emb", n_components=10, rng=0)


# ---------------------------------------------------------------- oracles and scale
# Every case above runs on a handful of nodes, which is fewer than NUMBA_NUM_THREADS, so the
# BFS kernel's thread striping is a no-op there. These run past that.


def _random_graph(n: int, seed: int, *, directed: bool = False, self_loops: bool = False):
    rng = np.random.default_rng(seed)
    dense = (rng.random((n, n)) < 0.06).astype(float)
    np.fill_diagonal(dense, 1.0 if self_loops else 0.0)
    if not directed:
        dense = np.maximum(dense, dense.T)
        if not self_loops:
            np.fill_diagonal(dense, 0.0)
    return csr_matrix(dense)


@pytest.mark.parametrize("n", [50, 137, 200])
@pytest.mark.parametrize("directed", [False, True])
def test_hop_rings_match_dijkstra(n: int, directed: bool):
    """Ring k must hold exactly the pairs at shortest-path distance k + 1.

    An independent oracle, and at an `n` well past the thread count, so the kernel's
    `range(thread, n, n_threads)` striping is actually exercised.
    """
    adjacency = _random_graph(n, seed=n, directed=directed)
    max_hop = 4
    rings = compute_hop_adjacency_matrices(adjacency, max_hop=max_hop)

    distances = dijkstra(adjacency, directed=directed, unweighted=True)
    for hop in range(1, max_hop):  # ring 0 is the graph itself, self-loops included
        expected = (distances == hop + 1).astype(float)
        np.testing.assert_array_equal(
            _toarray(rings[hop]).astype(float), expected, err_msg=f"ring {hop} (hop {hop + 1})"
        )


@pytest.mark.parametrize("n_jobs", [1, 2, 3, 8])
def test_hop_rings_do_not_depend_on_the_thread_count(n_jobs: int):
    "The striping partitions sources across threads; the result must not depend on how."
    adjacency = _random_graph(157, seed=3)
    reference = compute_hop_adjacency_matrices(adjacency, max_hop=3, n_jobs=1)
    rings = compute_hop_adjacency_matrices(adjacency, max_hop=3, n_jobs=n_jobs)
    for hop, (got, want) in enumerate(zip(rings, reference, strict=True)):
        assert (got != want).nnz == 0, f"ring {hop} differs at n_jobs={n_jobs}"


# ---------------------------------------------------------------- cellcharter numerics


def test_aggregate_over_variance_matches_the_definition():
    "E[x^2] - E[x]^2 over each neighborhood; a sign slip here is invisible to the flavor tests."
    adjacency = csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=float))
    features = np.array([[1.0, 4.0], [3.0, 0.0], [5.0, 2.0]])

    got = to_dense(_aggregate_over(adjacency, features, "variance"))

    # row i averages over the neighbors adjacency[i] selects
    expected = np.empty_like(got)
    for i in range(3):
        neighbors = features[np.asarray(adjacency[i].todense()).ravel() > 0]
        expected[i] = neighbors.mean(axis=0) ** 2 * -1 + (neighbors**2).mean(axis=0)
    np.testing.assert_allclose(got, expected, rtol=1e-10)


@pytest.mark.parametrize("distance", [1, 2])
def test_cellcharter_concatenates_hop_zero_with_every_ring(monkeypatch, distance: int):
    "The embedding is the raw features plus one block per ring; dropping hop 0 must be visible."
    adata = _tiny(n=60)
    seen: list[int] = []
    original = sc.pp.pca

    def spy(matrix, *args, **kwargs):
        seen.append(matrix.shape[1])
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(sc.pp, "pca", spy)
    calculate_niche_cellcharter(adata, distance=distance, n_components=2, rng=0)
    assert seen == [(distance + 1) * adata.n_vars], f"PCA was handed {seen} columns"


@pytest.mark.parametrize("sparse", [True, False])
def test_cellcharter_keeps_the_container_through_the_embedding(sparse: bool):
    "A sparse X must reach PCA through `sparse_hstack`, not be densified on the way."
    rng = np.random.default_rng(0)
    X = rng.random((50, 6)).astype(np.float32)
    adata = AnnData(X=csr_matrix(X) if sparse else X)
    adata.obsm["spatial"] = rng.random((50, 2)) * 10
    spatial_neighbors_knn(adata, n_neighs=4)

    calculate_niche_cellcharter(adata, distance=2, n_components=3, rng=0)
    assert "cellcharter_niche" in adata.obs
    assert str(adata.obs["cellcharter_niche"].dtype) == "category"


def test_cellcharter_with_a_library_key():
    "The stratified GMM path had no test at all once the seeding test was removed."
    adata = _tiny(n=60, libraries=["s1"] * 30 + ["s2"] * 30)
    calculate_niche_cellcharter(adata, distance=2, n_components=2, rng=0, library_key="library")

    labels = adata.obs["cellcharter_niche"].astype(str)
    assert str(adata.obs["cellcharter_niche"].dtype) == "category"
    # every label carries its own library's prefix, and both libraries produced some
    assert {label.split("_")[0] for label in labels} == {"lib=s1", "lib=s2"}


@pytest.mark.parametrize("flavor_fn", [calculate_niche_utag, calculate_niche_neighborhood])
def test_resolutions_reject_a_pair_outside_spatialleiden(flavor_fn):
    "A (latent, spatial) pair used to reach scanpy as `must be real number, not tuple`."
    adata = _tiny()
    kwargs = {"groups": "ct"} if flavor_fn is calculate_niche_neighborhood else {}
    with pytest.raises(TypeError, match=r"only the 'spatialleiden' flavor takes"):
        flavor_fn(adata, resolutions=(0.5, 1.0), n_neighbors=4, rng=0, **kwargs)


def test_resolutions_reject_repeated_values():
    "Repeats collided on the column name and silently produced one clustering, not two."
    adata = _tiny()
    with pytest.raises(ValueError, match=r"'resolutions' repeats 0.5"):
        calculate_niche_utag(adata, resolutions=[0.5, 0.5], n_neighbors=4, rng=0)


def test_resolutions_reject_an_empty_sequence():
    adata = _tiny()
    with pytest.raises(ValueError, match=r"'resolutions' is empty"):
        calculate_niche_utag(adata, resolutions=[], n_neighbors=4, rng=0)


def test_resolutions_accept_a_numpy_array():
    "Only `list` was unpacked, so an ndarray was clustered as a single resolution."
    adata = _tiny()
    calculate_niche_utag(adata, resolutions=np.array([0.5, 1.0]), n_neighbors=4, rng=0)
    assert {"utag_niche_res=0.5", "utag_niche_res=1.0"} <= set(adata.obs.columns)


def test_the_new_entry_points_validate_resolutions_too(dummy_adata2: AnnData):
    "The checks lived in `_validate_niche_args`, which only `calculate_niche` ever called."
    with pytest.raises(TypeError, match=r"'resolutions' must be numbers"):
        calculate_niche_utag(dummy_adata2, resolutions="high", n_neighbors=3, rng=0)
    with pytest.warns(FutureWarning), pytest.raises(TypeError, match=r"'resolutions' must be numbers"):
        calculate_niche(dummy_adata2, flavor="utag", resolutions="high", n_neighbors=3, rng=0)
