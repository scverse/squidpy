from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from fast_array_utils.conv import to_dense
from pandas import Series
from scanpy.pp import neighbors
from scipy.sparse import csr_matrix, identity, issparse
from scipy.sparse import hstack as sparse_hstack
from spatialdata import SpatialData
from spatialdata.models import TableModel

from squidpy.gr import (
    _niche,
    calculate_niche,
    calculate_niche_cellcharter,
    calculate_niche_neighborhood,
    spatial_neighbors_knn,
    spatial_neighbors_radius,
)
from squidpy.gr._nhood import (
    _compute_hop_adjacency_matrices,
    _nhood_blocks,
    _shell_adjacencies,
    nhood_aggregate,
)
from squidpy.gr._niche import _nhood_profile_embedding

N_NEIGHBORS = 20
GROUPS = "celltype_mapped_refined"

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


def test_niche_cellcharter_library_seeds_are_independent(dummy_adata2: AnnData, monkeypatch):
    "Each library must be fitted with its own seed, while the whole run stays reproducible."
    dummy_adata2.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5
    kwargs = {"distance": 2, "aggregation": "mean", "library_key": "batch", "n_components": 2}

    first = calculate_niche_cellcharter(dummy_adata2, rng=np.random.default_rng(0), copy=True, **kwargs)
    second = calculate_niche_cellcharter(dummy_adata2, rng=np.random.default_rng(0), copy=True, **kwargs)
    assert (first.obs["cellcharter_niche"] == second.obs["cellcharter_niche"]).all()

    # the clusterer is built once and reused for every library, so record what each fit
    # is actually seeded with
    seen: list[int] = []
    original = _niche.GaussianMixture

    def spy(*args, **kwargs):
        seen.append(kwargs["random_state"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_niche, "GaussianMixture", spy)
    calculate_niche_cellcharter(dummy_adata2, rng=np.random.default_rng(0), copy=True, **kwargs)

    assert len(seen) == 2, "expected one mixture fit per library"
    assert seen[0] != seen[1], "libraries were fitted with the same seed"


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


def _toarray(mat) -> np.ndarray:
    """Densify a sparse (or already-dense) matrix for easy comparison in assertions."""
    return np.asarray(mat.todense()) if hasattr(mat, "todense") else np.asarray(mat)


def test_hop_adjacency_invalid_max_hop_raises():
    "max_hop must be >= 1; anything smaller is rejected."
    adj = csr_matrix(np.array([[0, 1], [1, 0]]))
    with pytest.raises(ValueError, match="max_hop must be >= 1"):
        _compute_hop_adjacency_matrices(adj, max_hop=0)
    with pytest.raises(ValueError, match="max_hop must be >= 1"):
        _compute_hop_adjacency_matrices(adj, max_hop=-3)


def test_hop_adjacency_output_length_matches_max_hop():
    "The returned list always has exactly `max_hop` entries."
    adj = csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    for max_hop in (1, 2, 3, 5):
        result = _compute_hop_adjacency_matrices(adj, max_hop=max_hop)
        assert len(result) == max_hop


def test_hop_adjacency_first_entry_is_input_unmodified():
    "adj_mat_list[0] must be exactly the input matrix, self-loops and all."
    adj = csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]]))
    result = _compute_hop_adjacency_matrices(adj, max_hop=1)
    assert np.array_equal(_toarray(result[0]), _toarray(adj))


def test_hop_adjacency_path_graph_two_hop_layer():
    "On a 3-node path 0-1-2, the 2-hop layer connects only (0, 2): reachable in 2 steps but not adjacent."
    adj = csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    result = _compute_hop_adjacency_matrices(adj, max_hop=2)

    expected_hop2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    assert np.array_equal(_toarray(result[1]), expected_hop2)


def test_hop_adjacency_path_graph_beyond_diameter_is_empty():
    "A 3-node path has diameter 2, so the 3-hop layer must be all zeros: no pair is exactly 3 apart."
    adj = csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    result = _compute_hop_adjacency_matrices(adj, max_hop=3)

    assert np.array_equal(_toarray(result[2]), np.zeros((3, 3)))


def test_hop_adjacency_triangle_graph_two_hop_layer_is_empty():
    "In a fully-connected triangle every pair is already 1-hop apart, so the 2-hop layer adds nothing new."
    adj = csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    result = _compute_hop_adjacency_matrices(adj, max_hop=2)

    assert np.array_equal(_toarray(result[1]), np.zeros((3, 3)))


def test_hop_adjacency_self_loops_only_never_propagate():
    "If the input has no off-diagonal edges, no cell can reach any other cell at any hop distance."
    adj = identity(4, format="csr")
    result = _compute_hop_adjacency_matrices(adj, max_hop=3)

    assert np.array_equal(_toarray(result[0]), _toarray(adj))
    for hop_layer in result[1:]:
        assert np.array_equal(_toarray(hop_layer), np.zeros((4, 4)))


def test_hop_adjacency_isolated_node_stays_isolated():
    "A node with no edges at all must remain disconnected from everything, including itself, at every hop."
    adj = csr_matrix(np.zeros((1, 1)))
    result = _compute_hop_adjacency_matrices(adj, max_hop=3)

    for hop_layer in result:
        assert np.array_equal(_toarray(hop_layer), np.zeros((1, 1)))


def test_hop_adjacency_layers_are_binary_despite_multiple_paths():
    "A 4-cycle gives two distinct 2-hop paths between opposite corners; the output must read 0/1, not a path count."
    # 4-cycle: 0-1-2-3-0
    adj = csr_matrix(
        np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        )
    )
    result = _compute_hop_adjacency_matrices(adj, max_hop=2)
    hop2 = _toarray(result[1])

    assert set(np.unique(hop2)) <= {0, 1}
    expected_hop2 = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    assert np.array_equal(hop2, expected_hop2)


# older tests


def test_niche_calc_nhood(adata_seqfish: AnnData):
    """Check whether niche calculation using neighborhood profile approach works as intended."""
    spatial_neighbors_knn(adata_seqfish, n_neighs=N_NEIGHBORS)
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
    kwargs = {"groups": "celltype", "n_neighbors": 3, "resolutions": 1.0}

    out = calculate_niche_neighborhood(dummy_adata2, copy=True, **kwargs)
    assert key in out.obs.columns
    assert key not in dummy_adata2.obs.columns

    assert calculate_niche_neighborhood(dummy_adata2, **kwargs) is None
    assert (dummy_adata2.obs[key] == out.obs[key]).all()


def test_hop_adjacency_excludes_visited_pairs_on_a_weighted_graph():
    # 0 and 1 are direct neighbours and also share three common neighbours
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (4, 5)]

    for weight in (1.0, 0.5):
        adjacency = np.zeros((6, 6))
        for i, j in edges:
            adjacency[i, j] = adjacency[j, i] = weight

        shells = _compute_hop_adjacency_matrices(csr_matrix(adjacency), max_hop=2)
        assert not shells[1][0, 1], f"a one-hop neighbour reappeared in the two-hop shell at weight {weight}"
        assert shells[1][0, 5], "a genuinely new two-hop pair went missing"


def test_niche_neighborhood_rejects_too_few_hop_weights(dummy_adata2: AnnData):
    with pytest.raises(ValueError, match=r"less than hops requested"):
        calculate_niche_neighborhood(
            dummy_adata2, groups="celltype", resolutions=1.0, n_neighbors=3, distance=3, n_hop_weights=[1.0]
        )


def test_hop_adjacency_power_mode_differs_from_shells():
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (4, 5)]
    adjacency = np.zeros((6, 6))
    for i, j in edges:
        adjacency[i, j] = adjacency[j, i] = 1
    adjacency = csr_matrix(adjacency)

    shells = _compute_hop_adjacency_matrices(adjacency, max_hop=2)
    powers = [adjacency, adjacency @ adjacency]

    # hop 1 is the graph itself either way
    assert (shells[0] != powers[0]).nnz == 0
    # hop 2 is not: powers count every 2-walk with multiplicity, shells only new pairs
    assert powers[1][0, 1] == 3, "0 and 1 are joined by three 2-step paths"
    assert not shells[1][0, 1], "0 and 1 were already joined at one hop"


def test_neighborhood_profile_weights_by_path_count(dummy_adata2: AnnData):
    spatial_neighbors_knn(dummy_adata2, n_neighs=3)
    adj = dummy_adata2.obsp["spatial_connectivities"]
    one_hot = pd.get_dummies(dummy_adata2.obs["celltype"], dtype=np.float64).to_numpy()

    got = np.asarray(
        _nhood_profile_embedding(
            dummy_adata2,
            groups="celltype",
            spatial_connectivities_key="spatial_connectivities",
            scale=False,
            distance=3,
            abs_nhood=False,
            n_hop_weights=None,
        )
    )

    expected, power = np.zeros_like(got), adj
    for hop in range(3):
        if hop:
            power = power @ adj
        profile = power @ one_hot
        total = profile.sum(axis=1)[:, None]
        expected += np.divide(profile, total, out=np.zeros_like(profile), where=total != 0)
    np.testing.assert_allclose(got, expected / 3)


def test_calculate_niche_deprecation_is_a_future_warning(dummy_adata2: AnnData):
    with pytest.warns(FutureWarning, match=r"`calculate_niche` is deprecated"):
        calculate_niche(dummy_adata2, flavor="utag", n_neighbors=3, resolutions=1.0, rng=0)


def test_hop_adjacency_shells_are_boolean_on_a_weighted_graph():
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (4, 5)]
    for weight in (1.0, 0.5):
        adjacency = np.zeros((6, 6))
        for i, j in edges:
            adjacency[i, j] = adjacency[j, i] = weight
        shells = _compute_hop_adjacency_matrices(csr_matrix(adjacency), max_hop=2)

        assert shells[0].dtype == bool, "the graph is cast to bool, as CellCharter does"
        assert not shells[1][0, 1], f"one-hop neighbour reappeared in the two-hop shell at weight {weight}"
        assert shells[1][0, 5]


def test_hop_adjacency_powers_stay_numeric():
    """`power` counts walks, so it must not be binarised along with the shells."""
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (4, 5)]
    adjacency = np.zeros((6, 6))
    for i, j in edges:
        adjacency[i, j] = adjacency[j, i] = 1
    adjacency = csr_matrix(adjacency)
    powers = [adjacency, adjacency @ adjacency]
    assert powers[1][0, 1] == 3, "three 2-step paths, and the multiplicity has to survive"


@pytest.mark.parametrize("sparse", [True, False])
def test_nhood_blocks_stack_keeps_the_container(sparse: bool):
    rng = np.random.default_rng(0)
    X = rng.random((40, 5))
    adata = AnnData(X=csr_matrix(X) if sparse else X)
    adata.obsm["spatial"] = rng.random((40, 2)) * 10
    spatial_neighbors_knn(adata, n_neighs=4)

    blocks = _nhood_blocks(adata, hops=range(3), hop_mode="shell")
    stacked = sparse_hstack(blocks, format="csr") if sparse else np.hstack([to_dense(b) for b in blocks])
    assert issparse(stacked) is sparse, "a sparse input should not be densified on the way out"
    assert stacked.shape == (40, 15)

    # the pooled path keeps the container too
    pooled = nhood_aggregate(adata, hops=range(1, 3))
    assert issparse(pooled) is sparse
    assert pooled.shape == (40, 5)


@pytest.mark.parametrize("max_hop", [1, 2, 3, 4])
@pytest.mark.parametrize("self_loops", [False, True])
@pytest.mark.parametrize("weight", [1.0, 0.5])
def test_bfs_shells_match_the_matmul_definition(max_hop: int, self_loops: bool, weight: float):
    rng = np.random.default_rng(0)
    points = np.vstack([rng.random((120, 2)) * 10, rng.random((120, 2)) * 10 + [60, 0], rng.random((4, 2)) + [30, 30]])
    adata = AnnData(X=np.zeros((len(points), 1), dtype=np.float32))
    adata.obsm["spatial"] = points
    spatial_neighbors_radius(adata, radius=1.6)

    adj = adata.obsp["spatial_connectivities"].astype(float) * weight
    if self_loops:
        adj = adj.tolil()
        adj.setdiag(weight)
        adj = adj.tocsr()

    # the definition the search replaced: boolean matmul minus everything already reached
    boolean = adj.astype(bool)
    expected = [boolean]
    visited = boolean.copy()
    visited.setdiag(1)
    frontier = boolean
    for _ in range(1, max_hop):
        frontier = (frontier @ boolean) > visited
        visited = visited + frontier
        expected.append(frontier)

    got = _shell_adjacencies(adj, max_hop)
    assert len(got) == max_hop
    for hop, (want, have) in enumerate(zip(expected, got, strict=True), start=1):
        assert (want != have).nnz == 0, f"hop {hop} differs"
