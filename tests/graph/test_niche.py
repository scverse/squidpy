from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pandas import Series
from scanpy.pp import neighbors
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from spatialdata.models import TableModel

from squidpy.gr import (
    calculate_niche,
    calculate_niche_cellcharter,
    calculate_niche_neighborhood,
    spatial_neighbors_knn,
)
from squidpy.gr._niche import compute_hop_adjacency_matrices

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
    kwargs = {"groups": "celltype", "n_neighbors": 3, "resolutions": 1.0}

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
