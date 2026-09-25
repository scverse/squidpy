from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData, read_h5ad
from fast_array_utils.conv import to_dense
from pandas import Series
from pandas.testing import assert_frame_equal
from scanpy.pp import neighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from spatialdata import SpatialData
from spatialdata.models import TableModel

from squidpy.gr import (
    _niche,
    calculate_niche,
    calculate_niche_cellcharter,
    calculate_niche_neighborhood,
    calculate_niche_spatialleiden,
    calculate_niche_utag,
    spatial_neighbors_knn,
)
from squidpy.gr._nhood import _aggregate_over, nhood_aggregate
from squidpy.gr._niche import _fit_clusterers, compute_hop_adjacency_matrices

# Many tests here exercise the deprecated umbrella, which keeps working until v1.9.0;
# `test_calculate_niche_deprecation_is_a_future_warning` checks the deprecation itself.
pytestmark = pytest.mark.filterwarnings("ignore:Calling `calculate_niche` is deprecated:FutureWarning")

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
    "Without library_key the labels are one vocabulary over every batch."
    dummy_adata2.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5

    calculate_niche(dummy_adata2, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.5)

    niches = _assert_all_assigned(dummy_adata2, "nhood_niche_res=1.5")
    # one model over both batches now, so the labels are a shared vocabulary with no lib= prefix
    assert not any(label.startswith("lib=") for _, label in niches.items())


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
        calculate_niche_cellcharter(dummy_adata2, distance=distance, n_clusters=2, rng=0)


def test_neighborhood_profile_counts_each_neighbor_once(dummy_adata2: AnnData):
    """Through the public call, with `scale=False` so the raw profile survives.

    Past hop 1 the reach is a set, so a cell two paths away counts once, not twice.
    """
    spatial_neighbors_knn(dummy_adata2, n_neighs=3)
    adj = _toarray(dummy_adata2.obsp["spatial_connectivities"])
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

    expected = np.zeros_like(got)
    for hop in range(1, 4):
        # hop 1 keeps the supplied edge weights; beyond it only reachability is defined
        reach = adj if hop == 1 else (np.linalg.matrix_power(adj != 0, hop) > 0).astype(np.float64)
        profile = reach @ one_hot
        total = profile.sum(axis=1)[:, None]
        expected += np.divide(profile, total, out=np.zeros_like(profile), where=total != 0)
    np.testing.assert_allclose(got, expected / 3, rtol=1e-6, atol=1e-7)


def test_neighborhood_profile_on_an_irregular_graph():
    """A cell with fewer neighbors than the densest one still gets its own composition.

    The pre-refactor profile padded every row out to the maximum degree and counted the
    padding as the last cell's category, so all rows came back identical.
    """
    # degrees 3, 2, 2, 1 - cell 3 is the low-degree one, and 'b' is what padding injected
    edges = [(0, 1), (0, 2), (0, 3), (1, 2)]
    adata = AnnData(
        np.zeros((4, 1), dtype=np.float32),
        obs=pd.DataFrame({"celltype": pd.Categorical(["a", "b", "a", "b"])}, index=list("wxyz")),
    )
    adj = np.zeros((4, 4))
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1.0
    adata.obsp["spatial_connectivities"] = csr_matrix(adj)

    profile = to_dense(nhood_aggregate(adata, groups="celltype", hops=(1,), aggregation="mean"))
    expected = np.array(
        [
            [1 / 3, 2 / 3],  # neighbors 1, 2, 3 -> b, a, b
            [1.0, 0.0],  # neighbors 0, 2    -> a, a
            [0.5, 0.5],  # neighbors 0, 1    -> a, b
            [1.0, 0.0],  # neighbor  0       -> a
        ]
    )
    np.testing.assert_allclose(profile, expected)


def test_neighborhood_profile_skips_unlabelled_neighbors():
    """A neighbor with no category counts towards no one's neighbor count."""
    # cell 0 neighbors cells 1-4: a, a, b and one with no celltype
    adata = AnnData(
        np.zeros((5, 1), dtype=np.float32),
        obs=pd.DataFrame({"celltype": pd.Categorical(["a", "a", "a", "b", None])}, index=list("vwxyz")),
    )
    adj = np.zeros((5, 5))
    adj[0, 1:] = adj[1:, 0] = 1.0
    adata.obsp["spatial_connectivities"] = csr_matrix(adj)

    profile = to_dense(nhood_aggregate(adata, groups="celltype", hops=(1,), aggregation="mean"))
    # [2/3, 1/3] over the three labelled neighbors, not [1/2, 1/4] over all four
    np.testing.assert_allclose(profile[0], [2 / 3, 1 / 3])


def _weighted_square() -> AnnData:
    """Four cells in a ring, categories a b a b, with one heavy edge (0-1)."""
    adata = AnnData(
        np.zeros((4, 1), dtype=np.float32),
        obs=pd.DataFrame({"celltype": pd.Categorical(["a", "b", "a", "b"])}, index=list("wxyz")),
    )
    adj = np.zeros((4, 4))
    for (i, j), weight in zip([(0, 1), (0, 2), (1, 3), (2, 3)], [3.0, 1.0, 1.0, 1.0], strict=True):
        adj[i, j] = adj[j, i] = weight
    adata.obsp["spatial_connectivities"] = csr_matrix(adj)
    return adata


def test_neighborhood_profile_weights_hop_one_only():
    """Edge weights apply to edges, so hop 1 uses them and the hops past it cannot."""
    weighted = _weighted_square()
    binary = _weighted_square()
    binary.obsp["spatial_connectivities"] = csr_matrix(
        (_toarray(binary.obsp["spatial_connectivities"]) != 0).astype(np.float64)
    )

    def profile(adata: AnnData, hop: int) -> np.ndarray:
        return to_dense(nhood_aggregate(adata, groups="celltype", hops=(hop,), aggregation="mean"))

    # cell 0 neighbors 1 ('b', weight 3) and 2 ('a', weight 1)
    np.testing.assert_allclose(profile(weighted, 1)[0], [0.25, 0.75])
    np.testing.assert_allclose(profile(binary, 1)[0], [0.5, 0.5])
    # hop 2 is reachability, so the heavy edge cannot tilt it
    np.testing.assert_allclose(profile(weighted, 2), profile(binary, 2))


def test_neighborhood_warns_once_on_a_weighted_graph():
    """The warning is actionable: setting the weights to 1 is what it asks for."""
    with pytest.warns(UserWarning, match=r"non-binary edge weights.*Set them to 1") as caught:
        calculate_niche_neighborhood(
            _weighted_square(), groups="celltype", resolutions=1.0, n_neighbors=2, copy=True, rng=0
        )
    assert caught[0].filename == __file__, f"attributed to {caught[0].filename}"


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


@pytest.mark.parametrize(
    ("adjacency", "max_hop", "expected"),
    [
        pytest.param(_PATH3, 2, [_PATH3, _PATH3_RING2], id="path, second ring is the far pair"),
        pytest.param(_PATH3, 3, [_PATH3, _PATH3_RING2, np.zeros((3, 3))], id="path, nothing past the diameter"),
        pytest.param(_stored_zero_path3(), 2, [_PATH3, _PATH3_RING2], id="stored zeros are not edges"),
        pytest.param(_TRIANGLE, 2, [_TRIANGLE, np.zeros((3, 3))], id="triangle has no second ring"),
        pytest.param(_SELF_LOOPED, 1, [_SELF_LOOPED], id="hop 1 keeps a self-loop"),
        pytest.param(np.eye(3), 3, [np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))], id="self-loops never propagate"),
        pytest.param(np.zeros((1, 1)), 3, [np.zeros((1, 1))] * 3, id="an isolated node stays isolated"),
        pytest.param(_CYCLE4, 2, [_CYCLE4, np.roll(np.eye(4), 2, axis=1)], id="rings are binary, not path counts"),
    ],
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


def _tiny(n: int = 40, libraries: list[str] | None = None, *, embedding_cols: int | None = None) -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(X=csr_matrix(rng.random((n, 6)).astype(np.float32)))
    adata.obsm["spatial"] = rng.random((n, 2)) * 10
    adata.obs["ct"] = pd.Categorical([f"t{k}" for k in rng.integers(0, 3, n)])
    if libraries is not None:
        adata.obs["library"] = libraries
    if embedding_cols is not None:
        adata.obsm["emb"] = rng.random((n, embedding_cols))
    spatial_neighbors_knn(adata, n_neighs=4)
    return adata


def test_hop_adjacency_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match=r"must be square"):
        compute_hop_adjacency_matrices(csr_matrix(np.ones((3, 5), dtype=float)), max_hop=2)


def test_hop_adjacency_accepts_an_array_like():
    "A plain nested list still works; the square check must not read `.shape` off it."
    rings = compute_hop_adjacency_matrices([[0, 1, 0], [1, 0, 1], [0, 1, 0]], max_hop=2)
    assert np.array_equal(_toarray(rings[1]).astype(float), _PATH3_RING2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        # hop 0 returns the features unaggregated, so it never reaches the per-hop check
        ({"hops": (0,), "aggregation": "median"}, r"'aggregation' must be"),
        ({"hops": (1,), "aggregation": "median"}, r"'aggregation' must be"),
        ({"hops": (1, 2), "hop_weights": [1.0]}, r"'hop_weights' has 1 value"),
        # averaging over a zero total used to return an all-NaN matrix silently
        ({"hops": (1, 2), "hop_weights": [1.0, -1.0], "aggregation": "mean"}, r"must not sum to zero"),
    ],
)
def test_nhood_aggregate_rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        nhood_aggregate(_tiny(), **kwargs)


def test_nhood_aggregate_sums_cancelling_weights():
    "Counts stay counts, so `sum` never divides and the weights may cancel."
    assert nhood_aggregate(_tiny(), hops=(1, 2), hop_weights=[1.0, -1.0], aggregation="sum") is not None


@pytest.mark.parametrize("key", [None, "", 5])
def test_niche_rejects_an_unusable_embedding_key(key):
    "`obsm[None]` is accepted by AnnData and only fails later, at write_h5ad."
    with pytest.raises(ValueError, match=r"'embedding_key_added' must be a non-empty string"):
        calculate_niche_cellcharter(_tiny(), distance=2, n_clusters=2, rng=0, embedding_key_added=key)


def test_weighted_graph_warning_points_at_the_caller():
    "`stacklevel` must skip squidpy's own frames, including the `functools.partial` hop."
    adata = _tiny()
    spatial_neighbors_knn(adata, n_neighs=4, transform="spectral")
    with pytest.warns(UserWarning, match="non-binary") as caught:
        calculate_niche_cellcharter(adata, distance=2, n_clusters=2, rng=0)
    assert caught[0].filename == __file__, f"attributed to {caught[0].filename}"


def test_clusterer_without_a_random_state_is_rejected():
    "A deterministic estimator satisfies the protocol and then rejects the seed the pipeline sets."
    from sklearn.cluster import DBSCAN

    adata = _tiny()
    with pytest.raises(TypeError, match=r"no 'random_state'"):
        _fit_clusterers(adata, np.asarray(to_dense(adata.X)), {"c": DBSCAN(eps=3.0)}, np.random.default_rng(0))


def test_non_boolean_mask_raises():
    "`to_numpy(dtype=bool)` reads every non-empty string as True, so this must not pass silently."
    adata = _tiny(n=60)
    strings = Series(["False"] * 20 + ["True"] * 40, index=adata.obs_names)
    with pytest.raises(TypeError, match=r"'cluster_mask' must be a boolean Series, got dtype"):
        calculate_niche_utag(adata, resolutions=1.0, rng=0, cluster_mask=strings)


@pytest.mark.parametrize(
    ("flavor", "kwargs"),
    [
        ("neighborhood", {"groups": "ct", "resolutions": 1.0}),
        ("utag", {"resolutions": 1.0}),
        ("cellcharter", {"distance": 1, "n_clusters": 2}),
        ("spatialleiden", {"resolutions": 1.0, "latent_connectivities_key": "spatial_connectivities"}),
    ],
)
def test_no_flavor_takes_a_library_key(flavor, kwargs):
    "Each fits one model over everything, so labels compare across libraries."
    assert "library_key" not in inspect.signature(globals()[f"calculate_niche_{flavor}"]).parameters
    adata = _tiny(n=40, libraries=["a"] * 20 + ["b"] * 20)
    with pytest.raises(TypeError, match="library_key"):
        globals()[f"calculate_niche_{flavor}"](adata, rng=0, library_key="library", **kwargs)


@pytest.mark.parametrize(
    ("flavor", "kwargs"),
    [
        ("neighborhood", {"groups": "ct", "n_neighbors": 4, "resolutions": 1.0}),
        ("utag", {"n_neighbors": 4, "resolutions": 1.0}),
        ("cellcharter", {"n_components": 2, "distance": 1}),
        ("spatialleiden", {"resolutions": 1.0, "latent_connectivities_key": "spatial_connectivities"}),
    ],
)
def test_the_umbrella_still_fits_per_library_and_warns(flavor, kwargs):
    "Released with library_key, so until v1.9.0 it keeps v1.8.3's behaviour and says it is going."
    adata = _tiny(n=40, libraries=["a"] * 20 + ["b"] * 18 + [None] * 2)
    with pytest.warns(FutureWarning, match=r"'library_key' is deprecated") as caught:
        calculate_niche(adata, flavor=flavor, library_key="library", rng=0, **kwargs)
    ours = next(w for w in caught if "'library_key' is deprecated" in str(w.message))
    assert ours.filename == __file__, f"attributed to {ours.filename}"

    column = next(c for c in adata.obs.columns if "niche" in c or c.startswith("spatialleiden"))
    labels = adata.obs[column].astype(str)
    assert adata.obs[column].dtype == "category"
    # each library is its own vocabulary, and an observation with no library gets no niche
    for library in ("a", "b"):
        kept = labels[adata.obs["library"] == library]
        assert all(label == "not_a_niche" or label.startswith(f"lib={library}_") for label in kept)
    assert (labels[adata.obs["library"].isna()] == "not_a_niche").all()


def test_integer_features_keep_their_ring_means():
    "The block the rings are written into took the features' dtype, so counts truncated the mean."
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 20, (60, 4)).astype(np.int64)
    adata = AnnData(X=counts)
    adata.obs_names = [f"c{i}" for i in range(60)]
    adata.obsm["spatial"] = rng.random((60, 2)) * 10
    spatial_neighbors_knn(adata, n_neighs=5)

    calculate_niche_cellcharter(adata, use_rep="X", distance=1, n_clusters=2, rng=0)
    ring = np.asarray(adata.obsm["niche_embedding"][:, 4:8], dtype=np.float64)
    expected = to_dense(
        _aggregate_over(
            adata.obsp["spatial_connectivities"].astype(bool),
            counts.astype(np.float64),
            "mean",
        )
    )
    np.testing.assert_allclose(ring, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dtype", [bool, np.int64])
def test_non_float_features_aggregate_exactly(dtype):
    "`bool @ bool` saturates to True instead of summing, so widening the product is too late."
    rng = np.random.default_rng(0)
    n = 40
    adj = csr_matrix((np.ones(n * 4, bool), (np.repeat(np.arange(n), 4), rng.integers(0, n, n * 4))), shape=(n, n))
    features = (rng.random((n, 4)) > 0.5) if dtype is bool else rng.integers(0, 20, (n, 4))
    got = to_dense(_aggregate_over(adj, features, "mean"))
    expected = to_dense(_aggregate_over(adj.astype(np.float64), features.astype(np.float64), "mean"))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_niche_categories_are_in_numeric_order():
    "They order the legend and the colors, so 10 goes after 9 rather than after 1, as `sc.tl.leiden` has it."
    labels = np.array(["10", "2", "not_a_niche", "0", "1", "9", "2"])
    assert list(_niche._niche_labels(labels, None).categories) == ["0", "1", "2", "9", "10", "not_a_niche"]


@pytest.mark.parametrize("min_niche_size", [None, 3])
@pytest.mark.parametrize("flavor", ["neighborhood", "utag", "cellcharter", "spatialleiden"])
def test_niche_labels_are_strings_on_every_flavor(flavor, min_niche_size):
    "spatialleiden writes its own column, and used to leave integers there unless min_niche_size relabeled it."
    adata = _tiny(n=60)
    neighbors(adata, n_neighbors=8, use_rep="X")
    calculate_niche(
        adata, flavor=flavor, groups="ct", n_neighbors=8, resolutions=1.0, min_niche_size=min_niche_size, rng=0
    )
    column = next(c for c in adata.obs.columns if "niche" in c or c.startswith("spatialleiden"))
    assert adata.obs[column].dtype == "category"
    assert all(isinstance(label, str) for label in adata.obs[column].cat.categories)


@pytest.mark.parametrize("flavor", ["neighborhood", "utag", "cellcharter", "spatialleiden"])
def test_min_niche_size_is_not_reported_unused(flavor, caplog):
    "Every flavor applies it, so no flavor should call it unused."
    adata = _tiny(n=60)
    neighbors(adata, n_neighbors=8, use_rep="X")
    with caplog.at_level(logging.WARNING):
        calculate_niche(adata, flavor=flavor, groups="ct", n_neighbors=8, resolutions=1.0, min_niche_size=3, rng=0)
    assert "min_niche_size" not in caplog.text


def test_utag_use_rep_replaces_the_pca_it_would_fit():
    "A representation is already reduced, so utag aggregates it and stops."
    adata = _tiny(n=60, embedding_cols=5)
    calculate_niche_utag(adata, resolutions=1.0, n_neighbors=8, rng=0, use_rep="emb")
    assert adata.obsm["niche_embedding"].shape == (60, 5)

    on_x = _tiny(n=60)
    calculate_niche_utag(on_x, resolutions=1.0, n_neighbors=8, rng=0, use_rep="X")
    assert on_x.obsm["niche_embedding"].shape == (60, 6), "'X' is the spelling scanpy takes"

    derived = _tiny(n=60)
    calculate_niche_utag(derived, resolutions=1.0, n_neighbors=8, rng=0)
    assert derived.obsm["niche_embedding"].shape[1] == 5, "without it, the PCA still runs"


def test_utag_use_rep_and_use_layer_are_exclusive():
    adata = _tiny(embedding_cols=4)
    adata.layers["counts"] = adata.X.copy()
    with pytest.raises(ValueError, match=r"at most one of 'groups', 'use_rep' and 'layer'"):
        calculate_niche_utag(adata, resolutions=1.0, rng=0, use_rep="emb", use_layer="counts")


def test_neighborhood_takes_no_use_rep():
    "Its columns are the `groups` categories, shared across libraries already."
    assert "use_rep" not in inspect.signature(calculate_niche_neighborhood).parameters


def test_cellcharter_use_rep_x_skips_the_pca():
    adata = _tiny(n=60)
    calculate_niche_cellcharter(adata, n_clusters=3, distance=1, rng=0, use_rep="X")
    assert adata.obsm["niche_embedding"].shape == (60, 6 * 2), "X itself, then one ring of it"


def test_use_rep_is_aggregated_over_the_hop_rings():
    "CellCharter aggregates the representation; `use_rep` used to replace the whole embedder."
    adata = _tiny(n=60, embedding_cols=6)
    scrambled = _tiny(n=60, embedding_cols=6)
    rng = np.random.default_rng(7)
    scrambled.obsm["spatial"] = rng.random((60, 2)) * 10
    del scrambled.obsp["spatial_connectivities"], scrambled.obsp["spatial_distances"]
    spatial_neighbors_knn(scrambled, n_neighs=4)

    labels = []
    for a in (adata, scrambled):
        calculate_niche_cellcharter(a, use_rep="emb", n_clusters=3, distance=2, rng=0)
        labels.append(np.asarray(a.obs["cellcharter_niche"].astype(str)))
    assert not (labels[0] == labels[1]).all(), "the spatial graph did not affect the result"


def test_use_rep_narrower_than_n_clusters_is_accepted():
    "A k-cluster GMM is well posed in any dimensionality; the old guard required k columns."
    adata = _tiny(embedding_cols=2)
    calculate_niche_cellcharter(adata, use_rep="emb", n_clusters=5, distance=1, rng=0)
    assert adata.obs["cellcharter_niche"].nunique() <= 5


def test_n_pca_components_sizes_the_pca():
    adata = _tiny(n=60)
    calculate_niche_cellcharter(adata, n_clusters=3, n_pca_components=4, distance=1, rng=0)
    assert adata.obsm["niche_embedding"].shape[1] == 4 * 2


def test_n_pca_components_is_rejected_with_use_rep():
    with pytest.raises(ValueError, match=r"'n_pca_components' sizes the PCA, which 'use_rep' replaces"):
        calculate_niche_cellcharter(_tiny(embedding_cols=6), use_rep="emb", n_pca_components=3, rng=0)


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


def test_composition_profile_rows_sum_exactly():
    "`spatial_neighbors` leaves obsp float32, and a float32 1/k puts a 1e-8 error in every row."
    adata = _tiny(n=200)
    profile = nhood_aggregate(adata, groups="ct", connectivity_key="spatial_connectivities")
    assert profile.dtype == np.float64, "a float32 adjacency must not set the profile's precision"
    # every row is a distribution over the categories, so each sums to one and the total is n_obs
    np.testing.assert_array_almost_equal(to_dense(profile).sum(axis=1), 1.0, decimal=15)


def test_a_feature_matrix_keeps_its_own_precision():
    "Supplied features are not copied into another width, so a float32 X aggregates in float32."
    adata = _tiny(n=60)
    assert adata.X.dtype == np.float32
    assert nhood_aggregate(adata, connectivity_key="spatial_connectivities").dtype == np.float32
    adata.obsm["emb32"] = np.ones((60, 4), dtype=np.float32)
    got = nhood_aggregate(adata, use_rep="emb32", connectivity_key="spatial_connectivities")
    assert got.dtype == np.float32, "a representation the caller supplied must not be widened"


def test_float32_features_aggregate_exactly():
    "`spatial_neighbors` leaves obsp float32, and pre-dividing it put 1/k's rounding in every row."
    rng = np.random.default_rng(0)
    n = 4000
    pts = rng.random((n, 2)) * 26
    # a radius graph: degrees vary, so 1/k is inexact for most rows
    pairs = cKDTree(pts).query_pairs(r=1.0, output_type="ndarray")
    adj = csr_matrix((np.ones(len(pairs), np.float32), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    adj = ((adj + adj.T) > 0).astype(np.float32)
    assert len(set(np.asarray(adj.sum(1)).ravel().astype(int))) > 5, "degrees must vary"

    adata = _tiny(n=n)
    adata.obsp["spatial_connectivities"] = adj
    adata.obsm["ones"] = np.ones((n, 8), dtype=np.float32)
    # the mean of an all-ones matrix is exactly 1.0 and representable at float32, so any deviation
    # is the division, not the width
    got = to_dense(nhood_aggregate(adata, use_rep="ones", connectivity_key="spatial_connectivities"))
    assert got.dtype == np.float32
    reached = np.asarray(adj.sum(1)).ravel() > 0
    np.testing.assert_array_equal(got[reached], 1.0)


@pytest.mark.parametrize("sparse", [False, True])
def test_integer_features_aggregate(sparse):
    "The hop rings are bool, so an integer X sums to an integer, which cannot hold a mean."
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 20, (60, 4)).astype(np.int64)
    adata = AnnData(X=csr_matrix(counts) if sparse else counts)
    adata.obs_names = [f"c{i}" for i in range(60)]
    adata.obsm["spatial"] = rng.random((60, 2)) * 10
    spatial_neighbors_knn(adata, n_neighs=5)

    ring = _aggregate_over(adata.obsp["spatial_connectivities"].astype(bool), adata.X, "mean")
    assert np.issubdtype(to_dense(ring).dtype, np.floating), "a mean of counts is not a count"
    # and the whole flavor still runs, which it did not when the quotient went back into an int
    calculate_niche_cellcharter(adata, use_rep="X", distance=1, n_clusters=2, rng=0)


def test_an_isolated_observation_aggregates_to_zero():
    "`normalize` left a zero row alone; the reciprocal that replaced it must too."
    adata = _tiny(n=40)
    graph = adata.obsp["spatial_connectivities"].tolil()
    graph[7, :] = 0
    graph[:, 7] = 0
    adata.obsp["spatial_connectivities"] = graph.tocsr()
    adata.obsp["spatial_connectivities"].eliminate_zeros()

    for kwargs in ({"groups": "ct"}, {}):
        got = to_dense(nhood_aggregate(adata, connectivity_key="spatial_connectivities", **kwargs))
        assert np.isfinite(got).all(), f"{kwargs} left a nan or an inf"
        np.testing.assert_array_equal(got[7], 0.0)


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
def test_cellcharter_concatenates_hop_zero_with_every_ring(distance: int):
    "The embedding is the PCA of X plus one block per ring; dropping hop 0 must be visible."
    adata = _tiny(n=60)
    calculate_niche_cellcharter(adata, distance=distance, n_clusters=2, n_pca_components=3, rng=0)

    embedding = adata.obsm["niche_embedding"]
    assert embedding.shape[1] == (distance + 1) * 3
    # the concatenation is `distance + 1` times as wide as the features, so it keeps their dtype
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(embedding[:, :3], sc.pp.pca(adata.X, n_comps=3), rtol=1e-5)


def test_cellcharter_reduces_before_aggregating():
    "PCA of X is the fallback for `use_rep`, so passing that PCA as `use_rep` must change nothing."
    adata = _tiny(n=60)
    adata.obsm["X_pca"] = sc.pp.pca(adata.X, n_comps=3)
    fallback = calculate_niche_cellcharter(adata, distance=2, n_clusters=3, n_pca_components=3, rng=0, copy=True)
    given = calculate_niche_cellcharter(adata, distance=2, n_clusters=3, use_rep="X_pca", rng=0, copy=True)
    np.testing.assert_allclose(fallback.obsm["niche_embedding"], given.obsm["niche_embedding"], rtol=1e-5)
    fallback_niches = fallback.obs["cellcharter_niche"].astype(str).to_numpy()
    np.testing.assert_array_equal(fallback_niches, given.obs["cellcharter_niche"].astype(str).to_numpy())


def test_cellcharter_pca_fallback_handles_a_single_feature():
    "The PCA width is clamped to what X can give, down to a one-marker panel."
    adata = _tiny(n=40)[:, :1].copy()
    calculate_niche_cellcharter(adata, distance=1, n_clusters=2, rng=0)
    assert adata.obsm["niche_embedding"].shape == (40, 2)


def test_cellcharter_rejects_n_pca_components_wider_than_x():
    "The PCA runs on X, so its ceiling is X's, and the message must say so."
    with pytest.raises(ValueError, match=r"'n_pca_components' must be between 1 and 5"):
        calculate_niche_cellcharter(_tiny(n=60), n_pca_components=30, rng=0)


def test_cellcharter_pca_respects_highly_variable():
    "`sc.pp.pca(adata)` masks by `highly_variable`; the fallback must not silently use every gene."
    adata = _tiny(n=60)
    adata.var["highly_variable"] = [True, True, True, False, False, False]
    calculate_niche_cellcharter(adata, distance=1, n_clusters=2, n_pca_components=2, rng=0)
    np.testing.assert_allclose(adata.obsm["niche_embedding"][:, :2], sc.pp.pca(adata.X[:, :3], n_comps=2), rtol=1e-5)


def test_cellcharter_niches_follow_neighborhoods_not_cell_identity():
    "Same two cell types, mixed on the left and in pure blocks on the right: niches must split them."
    width, height = 40, 20
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    x, y = x.ravel(), y.ravel()
    checkerboard = x < width // 2
    is_a = np.where(checkerboard, (x + y) % 2 == 0, x < 3 * width // 4)

    rng = np.random.default_rng(0)
    adata = AnnData(X=csr_matrix((x.size, 1), dtype=np.float32))
    adata.obsm["spatial"] = np.column_stack([x, y]).astype(float)
    adata.obsm["emb"] = np.column_stack([is_a, ~is_a]).astype(float) + rng.normal(0, 0.05, (x.size, 2))
    spatial_neighbors_knn(adata, n_neighs=4)

    calculate_niche_cellcharter(adata, use_rep="emb", distance=1, n_clusters=4, rng=0)
    labels = adata.obs["cellcharter_niche"].to_numpy()
    # away from the region and block borders every neighborhood is pure, so the niches are too
    interior = (
        (y > 0)
        & (y < height - 1)
        & ~np.isin(x, [0, width // 2 - 1, width // 2, 3 * width // 4 - 1, 3 * width // 4, width - 1])
    )
    for cell_type in (is_a, ~is_a):
        mixed = set(labels[interior & cell_type & checkerboard])
        pure = set(labels[interior & cell_type & ~checkerboard])
        assert len(mixed) == len(pure) == 1
        assert mixed != pure, "a cell type got one niche regardless of its neighbors"


@pytest.mark.parametrize("sparse", [True, False])
def test_cellcharter_accepts_sparse_and_dense_x(sparse: bool):
    rng = np.random.default_rng(0)
    X = rng.random((50, 6)).astype(np.float32)
    adata = AnnData(X=csr_matrix(X) if sparse else X)
    adata.obsm["spatial"] = rng.random((50, 2)) * 10
    spatial_neighbors_knn(adata, n_neighs=4)

    calculate_niche_cellcharter(adata, distance=2, n_clusters=3, rng=0)
    assert "cellcharter_niche" in adata.obs
    assert str(adata.obs["cellcharter_niche"].dtype) == "category"


@pytest.mark.parametrize(
    ("resolutions", "exc", "match"),
    [
        # a pair is the spatialleiden (latent, spatial) argument; elsewhere it reached scanpy
        # as `must be real number, not tuple`
        ((0.5, 1.0), TypeError, r"only the 'spatialleiden' flavor takes"),
        # repeats collided on the column name and silently produced one clustering, not two
        ([0.5, 0.5], ValueError, r"'resolutions' repeats 0.5"),
        ([], ValueError, r"'resolutions' is empty"),
        ("high", TypeError, r"'resolutions' must be numbers"),
    ],
)
def test_resolutions_are_rejected(resolutions, exc, match):
    with pytest.raises(exc, match=match):
        calculate_niche_utag(_tiny(), resolutions=resolutions, n_neighbors=4, rng=0)


def test_a_resolution_pair_is_rejected_by_every_leiden_flavor():
    "Both flavors funnel through `_leiden_clusterers`, which is where the check lives."
    with pytest.raises(TypeError, match=r"only the 'spatialleiden' flavor takes"):
        calculate_niche_neighborhood(_tiny(), groups="ct", resolutions=(0.5, 1.0), n_neighbors=4, rng=0)


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


@pytest.mark.parametrize(
    ("fn", "kwargs", "column"),
    [
        pytest.param(calculate_niche_utag, {"resolutions": 1.0, "n_neighbors": 8}, "utag_niche_res=1.0", id="utag"),
        pytest.param(
            calculate_niche_cellcharter, {"n_clusters": 3, "distance": 2}, "cellcharter_niche", id="cellcharter"
        ),
    ],
)
def test_cluster_mask_reaches_the_other_flavors(fn, kwargs, column):
    "v1.8.3 documented a mask for every flavor and applied it to one; these now honour it."
    adata = _tiny(n=80)
    keep = Series(np.arange(80) < 60, index=adata.obs_names)
    fn(adata, rng=0, cluster_mask=keep, **kwargs)
    labels = adata.obs[column].astype(str)
    assert (labels[60:] == "not_a_niche").all()
    assert (labels[:60] != "not_a_niche").all()


def test_mask_excludes_cells_from_the_clustering():
    "Masked cells used to be clustered and then relabelled, so they still shaped the niches."
    adata = _tiny(n=120)
    spied: list[int] = []
    original = _niche.LeidenClusterer.fit

    def spy(self, X, y=None):
        spied.append(X.shape[0])
        return original(self, X, y)

    keep = pd.Series(np.arange(120) < 80, index=adata.obs_names)
    _niche.LeidenClusterer.fit = spy
    try:
        calculate_niche_neighborhood(adata, groups="ct", resolutions=1.0, n_neighbors=8, rng=0, cluster_mask=keep)
    finally:
        _niche.LeidenClusterer.fit = original
    assert spied == [80], f"the clusterer was fitted on {spied} observations, not the kept 80"
    labels = adata.obs["nhood_niche_res=1.0"].astype(str)
    assert (labels[80:] == "not_a_niche").all()
    assert (labels[:80] != "not_a_niche").all()


def test_mask_accepts_a_partial_index():
    "The documented example is a three-entry mask; it used to raise an IndexingError."
    adata = _tiny(n=60)
    partial = Series([False, False, True], index=["0", "1", "2"])
    partial.index = adata.obs_names[:3]
    calculate_niche_neighborhood(adata, groups="ct", resolutions=1.0, n_neighbors=8, rng=0, cluster_mask=partial)
    labels = adata.obs["nhood_niche_res=1.0"].astype(str)
    assert (labels[:2] == "not_a_niche").all(), "the two False entries must be excluded"
    assert (labels[2:] != "not_a_niche").all(), "everything the mask omits is kept"


@pytest.mark.parametrize(
    ("index", "match"),
    [
        pytest.param(["zz", "yy"], r"shares no index value", id="wrong index entirely"),
        pytest.param(None, r"excludes every observation", id="excludes everything"),
    ],
)
def test_mask_rejects_what_it_cannot_mean(index, match):
    adata = _tiny(n=40)
    mask = (
        Series([False, False], index=index)
        if index is not None
        else Series(np.zeros(40, dtype=bool), index=adata.obs_names)
    )
    with pytest.raises(ValueError, match=match):
        calculate_niche_neighborhood(adata, groups="ct", resolutions=1.0, n_neighbors=8, rng=0, cluster_mask=mask)


def test_spatialleiden_refuses_a_cluster_mask():
    "It clusters the graphs, so an observation cannot be kept as a neighbor but dropped from the fit."
    adata = _tiny(n=40)
    keep = Series(np.arange(40) < 30, index=adata.obs_names)
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'cluster_mask'"):
        calculate_niche_spatialleiden(adata, resolutions=0.5, rng=0, cluster_mask=keep)


def test_the_umbrella_refuses_a_mask_for_spatialleiden():
    "`calculate_niche` keeps the released spelling `mask`; it maps to `cluster_mask`."
    adata = _tiny(n=40)
    keep = Series(np.arange(40) < 30, index=adata.obs_names)
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match=r"'spatialleiden' cannot"):
        calculate_niche(adata, flavor="spatialleiden", resolutions=0.5, rng=0, mask=keep)


@pytest.mark.parametrize(
    ("fn", "kwargs", "expected"),
    [
        pytest.param(
            calculate_niche_neighborhood,
            {"groups": "ct", "resolutions": [0.5, 1.0], "n_neighbors": 8},
            ["mine_res=0.5", "mine_res=1.0"],
            id="neighborhood",
        ),
        pytest.param(calculate_niche_utag, {"resolutions": 1.0, "n_neighbors": 8}, ["mine_res=1.0"], id="utag"),
        pytest.param(calculate_niche_cellcharter, {"n_clusters": 3, "distance": 2}, ["mine"], id="cellcharter"),
    ],
)
def test_key_added_names_the_columns(fn, kwargs, expected):
    "A stem for the flavors that write one column per resolution, exact for the one that writes one."
    adata = _tiny(n=60)
    fn(adata, rng=0, key_added="mine", **kwargs)
    assert [c for c in adata.obs.columns if c.startswith("mine")] == expected


def test_key_added_lets_two_runs_coexist():
    "Without it the second call silently overwrote the first."
    adata = _tiny(n=60)
    calculate_niche_utag(adata, resolutions=1.0, n_neighbors=8, rng=0, key_added="runA")
    calculate_niche_utag(adata, resolutions=1.0, n_neighbors=8, rng=1, key_added="runB")
    assert "runA_res=1.0" in adata.obs.columns
    assert "runB_res=1.0" in adata.obs.columns


def test_key_added_defaults_reproduce_the_derived_names():
    adata = _tiny(n=60)
    calculate_niche_utag(adata, resolutions=1.0, n_neighbors=8, rng=0)
    assert "utag_niche_res=1.0" in adata.obs.columns


# the stability sweep selecting the number of mixture components


def test_niche_cellcharter_auto_k_stores_per_k_diagnostics(dummy_adata2: AnnData):
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(dummy_adata2, distance=2, aggregation="mean", rng=0, n_clusters=(2, 3), max_runs=2)

    assert "cellcharter_niche" in dummy_adata2.obs.columns

    diagnostics = dummy_adata2.uns["cellcharter_niche_autok"]
    assert set(diagnostics) == {"table", "stability", "best_k", "n_runs", "converged"}

    table = diagnostics["table"]
    assert list(table.index) == [1, 2, 3, 4], "a (min, max) request gains a +-1 halo"
    assert table.loc[[1, 4], "stability_mean"].isna().all(), "the halo is fitted but not scored"

    interior = table.index[table["stability_mean"].notna()].tolist()
    assert interior == [2, 3]
    assert diagnostics["stability"].shape[0] == len(interior)
    assert table["nll"].notna().all()
    assert diagnostics["best_k"] in interior


def test_niche_cellcharter_auto_k_store_labels(dummy_adata2: AnnData):
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(
        dummy_adata2, distance=2, aggregation="mean", rng=0, n_clusters=(2, 3), max_runs=2, store_labels=True
    )

    for k in dummy_adata2.uns["cellcharter_niche_autok"]["table"].index:
        column = f"cellcharter_niche_k{k}"
        assert column in dummy_adata2.obs.columns
        assert dummy_adata2.obs[column].nunique() == k


def test_niche_cellcharter_auto_k_labels_go_through_postprocessing(dummy_adata2: AnnData):
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(
        dummy_adata2,
        distance=2,
        aggregation="mean",
        rng=0,
        n_clusters=(2, 3),
        max_runs=2,
        store_labels=True,
        min_niche_size=100,  # larger than the object, so every label is dropped
    )

    for k in dummy_adata2.uns["cellcharter_niche_autok"]["table"].index:
        assert (dummy_adata2.obs[f"cellcharter_niche_k{k}"] == "not_a_niche").all()


def test_niche_cellcharter_auto_k_diagnostics_roundtrip_h5ad(dummy_adata2: AnnData, tmp_path):
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(dummy_adata2, distance=2, aggregation="mean", rng=0, n_clusters=(2, 3), max_runs=2)

    path = tmp_path / "niche.h5ad"
    dummy_adata2.write_h5ad(path)
    restored = read_h5ad(path)

    original = dummy_adata2.uns["cellcharter_niche_autok"]
    reloaded = restored.uns["cellcharter_niche_autok"]
    assert set(reloaded) == set(original)
    assert_frame_equal(reloaded["table"], original["table"])
    assert reloaded["converged"] == original["converged"]
    assert reloaded["best_k"] == original["best_k"]
