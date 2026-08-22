from __future__ import annotations

import pytest
from anndata import AnnData, read_h5ad
from pandas import Categorical, Series
from pandas.testing import assert_frame_equal
from scanpy.pp import neighbors
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from spatialdata.models import TableModel

from squidpy.gr import (
    _niche,
    calculate_niche,
    calculate_niche_cellcharter,
    calculate_niche_neighborhood,
    spatial_neighbors_knn,
)

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
    calculate_niche(dummy_adata2, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.0, seed=0)
    niches = _assert_all_assigned(dummy_adata2, "nhood_niche_res=1.0")

    # a fixed seed gives reproducible niches
    calculate_niche(rerun, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.0, seed=0)
    assert (niches.to_numpy() == rerun.obs["nhood_niche_res=1.0"].to_numpy()).all()


def test_niche_calc_utag_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using utag approach works as intended for dummy_adata2."
    rerun = dummy_adata2.copy()
    calculate_niche(dummy_adata2, flavor="utag", n_neighbors=3, resolutions=1.0, seed=0)
    niches = _assert_all_assigned(dummy_adata2, "utag_niche_res=1.0")

    # a fixed seed gives reproducible niches
    calculate_niche(rerun, flavor="utag", n_neighbors=3, resolutions=1.0, seed=0)
    assert (niches.to_numpy() == rerun.obs["utag_niche_res=1.0"].to_numpy()).all()


def test_niche_calc_cellcharter_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using cellcharter approach works as intended for dummy_adata2."

    # since cellcharter throws an error if the object's expression matrix is not sparse, first ensure that is the case
    dummy_adata2.X = csr_matrix(dummy_adata2.X)

    calculate_niche(dummy_adata2, flavor="cellcharter", distance=2, aggregation="mean", seed=0)

    _assert_all_assigned(dummy_adata2, "cellcharter_niche")

    # cellcharter clusters with a GMM, not Leiden, so unlike the flavors above its labels are
    # fully determined by `seed` and can still be pinned. See scverse/squidpy#1260.
    expected_niches = Series(
        Categorical([6, 0, 1, 4, 9, 2, 3, 7, 5, 8], categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="cellcharter_niche",
    )
    assert (expected_niches == dummy_adata2.obs["cellcharter_niche"]).all()


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
        seed=0,
    )

    _assert_all_assigned(dummy_adata2, "spatialleiden_res=1.0")


# seed handling


def test_niche_cellcharter_seed_reproducible(dummy_adata2: AnnData):
    "The same `seed` must give the same niches, a different one must be free to differ."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    kwargs = {"distance": 2, "aggregation": "mean"}

    first = calculate_niche_cellcharter(dummy_adata2, seed=0, copy=True, **kwargs)
    second = calculate_niche_cellcharter(dummy_adata2, seed=0, copy=True, **kwargs)
    assert (first.obs["cellcharter_niche"] == second.obs["cellcharter_niche"]).all()

    # not a guarantee about the labels themselves, only that the seed is actually wired through
    other = calculate_niche_cellcharter(dummy_adata2, seed=1, copy=True, **kwargs)
    assert list(other.obs["cellcharter_niche"]) != list(first.obs["cellcharter_niche"])


def test_niche_cellcharter_seed_none_runs(dummy_adata2: AnnData):
    "`seed=None` (the default) must work: it means 'draw from OS entropy', not 'missing argument'."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(dummy_adata2, distance=2, aggregation="mean")
    assert "cellcharter_niche" in dummy_adata2.obs.columns


def test_niche_cellcharter_library_seeds_are_independent(dummy_adata2: AnnData, monkeypatch):
    "Each library must be fitted with its own seed, while the whole run stays reproducible."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    dummy_adata2.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5
    kwargs = {"distance": 2, "aggregation": "mean", "library_key": "batch", "n_components": 2}

    first = calculate_niche_cellcharter(dummy_adata2, seed=0, copy=True, **kwargs)
    second = calculate_niche_cellcharter(dummy_adata2, seed=0, copy=True, **kwargs)
    assert (first.obs["cellcharter_niche"] == second.obs["cellcharter_niche"]).all()

    # the clusterer is built once and reused for every library, so record what each fit
    # is actually seeded with
    seen: list[int] = []
    original = _niche.GaussianMixture

    def spy(*args, **kwargs):
        seen.append(kwargs["random_state"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_niche, "GaussianMixture", spy)
    calculate_niche_cellcharter(dummy_adata2, seed=0, copy=True, **kwargs)

    assert len(seen) == 2, "expected one mixture fit per library"
    assert seen[0] != seen[1], "libraries were fitted with the same seed"


def test_niche_cellcharter_seeds_the_pca(dummy_adata2: AnnData, monkeypatch):
    "PCA is part of the embedding, so `seed` must reach it too, not only the mixture fits."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    kwargs = {"distance": 2, "aggregation": "mean", "copy": True}

    seen: list[int] = []
    original = _niche.sc.tl.pca

    def spy(*args, **kwargs_):
        seen.append(kwargs_["random_state"])
        return original(*args, **kwargs_)

    monkeypatch.setattr(_niche.sc.tl, "pca", spy)
    for seed in (0, 0, 1):
        calculate_niche_cellcharter(dummy_adata2, seed=seed, **kwargs)

    assert seen[0] == seen[1], "the same seed must give the same PCA"
    assert seen[2] != seen[0], "a different seed must reach the PCA"


def test_niche_neighborhood_seeds_the_knn_graph(dummy_adata2: AnnData, monkeypatch):
    "`sc.pp.neighbors` is approximate above ~4096 obs, so `seed` has to control it as well."
    kwargs = {"groups": "celltype", "n_neighbors": 3, "resolutions": 1.0, "copy": True}

    seen: list[int] = []
    original = _niche.sc.pp.neighbors

    def spy(*args, **kwargs_):
        seen.append(kwargs_["random_state"])
        return original(*args, **kwargs_)

    monkeypatch.setattr(_niche.sc.pp, "neighbors", spy)
    for seed in (0, 0, 1):
        calculate_niche_neighborhood(dummy_adata2, seed=seed, **kwargs)

    assert seen[0] == seen[1], "the same seed must build the same graph"
    assert seen[2] != seen[0], "a different seed must reach the graph"


def test_niche_leiden_resolution_seeds_are_independent(dummy_adata2: AnnData, monkeypatch):
    "Each resolution is its own clustering run, so each must be seeded independently."
    seen: list[int] = []
    original = _niche.sc.tl.leiden

    def spy(*args, **kwargs):
        seen.append(kwargs["random_state"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_niche.sc.tl, "leiden", spy)
    calculate_niche_neighborhood(dummy_adata2, groups="celltype", n_neighbors=3, resolutions=[0.5, 1.0], seed=0)

    assert len(seen) == 2, "expected one leiden run per resolution"
    assert seen[0] != seen[1], "resolutions were clustered with the same seed"


def test_niche_leiden_library_seeds_are_independent(dummy_adata2: AnnData, monkeypatch):
    "The clusterer is reused across libraries, so it must not replay the same seeds for each one."
    dummy_adata2.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5
    seen: list[int] = []
    original = _niche.sc.tl.leiden

    def spy(*args, **kwargs):
        seen.append(kwargs["random_state"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_niche.sc.tl, "leiden", spy)
    calculate_niche_neighborhood(
        dummy_adata2, groups="celltype", n_neighbors=3, resolutions=[0.5, 1.0], seed=0, library_key="batch"
    )

    assert len(seen) == 4, "expected one leiden run per library per resolution"
    assert len(set(seen)) == 4, "libraries were clustered with the same seeds"


@pytest.mark.parametrize("aggregation", ["mean", "variance"])
def test_niche_cellcharter_accepts_a_dense_x(dummy_adata2: AnnData, aggregation: str):
    "A dense `adata.X` used to reach `scipy.sparse.hstack`/`.toarray()` with nothing sparse to work on."
    dense = dummy_adata2.copy()
    sparse = dummy_adata2.copy()
    sparse.X = csr_matrix(sparse.X)
    kwargs = {"distance": 2, "aggregation": aggregation, "n_components": 2, "seed": 0, "copy": True}

    from_dense = calculate_niche_cellcharter(dense, **kwargs)
    from_sparse = calculate_niche_cellcharter(sparse, **kwargs)
    assert (from_dense.obs["cellcharter_niche"] == from_sparse.obs["cellcharter_niche"]).all()


# selecting the number of clusters by stability


def test_niche_cellcharter_n_clusters_none_keeps_a_single_fit(dummy_adata2: AnnData):
    "`n_clusters=None` must behave exactly like today: one fit at `n_components`, no diagnostics."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    kwargs = {"distance": 2, "aggregation": "mean", "n_components": 4, "seed": 0, "copy": True}

    default = calculate_niche_cellcharter(dummy_adata2, **kwargs)
    explicit = calculate_niche_cellcharter(dummy_adata2, n_clusters=4, **kwargs)

    assert "cellcharter_niche_autok" not in default.uns
    assert (default.obs["cellcharter_niche"] == explicit.obs["cellcharter_niche"]).all()


def test_niche_cellcharter_auto_k_stores_per_k_diagnostics(dummy_adata2: AnnData):
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(dummy_adata2, distance=2, aggregation="mean", seed=0, n_clusters=(2, 3), max_runs=2)

    # the niche column stays independent of the selected K
    assert "cellcharter_niche" in dummy_adata2.obs.columns

    diagnostics = dummy_adata2.uns["cellcharter_niche_autok"]
    assert set(diagnostics) == {"table", "stability", "best_k", "n_runs", "converged"}

    table = diagnostics["table"]
    assert list(table.index) == [1, 2, 3, 4], "a (min, max) request gains a +-1 halo"
    assert table.loc[[1, 4], "stability_mean"].isna().all(), "the halo is fitted but not scored"

    # the scored K values are exactly the interior ones, readable off the table
    interior = table.index[table["stability_mean"].notna()].tolist()
    assert interior == [2, 3]
    assert diagnostics["stability"].shape[0] == len(interior)
    assert table["nll"].notna().all()
    assert diagnostics["best_k"] in interior


def test_niche_cellcharter_auto_k_store_labels(dummy_adata2: AnnData):
    "`store_labels` must emit one obs column per fitted K, usable as a `color=` key."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(
        dummy_adata2, distance=2, aggregation="mean", seed=0, n_clusters=(2, 3), max_runs=2, store_labels=True
    )

    for k in dummy_adata2.uns["cellcharter_niche_autok"]["table"].index:
        column = f"cellcharter_niche_k{k}"
        assert column in dummy_adata2.obs.columns
        assert dummy_adata2.obs[column].nunique() <= k


def test_niche_cellcharter_auto_k_labels_go_through_postprocessing(dummy_adata2: AnnData):
    "Per-K columns are returned from `cluster()`, so `min_niche_size` must apply to them too."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(
        dummy_adata2,
        distance=2,
        aggregation="mean",
        seed=0,
        n_clusters=(2, 3),
        max_runs=2,
        store_labels=True,
        min_niche_size=100,  # larger than the object, so every label is dropped
    )

    for k in dummy_adata2.uns["cellcharter_niche_autok"]["table"].index:
        assert (dummy_adata2.obs[f"cellcharter_niche_k{k}"] == "not_a_niche").all()


def test_niche_cellcharter_auto_k_is_keyed_by_library(dummy_adata2: AnnData):
    "Each library sweeps independently, so the diagnostics must be stored per library id."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    dummy_adata2.obs["batch"] = ["batch1"] * 5 + ["batch2"] * 5

    calculate_niche_cellcharter(
        dummy_adata2,
        distance=2,
        aggregation="mean",
        seed=0,
        n_clusters=(2, 3),
        max_runs=2,
        library_key="batch",
        store_labels=True,
    )

    diagnostics = dummy_adata2.uns["cellcharter_niche_autok"]
    assert sorted(diagnostics) == ["batch1", "batch2"]
    for per_library in diagnostics.values():
        table = per_library["table"]
        assert per_library["best_k"] in table.index[table["stability_mean"].notna()]

    # the per-library merge carries obs columns, including the per-K ones, with a lib prefix
    assert dummy_adata2.obs["cellcharter_niche"].str.startswith("lib=").all()
    assert dummy_adata2.obs["cellcharter_niche_k2"].str.startswith("lib=").all()


def test_niche_cellcharter_auto_k_diagnostics_roundtrip_h5ad(dummy_adata2: AnnData, tmp_path):
    "The diagnostics land in `uns`, so they have to survive being written out."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(dummy_adata2, distance=2, aggregation="mean", seed=0, n_clusters=(2, 3), max_runs=2)

    path = tmp_path / "niche.h5ad"
    dummy_adata2.write_h5ad(path)
    restored = read_h5ad(path)

    original = dummy_adata2.uns["cellcharter_niche_autok"]
    reloaded = restored.uns["cellcharter_niche_autok"]
    assert set(reloaded) == set(original)
    assert_frame_equal(reloaded["table"], original["table"])
    assert reloaded["converged"] == original["converged"]
    assert reloaded["best_k"] == original["best_k"]


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
        seed=0,
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
    assert (real.value_counts() >= 3).all()


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
