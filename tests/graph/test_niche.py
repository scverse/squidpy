from __future__ import annotations

import pytest
from anndata import AnnData, read_h5ad
from pandas import Categorical, Series
from pandas.testing import assert_frame_equal
from scanpy.pp import neighbors
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from spatialdata.models import TableModel

from squidpy.gr import _niche, calculate_niche, calculate_niche_cellcharter, spatial_neighbors_knn

N_NEIGHBORS = 20
GROUPS = "celltype_mapped_refined"

# test if calculate_niche() gives appropriate output for dummy_adata2 for the different flavors


def test_niche_calc_nhood_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using neighborhood profile approach works as intended for dummy_adata2."
    calculate_niche(dummy_adata2, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.0)
    assert "nhood_niche_res=1.0" in dummy_adata2.obs.columns
    expected_niches = Series(
        ["0", "2", "0", "2", "1", "0", "0", "1", "0", "1"],
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="nhood_niche_res=1.0",
    )
    assert (expected_niches == dummy_adata2.obs["nhood_niche_res=1.0"]).all()


def test_niche_calc_utag_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using utag approach works as intended for dummy_adata2."
    calculate_niche(dummy_adata2, flavor="utag", n_neighbors=3, resolutions=1.0)
    assert "utag_niche_res=1.0" in dummy_adata2.obs.columns
    expected_niches = Series(
        Categorical(["1", "0", "0", "0", "1", "0", "0", "1", "1", "0"], categories=["0", "1"]),
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="utag_niche_res=1.0",
    )
    assert (expected_niches == dummy_adata2.obs["utag_niche_res=1.0"]).all()


def test_niche_calc_cellcharter_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using cellcharter approach works as intended for dummy_adata2."

    # since cellcharter throws an error if the object's expression matrix is not sparse, first ensure that is the case
    dummy_adata2.X = csr_matrix(dummy_adata2.X)

    calculate_niche(dummy_adata2, flavor="cellcharter", distance=2, aggregation="mean", seed=0)

    assert "cellcharter_niche" in dummy_adata2.obs.columns

    expected_niches = Series(
        Categorical([2, 6, 4, 9, 3, 0, 1, 5, 8, 7], categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="cellcharter_niche",
    )
    assert (expected_niches == dummy_adata2.obs["cellcharter_niche"]).all()


def test_niche_calc_spatialleiden_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation using spatialleiden approach works as intended for dummy_adata2."

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

    assert "spatialleiden_res=1.0" in dummy_adata2.obs.columns
    expected_niches = Series(
        Categorical([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], categories=[0, 1, 2]),
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="spatialleiden_res=1.0",
    )

    assert (expected_niches == dummy_adata2.obs["spatialleiden_res=1.0"]).all()


# seed handling


def test_niche_random_state_removed(dummy_adata2: AnnData):
    "`random_state` was replaced by `seed`; passing it must fail with a pointer to `seed`."
    with pytest.raises(TypeError, match=r"'random_state' is no longer supported, provide 'seed' instead"):
        calculate_niche(dummy_adata2, flavor="cellcharter", distance=2, aggregation="mean", random_state=0)


def test_niche_cellcharter_seed_reproducible(dummy_adata2: AnnData):
    "The same `seed` must give the same niches, a different one must be free to differ."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    kwargs = {"distance": 2, "aggregation": "mean"}

    first = calculate_niche_cellcharter(dummy_adata2, seed=0, inplace=False, **kwargs)
    second = calculate_niche_cellcharter(dummy_adata2, seed=0, inplace=False, **kwargs)
    assert (first.obs["cellcharter_niche"] == second.obs["cellcharter_niche"]).all()

    # not a guarantee about the labels themselves, only that the seed is actually wired through
    other = calculate_niche_cellcharter(dummy_adata2, seed=1, inplace=False, **kwargs)
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

    first = calculate_niche_cellcharter(dummy_adata2, seed=0, inplace=False, **kwargs)
    second = calculate_niche_cellcharter(dummy_adata2, seed=0, inplace=False, **kwargs)
    assert (first.obs["cellcharter_niche"] == second.obs["cellcharter_niche"]).all()

    # the clusterer is built once and reused for every library, so record what each fit
    # is actually seeded with
    seen: list[int] = []
    original = _niche.GaussianMixture

    def spy(*args, **kwargs):
        seen.append(kwargs["random_state"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_niche, "GaussianMixture", spy)
    calculate_niche_cellcharter(dummy_adata2, seed=0, inplace=False, **kwargs)

    assert len(seen) == 2, "expected one mixture fit per library"
    assert seen[0] != seen[1], "libraries were fitted with the same seed"


# selecting the number of clusters by stability


def test_niche_cellcharter_n_clusters_none_keeps_a_single_fit(dummy_adata2: AnnData):
    "`n_clusters=None` must behave exactly like today: one fit at `n_components`, no diagnostics."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    kwargs = {"distance": 2, "aggregation": "mean", "n_components": 4, "seed": 0, "inplace": False}

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
    assert diagnostics["n_clusters"] == [1, 2, 3, 4], "a (min, max) request gains a +-1 halo"
    assert diagnostics["interior"] == [2, 3]
    assert diagnostics["best_k"] in diagnostics["interior"]
    assert diagnostics["stability"].shape[0] == len(diagnostics["interior"])

    table = diagnostics["table"]
    assert list(table.index) == diagnostics["n_clusters"]
    assert table.loc[[1, 4], "stability_mean"].isna().all(), "the halo is fitted but not scored"
    assert table.loc[diagnostics["interior"], "stability_mean"].notna().all()
    assert table["nll"].notna().all()
    assert table.index[table["is_best"]].tolist() == [diagnostics["best_k"]]


def test_niche_cellcharter_auto_k_store_labels(dummy_adata2: AnnData):
    "`store_labels` must emit one obs column per fitted K, usable as a `color=` key."
    dummy_adata2.X = csr_matrix(dummy_adata2.X)
    calculate_niche_cellcharter(
        dummy_adata2, distance=2, aggregation="mean", seed=0, n_clusters=(2, 3), max_runs=2, store_labels=True
    )

    for k in dummy_adata2.uns["cellcharter_niche_autok"]["n_clusters"]:
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

    for k in dummy_adata2.uns["cellcharter_niche_autok"]["n_clusters"]:
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
        assert per_library["best_k"] in per_library["interior"]

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
    assert reloaded["best_k"] == original["best_k"]
    assert_frame_equal(reloaded["table"], original["table"])


# more special test cases


def test_niche_calc_library_key_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation when library_key is supplied works as intended for dummy_adata2."

    # add library_key information in dummy_adata
    dummy_adata2.obs["batch"] = [
        "batch1",
        "batch1",
        "batch1",
        "batch1",
        "batch1",
        "batch2",
        "batch2",
        "batch2",
        "batch2",
        "batch2",
    ]

    calculate_niche(
        dummy_adata2, flavor="neighborhood", groups="celltype", n_neighbors=3, resolutions=1.5, library_key="batch"
    )

    assert "nhood_niche_res=1.5" in dummy_adata2.obs.columns

    expected_niches = Series(
        [
            "lib=batch1_1",
            "lib=batch1_0",
            "lib=batch1_2",
            "lib=batch1_0",
            "lib=batch1_1",
            "lib=batch2_2",
            "lib=batch2_0",
            "lib=batch2_1",
            "lib=batch2_0",
            "lib=batch2_1",
        ],
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="nhood_niche_res=1.5",
        dtype=str,
    )

    assert (expected_niches == dummy_adata2.obs["nhood_niche_res=1.5"]).all()


def test_niche_calc_spatialleiden_library_key_dummy_adata(dummy_adata2: AnnData):
    "Check whether niche calculation for spatialleiden works as intended for dummy_adata2 when library_key is supplied."

    # need the latent_connectivities_key, meaning have to run the graph construction
    neighbors(dummy_adata2, n_neighbors=3, use_rep="X")

    # add library_key information in dummy_adata
    dummy_adata2.obs["batch"] = [
        "batch1",
        "batch1",
        "batch1",
        "batch1",
        "batch1",
        "batch2",
        "batch2",
        "batch2",
        "batch2",
        "batch2",
    ]

    calculate_niche(
        dummy_adata2,
        flavor="spatialleiden",
        latent_connectivities_key="connectivities",
        spatial_connectivities_key="spatial_connectivities",
        resolutions=1.0,
        library_key="batch",
        seed=0,
    )

    assert "spatialleiden_res=1.0" in dummy_adata2.obs.columns

    expected_niches = Series(
        [
            "lib=batch1_1",
            "lib=batch1_0",
            "lib=batch1_0",
            "lib=batch1_1",
            "lib=batch1_0",
            "lib=batch2_1",
            "lib=batch2_1",
            "lib=batch2_0",
            "lib=batch2_0",
            "lib=batch2_0",
        ],
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="spatialleiden_res=1.0",
        dtype=str,
    )

    assert (expected_niches == dummy_adata2.obs["spatialleiden_res=1.0"]).all()


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
    assert "nhood_niche_res=1.0" in dummy_adata2.obs.columns
    expected_niches = Series(
        ["not_a_niche", "not_a_niche", "0", "not_a_niche", "1", "0", "0", "1", "0", "1"],
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="nhood_niche_res=1.0",
    )
    assert (expected_niches == dummy_adata2.obs["nhood_niche_res=1.0"]).all()


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

    assert "nhood_niche_res_1.0" in sdata["adata"].obs.columns

    expected_niches = Series(
        ["0", "2", "0", "2", "1", "0", "0", "1", "0", "1"],
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="nhood_niche_res_1.0",
        dtype=str,
    )

    assert (expected_niches == sdata["adata"].obs["nhood_niche_res_1.0"]).all()


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
