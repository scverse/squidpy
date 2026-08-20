from __future__ import annotations

from anndata import AnnData
from pandas import Categorical, Series
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


def test_niche_copy_semantics(dummy_adata2: AnnData):
    "copy=True returns an annotated copy and leaves the input untouched; copy=False mutates and returns None."
    key = "nhood_niche_res=1.0"
    kwargs = {"groups": "celltype", "n_neighbors": 3, "resolutions": 1.0}

    out = calculate_niche_neighborhood(dummy_adata2, copy=True, **kwargs)
    assert key in out.obs.columns
    assert key not in dummy_adata2.obs.columns

    assert calculate_niche_neighborhood(dummy_adata2, **kwargs) is None
    assert (dummy_adata2.obs[key] == out.obs[key]).all()
