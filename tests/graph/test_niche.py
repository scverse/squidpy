from __future__ import annotations

from anndata import AnnData
from pandas import Categorical, DataFrame, Series
from scanpy.pp import neighbors
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from spatialdata.models import TableModel

from squidpy.gr import calculate_niche, spatial_neighbors_knn
from squidpy.gr._niche import NhoodProfileEmbedder, UtagEmbedder

SPATIAL_CONNECTIVITIES_KEY = "spatial_connectivities"
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

    calculate_niche(dummy_adata2, flavor="cellcharter", distance=2, aggregation="mean", random_state=0)

    assert "cellcharter_niche" in dummy_adata2.obs.columns

    expected_niches = Series(
        Categorical([8, 4, 0, 7, 2, 9, 5, 6, 1, 3], categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
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
    )

    assert "spatialleiden_res=1.0" in dummy_adata2.obs.columns
    expected_niches = Series(
        Categorical([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], categories=[0, 1, 2]),
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        name="spatialleiden_res=1.0",
    )

    assert (expected_niches == dummy_adata2.obs["spatialleiden_res=1.0"]).all()


# test if calculate_niche() gives appropriate output with library_key and sdata format too


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

    assert "nhood_niche_res=1.5_renamed" in dummy_adata2.obs.columns

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

    assert (expected_niches == dummy_adata2.obs["nhood_niche_res=1.5_renamed"]).all()


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
    )

    assert "spatialleiden_res=1.0_renamed" in dummy_adata2.obs.columns

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

    assert (expected_niches == dummy_adata2.obs["spatialleiden_res=1.0_renamed"]).all()


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


def test_calculate_neighborhood_profile(dummy_adata2: AnnData):
    "calculate_neighborhood_profile function needs to be tested, as it is at the base of the functionality of neighborhood flavor"

    # since now this has become a method of an embedder class, will initialize the class and call the method for that class
    embedder1 = NhoodProfileEmbedder(
        groups="celltype",
        spatial_connectivities_key="spatial_connectivities",
        scale=True,  # doesnt make a difference in this test function
        distance=1,  # doesnt make a difference in this test function
        abs_nhood=True,
        n_hop_weights=None,  # doesnt make a difference in this test function
    )

    embedder2 = NhoodProfileEmbedder(
        groups="celltype",
        spatial_connectivities_key="spatial_connectivities",
        scale=True,  # doesnt make a difference in this test function
        distance=1,  # doesnt make a difference in this test function
        abs_nhood=False,
        n_hop_weights=None,  # doesnt make a difference in this test function
    )

    matrix = dummy_adata2.obsp["spatial_connectivities"].tocoo()
    nhood_profile = embedder1._calculate_neighborhood_profile(dummy_adata2, matrix)
    relative_nhood_profile = embedder2._calculate_neighborhood_profile(dummy_adata2, matrix)

    # nhood_profile and relative_nhood_profile should have same entries as this manually determined version for dummy_adata2
    expected_nhood_profile = DataFrame(
        {
            0: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            1: [1, 0, 0, 0, 3, 1, 1, 3, 0, 1],
            2: [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        },
        index=list("abcdefghij"),
        dtype="float",
    )
    total_neighs = expected_nhood_profile.sum(axis=1)
    expected_relative_nhood_profile = expected_nhood_profile.div(total_neighs, axis=0)
    expected_relative_nhood_profile = expected_relative_nhood_profile.fillna(0.0)

    # compare
    assert (nhood_profile.values == expected_nhood_profile.values).all()
    assert (relative_nhood_profile.values == expected_relative_nhood_profile.values).all()


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
    niches = adata_seqfish.obs["nhood_niche_res=0.1_size_filter"]

    # assert no nans, more niche labels than non-niche labels, and at least 100 obs per niche
    assert niches.isna().sum() == 0
    assert len(niches[niches != "not_a_niche"]) > len(niches[niches == "not_a_niche"])
    for label in niches.unique():
        if label != "not_a_niche":
            assert len(niches[niches == label]) >= 100

    # get obs x neighbor matrix from sparse matrix
    matrix = adata_seqfish.obsp[SPATIAL_CONNECTIVITIES_KEY].tocoo()

    # since now this has become a method of an embedder class, will initialize the class and call the method for that class
    embedder1 = NhoodProfileEmbedder(
        groups=GROUPS,
        spatial_connectivities_key="spatial_connectivities",
        scale=True,  # doesnt make a difference in this test function
        distance=1,  # doesnt make a difference in this test function
        abs_nhood=True,
        n_hop_weights=None,  # doesnt make a difference in this test function
    )

    embedder2 = NhoodProfileEmbedder(
        groups=GROUPS,
        spatial_connectivities_key="spatial_connectivities",
        scale=True,  # doesnt make a difference in this test function
        distance=1,  # doesnt make a difference in this test function
        abs_nhood=False,
        n_hop_weights=None,  # doesnt make a difference in this test function
    )

    # get obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood
    abs_nhood_profile = embedder1._calculate_neighborhood_profile(adata_seqfish, matrix)
    rel_nhood_profile = embedder2._calculate_neighborhood_profile(adata_seqfish, matrix)
    # assert shape obs x groups
    assert rel_nhood_profile.shape == (
        adata_seqfish.n_obs,
        len(adata_seqfish.obs[GROUPS].cat.categories),
    )
    assert abs_nhood_profile.shape == rel_nhood_profile.shape
    # normalization
    assert int(rel_nhood_profile.sum(axis=1).sum()) == adata_seqfish.n_obs
    assert round(rel_nhood_profile.sum(axis=1).max(), 2) == 1
    # maximum amount of categories equals n_neighbors
    assert abs_nhood_profile.sum(axis=1).max() == N_NEIGHBORS


def test_niche_calc_utag(adata_seqfish: AnnData):
    """Check whether niche calculation using UTAG approach works as intended."""
    spatial_neighbors_knn(adata_seqfish, n_neighs=N_NEIGHBORS)
    calculate_niche(adata_seqfish, flavor="utag", n_neighbors=N_NEIGHBORS, resolutions=[0.1, 1.0])

    niches = adata_seqfish.obs["utag_niche_res=1.0"]
    niches_low_res = adata_seqfish.obs["utag_niche_res=0.1"]

    assert niches.isna().sum() == 0
    assert niches.nunique() > niches_low_res.nunique()

    embedder = UtagEmbedder(spatial_connectivities_key=SPATIAL_CONNECTIVITIES_KEY)
    embedding1 = embedder.get_embedding(adata_seqfish)  # this is pca of the utag feature matrix

    # assert shape[0] of matrices same
    assert embedding1.shape[0] == adata_seqfish.X.shape[0]

    spatial_neighbors_knn(adata_seqfish, n_neighs=40)
    embedding2 = embedder.get_embedding(adata_seqfish)  # this is pca of the utag feature matrix

    # matrix products (and hence pca) should differ when using different amount of neighbors
    assert (embedding1 != embedding2).all()
