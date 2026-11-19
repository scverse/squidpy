from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad

from squidpy._constants._pkg_constants import Key
from squidpy.gr import (
    SpatialGraphWarning,
    graph_summary,
    spatial_neighbors_delaunay,
    spatial_neighbors_knn,
    spatial_neighbors_radius,
)


def _lattice(n: int = 10, spacing: float = 10.0, **obs) -> AnnData:
    xy = np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1).reshape(-1, 2) * spacing
    adata = AnnData(np.zeros((len(xy), 1)), obsm={Key.obsm.spatial: xy.astype(float)})
    for k, v in obs.items():
        adata.obs[k] = pd.Categorical(v)
    return adata


class TestBuildWarning:
    def test_radius_below_spacing_warns(self):
        adata = _lattice()
        with pytest.warns(SpatialGraphWarning, match=r"100\.0% of cells have no neighbour") as record:
            spatial_neighbors_radius(adata, radius=5.0)
        message = str(record[0].message)
        assert "radius 5 vs median nearest-neighbour distance 10" in message
        assert "pixels vs micrometres" in message
        # the warning points at the caller, not at squidpy internals
        assert record[0].filename == __file__

    @pytest.mark.parametrize(
        ("func", "kwargs"),
        [
            (spatial_neighbors_radius, {"radius": 15.0}),
            (spatial_neighbors_knn, {"n_neighs": 4}),
            (spatial_neighbors_delaunay, {}),
            (spatial_neighbors_delaunay, {"radius": (0.0, 15.0)}),
        ],
        ids=["radius", "knn", "delaunay", "delaunay_pruned"],
    )
    def test_healthy_graph_does_not_warn(self, func, kwargs):
        adata = _lattice()
        with warnings.catch_warnings():
            warnings.simplefilter("error", SpatialGraphWarning)
            func(adata, **kwargs)

    def test_set_diag_does_not_hide_isolated_cells(self):
        with pytest.warns(SpatialGraphWarning):
            spatial_neighbors_radius(_lattice(), radius=5.0, set_diag=True)

    def test_knn_percentile_pruning_can_warn(self):
        xy = np.random.default_rng(0).uniform(0, 100, size=(200, 2))
        adata = AnnData(np.zeros((200, 1)), obsm={Key.obsm.spatial: xy})
        with pytest.warns(SpatialGraphWarning, match="mean degree"):
            spatial_neighbors_knn(adata, n_neighs=4, percentile=10)

    def test_only_bad_library_is_named(self):
        adata = _lattice(lib=["a"] * 50 + ["b"] * 50)
        adata.obsm[Key.obsm.spatial][50:] *= 100  # library "b" is 100x sparser
        with pytest.warns(SpatialGraphWarning, match="library 'b'") as record:
            spatial_neighbors_radius(adata, radius=15.0, library_key="lib")
        assert "library 'a'" not in str(record[0].message)

    def test_multiple_libraries_without_library_key_warns(self):
        adata = _lattice()
        adata.uns[Key.uns.spatial] = {"slide1": {}, "slide2": {}}
        with pytest.warns(SpatialGraphWarning, match="no `library_key` was given"):
            spatial_neighbors_radius(adata, radius=15.0)

    def test_warning_is_filterable_by_category(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=SpatialGraphWarning)
            spatial_neighbors_radius(_lattice(), radius=5.0)

    def test_uns_metadata_roundtrips(self, tmp_path):
        adata = _lattice(lib=["a"] * 50 + ["b"] * 50)
        spatial_neighbors_radius(adata, radius=15.0, library_key="lib")
        adata.write_h5ad(tmp_path / "adata.h5ad")

        uns = read_h5ad(tmp_path / "adata.h5ad").uns[Key.uns.spatial_neighs()]
        assert uns["params"]["builder"] == "RadiusBuilder"
        assert uns["params"]["library_key"] == "lib"
        assert uns["params"]["spatial_key"] == Key.obsm.spatial
        assert "percentile" not in uns["params"]
        assert uns["summary"]["n_obs"] == 100
        assert uns["summary"]["n_isolated"] == 0
        assert uns["summary"]["nn_dist_median"] == pytest.approx(10.0)


class TestGraphSummary:
    def test_lattice_values(self):
        adata = _lattice()
        spatial_neighbors_radius(adata, radius=10.5)
        df = graph_summary(adata)

        assert isinstance(df.index, pd.RangeIndex)
        (row,) = df.to_dict("records")
        assert row["library"] == "all"
        assert np.isnan(row["radius"])
        assert row["n_obs"] == 100
        # 64 interior cells with 4 neighbours, 32 border cells with 3, 4 corners with 2
        assert row["n_edges"] == 4 * 64 + 3 * 32 + 2 * 4
        assert (row["degree_min"], row["degree_max"]) == (2, 4)
        assert row["n_isolated"] == 0
        assert row["n_components"] == 1
        assert row["reciprocity"] == 1.0
        assert row["nn_dist_median"] == 10.0
        assert row["n_duplicate_coords"] == 0

    def test_directed_knn_in_degree_zero(self):
        # a tight cluster plus one distant cell: the distant cell points into the cluster,
        # but no cell picks it as a neighbour
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [50, 50]], dtype=float)
        adata = AnnData(np.zeros((5, 1)), obsm={Key.obsm.spatial: xy})
        spatial_neighbors_knn(adata, n_neighs=2)
        row = graph_summary(adata).iloc[0]

        assert row["n_isolated"] == 0
        assert row["n_in_degree_zero"] == 1
        assert row["reciprocity"] < 1

    def test_cross_library_edges(self):
        lib = ["a"] * 100 + ["b"] * 100
        adata = _lattice()  # both libraries share the same coordinates
        adata = AnnData(
            np.zeros((200, 1)),
            obs=pd.DataFrame({"lib": pd.Categorical(lib)}, index=[str(i) for i in range(200)]),
            obsm={Key.obsm.spatial: np.concatenate([adata.obsm[Key.obsm.spatial]] * 2)},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpatialGraphWarning)
            spatial_neighbors_radius(adata, radius=10.5)
        joined = graph_summary(adata, library_key="lib").set_index("library")
        assert joined.loc["all", "n_cross_library_edges"] > 0
        assert joined.loc["a", "n_duplicate_coords"] == 0
        assert graph_summary(adata).loc[0, "n_duplicate_coords"] == 200

        spatial_neighbors_radius(adata, radius=10.5, library_key="lib")
        per_lib = graph_summary(adata).set_index("library")  # library_key read from the stored params
        assert list(per_lib.index) == ["a", "b", "all"]
        assert per_lib.loc["all", "n_cross_library_edges"] == 0

    def test_radii_match_rebuilt_graphs(self):
        adata = _lattice(lib=["a"] * 50 + ["b"] * 50)
        spatial_neighbors_radius(adata, radius=10.5, library_key="lib")
        df = graph_summary(adata, radii=[5.0, 10.5, 15.0])
        swept = df[df["radius"].notna()].set_index(["library", "radius"])

        for r in (10.5, 15.0):
            spatial_neighbors_radius(adata, radius=r, library_key="lib", key_added="check")
            rebuilt = graph_summary(adata, connectivity_key="check_connectivities").set_index("library")
            for lib in ("a", "b", "all"):
                assert swept.loc[(lib, r), "degree_mean"] == pytest.approx(rebuilt.loc[lib, "degree_mean"])
        assert swept.loc[("all", 5.0), "frac_isolated"] == 1.0

    def test_recomputes_after_subsetting(self):
        adata = _lattice()
        spatial_neighbors_radius(adata, radius=10.5)
        sub = adata[:30].copy()
        assert graph_summary(sub).loc[0, "n_obs"] == 30
        assert sub.uns[Key.uns.spatial_neighs()]["summary"]["n_obs"] == 100  # stale by design

    def test_possible_library_columns_in_attrs(self):
        adata = _lattice(sample_id=["s1"] * 50 + ["s2"] * 50, cell_type=["t"] * 100)
        spatial_neighbors_radius(adata, radius=10.5)
        assert graph_summary(adata).attrs["possible_library_columns"] == ["sample_id"]

    def test_missing_graph_raises(self):
        with pytest.raises(KeyError, match="not found"):
            graph_summary(_lattice())
