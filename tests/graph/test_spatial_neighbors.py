from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import spatialdata as sd
from anndata import AnnData
from numpy.random import default_rng
from scipy.sparse import isspmatrix_csr
from shapely import Point
from spatialdata.datasets import blobs

from squidpy._constants._pkg_constants import Key
from squidpy.gr import mask_graph, spatial_neighbors
from squidpy.gr._build import _build_connectivity


@pytest.fixture(params=[None, "spectral", "cosine"])
def transform(request):
    return request.param


class TestSpatialNeighbors:
    # ground-truth Delaunay distances
    _gt_ddist = np.array(
        [
            [0.0, 2.0, 0.0, 4.12310563],
            [2.0, 0.0, 6.32455532, 5.0],
            [0.0, 6.32455532, 0.0, 5.38516481],
            [4.12310563, 5.0, 5.38516481, 0.0],
        ]
    )
    # ground-truth Delaunay graph
    _gt_dgraph = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )

    @staticmethod
    def _adata_concat(adata1, adata2):
        adata2.uns["spatial"] = {"library2": None}  # needed to trigger grid building
        batch1, batch2 = "batch1", "batch2"
        adata_concat = ad.concat(
            {batch1: adata1, batch2: adata2},
            label="library_id",
            uns_merge="unique",
        )
        return adata_concat, batch1, batch2

    # TODO: add edge cases
    # TODO(giovp): test with reshuffling
    @pytest.mark.parametrize(("n_rings", "n_neigh", "sum_dist"), [(1, 6, 0), (2, 18, 30), (3, 36, 84)])
    def test_spatial_neighbors_visium(self, visium_adata: AnnData, n_rings: int, n_neigh: int, sum_dist: int):
        """
        check correctness of neighborhoods for visium coordinates
        """
        spatial_neighbors(visium_adata, n_rings=n_rings)
        assert visium_adata.obsp[Key.obsp.spatial_conn()][0].sum() == n_neigh
        assert visium_adata.uns[Key.uns.spatial_neighs()]["distances_key"] == Key.obsp.spatial_dist()
        if n_rings > 1:
            assert visium_adata.obsp[Key.obsp.spatial_dist()][0].sum() == sum_dist

        # test for library_key
        visium_adata2 = visium_adata.copy()
        adata_concat, batch1, batch2 = TestSpatialNeighbors._adata_concat(visium_adata, visium_adata2)
        spatial_neighbors(visium_adata2, n_rings=n_rings)
        spatial_neighbors(adata_concat, library_key="library_id", n_rings=n_rings)
        assert adata_concat.obsp[Key.obsp.spatial_conn()][0].sum() == n_neigh
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch1].obsp[Key.obsp.spatial_conn()].toarray(),
            visium_adata.obsp[Key.obsp.spatial_conn()].toarray(),
        )
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch2].obsp[Key.obsp.spatial_conn()].toarray(),
            visium_adata2.obsp[Key.obsp.spatial_conn()].toarray(),
        )

    @pytest.mark.parametrize(("n_rings", "n_neigh", "sum_neigh"), [(1, 4, 4), (2, 4, 12), (3, 4, 24)])
    def test_spatial_neighbors_squaregrid(self, adata_squaregrid: AnnData, n_rings: int, n_neigh: int, sum_neigh: int):
        """
        check correctness of neighborhoods for visium coordinates
        """
        adata = adata_squaregrid
        spatial_neighbors(adata, n_neighs=n_neigh, n_rings=n_rings, coord_type="grid")
        assert np.diff(adata.obsp[Key.obsp.spatial_conn()].indptr).max() == sum_neigh
        assert adata.uns[Key.uns.spatial_neighs()]["distances_key"] == Key.obsp.spatial_dist()

        # test for library_key
        adata2 = adata.copy()
        adata_concat, batch1, batch2 = TestSpatialNeighbors._adata_concat(adata, adata2)
        spatial_neighbors(adata2, n_neighs=n_neigh, n_rings=n_rings, coord_type="grid")
        spatial_neighbors(
            adata_concat,
            library_key="library_id",
            n_neighs=n_neigh,
            n_rings=n_rings,
            coord_type="grid",
        )
        assert np.diff(adata_concat.obsp[Key.obsp.spatial_conn()].indptr).max() == sum_neigh
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch1].obsp[Key.obsp.spatial_conn()].toarray(),
            adata.obsp[Key.obsp.spatial_conn()].toarray(),
        )
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch2].obsp[Key.obsp.spatial_conn()].toarray(),
            adata2.obsp[Key.obsp.spatial_conn()].toarray(),
        )

    @pytest.mark.parametrize("type_rings", [("grid", 1), ("grid", 6), ("generic", 1)])
    @pytest.mark.parametrize("set_diag", [False, True])
    def test_set_diag(self, adata_squaregrid: AnnData, set_diag: bool, type_rings: tuple[str, int]):
        typ, n_rings = type_rings
        spatial_neighbors(adata_squaregrid, coord_type=typ, set_diag=set_diag, n_rings=n_rings)
        G = adata_squaregrid.obsp[Key.obsp.spatial_conn()]
        D = adata_squaregrid.obsp[Key.obsp.spatial_dist()]

        np.testing.assert_array_equal(G.diagonal(), float(set_diag))
        np.testing.assert_array_equal(D.diagonal(), 0.0)

    def test_spatial_neighbors_non_visium(self, non_visium_adata: AnnData):
        """
        check correctness of neighborhoods for non-visium coordinates
        """
        correct_knn_graph = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )

        correct_radius_graph = np.array(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )

        spatial_neighbors(non_visium_adata, n_neighs=3, coord_type=None)
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].toarray()

        np.testing.assert_array_equal(spatial_graph, correct_knn_graph)

        spatial_neighbors(non_visium_adata, radius=5.0, coord_type=None)
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].toarray()

        np.testing.assert_array_equal(spatial_graph, correct_radius_graph)

        spatial_neighbors(non_visium_adata, delaunay=True, coord_type=None)
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].toarray()
        spatial_dist = non_visium_adata.obsp[Key.obsp.spatial_dist()].toarray()

        np.testing.assert_array_equal(spatial_graph, self._gt_dgraph)
        np.testing.assert_allclose(spatial_dist, self._gt_ddist)

        # test for library_key
        non_visium_adata2 = non_visium_adata.copy()
        adata_concat, batch1, batch2 = TestSpatialNeighbors._adata_concat(non_visium_adata, non_visium_adata2)
        spatial_neighbors(
            adata_concat,
            library_key="library_id",
            delaunay=True,
            coord_type=None,
        )
        spatial_neighbors(
            non_visium_adata2,
            delaunay=True,
            coord_type=None,
        )

        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch1].obsp[Key.obsp.spatial_conn()].toarray(),
            non_visium_adata.obsp[Key.obsp.spatial_conn()].toarray(),
        )
        np.testing.assert_array_equal(
            adata_concat[adata_concat.obs["library_id"] == batch2].obsp[Key.obsp.spatial_conn()].toarray(),
            non_visium_adata2.obsp[Key.obsp.spatial_conn()].toarray(),
        )

    @pytest.mark.parametrize("set_diag", [False, True])
    @pytest.mark.parametrize("radius", [(0, np.inf), (2.0, 4.0), (-42, -420), (100, 200)])
    def test_radius_min_max(self, non_visium_adata: AnnData, radius: tuple[float, float], set_diag: bool):
        gt_ddist = self._gt_ddist.copy()
        gt_dgraph = self._gt_dgraph.copy()

        minn, maxx = sorted(radius)
        mask = (gt_ddist < minn) | (gt_ddist > maxx)
        gt_ddist[mask] = 0.0
        gt_dgraph[mask] = 0.0
        if set_diag:
            ixs = np.arange(len(gt_dgraph))
            gt_dgraph[ixs, ixs] = 1.0

        spatial_neighbors(
            non_visium_adata,
            delaunay=True,
            coord_type=None,
            radius=radius,
            set_diag=set_diag,
        )
        spatial_dist = non_visium_adata.obsp[Key.obsp.spatial_dist()].toarray()
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].toarray()

        np.testing.assert_allclose(spatial_graph, gt_dgraph)
        np.testing.assert_allclose(spatial_dist, gt_ddist)

    def test_copy(self, non_visium_adata: AnnData):
        conn, dist = spatial_neighbors(non_visium_adata, delaunay=True, coord_type=None, copy=True)

        assert isspmatrix_csr(conn)
        assert isspmatrix_csr(dist)
        assert Key.obsp.spatial_conn() not in non_visium_adata.obsp
        assert Key.obsp.spatial_dist() not in non_visium_adata.obsp
        np.testing.assert_allclose(dist.toarray(), self._gt_ddist)
        np.testing.assert_allclose(conn.toarray(), self._gt_dgraph)

    @pytest.mark.parametrize("percentile", [99.0, 95.0])
    def test_percentile_filtering(self, adata_hne: AnnData, percentile: float, coord_type="generic"):
        conn, dist = spatial_neighbors(adata_hne, coord_type=coord_type, copy=True)
        conn_filtered, dist_filtered = spatial_neighbors(
            adata_hne, coord_type=coord_type, percentile=percentile, copy=True
        )

        # check whether there are less connectivities in the filtered graph and whether the max distance is smaller
        assert not ((conn != conn_filtered).nnz == 0)
        assert dist.max() > dist_filtered.max()

        Adj, Dst = _build_connectivity(adata_hne.obsm["spatial"], n_neighs=6, return_distance=True, set_diag=False)
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0
        Adj.eliminate_zeros()
        Dst.eliminate_zeros()

        assert dist_filtered.max() == Dst.max()

    @pytest.mark.parametrize("n_neighs", [5, 10, 20])
    def test_spatial_neighbors_generic(self, n_neighs: int):
        rng = np.random.default_rng(42)
        adata = ad.AnnData(shape=(512, 1))
        adata.obsm["spatial"] = rng.random(size=(512, 2))

        spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic", radius=None)
        graph = adata.obsp[Key.obsp.spatial_conn()]
        actual = np.array(graph.sum(axis=1)).flatten()

        np.testing.assert_array_equal(actual, n_neighs)

    @pytest.mark.parametrize("n_neighs", [2, 3, 4])
    def test_spatial_neighbors_sdata(self, n_neighs: int):
        TABLE_NAME = "table"
        REGION_NAME = "blobs_labels"
        sdata = blobs()

        spatial_neighbors(
            sdata,
            table_key=TABLE_NAME,
            elements_to_coordinate_systems={REGION_NAME: "global"},
            n_neighs=n_neighs,
            coord_type="generic",
        )
        graph = sdata.tables[TABLE_NAME].obsp[Key.obsp.spatial_conn()]
        actual = np.array(graph.sum(axis=1)).flatten()

        np.testing.assert_array_equal(actual, n_neighs)

        TABLE_NAME = "table_circles"
        REGION_NAME = "blobs_circles"
        rng = default_rng(42)
        X = rng.normal(size=(len(sdata.shapes[REGION_NAME]), 2))
        adata_circles = AnnData(X)
        adata_circles.obs["region"] = REGION_NAME
        adata_circles.obs["instance_id"] = sdata[REGION_NAME].index.values

        sdata[TABLE_NAME] = adata_circles
        sdata.set_table_annotates_spatialelement(
            TABLE_NAME,
            REGION_NAME,
            region_key="region",
            instance_key="instance_id",
        )

        spatial_neighbors(
            sdata,
            table_key=TABLE_NAME,
            elements_to_coordinate_systems={REGION_NAME: "global"},
            n_neighs=n_neighs,
            coord_type="generic",
        )
        graph = sdata.tables[TABLE_NAME].obsp[Key.obsp.spatial_conn()]
        actual = np.array(graph.sum(axis=1)).flatten()

        np.testing.assert_array_equal(actual, n_neighs)

    @pytest.mark.parametrize("key_added", ["mask", "mask2"])
    @pytest.mark.parametrize("delaunay", [True, False])
    def test_mask_graph(
        self,
        sdata_mask_graph: sd.SpatialData,
        key_added: str,
        delaunay: bool,
    ):
        sdata = sdata_mask_graph
        mask_polygon = sdata["polygon"].geometry[0]

        neighs_key = Key.uns.spatial_neighs("spatial")
        conns_key = Key.obsp.spatial_conn("spatial")
        dists_key = Key.obsp.spatial_dist("spatial")

        mask_conns_key = f"{key_added}_{conns_key}"
        mask_dists_key = f"{key_added}_{dists_key}"
        mask_neighs_key = f"{key_added}_{neighs_key}"

        spatial_neighbors(
            sdata,
            elements_to_coordinate_systems={"circles": "global"},
            table_key="table",
            delaunay=delaunay,
        )

        mask_graph(
            sdata,
            "table",
            mask_polygon,
            negative_mask=False,
            key_added=key_added,
        )

        graph_original = sdata["table"].obsp[conns_key].copy()
        graph_positive_filter = sdata["table"].obsp[mask_conns_key].copy()
        mask_graph(
            sdata,
            "table",
            mask_polygon,
            negative_mask=True,
            key_added=key_added,
        )

        graph_negative_filter = sdata["table"].obsp[mask_conns_key].copy()

        assert graph_original.toarray().sum() == sum(
            [
                graph_positive_filter.toarray().sum(),
                graph_negative_filter.toarray().sum(),
            ]
        )
        assert mask_conns_key in sdata["table"].obsp
        assert mask_dists_key in sdata["table"].obsp
        assert mask_neighs_key in sdata["table"].uns
        uns = sdata["table"].uns
        assert uns[mask_neighs_key]["distances_key"] == mask_dists_key
        assert uns[mask_neighs_key]["connectivities_key"] == mask_conns_key
        assert uns[mask_neighs_key]["params"]["negative_mask"]

        mask_polygon = sdata["polygon"]
        with pytest.raises(
            ValueError,
            match="`polygon_mask` should be of type `Polygon` or `MultiPolygon`, got",
        ):
            mask_graph(
                sdata,
                "table",
                Point((0, 1)),
                negative_mask=True,
                key_added=key_added,
            )

    @pytest.mark.parametrize("coord_type", ["generic", "grid"])
    @pytest.mark.parametrize("transform", ["spectral", "cosine"])
    def test_transform_properties(self, coord_type: str, transform: str):
        """Test that transforms produce expected matrix properties."""
        # Create test data
        if coord_type == "generic":
            rng = np.random.default_rng(42)
            adata = ad.AnnData(shape=(20, 5))
            adata.obsm["spatial"] = rng.random(size=(20, 2)) * 10
            kwargs = {"n_neighs": 5, "coord_type": "generic"}
        else:  # grid
            # Create a small visium-like dataset
            adata = ad.AnnData(shape=(9, 5))
            # 3x3 grid coordinates
            coords = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
            adata.obsm["spatial"] = coords
            adata.uns["spatial"] = {"library1": {"scalefactors": {}}}
            kwargs = {"n_neighs": 4, "coord_type": "grid"}

        # Test without transform (baseline)
        conn_orig, dist_orig = spatial_neighbors(adata, copy=True, transform=None, **kwargs)

        # Test with transform
        conn_trans, dist_trans = spatial_neighbors(adata, copy=True, transform=transform, **kwargs)

        # Basic assertions
        assert isspmatrix_csr(conn_trans)
        assert isspmatrix_csr(dist_trans)
        assert conn_trans.shape == conn_orig.shape
        assert dist_trans.shape == dist_orig.shape

        # Transform-specific assertions
        if transform == "spectral":
            # Spectral transform should normalize the matrix
            # Row sums should be close to 1.0 for non-isolated nodes
            row_sums = np.array(conn_trans.sum(axis=1)).flatten()
            non_zero_rows = row_sums > 0
            if np.any(non_zero_rows):
                # For connected components, normalized row sums should be reasonable
                assert np.all(row_sums[non_zero_rows] <= 2.0)  # Reasonable upper bound
                assert np.all(row_sums[non_zero_rows] >= 0.1)  # Reasonable lower bound
            print(conn_trans.toarray())
            print(conn_trans.T.toarray())
            # Matrix should still be symmetric
            assert np.allclose(conn_trans.toarray(), conn_trans.T.toarray(), rtol=1e-10)

        elif transform == "cosine":
            # Cosine similarity matrix should have values between 0 and 1
            assert conn_trans.min() >= 0.0
            assert conn_trans.max() <= 1.0

            # Diagonal should be 1.0 for non-zero rows
            diag = conn_trans.diagonal()
            row_sums = np.array(conn_orig.sum(axis=1)).flatten()
            non_zero_rows = row_sums > 0
            if np.any(non_zero_rows):
                assert np.allclose(diag[non_zero_rows], 1.0, rtol=1e-10)

        # Distance matrix should be unchanged by transforms
        np.testing.assert_allclose(dist_trans.toarray(), dist_orig.toarray(), rtol=1e-12)

    @pytest.mark.parametrize("transform", ["spectral", "cosine"])
    def test_transform_parameters_stored(self, non_visium_adata: AnnData, transform: str):
        """Test that transform parameters are correctly stored in uns."""
        spatial_neighbors(non_visium_adata, coord_type="generic", n_neighs=3, transform=transform)

        # Check that transform is stored in parameters
        params = non_visium_adata.uns[Key.uns.spatial_neighs()]["params"]
        assert "transform" in params
        assert params["transform"] == transform

    def test_transform_none_equivalent_to_no_transform(self, non_visium_adata: AnnData):
        """Test that transform=None produces the same result as no transform parameter."""
        adata1 = non_visium_adata.copy()
        adata2 = non_visium_adata.copy()

        # One with transform=None, one without transform parameter
        spatial_neighbors(adata1, coord_type="generic", n_neighs=3, transform=None)
        spatial_neighbors(adata2, coord_type="generic", n_neighs=3)

        # Results should be identical
        np.testing.assert_allclose(
            adata1.obsp[Key.obsp.spatial_conn()].toarray(), adata2.obsp[Key.obsp.spatial_conn()].toarray()
        )
        np.testing.assert_allclose(
            adata1.obsp[Key.obsp.spatial_dist()].toarray(), adata2.obsp[Key.obsp.spatial_dist()].toarray()
        )

    @pytest.mark.parametrize("transform", ["spectral", "cosine"])
    def test_transform_with_copy(self, non_visium_adata: AnnData, transform: str):
        """Test transform functionality with copy=True."""
        conn, dist = spatial_neighbors(
            non_visium_adata, coord_type="generic", n_neighs=3, transform=transform, copy=True
        )

        # Basic checks
        assert isspmatrix_csr(conn)
        assert isspmatrix_csr(dist)

        # Should not modify original data
        assert Key.obsp.spatial_conn() not in non_visium_adata.obsp
        assert Key.obsp.spatial_dist() not in non_visium_adata.obsp

        # Transform-specific checks
        if transform == "spectral":
            # Should be normalized
            row_sums = np.array(conn.sum(axis=1)).flatten()
            non_zero_rows = row_sums > 0
            if np.any(non_zero_rows):
                assert np.all(row_sums[non_zero_rows] <= 2.0)
        elif transform == "cosine":
            # Should have cosine similarity properties
            assert conn.min() >= 0.0
            assert conn.max() <= 1.0

    @pytest.mark.parametrize("transform", ["spectral", "cosine"])
    def test_transform_preserves_sparsity_pattern_structure(self, non_visium_adata: AnnData, transform: str):
        """Test that transforms preserve basic connectivity structure when appropriate."""
        # Get original connectivity
        conn_orig, _ = spatial_neighbors(non_visium_adata, coord_type="generic", n_neighs=3, transform=None, copy=True)

        # Get transformed connectivity
        conn_trans, _ = spatial_neighbors(
            non_visium_adata, coord_type="generic", n_neighs=3, transform=transform, copy=True
        )

        if transform == "spectral":
            # Spectral transform should preserve the sparsity pattern
            # (same non-zero locations)
            np.testing.assert_array_equal((conn_orig > 0).toarray(), (conn_trans > 0).toarray())
        elif transform == "cosine":
            # Cosine transform creates a dense similarity matrix, so sparsity pattern changes
            # But it should still be a valid similarity matrix
            assert conn_trans.nnz >= conn_orig.nnz  # Usually becomes denser

    def test_transform_invalid_option(self, non_visium_adata: AnnData):
        """Test that invalid transform options raise appropriate errors."""
        with pytest.raises((ValueError, NotImplementedError)):
            spatial_neighbors(non_visium_adata, coord_type="generic", n_neighs=3, transform="invalid_transform")

    @pytest.mark.parametrize("transform", ["spectral", "cosine"])
    def test_transform_with_different_coord_types(self, transform: str):
        """Test transforms work with different coordinate types."""
        # Test with generic coordinates
        rng = np.random.default_rng(42)
        adata_generic = ad.AnnData(shape=(16, 5))
        adata_generic.obsm["spatial"] = rng.random(size=(16, 2)) * 10

        spatial_neighbors(adata_generic, coord_type="generic", n_neighs=4, transform=transform)

        # Test with grid coordinates
        adata_grid = ad.AnnData(shape=(9, 5))
        coords = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
        adata_grid.obsm["spatial"] = coords
        adata_grid.uns["spatial"] = {"library1": {"scalefactors": {}}}

        spatial_neighbors(adata_grid, coord_type="grid", n_rings=1, transform=transform)

        # Both should complete without error and have transformed matrices
        for adata in [adata_generic, adata_grid]:
            conn = adata.obsp[Key.obsp.spatial_conn()]
            assert conn.nnz > 0  # Should have connections

            if transform == "spectral":
                # Check normalization properties
                row_sums = np.array(conn.sum(axis=1)).flatten()
                non_zero_rows = row_sums > 0
                if np.any(non_zero_rows):
                    assert np.all(row_sums[non_zero_rows] <= 2.0)
            elif transform == "cosine":
                # Check cosine properties
                assert conn.min() >= 0.0
                assert conn.max() <= 1.0
