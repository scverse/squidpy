from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from anndata import AnnData

from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import _shuffle_group, extract_adata


class TestExtractAdata:
    def test_passthrough_anndata(self):
        adata = AnnData(np.zeros((3, 2)))
        assert extract_adata(adata) is adata

    def test_sdata_default_table_key(self, sdata_mask_graph: sd.SpatialData):
        result = extract_adata(sdata_mask_graph)
        assert isinstance(result, AnnData)
        assert result is sdata_mask_graph.tables["table"]

    def test_sdata_custom_table_key(self, sdata_mask_graph: sd.SpatialData):
        sdata_mask_graph["my_table"] = sdata_mask_graph.tables["table"].copy()
        result = extract_adata(sdata_mask_graph, table_key="my_table")
        assert isinstance(result, AnnData)

    def test_sdata_missing_table_key(self, sdata_mask_graph: sd.SpatialData):
        with pytest.raises(ValueError, match="not found in SpatialData"):
            extract_adata(sdata_mask_graph, table_key="nonexistent")

    def test_sdata_missing_default_table(self):
        sdata = sd.SpatialData()
        with pytest.raises(ValueError, match="'table' not found in SpatialData"):
            extract_adata(sdata)


class TestObspSpatialKey:
    def test_spatial_conn_none(self):
        assert Key.obsp.spatial_conn() == "spatial_connectivities"

    def test_spatial_dist_none(self):
        assert Key.obsp.spatial_dist() == "spatial_distances"

    def test_spatial_conn_custom_prefix(self):
        assert Key.obsp.spatial_conn("mykey") == "mykey_connectivities"

    def test_spatial_dist_custom_prefix(self):
        assert Key.obsp.spatial_dist("mykey") == "mykey_distances"

    def test_spatial_conn_idempotent(self):
        once = Key.obsp.spatial_conn("custom")
        twice = Key.obsp.spatial_conn(once)
        assert once == twice == "custom_connectivities"

    def test_spatial_dist_idempotent(self):
        once = Key.obsp.spatial_dist("custom")
        twice = Key.obsp.spatial_dist(once)
        assert once == twice == "custom_distances"

    def test_spatial_conn_already_suffixed(self):
        assert Key.obsp.spatial_conn("foo_connectivities") == "foo_connectivities"

    def test_spatial_dist_already_suffixed(self):
        assert Key.obsp.spatial_dist("foo_distances") == "foo_distances"

    def test_spatial_key_suffix_not_partial_match(self):
        assert Key.obsp.spatial_conn("my_conn") == "my_conn_connectivities"
        assert Key.obsp.spatial_dist("my_dist") == "my_dist_distances"


class TestUtils:
    @pytest.mark.parametrize("cluster_annotations_type", [int, str])
    @pytest.mark.parametrize("library_annotations_type", [int, str])
    @pytest.mark.parametrize("seed", [422, 422222])
    def test_shuffle_group(self, cluster_annotations_type: type, library_annotations_type: type, seed: int):
        size = 6
        rng = np.random.default_rng(seed)
        if isinstance(cluster_annotations_type, int):
            libraries = pd.Series(rng.choice([1, 2, 3, 4], size=(size,)), dtype="category")
        else:
            libraries = pd.Series(rng.choice(["a", "b", "c"], size=(size,)), dtype="category")

        if isinstance(library_annotations_type, int):
            cluster_annotations = rng.choice([1, 2, 3, 4], size=(size,))
        else:
            cluster_annotations = rng.choice(["X", "Y", "Z"], size=(size,))
        out = _shuffle_group(cluster_annotations, libraries, rng)
        for c in libraries.cat.categories:
            assert set(out[libraries == c]) == set(cluster_annotations[libraries == c])
