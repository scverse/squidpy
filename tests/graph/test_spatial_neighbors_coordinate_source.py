from __future__ import annotations

import inspect
from copy import deepcopy
from unittest.mock import PropertyMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, concat, read_h5ad
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel
from spatialdata.transformations import Scale, set_transformation

from squidpy import gr
from squidpy.gr import _build
from squidpy.gr.neighbors import KNNBuilder

pytestmark = pytest.mark.filterwarnings("ignore:Calling `spatial_neighbors` is deprecated:FutureWarning")

INTERFACES = [
    pytest.param(gr.spatial_neighbors, {"coord_type": "generic", "n_neighs": 2}, id="legacy"),
    pytest.param(gr.spatial_neighbors_knn, {"n_neighs": 2}, id="knn"),
    pytest.param(gr.spatial_neighbors_radius, {"radius": 2.0}, id="radius"),
    pytest.param(gr.spatial_neighbors_delaunay, {}, id="delaunay"),
    pytest.param(gr.spatial_neighbors_grid, {"n_neighs": 4}, id="grid"),
    pytest.param(gr.spatial_neighbors_from_builder, {"builder": KNNBuilder(n_neighs=2)}, id="builder"),
]


@pytest.fixture
def prepared_sdata(sdata_mask_graph):
    sdata = sdata_mask_graph
    coords = _build.get_centroids(sdata.shapes["circles"], coordinate_system="global")[["x", "y"]].compute()
    sdata.tables["table"].obsm["spatial"] = coords.to_numpy()
    return sdata


def assert_graph_equal(actual, expected):
    for name in ("connectivities", "distances"):
        np.testing.assert_allclose(getattr(actual, name).toarray(), getattr(expected, name).toarray())


@pytest.mark.parametrize(("function", "kwargs"), INTERFACES)
@pytest.mark.parametrize("spatial_key", ["spatial", "precomputed_spatial"])
@pytest.mark.parametrize("copy", [True, False])
def test_table_coordinates_all_interfaces(prepared_sdata, mocker, function, kwargs, spatial_key, copy):
    table = prepared_sdata.tables["table"]
    coords = table.obsm.pop("spatial")
    table.obsm[spatial_key] = coords
    reference = AnnData(obs=table.obs.copy(), obsm={spatial_key: coords.copy()})
    expected = function(reference, spatial_key=spatial_key, library_key="region", copy=True, **kwargs)
    before_obs, before_var, before_uns = table.obs.copy(), table.var.copy(), deepcopy(table.uns)
    before_x = table.X.copy()
    centroid = mocker.patch.object(_build, "get_centroids", side_effect=AssertionError("Centroids must not be read"))

    result = function(
        prepared_sdata,
        table_key="table",
        # The value need not name an existing coordinate system in table mode.
        elements_to_coordinate_systems={"circles": "already_prepared"},
        coordinate_source="table",
        spatial_key=spatial_key,
        key_added="reused",
        copy=copy,
        **kwargs,
    )

    centroid.assert_not_called()
    assert table.obsm[spatial_key] is coords
    np.testing.assert_array_equal(table.X, before_x)
    pd.testing.assert_frame_equal(table.obs, before_obs)
    pd.testing.assert_frame_equal(table.var, before_var)
    if copy:
        assert_graph_equal(result, expected)
        assert not table.obsp
        assert table.uns == before_uns
    else:
        assert result is None
        np.testing.assert_allclose(table.obsp["reused_connectivities"].toarray(), expected.connectivities.toarray())
        np.testing.assert_allclose(table.obsp["reused_distances"].toarray(), expected.distances.toarray())
        assert table.uns["reused_neighbors"]["connectivities_key"] == "reused_connectivities"


@pytest.mark.parametrize(("function", "kwargs"), INTERFACES)
def test_default_elements_and_table_agree(prepared_sdata, mocker, function, kwargs):
    params = {"table_key": "table", "elements_to_coordinate_systems": {"circles": "global"}, "copy": True, **kwargs}
    centroid = mocker.spy(_build, "get_centroids")
    expected = function(prepared_sdata, **params)
    assert centroid.call_count == 1
    centroid.reset_mock()
    actual = function(prepared_sdata, coordinate_source="table", **params)
    centroid.assert_not_called()
    assert_graph_equal(actual, expected)


@pytest.mark.parametrize("copy", [True, False])
def test_table_mode_does_not_transform_coordinates(prepared_sdata, copy):
    table = prepared_sdata.tables["table"]
    coords = table.obsm["spatial"].copy()
    set_transformation(
        prepared_sdata.shapes["circles"], Scale([10.0, 10.0], axes=("x", "y")), to_coordinate_system="scaled"
    )
    params = {"table_key": "table", "elements_to_coordinate_systems": {"circles": "scaled"}, "radius": 2.0}
    expected = gr.spatial_neighbors_radius(AnnData(obsm={"spatial": coords}), radius=2.0, copy=True)
    actual = gr.spatial_neighbors_radius(prepared_sdata, coordinate_source="table", copy=copy, **params)
    if copy:
        assert_graph_equal(actual, expected)
    else:
        np.testing.assert_allclose(table.obsp["spatial_distances"].toarray(), expected.distances.toarray())
    np.testing.assert_array_equal(table.obsm["spatial"], coords)
    # The default still recomputes and transforms centroids, overwriting stored coordinates.
    gr.spatial_neighbors_radius(prepared_sdata, **params)
    np.testing.assert_allclose(table.obsm["spatial"], coords * 10.0)
    assert table.obsp["spatial_connectivities"].nnz < expected.connectivities.nnz


def test_copy_isolates_coordinates_even_for_a_mutating_builder(prepared_sdata, mocker):
    table = prepared_sdata.tables["table"]
    coords = table.obsm["spatial"]
    before = coords.copy()
    builder = KNNBuilder(n_neighs=2)
    original = builder.build_graph

    def mutate(received):
        assert not np.shares_memory(received, coords)
        received[:] += 100.0
        return original(received)

    mocker.patch.object(builder, "build_graph", side_effect=mutate)
    run = mocker.spy(_build, "_run_spatial_neighbors")
    gr.spatial_neighbors_from_builder(
        prepared_sdata,
        builder=builder,
        table_key="table",
        elements_to_coordinate_systems={"circles": "global"},
        coordinate_source="table",
        copy=True,
    )
    resolved = run.call_args.args[0]
    assert resolved.X is None
    assert not np.shares_memory(resolved.obsm["spatial"], coords)
    np.testing.assert_array_equal(coords, before)
    assert not table.obsp


def test_table_copy_preserves_input_on_builder_failure(prepared_sdata):
    table = prepared_sdata.tables["table"]
    coords = table.obsm["spatial"].copy()
    with pytest.raises(ValueError, match="n_neighbors"):
        gr.spatial_neighbors_knn(
            prepared_sdata,
            n_neighs=table.n_obs + 1,
            table_key="table",
            elements_to_coordinate_systems={"circles": "global"},
            coordinate_source="table",
            copy=True,
        )
    np.testing.assert_array_equal(table.obsm["spatial"], coords)
    assert not table.obsp


@pytest.mark.parametrize(("function", "kwargs"), INTERFACES)
@pytest.mark.parametrize("coordinate_source", ["auto", "", None, 42])
@pytest.mark.parametrize("input_type", ["anndata", "spatialdata"])
def test_invalid_source(prepared_sdata, function, kwargs, coordinate_source, input_type):
    data = prepared_sdata if input_type == "spatialdata" else prepared_sdata.tables["table"]
    with pytest.raises(ValueError, match="coordinate_source"):
        function(data, coordinate_source=coordinate_source, copy=True, **kwargs)


@pytest.mark.parametrize(("function", "kwargs"), INTERFACES)
def test_anndata_sources_use_existing_coordinates(prepared_sdata, function, kwargs):
    table = prepared_sdata.tables["table"]
    expected = function(table, copy=True, **kwargs)
    actual = function(table, coordinate_source="table", copy=True, **kwargs)
    assert_graph_equal(actual, expected)
    parameter = inspect.signature(function).parameters["coordinate_source"]
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "elements"


@pytest.mark.parametrize("copy", [True, False])
def test_missing_coordinates_do_not_fall_back(sdata_mask_graph, mocker, copy):
    centroid = mocker.patch.object(_build, "get_centroids")
    with pytest.raises(KeyError, match="precomputed_spatial"):
        gr.spatial_neighbors_knn(
            sdata_mask_graph,
            table_key="table",
            spatial_key="precomputed_spatial",
            elements_to_coordinate_systems={"circles": "global"},
            coordinate_source="table",
            copy=copy,
        )
    centroid.assert_not_called()
    assert not sdata_mask_graph.tables["table"].obsm
    assert not sdata_mask_graph.tables["table"].obsp


@pytest.mark.parametrize("dtype", [bool, complex, object, str])
def test_invalid_coordinate_dtype(prepared_sdata, dtype):
    table = prepared_sdata.tables["table"]
    table.obsm["spatial"] = table.obsm["spatial"].astype(dtype)
    with pytest.raises(TypeError, match="real numeric NumPy array"):
        gr.spatial_neighbors_knn(
            prepared_sdata,
            table_key="table",
            elements_to_coordinate_systems={"circles": "global"},
            coordinate_source="table",
            copy=True,
        )


@pytest.mark.parametrize("container", [pd.DataFrame, csr_matrix, da.from_array])
def test_non_numpy_coordinates(prepared_sdata, container):
    table = prepared_sdata.tables["table"]
    value = container(table.obsm["spatial"])
    if isinstance(value, pd.DataFrame):
        value.index = table.obs_names
    table.obsm["spatial"] = value
    with pytest.raises(TypeError, match="real numeric NumPy array"):
        gr.spatial_neighbors_knn(
            prepared_sdata,
            table_key="table",
            elements_to_coordinate_systems={"circles": "global"},
            coordinate_source="table",
            copy=True,
        )


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32, np.float64])
def test_real_numeric_dtypes(prepared_sdata, dtype):
    table = prepared_sdata.tables["table"]
    table.obsm["spatial"] = np.arange(table.n_obs * 2).reshape(table.n_obs, 2).astype(dtype)
    result = gr.spatial_neighbors_knn(
        prepared_sdata,
        n_neighs=2,
        table_key="table",
        elements_to_coordinate_systems={"circles": "global"},
        coordinate_source="table",
        copy=True,
    )
    assert result.connectivities.nnz == table.n_obs * 2


@pytest.mark.parametrize("trailing_shape", [(1,), (3,), (2, 1)])
def test_invalid_coordinate_shape(prepared_sdata, trailing_shape):
    table = prepared_sdata.tables["table"]
    table.obsm["spatial"] = np.ones((table.n_obs, *trailing_shape))
    with pytest.raises(ValueError, match="must have shape"):
        gr.spatial_neighbors_knn(
            prepared_sdata,
            table_key="table",
            elements_to_coordinate_systems={"circles": "global"},
            coordinate_source="table",
            copy=True,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates(prepared_sdata, bad_value):
    table = prepared_sdata.tables["table"]
    table.obsm["spatial"][0, 0] = bad_value
    with pytest.raises(ValueError, match="only finite coordinates"):
        gr.spatial_neighbors_knn(
            prepared_sdata,
            table_key="table",
            elements_to_coordinate_systems={"circles": "global"},
            coordinate_source="table",
            copy=True,
        )
    assert not table.obsp


def test_multiple_regions_preserve_row_order_and_separate_graphs(prepared_sdata, mocker):
    sdata = prepared_sdata
    sdata.shapes["other"] = sdata.shapes["circles"].copy()
    table = concat({"circles": sdata.tables["table"], "other": sdata.tables["table"]}, label="region", index_unique="-")
    sdata.tables["table"] = TableModel.parse(
        table, region=["circles", "other"], region_key="region", instance_key="instance_id"
    )
    before_obs, before_coords = table.obs.copy(), table.obsm["spatial"].copy()
    params = {
        "table_key": "table",
        "elements_to_coordinate_systems": {"circles": "global", "other": "global"},
        "n_neighs": 2,
    }
    expected = gr.spatial_neighbors_knn(sdata, copy=True, **params)
    centroid = mocker.patch.object(_build, "get_centroids", side_effect=AssertionError("Must reuse table coordinates"))
    result = gr.spatial_neighbors_knn(sdata, coordinate_source="table", copy=True, n_jobs=2, **params)
    centroid.assert_not_called()
    assert_graph_equal(result, expected)
    split = table.n_obs // 2
    assert result.connectivities[:split, split:].nnz == 0
    assert result.connectivities[split:, :split].nnz == 0
    pd.testing.assert_frame_equal(table.obs, before_obs)
    np.testing.assert_array_equal(table.obsm["spatial"], before_coords)


def test_reordered_observations_and_custom_table_key(prepared_sdata, mocker):
    table = prepared_sdata.tables.pop("table")[::-1].copy()
    prepared_sdata.tables["cells_table"] = table
    expected = gr.spatial_neighbors_knn(table, n_neighs=2, copy=True)
    centroid = mocker.patch.object(_build, "get_centroids", side_effect=AssertionError("Must reuse table coordinates"))
    actual = gr.spatial_neighbors_knn(
        prepared_sdata,
        n_neighs=2,
        table_key="cells_table",
        elements_to_coordinate_systems={"circles": "global"},
        coordinate_source="table",
        copy=True,
    )
    centroid.assert_not_called()
    assert_graph_equal(actual, expected)


@pytest.mark.parametrize("problem", ["missing_mapping", "missing_region", "unknown_instance", "region_order"])
def test_association_checks_remain_required(prepared_sdata, problem):
    sdata = prepared_sdata
    table = sdata.tables["table"]
    mapping = {"circles": "global"}
    if problem == "missing_mapping":
        mapping = None
    elif problem == "missing_region":
        mapping = {"polygon": "global"}
    elif problem == "unknown_instance":
        table.obs.loc[table.obs_names[0], "instance_id"] = 999_999
    else:
        sdata.shapes["other"] = sdata.shapes["circles"].copy()
        table = concat({"other": table, "circles": table}, label="region", index_unique="-")
        sdata.tables["table"] = TableModel.parse(
            table, region=["circles", "other"], region_key="region", instance_key="instance_id"
        )
        mapping = {"circles": "global", "other": "global"}
    with pytest.raises((AssertionError, ValueError, KeyError), match="elements|ordered|circles"):
        gr.spatial_neighbors_knn(
            sdata, table_key="table", elements_to_coordinate_systems=mapping, coordinate_source="table", copy=True
        )


def test_labels_keep_background_and_instance_checks(mocker):
    labels = Labels2DModel.parse(np.array([[0, 1, 1], [2, 2, 3], [0, 3, 3]], dtype=np.int32), dims=("y", "x"))
    table = TableModel.parse(
        AnnData(obs=pd.DataFrame({"region": pd.Categorical(["cells"] * 3), "instance_id": [1, 2, 3]})),
        region="cells",
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData(labels={"cells": labels}, tables={"table": table})
    params = {"table_key": "table", "elements_to_coordinate_systems": {"cells": "global"}, "n_neighs": 1}
    gr.spatial_neighbors_knn(sdata, **params)
    expected = gr.spatial_neighbors_knn(sdata, copy=True, **params)
    centroid = mocker.patch.object(_build, "get_centroids", side_effect=AssertionError("Must reuse coordinates"))
    actual = gr.spatial_neighbors_knn(sdata, coordinate_source="table", copy=True, **params)
    centroid.assert_not_called()
    assert_graph_equal(actual, expected)
    table.obs["instance_id"] = [0, 1, 2]
    with pytest.raises(ValueError, match="ordered"):
        gr.spatial_neighbors_knn(sdata, coordinate_source="table", copy=True, **params)


@pytest.mark.parametrize("fail", [True, False])
def test_backed_table_does_not_load_expression(prepared_sdata, tmp_path, fail):
    original = prepared_sdata.tables["table"]
    path = tmp_path / "table.h5ad"
    original.write_h5ad(path)
    table = read_h5ad(path, backed="r")
    prepared_sdata.tables["table"] = table
    params = {
        "table_key": "table",
        "elements_to_coordinate_systems": {"circles": "global"},
        "coordinate_source": "table",
        "copy": True,
        "n_neighs": table.n_obs + 1 if fail else 2,
    }
    # A graph needs observations and coordinates only; any access to X is a regression.
    with patch.object(
        AnnData, "X", new_callable=PropertyMock, side_effect=AssertionError("Expression must not be loaded")
    ):
        try:
            if fail:
                with pytest.raises(ValueError, match="n_neighbors"):
                    gr.spatial_neighbors_knn(prepared_sdata, **params)
            else:
                actual = gr.spatial_neighbors_knn(prepared_sdata, **params)
                expected = gr.spatial_neighbors_knn(original, n_neighs=2, library_key="region", copy=True)
                assert_graph_equal(actual, expected)
            assert table.isbacked
            np.testing.assert_array_equal(table.obsm["spatial"], original.obsm["spatial"])
            pd.testing.assert_frame_equal(table.obs, original.obs)
            assert table.uns == original.uns
            assert not table.obsp
        finally:
            table.file.close()
    on_disk = read_h5ad(path)
    np.testing.assert_array_equal(on_disk.X, original.X)
    np.testing.assert_array_equal(on_disk.obsm["spatial"], original.obsm["spatial"])
    assert not on_disk.obsp
