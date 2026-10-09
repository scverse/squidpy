from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import concat, read_h5ad
from spatialdata.models import TableModel

from squidpy import gr
from squidpy.gr.neighbors import KNNBuilder


@pytest.mark.parametrize("spatial_key", ["spatial", "custom_coordinates"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (gr.spatial_neighbors, {"coord_type": "generic", "n_neighs": 2}),
        (gr.spatial_neighbors_knn, {"n_neighs": 2}),
        (gr.spatial_neighbors_radius, {"radius": 2.0}),
        (gr.spatial_neighbors_delaunay, {}),
        (gr.spatial_neighbors_grid, {"n_neighs": 4}),
        (gr.spatial_neighbors_from_builder, {"builder": KNNBuilder(n_neighs=2)}),
    ],
)
def test_spatialdata_copy_preserves_input(sdata_mask_graph, function, kwargs, existing, spatial_key):
    sdata = sdata_mask_graph
    table = sdata.tables["table"]
    if existing:
        table.obsm[spatial_key] = np.full((table.n_obs, 2), -100.0)
    before = table.copy()
    params = {
        "table_key": "table",
        "elements_to_coordinate_systems": {"circles": "global"},
        "spatial_key": spatial_key,
        **kwargs,
    }

    result = function(sdata, copy=True, **params)

    assert sdata.tables["table"] is table
    assert set(table.obsm) == set(before.obsm)
    for key in before.obsm:
        np.testing.assert_array_equal(table.obsm[key], before.obsm[key])
    np.testing.assert_array_equal(table.X, before.X)
    pd.testing.assert_frame_equal(table.obs, before.obs)
    pd.testing.assert_frame_equal(table.var, before.var)
    assert table.uns == before.uns
    assert not table.obsp

    # The in-place mode still saves coordinates and the same graph.
    assert function(sdata, copy=False, **params) is None
    assert spatial_key in table.obsm
    assert not np.all(table.obsm[spatial_key] == -100.0)
    np.testing.assert_allclose(result.connectivities.toarray(), table.obsp["spatial_connectivities"].toarray())
    np.testing.assert_allclose(result.distances.toarray(), table.obsp["spatial_distances"].toarray())


@pytest.mark.parametrize("spatial_key", ["spatial", "custom_coordinates"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_spatialdata_copy_backed(sdata_mask_graph, tmp_path, spatial_key, existing, fail):
    sdata = sdata_mask_graph
    original = sdata.tables["table"]
    if existing:
        original.obsm[spatial_key] = np.full((original.n_obs, 2), -100.0)
    path = tmp_path / "table.h5ad"
    original.write_h5ad(path)
    table = read_h5ad(path, backed="r")
    sdata.tables["table"] = table
    params = {
        "table_key": "table",
        "elements_to_coordinate_systems": {"circles": "global"},
        "spatial_key": spatial_key,
        "n_neighs": table.n_obs + 1 if fail else 2,
    }
    try:
        if fail:
            with pytest.raises(ValueError, match="n_neighbors"):
                gr.spatial_neighbors_knn(sdata, copy=True, **params)
        else:
            result = gr.spatial_neighbors_knn(sdata, copy=True, **params)
            # Compare to the existing in-place path on a separate in-memory table.
            sdata.tables["table"] = original.copy()
            gr.spatial_neighbors_knn(sdata, copy=False, **params)
            expected = sdata.tables["table"].obsp
            np.testing.assert_allclose(result.connectivities.toarray(), expected["spatial_connectivities"].toarray())
            np.testing.assert_allclose(result.distances.toarray(), expected["spatial_distances"].toarray())

        assert table.isbacked
        assert set(table.obsm) == set(original.obsm)
        if existing:
            np.testing.assert_array_equal(table.obsm[spatial_key], original.obsm[spatial_key])
        np.testing.assert_array_equal(table.X[:], original.X)
        pd.testing.assert_frame_equal(table.obs, original.obs)
        assert table.uns == original.uns
        assert not table.obsp
    finally:
        table.file.close()
    on_disk = read_h5ad(path)
    assert set(on_disk.obsm) == set(original.obsm)
    if existing:
        np.testing.assert_array_equal(on_disk.obsm[spatial_key], original.obsm[spatial_key])
    assert not on_disk.obsp


@pytest.mark.parametrize("spatial_key", ["spatial", "custom_coordinates"])
@pytest.mark.parametrize("existing", [False, True])
def test_spatialdata_copy_preserves_input_on_error(sdata_mask_graph, existing, spatial_key):
    sdata = sdata_mask_graph
    table = sdata.tables["table"]
    if existing:
        table.obsm[spatial_key] = np.full((table.n_obs, 2), -100.0)
    with pytest.raises(ValueError, match="n_neighbors"):
        gr.spatial_neighbors_knn(
            sdata,
            table_key="table",
            elements_to_coordinate_systems={"circles": "global"},
            spatial_key=spatial_key,
            n_neighs=table.n_obs + 1,
            copy=True,
        )
    if existing:
        np.testing.assert_array_equal(table.obsm[spatial_key], -100.0)
    else:
        assert spatial_key not in table.obsm
    assert not table.obsp


def test_spatialdata_copy_multiple_libraries(sdata_mask_graph):
    sdata = sdata_mask_graph
    sdata.shapes["other"] = sdata.shapes["circles"].copy()
    table = concat(
        {"circles": sdata.tables["table"], "other": sdata.tables["table"]},
        label="region",
        index_unique="-",
    )
    sdata.tables["table"] = TableModel.parse(
        table, region=["circles", "other"], region_key="region", instance_key="instance_id"
    )
    before = table.obs.copy()
    params = {
        "table_key": "table",
        "elements_to_coordinate_systems": {"circles": "global", "other": "global"},
        "n_neighs": 2,
    }
    result = gr.spatial_neighbors_knn(sdata, copy=True, **params)
    pd.testing.assert_frame_equal(table.obs, before)
    assert not table.obsm
    assert not table.obsp
    split = table.n_obs // 2
    assert result.connectivities[:split, split:].nnz == 0
    assert result.connectivities[split:, :split].nnz == 0
    gr.spatial_neighbors_knn(sdata, copy=False, **params)
    np.testing.assert_allclose(result.connectivities.toarray(), table.obsp["spatial_connectivities"].toarray())
    np.testing.assert_allclose(result.distances.toarray(), table.obsp["spatial_distances"].toarray())
