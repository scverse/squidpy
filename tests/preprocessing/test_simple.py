from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
from spatialdata.datasets import blobs_annotating_element

import squidpy as sq


def _make_sdata(name: str, num_counts: int, count_value: int):
    assert num_counts <= 5, "num_counts must be less than 5"
    sdata_temp = blobs_annotating_element(name)
    m, _ = sdata_temp.tables["table"].shape
    n = m
    X = np.zeros((m, n))
    # random choice of row
    row_indices = np.random.choice(m, num_counts, replace=False)
    col_indices = np.random.choice(n, num_counts, replace=False)
    X[row_indices, col_indices] = count_value

    sdata_temp.tables["table"] = ad.AnnData(
        X=X,
        obs=sdata_temp.tables["table"].obs,
        var={"gene": ["gene" for _ in range(n)]},
        uns=sdata_temp.tables["table"].uns,
    )
    return sdata_temp


@pytest.mark.parametrize("name", ["blobs_labels", "blobs_circles", "blobs_points", "blobs_multiscale_labels"])
def test_filter_cells(name: str):
    filtered_cells = 3
    sdata = _make_sdata(name, num_counts=filtered_cells, count_value=100)
    num_cells = sdata.tables["table"].shape[0]
    adata_copy = sdata.tables["table"].copy()
    sc.pp.filter_cells(adata_copy, max_counts=50, inplace=True)
    sq.pp.filter_cells(sdata, max_counts=50, inplace=True, filter_labels=True)

    assert np.all(sdata.tables["table"].X == adata_copy.X), "Filtered cells are not the same as scanpy"
    assert np.all(sdata.tables["table"].obs["instance_id"] == adata_copy.obs["instance_id"]), (
        "Filtered cells are not the same as scanpy"
    )
    assert sdata.tables["table"].shape[0] == (num_cells - filtered_cells), (
        f"Expected {num_cells - filtered_cells} cells, got {sdata.tables['table'].shape[0]}"
    )

    if name == "blobs_labels":
        unique_labels = np.unique(adata_copy.obs["instance_id"])
        unique_labels_sdata = np.unique(sdata.labels["blobs_labels"].data.compute())
        assert set(unique_labels) == set(unique_labels_sdata).difference([0]), (
            f"Filtered labels {unique_labels} are not the same as scanpy {unique_labels_sdata}"
        )


def test_filter_cells_empty_fail():
    sdata = _make_sdata("blobs_labels", num_counts=5, count_value=200)
    with pytest.raises(ValueError, match="Filter results in empty table when filtering table `table`."):
        sq.pp.filter_cells(sdata, max_counts=100, inplace=True)
