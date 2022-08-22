from __future__ import annotations

from typing import List
import pytest

from anndata import AnnData


@pytest.mark.parametrize("covariates", ["array_col", None])
@pytest.mark.parametrize("groups", ["Cortex_1", ["Cortex_1", "Cortex_2", "Striatum"], None])
@pytest.mark.parametrize("batch_key", ["library_id", None])
def test_exp_dist(
    adata_hne_concat: AnnData,
    groups: str | List[str] | None,
    batch_key: str | None,
    covariates: str | List[str] | None,
    cluster_key: str = "cluster",
):
    # TODO(LLehren): write tests for above parametrization
    # check for dataframe shape, types of columns etc.
    # general behaviour with different arguments passed
    # catch correct warnings
    assert True
