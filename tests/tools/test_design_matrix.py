import pytest

from typing import List

import numpy as np
import pytest
from squidpy.tl._exp_dist import exp_dist


@pytest.mark.parametrize("groups", ["NMP", ["NMP", "Spinal cord"], None])
@pytest.mark.parametrize("covariates")
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_exp_dist(
    groups: str | List[str] | None, batch_key: str | None, covariates: str | List[str] | None, metric: str
):
    """Check whether the design matrix is being built correctly by analyzing shape, type and normalization"""
    adata = sq.datasets.seqfish()

    df = exp_dist(
        adata=adata,
        cluster_key="celltype_mapped_refined",
        groups=groups,
        batch_key=batch_key,
        covariates=covariates,
        metric=metric,
        copy=True,
    )

    if isinstance(group, str):
        groups = [groups]
    if isinstance(covariates, str):
        covariates = [covariates]
    batch = int(batch_key is not None)
    min_length = len(groups)
    max_length = 1 + batch + len(groups) * 2 + len(covariates)

    # shape
    assert df.index == adata.obs.index
    assert min_length <= len(df.columns) <= max_length
    # type
    assert isinstance(df.iloc[:, 0], CategoricalDtype)
    assert isinstance(df["batch"], CategoricalDtype)
    # normalization
    assert np.max(df[groups].values) <= 1
    assert df[groups].value_counts()[1] <= 1
