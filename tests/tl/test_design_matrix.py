from typing import List

import numpy as np
import pytest
from anndata import AnnData
from pandas.core.dtypes.common import is_categorical_dtype

import squidpy as sq


class TestDesignMatrix:
    # @pytest.mark.parametrize("groups", ["NMP", ["NMP", "Spinal cord"], None])
    # @pytest.mark.parametrize("covariates", None)
    # @pytest.mark.parametrize("batch_key", None)
    # @pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
    def test_exp_dist(
        self,
        adata_mibitof: AnnData,
        # groups: str | List[str] | None,
        # batch_key: str | None,
        # covariates: str | List[str] | None,
        # metric: str,
    ):
        """Check whether the design matrix is being built correctly by analyzing shape, type and normalization"""
        adata = adata_mibitof
        assert isinstance(adata, AnnData)
        # df = exp_dist(
        #     adata=adata,
        #     cluster_key="celltype_mapped_refined",
        #     groups=groups,
        #     batch_key=batch_key,
        #     covariates=covariates,
        #     metric=metric,
        #     copy=True,
        # )

        # if isinstance(groups, str):
        #     groups = [groups]
        # if isinstance(covariates, str):
        #     covariates = [covariates]
        # batch = int(batch_key is not None)
        # min_length = len(groups)
        # max_length = 1 + batch + len(groups) * 2 + len(covariates)

        # # shape
        # assert df.index == adata.obs.index
        # assert min_length <= len(df.columns) <= max_length
        # # check type
        # assert is_categorical_dtype(df["batch"])
        # # normalization
        # assert np.max(df[groups].values) <= 1
        # assert df[groups].value_counts()[1] <= 1
