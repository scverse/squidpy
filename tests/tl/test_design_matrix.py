from __future__ import annotations

import pytest
from typing import List
import squidpy as sq
from anndata import AnnData

class TestDesignMatrix:
    @pytest.mark.parametrize("groups", ["Endothelial", ["Endothelial", "Epitheliel", "Fibroblast"]])
    @pytest.mark.parametrize("covariates", ["category", ["category", "cell_size"], None])
    @pytest.mark.parametrize("library_key", ["library_id", None])
    def test_design_matrix(adata_mibitof: AnnData,
                        groups: str | List[str],
                        library_key: str | None,
                        covariates: str | List[str] | None = None):
        df = sq.tl._exp_dist(adata_mibitof, groups=groups, cluster_key="Cluster", library_key=library_key, covariates=covariates, copy=True)

        if isinstance(groups, str):
            groups = [groups]
        elif isinstance(groups, None):
            groups = [None]
        if isinstance(covariates, str) or isinstance(covariates, None):
            n_covariates = 1
        else:
            n_covariates = len(covariates)
        if isinstance(library_key, str):
            slides = True
        
        assert len(df) == adata_mibitof.n_obs #correct amount of rows
        assert len(df.columns) == 1 + 2*len(groups) + int(slides) + n_covariates #correct amount of columns

        for anchor in groups:
            assert min(df[anchor]) == 0 and max(df[anchor]) <= 1 #correct normalized range
        assert df.Cluster.dtype.name == "category" # correct dtype

        if not isinstance(library_key, None):
            assert df[library_key].nunique() > 1 # more than one slide

        for anchor in groups:
            anchor_ids_adata = adata_mibitof.obs.index[adata_mibitof.obs["Cluster"] == anchor].tolist()
            anchor_ids_design_matrix = df.obs.index[df.obs["Cluster"] == anchor].tolist()
            zero_dist_ids = df.obs.index[df.obs["f{anchor}_raw"] == 0.0].tolist()
            nan_ids = df.obs.index[df.obs[anchor].isna()].tolist()

            assert anchor_ids_adata == anchor_ids_design_matrix # anchor point indices match before and after
            assert anchor_ids_adata == zero_dist_ids # anchor point indices have zero values in anchor_raw column
            assert nan_ids <= zero_dist_ids # zero value indices must be subset of indices with NaN values in anchor column

        return
