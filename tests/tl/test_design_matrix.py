from __future__ import annotations

from typing import List

import pytest
from anndata import AnnData

from squidpy.tl import exp_dist


class TestDesignMatrix:
    @pytest.mark.parametrize("groups", ["Endothelial", ["Endothelial", "Epitheliel", "Fibroblast"]])
    @pytest.mark.parametrize("cluster_key", ["Cluster"])
    @pytest.mark.parametrize("covariates", ["category", ["category", "cell_size"], None])
    @pytest.mark.parametrize("library_key", ["library_id", None])
    def test_design_matrix_several_slides(
        adata_mibitof: AnnData,
        groups: str | List[str],
        cluster_key: str,
        library_key: str | None,
        covariates: str | List[str] | None = None,
    ):
        df = exp_dist(
            adata_mibitof,
            cluster_key=cluster_key,
            groups=groups,
            library_key=library_key,
            covariates=covariates,
            copy=True,
        )

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

        assert len(df) == adata_mibitof.n_obs  # correct amount of rows
        assert len(df.columns) == 1 + 2 * len(groups) + int(slides) + n_covariates  # correct amount of columns

        for anchor in groups:
            assert min(df[anchor]) == 0 and max(df[anchor]) <= 1  # correct normalized range
        assert df.Cluster.dtype.name == "category"  # correct dtype

        if not isinstance(library_key, None):
            assert df[library_key].nunique() > 1  # more than one slide

        assert df[covariates].equals(
            adata_mibitof.obs[covariates]
        )  # covariate column in design matrix match covariate column in .obs

        for anchor in groups:
            anchor_ids_adata = adata_mibitof.obs.index[adata_mibitof.obs["Cluster"] == anchor].tolist()
            anchor_ids_design_matrix = df.obs.index[df.obs["Cluster"] == anchor].tolist()
            zero_dist_ids = df.obs.index[df.obs["f{anchor}_raw"] == 0.0].tolist()
            nan_ids = df.obs.index[df.obs[anchor].isna()].tolist()

            assert anchor_ids_adata == anchor_ids_design_matrix  # anchor point indices match before and after
            assert anchor_ids_adata == zero_dist_ids  # anchor point indices have zero values in anchor_raw column
            assert (
                nan_ids <= zero_dist_ids
            )  # zero value indices must be subset of indices with NaN values in anchor column

    @pytest.mark.parametrize("groups", ["celltype_mapped_refined"])
    @pytest.mark.parametrize("cluster_key", ["Spinal cord"])
    @pytest.mark.parametrize("library_key", [None])
    def test_design_matrix_single_slide(
        adata_seqfish: AnnData,
        groups: str,
        cluster_key: str,
        library_key: str | None,
    ):
        df = exp_dist(adata_seqfish, groups=groups, cluster_key=cluster_key, library_key=library_key, copy=True)

        assert len(df) == adata_seqfish.n_obs  # correct amount of rows
        assert len(df.columns) == len(groups) + len(cluster_key) * 2  # correct amount of columns

        anchor_ids = adata_seqfish.obs.index[adata_seqfish.obs[groups] == cluster_key]
        nan_ids = df.obs.index[df.obs[cluster_key].isna()].tolist()

        assert anchor_ids == nan_ids  # nan ids match anchor point ids
