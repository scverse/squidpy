from __future__ import annotations

from typing import List

import pytest
from anndata import AnnData
from squidpy.tl import var_by_distance
from squidpy.tl._var_by_distance import _normalize_distances


class TestVarDist:
    @pytest.mark.parametrize("groups", ["Endothelial", ["Endothelial", "Epithelial", "Fibroblast"]])
    @pytest.mark.parametrize("cluster_key", ["Cluster"])
    @pytest.mark.parametrize("covariates", ["category", ["category", "cell_size"], None])
    @pytest.mark.parametrize("library_key", ["point"])
    def test_design_matrix_several_slides(
        self,
        adata_mibitof: AnnData,
        groups: str | list[str],
        cluster_key: str,
        library_key: str | None,
        covariates: str | list[str] | None,
    ):
        df = var_by_distance(
            adata_mibitof,
            cluster_key=cluster_key,
            groups=groups,
            library_key=library_key,
            covariates=covariates,
            copy=True,
        )

        if not isinstance(groups, list):
            groups = [groups]
        if not isinstance(covariates, list) and covariates is not None:
            covariates = [covariates]
        n_covariates = len(covariates) if covariates is not None else 0
        slides = 1 if isinstance(library_key, str) else 0

        assert len(df) == adata_mibitof.n_obs  # correct amount of rows
        assert len(df.columns) == 1 + 2 * len(groups) + slides + n_covariates  # correct amount of columns

        for anchor in groups:
            assert df[anchor].min() == 0 and df[anchor].max() <= 1  # correct normalized range
        assert df.Cluster.dtype.name == "category"  # correct dtype

        if library_key is not None:
            assert df[library_key].nunique() > 1  # more than one slide

        if covariates is not None:
            assert df[covariates].equals(
                adata_mibitof.obs[covariates]
            )  # covariate column in design matrix match covariate column in .obs

        for anchor in groups:
            anchor_ids_adata = adata_mibitof.obs.index[adata_mibitof.obs[cluster_key] == anchor].tolist()
            anchor_ids_design_matrix = df.index[df["Cluster"] == anchor].tolist()
            zero_dist_ids = df.index[df[f"{anchor}_raw"] == 0.0].tolist()
            nan_ids = df.index[df[anchor].isna()].tolist()

            assert anchor_ids_adata == anchor_ids_design_matrix  # anchor point indices match before and after
            assert anchor_ids_adata == zero_dist_ids  # anchor point indices have zero values in anchor_raw column
            assert (
                nan_ids <= zero_dist_ids
            )  # zero value indices must be subset of indices with NaN values in anchor column

    @pytest.mark.parametrize("groups", ["Spinal cord"])
    @pytest.mark.parametrize("cluster_key", ["celltype_mapped_refined"])
    @pytest.mark.parametrize("library_key", [None])
    def test_design_matrix_single_slide(
        self,
        adata_seqfish: AnnData,
        groups: str,
        cluster_key: str,
        library_key: str | None,
    ):
        df = var_by_distance(adata_seqfish, groups=groups, cluster_key=cluster_key, library_key=library_key, copy=True)

        assert len(df) == adata_seqfish.n_obs  # correct amount of rows
        assert len(df.columns) == len([groups]) * 2 + len([cluster_key])  # correct amount of columns

        anchor_ids = adata_seqfish.obs.index[adata_seqfish.obs[cluster_key] == groups]
        nan_ids = df.index[df[groups].isna()]

        assert anchor_ids.equals(nan_ids)  # nan ids match anchor point ids
