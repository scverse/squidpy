from copy import deepcopy
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from squidpy import gr, pl

from tests.conftest import DPI, PlotTester, PlotTesterMeta

C_KEY = "leiden"


sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=40, color_map="viridis")

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be changed, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestGraph(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_interaction(self, adata: AnnData):
        gr.spatial_neighbors(adata)
        gr.interaction_matrix(adata, cluster_key=C_KEY)

        pl.interaction_matrix(adata, cluster_key=C_KEY)

    def test_plot_interaction_dendro(self, adata: AnnData):
        gr.spatial_neighbors(adata)
        gr.interaction_matrix(adata, cluster_key=C_KEY)

        pl.interaction_matrix(adata, cluster_key=C_KEY, method="single")

    def test_plot_centrality_scores(self, adata: AnnData):
        gr.spatial_neighbors(adata)
        gr.centrality_scores(adata, cluster_key=C_KEY)

        pl.centrality_scores(adata, cluster_key=C_KEY)

    def test_plot_centrality_scores_single(self, adata: AnnData):
        selected_score = "degree_centrality"
        gr.spatial_neighbors(adata)
        gr.centrality_scores(adata, cluster_key=C_KEY)

        pl.centrality_scores(adata, cluster_key=C_KEY, score=selected_score, dpi=DPI)

    def test_plot_nhood_enrichment(self, adata: AnnData):
        gr.spatial_neighbors(adata)
        gr.nhood_enrichment(adata, cluster_key=C_KEY)

        pl.nhood_enrichment(adata, cluster_key=C_KEY)

    def test_plot_nhood_enrichment_ax(self, adata: AnnData):
        gr.spatial_neighbors(adata)
        gr.nhood_enrichment(adata, cluster_key=C_KEY)

        fig, ax = plt.subplots(figsize=(2, 2), constrained_layout=True)
        pl.nhood_enrichment(adata, cluster_key=C_KEY, ax=ax)

    def test_plot_nhood_enrichment_dendro(self, adata: AnnData):
        gr.spatial_neighbors(adata)
        gr.nhood_enrichment(adata, cluster_key=C_KEY)

        # use count to avoid nan for scipy.cluster.hierarchy
        pl.nhood_enrichment(adata, cluster_key=C_KEY, mode="count", method="single")

    def test_plot_ripley_l(self, adata_ripley: AnnData):
        adata = adata_ripley
        gr.ripley(adata, cluster_key=C_KEY, mode="L")
        pl.ripley(adata, cluster_key=C_KEY, mode="L")

    def test_plot_ripley_f(self, adata_ripley: AnnData):
        adata = adata_ripley
        gr.ripley(adata, cluster_key=C_KEY, mode="F")
        pl.ripley(adata, cluster_key=C_KEY, mode="F")

    def test_plot_ripley_g(self, adata_ripley: AnnData):
        adata = adata_ripley
        gr.ripley(adata, cluster_key=C_KEY, mode="G")
        pl.ripley(adata, cluster_key=C_KEY, mode="G")

    def test_plot_ripley_f_nopalette(self, adata_ripley: AnnData):
        adata = adata_ripley
        adata.uns.pop(f"{C_KEY}_colors")
        gr.ripley(adata, cluster_key=C_KEY, mode="F")
        pl.ripley(adata, cluster_key=C_KEY, mode="F")

    def test_tol_plot_co_occurrence(self, adata: AnnData):
        gr.co_occurrence(adata, cluster_key=C_KEY)

        pl.co_occurrence(adata, cluster_key=C_KEY, clusters=["0", "2"])
        self.compare("Graph_co_occurrence", tolerance=70)

    def test_tol_plot_co_occurrence_palette(self, adata_palette: AnnData):
        adata = adata_palette
        gr.co_occurrence(adata, cluster_key=C_KEY)

        pl.co_occurrence(adata, cluster_key=C_KEY, clusters=["0", "2"])
        self.compare("Graph_co_occurrence_palette", tolerance=70)


class TestHeatmap(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_cbar_vmin_vmax(self, adata: AnnData):
        gr.spatial_neighbors(adata)
        gr.nhood_enrichment(adata, cluster_key=C_KEY)

        pl.nhood_enrichment(adata, cluster_key=C_KEY, vmin=10, vmax=20)

    def test_plot_cbar_kwargs(self, adata: AnnData):
        gr.spatial_neighbors(adata)
        gr.nhood_enrichment(adata, cluster_key=C_KEY)

        pl.nhood_enrichment(adata, cluster_key=C_KEY, cbar_kwargs={"label": "FOOBARBAZQUUX"})


class TestLigrec(PlotTester, metaclass=PlotTesterMeta):
    def test_invalid_type(self):
        with pytest.raises(TypeError, match=r"Expected `adata` .+ found `int`."):
            pl.ligrec(42)

    def test_invalid_key(self, adata: AnnData):
        with pytest.raises(KeyError, match=r"Key `foobar_ligrec` not found in `adata.uns`."):
            pl.ligrec(adata, cluster_key="foobar")

    def test_valid_key_invalid_object(self, adata: AnnData):
        adata.uns["foobar_ligrec"] = "baz"
        with pytest.raises(TypeError, match=r"Expected `adata` .+ found `str`."):
            pl.ligrec(adata, cluster_key="foobar")

    def test_invalid_alpha(self, ligrec_result: Mapping[str, pd.DataFrame]):
        with pytest.raises(ValueError, match=r"Expected `alpha`"):
            pl.ligrec(ligrec_result, alpha=1.2)

    def test_invalid_means_range_size(self, ligrec_result: Mapping[str, pd.DataFrame]):
        with pytest.raises(ValueError, match=r"Expected `means_range` to be a sequence of size `2`, found `3`."):
            pl.ligrec(ligrec_result, means_range=[0, 1, 2])

    def test_invalid_clusters(self, ligrec_result: Mapping[str, pd.DataFrame]):
        source_groups = "foo"
        target_groups = "bar"
        with pytest.raises(ValueError, match=r"Invalid cluster in .*"):
            pl.ligrec(ligrec_result, source_groups=source_groups, target_groups=target_groups)

    def test_all_interactions_empty(self, ligrec_result: Mapping[str, pd.DataFrame]):
        empty = pd.DataFrame(np.nan, index=ligrec_result["pvalues"].index, columns=ligrec_result["pvalues"].columns)

        with pytest.raises(ValueError, match=r"After removing rows with only NaN interactions, none remain."):
            pl.ligrec({"means": empty, "pvalues": empty, "metadata": empty}, remove_empty_interactions=True)

    def test_plot_source_clusters(self, ligrec_result: Mapping[str, pd.DataFrame]):
        src_cls = ligrec_result["pvalues"].columns.get_level_values(0)[0]
        pl.ligrec(ligrec_result, source_groups=src_cls)

    def test_plot_target_clusters(self, ligrec_result: Mapping[str, pd.DataFrame]):
        tgt_cls = ligrec_result["pvalues"].columns.get_level_values(1)[0]
        pl.ligrec(ligrec_result, target_groups=tgt_cls)

    def test_plot_no_remove_empty_interactions(self, ligrec_result: Mapping[str, pd.DataFrame]):
        tmp = deepcopy(ligrec_result)
        tmp["pvalues"].values[:2, :] = np.nan
        pl.ligrec(tmp, remove_empty_interactions=False)

    def test_plot_pvalue_threshold(self, ligrec_result: Mapping[str, pd.DataFrame]):
        pl.ligrec(ligrec_result, pvalue_threshold=0.05)

    def test_plot_means_range(self, ligrec_result: Mapping[str, pd.DataFrame]):
        pl.ligrec(ligrec_result, means_range=(0.5, 1))

    def test_plot_dendrogram_pairs(self, ligrec_result: Mapping[str, pd.DataFrame]):
        np.random.seed(42)
        pl.ligrec(ligrec_result, dendrogram="interacting_molecules")

    def test_plot_dendrogram_clusters(self, ligrec_result: Mapping[str, pd.DataFrame]):
        np.random.seed(42)
        pl.ligrec(ligrec_result, dendrogram="interacting_clusters")

    def test_plot_dendrogram_both(self, ligrec_result: Mapping[str, pd.DataFrame]):
        np.random.seed(42)
        pl.ligrec(ligrec_result, dendrogram="both")

    def test_plot_swap_axes(self, ligrec_result: Mapping[str, pd.DataFrame]):
        pl.ligrec(ligrec_result, swap_axes=True)

    def test_plot_swap_axes_dedrogram(self, ligrec_result: Mapping[str, pd.DataFrame]):
        pl.ligrec(ligrec_result, swap_axes=True, dendrogram="interacting_molecules")

    def test_plot_alpha(self, ligrec_result: Mapping[str, pd.DataFrame]):
        pl.ligrec(ligrec_result, alpha=1)

    def test_plot_alpha_none(self, ligrec_result: Mapping[str, pd.DataFrame]):
        pl.ligrec(ligrec_result, alpha=None)

    def test_plot_cmap(self, ligrec_result: Mapping[str, pd.DataFrame]):
        pl.ligrec(ligrec_result, cmap="inferno")

    def test_plot_kwargs(self, ligrec_result: Mapping[str, pd.DataFrame]):
        # color_on is intentionally ignored
        pl.ligrec(ligrec_result, grid=False, color_on="square", x_padding=2, y_padding=2)

    def test_plot_remove_nonsig_interactions(self, ligrec_result: Mapping[str, pd.DataFrame]):
        pl.ligrec(ligrec_result, remove_nonsig_interactions=True, alpha=1e-4)
