from copy import deepcopy

from conftest import DPI, PlotTester, PlotTesterMeta
import pytest

from anndata import AnnData
import scanpy as sc

import numpy as np
import pandas as pd

from squidpy import gr, pl
from squidpy.gr._ligrec import LigrecResult

C_KEY = "leiden"


sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=40, color_map="viridis")

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be change, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`


class TestGraph(PlotTester, metaclass=PlotTesterMeta):
    @pytest.mark.skip(reason="X_spatial not in obsm. Apparently already fixed in scanpy's master/dev.")
    def test_plot_spatial_graph(self, adata: AnnData):
        gr.spatial_connectivity(adata)
        pl.spatial_graph(adata)

    def test_plot_interaction(self, adata: AnnData):
        gr.spatial_connectivity(adata)
        gr.interaction_matrix(adata, cluster_key=C_KEY)

        pl.interaction_matrix(adata, cluster_key=C_KEY)

    def test_plot_centrality_scores(self, adata: AnnData):
        gr.spatial_connectivity(adata)
        gr.centrality_scores(adata, cluster_key=C_KEY)

        pl.centrality_scores(adata, cluster_key=C_KEY)

    def test_plot_centrality_scores_single(self, adata: AnnData):
        selected_score = "degree_centrality"
        gr.spatial_connectivity(adata)
        gr.centrality_scores(adata, cluster_key=C_KEY)

        pl.centrality_scores(adata, cluster_key=C_KEY, selected_score=selected_score, dpi=DPI)

    def test_plot_nhood_enrichment(self, adata: AnnData):
        gr.spatial_connectivity(adata)
        gr.nhood_enrichment(adata, cluster_key=C_KEY)

        pl.nhood_enrichment(adata, cluster_key=C_KEY)

    def test_plot_ripley_k(self, adata: AnnData):
        gr.spatial_connectivity(adata)
        gr.ripley_k(adata, cluster_key=C_KEY)

        pl.plot_ripley_k(adata, cluster_key=C_KEY)


class TestLigrec(PlotTester, metaclass=PlotTesterMeta):
    def test_invalid_type(self):
        with pytest.raises(TypeError, match=r"Expected `adata` .+ found `int`."):
            pl.ligrec(42)

    def test_invalid_key(self, adata: AnnData):
        with pytest.raises(KeyError, match=r"Key `foobar` not found in `adata.uns`."):
            pl.ligrec(adata, key="foobar")

    def test_valid_key_invalid_object(self, adata: AnnData):
        adata.uns["foobar"] = "baz"
        with pytest.raises(TypeError, match=r"Expected `adata` .+ found `str`."):
            pl.ligrec(adata, key="foobar")

    def test_invalid_alpha(self, ligrec_result: LigrecResult):
        with pytest.raises(ValueError, match=r"Expected `alpha`"):
            pl.ligrec(ligrec_result, alpha=1.2)

    def test_invalid_clusters(self, ligrec_result: LigrecResult):
        with pytest.raises(ValueError, match=r"No clusters have been selected"):
            pl.ligrec(ligrec_result, source_groups="foo", target_groups="bar")

    def test_all_interactions_empty(self, ligrec_result: LigrecResult):
        empty = pd.DataFrame(np.nan, index=ligrec_result.pvalues.index, columns=ligrec_result.pvalues.columns)
        tmp = type(ligrec_result)(empty, empty, empty)

        with pytest.raises(ValueError, match=r"After removing empty interactions, none remain."):
            pl.ligrec(tmp, remove_empty_interactions=True)

    def test_plot_source_clusters(self, ligrec_result: LigrecResult):
        src_cls = ligrec_result.pvalues.columns.get_level_values(0)[0]
        pl.ligrec(ligrec_result, source_groups=src_cls)

    def test_plot_target_clusters(self, ligrec_result: LigrecResult):
        tgt_cls = ligrec_result.pvalues.columns.get_level_values(1)[0]
        pl.ligrec(ligrec_result, target_groups=tgt_cls)

    def test_plot_remove_empty_interactions(self, ligrec_result: LigrecResult):
        tmp = deepcopy(ligrec_result)
        tmp.pvalues.values[:2, :] = np.nan
        pl.ligrec(tmp, remove_empty_interactions=True)

    def test_plot_dendrogram(self, ligrec_result: LigrecResult):
        pl.ligrec(ligrec_result, dendrogram=True)

    def test_plot_swap_axes(self, ligrec_result: LigrecResult):
        pl.ligrec(ligrec_result, swap_axes=True)

    def test_plot_swap_axes_dedrogram(self, ligrec_result: LigrecResult):
        pl.ligrec(ligrec_result, swap_axes=True, dendrogram=True)

    def test_plot_alpha(self, ligrec_result: LigrecResult):
        pl.ligrec(ligrec_result, alpha=1)

    def test_plot_cmap(self, ligrec_result: LigrecResult):
        pl.ligrec(ligrec_result, cmap="inferno")

    def test_plot_kwargs(self, ligrec_result: LigrecResult):
        # color_on is intentionally ignored
        pl.ligrec(ligrec_result, grid=False, color_on="square", x_padding=2, y_padding=2)
