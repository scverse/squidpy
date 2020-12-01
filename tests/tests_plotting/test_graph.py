from pathlib import Path

import pytest

import scanpy as sc
from anndata import AnnData

from squidpy import gr, pl

HERE: Path = Path(__file__).parents[1]

ROOT = HERE / "_images"
FIGS = HERE / "figures"
TOL = 50
DPI = 40

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=40, color_map="viridis")


@pytest.mark.skip(reason="X_spatial not in obsm. Apparently already fixed in scanpy's master/dev.")
def test_spatial_graph(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=TOL)

    gr.spatial_connectivity(adata)
    pl.spatial_graph(adata)
    save_and_compare_images("master_spatial_graph")


def test_interaction(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=TOL)

    c_key = "leiden"
    gr.spatial_connectivity(adata)
    gr.interaction_matrix(adata, cluster_key=c_key)

    pl.interaction_matrix(adata, cluster_key=c_key)
    save_and_compare_images("master_heatmap")


def test_centrality_scores(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=TOL)

    c_key = "leiden"
    gr.spatial_connectivity(adata)
    gr.centrality_scores(adata, cluster_key=c_key)

    pl.centrality_scores(adata, cluster_key=c_key)
    save_and_compare_images("master_scatter")


def test_centrality_scores_single(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=TOL)

    c_key = "leiden"
    selected_score = "degree_centrality"
    gr.spatial_connectivity(adata)
    gr.centrality_scores(adata, cluster_key=c_key)

    pl.centrality_scores(adata, cluster_key=c_key, selected_score=selected_score, dpi=DPI)
    save_and_compare_images("master_scatter_single")


def test_nhood_enrichment(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=TOL)

    c_key = "leiden"
    gr.spatial_connectivity(adata)
    gr.nhood_enrichment(adata, cluster_key=c_key)

    pl.nhood_enrichment(adata, cluster_key=c_key)
    save_and_compare_images("master_nhood_enrichment")


def test_ripley_k(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=TOL)

    c_key = "leiden"
    gr.spatial_connectivity(adata)
    gr.ripley_k(adata, cluster_key=c_key)

    pl.plot_ripley_k(adata, cluster_key=c_key)
    save_and_compare_images("master_ripley_k")
