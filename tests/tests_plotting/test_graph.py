from pathlib import Path

import scanpy as sc
from anndata import AnnData

from spatial_tools import graph, plotting

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=40, color_map="viridis")

HERE: Path = Path(__file__).parents[1]

ROOT = HERE / "_images"
FIGS = HERE / "figures"


def test_interaction(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=15)

    adata = adata
    c_key = "leiden"
    graph.spatial_connectivity(adata)
    graph.interaction_matrix(adata, cluster_key=c_key)

    plotting.interaction_matrix(adata, cluster_key=c_key)
    save_and_compare_images("master_heatmap")


def test_centrality_scores(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=15)

    adata = adata
    c_key = "leiden"
    graph.spatial_connectivity(adata)
    graph.centrality_scores(adata, cluster_key=c_key)

    plotting.centrality_scores(adata, cluster_key=c_key)
    save_and_compare_images("master_scatter")


def test_centrality_scores_single(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=15)

    adata = adata
    c_key = "leiden"
    selected_score = "degree_centrality"
    graph.spatial_connectivity(adata)
    graph.centrality_scores(adata, cluster_key=c_key)

    plotting.centrality_scores(adata, cluster_key=c_key, selected_score=selected_score)
    save_and_compare_images("master_scatter_single")


def test_nhood_enrichment(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=15)

    adata = adata
    c_key = "leiden"
    graph.spatial_connectivity(adata)
    graph.nhood_enrichment(adata, cluster_key=c_key)

    plotting.nhood_enrichment(adata, cluster_key=c_key)
    save_and_compare_images("master_nhood_enrichment")


def test_ripley_k(image_comparer, adata: AnnData):
    save_and_compare_images = image_comparer(ROOT, FIGS, tol=15)

    adata = adata
    c_key = "leiden"
    graph.spatial_connectivity(adata)
    graph.ripley_k(adata, cluster_key=c_key)

    plotting.plot_ripley_k(adata, cluster_key=c_key)
    save_and_compare_images("master_ripley_k")
