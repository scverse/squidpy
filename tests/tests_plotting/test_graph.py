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
