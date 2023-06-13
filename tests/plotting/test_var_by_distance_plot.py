import scanpy as sc
from anndata import AnnData
from squidpy import pl, tl

from tests.conftest import PlotTester, PlotTesterMeta

sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=40)

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be change, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestVarDist(PlotTester, metaclass=PlotTesterMeta):
    def test_tol_plot_var_by_distance(self, adata_mibitof: AnnData):
        tl.var_by_distance(
            adata_mibitof,
            cluster_key="Cluster",
            groups="Epithelial",
            library_key="point",
        )
        pl.var_by_distance(
            adata=adata_mibitof,
            design_matrix_key="design_matrix",
            var="HK1",
            anchor_key="Epithelial",
            color="HK1",
            figsize=(5, 4),
        )
        self.compare("var_by_distance_single_anchor_and_gene")  # tolerance added due to numerical errors of spline

    def test_tol_plot_var_by_distance_with_covariate(self, adata_mibitof: AnnData):
        tl.var_by_distance(
            adata_mibitof, cluster_key="Cluster", groups="Epithelial", library_key="point", covariates="donor"
        )
        pl.var_by_distance(
            adata=adata_mibitof,
            design_matrix_key="design_matrix",
            var="IDH2",
            anchor_key="Epithelial",
            covariate="donor",
            figsize=(5, 4),
        )
        self.compare(
            "var_by_distance_single_anchor_and_gene_two_categories"
        )  # tolerance added due to numerical errors of spline

    def test_tol_plot_var_by_distance_various_palettes(self, adata_mibitof: AnnData):
        tl.var_by_distance(
            adata_mibitof, cluster_key="Cluster", groups="Epithelial", library_key="point", covariates="donor"
        )
        pl.var_by_distance(
            adata=adata_mibitof,
            design_matrix_key="design_matrix",
            var=["IDH2", "H3", "vimentin", "CD98"],
            anchor_key="Epithelial",
            color="Cluster",
            covariate="donor",
            scatter_palette="plasma",
            line_palette=["red", "blue"],
            figsize=(10, 4),
        )
        self.compare(
            "var_by_distance_single_anchor_four_genes_two_categories_two_palettes"
        )  # tolerance added due to numerical errors of spline

    def test_tol_plot_var_by_distance_without_scatter(self, adata_mibitof: AnnData):
        tl.var_by_distance(
            adata_mibitof, cluster_key="Cluster", groups="Epithelial", library_key="point", covariates="donor"
        )
        pl.var_by_distance(
            adata=adata_mibitof,
            design_matrix_key="design_matrix",
            var="CD98",
            anchor_key="Epithelial",
            covariate="donor",
            line_palette=["blue", "orange"],
            show_scatter=False,
            figsize=(5, 4),
        )
        self.compare(
            "var_by_distance_single_anchor_one_gene_two_categories_without_scatter"
        )  # tolerance added due to numerical errors of spline
