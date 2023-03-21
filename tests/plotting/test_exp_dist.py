import scanpy as sc
from anndata import AnnData

from squidpy import pl, tl

# from squidpy.tl import exp_dist
from tests.conftest import PlotTester, PlotTesterMeta

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be change, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestSpatialExpDist(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_exp_dist(self, adata_mibitof: AnnData):
        tl.exp_dist(
            adata_mibitof,
            cluster_key="Cluster",
            groups="Epithelial",
            library_key="point",
        )
        pl.exp_dist(
            adata=adata_mibitof,
            design_matrix_key="design_matrix",
            var="HK1",
            anchor_key="Epithelial",
            color="HK1",
            dpi=200,
        )
        # self.compare("Exp_dist_single_anchor_one_gene", tolerance=70)
