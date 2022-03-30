from anndata import AnnData
import scanpy as sc

from squidpy import pl
from tests.conftest import PlotTester, PlotTesterMeta

C_KEY = "leiden"


sc.pl.set_rcParams_defaults()
sc.set_figure_params(dpi=40, color_map="viridis")

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be change, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestSpatialStatic(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_spatial_scatter_image(self, adata_hne: AnnData):
        pl.spatial_scatter(adata_hne, na_color="lightgrey")

    def test_plot_spatial_scatter_noimage(self, adata_hne: AnnData):
        pl.spatial_scatter(adata_hne, shape=None, na_color="lightgrey")
