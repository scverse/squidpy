from __future__ import annotations

import scanpy as sc
from anndata import AnnData

from tests.conftest import PlotTester, PlotTesterMeta

sc.set_figure_params(dpi=40, color_map="viridis")

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be change, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestSpatialStatic(PlotTester, metaclass=PlotTesterMeta):
    def test_single_cluster(self, adata_hne: AnnData):
        pass

    def test_multiple_clusters(self, adata_hne: AnnData):
        pass
