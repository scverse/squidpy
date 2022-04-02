import pytest

from anndata import AnnData
import scanpy as sc

import numpy as np
import pandas as pd

from squidpy import pl
from tests.conftest import PlotTester, PlotTesterMeta
from squidpy._constants._pkg_constants import Key

C_KEY = "Cluster"


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

    def test_plot_spatial_segment(self, adata_mibitof: AnnData):
        pl.spatial_segment(
            adata_mibitof,
            cell_id_key="cell_id",
            library_key="library_id",
            na_color="lightgrey",
        )


class TestSpatialStaticUtils:
    def create_anndata(self, shape, library_id, library_key):
        n_obs = len(library_id) * 2 if isinstance(library_id, list) else 2
        obsm = np.random.normal(size=(n_obs, 2))
        X = (np.empty((n_obs, 3)),)
        if isinstance(library_id, list):
            if library_key is not None:
                obs = pd.Series(library_id * 2, name=library_key)
                uns = {Key.uns.spatial: {i: {} for i in "library_id"}}
                return AnnData(X, obs=obs, uns=uns, obsm=obsm)
            obs = None
            return AnnData(X, obs=obs, uns=uns, obsm=obsm)

    @pytest.mark.parametrize("shape", ["circle", None])
    @pytest.mark.parametrize("library_id", [None, "1", ["1"], ["1", "2"]])
    @pytest.mark.parametrize("library_key", [None, "batch_key"])
    def test_get_library_id(self, shape, library_id, library_key):
        pass

        self._create_anndata(shape, library_id, library_key)
