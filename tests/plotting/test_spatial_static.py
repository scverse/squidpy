from functools import partial
import pytest

from anndata import AnnData
import scanpy as sc

import numpy as np
import pandas as pd

from squidpy import pl
from tests.conftest import PlotTester, PlotTesterMeta
from squidpy.pl._spatial_utils import _get_library_id
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

    def test_plot_spatial_segment_group(self, adata_mibitof: AnnData):
        pl.spatial_segment(
            adata_mibitof,
            color=["Cluster"],
            groups=["Fibroblast", "Endothelial"],
            library_key="library_id",
            cell_id_key="cell_id",
            img=False,
            seg=True,
            figsize=(5, 5),
            legend_na=False,
            scalebar_dx=2.0,
            scalebar_kwargs={"scale_loc": "bottom", "location": "lower right"},
        )

    def test_plot_spatial_segment_crop(self, adata_mibitof: AnnData):
        pl.spatial_segment(
            adata_mibitof,
            color=["Cluster", "cell_size"],
            groups=["Fibroblast", "Endothelial"],
            library_key="library_id",
            cell_id_key="cell_id",
            img=True,
            seg=True,
            seg_outline=True,
            seg_contourpx=15,
            figsize=(5, 5),
            cmap="magma",
            vmin=500,
            crop_coord=[[0, 500, 0, 500], [0, 500, 0, 500], [0, 500, 0, 500]],
            img_alpha=0.5,
        )


class TestSpatialStaticUtils:
    def _create_anndata(self, shape, library_id, library_key):
        n_obs = len(library_id) * 2 if isinstance(library_id, list) else 2
        X = np.empty((n_obs, 3))
        if not isinstance(library_id, list) and library_id is not None:
            library_id = [library_id]
        if library_id is not None:
            obs = pd.DataFrame(library_id * 2, columns=[library_key])
            uns = {Key.uns.spatial: {k: None for k in library_id}}
            return AnnData(X, obs=obs, uns=uns, dtype=X.dtype)
        else:
            return AnnData(X, dtype=X.dtype)

    @pytest.mark.parametrize("shape", ["circle", None])
    @pytest.mark.parametrize("library_id", [None, "1", ["1"], ["1", "2"]])
    @pytest.mark.parametrize("library_key", [None, "batch_key"])
    def test_get_library_id(self, shape, library_id, library_key):
        adata = self._create_anndata(shape, library_id, library_key)
        if not isinstance(library_id, list) and library_id is not None:
            library_id = [library_id]
        _get_libid = partial(
            _get_library_id,
            shape=shape,
            library_id=library_id,
            library_key=library_key,
        )
        if shape is None:
            if library_id is None:
                if library_key is None:
                    assert _get_libid(adata) == [""]
                else:
                    with pytest.raises(ValueError, match="library_key"):
                        _get_libid(adata)
            else:
                assert library_id == _get_libid(adata)
        else:
            if library_id is None:
                with pytest.raises(KeyError, match=Key.uns.spatial):
                    _get_libid(adata)
            else:
                assert library_id == _get_libid(adata)
