from __future__ import annotations

import platform
from functools import partial
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import ListedColormap
from squidpy import pl
from squidpy._constants._pkg_constants import Key
from squidpy.gr import spatial_neighbors
from squidpy.pl._spatial_utils import _get_library_id

from tests.conftest import PlotTester, PlotTesterMeta

sc.set_figure_params(dpi=40, color_map="viridis")

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be change, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestSpatialStatic(PlotTester, metaclass=PlotTesterMeta):
    def test_tol_plot_spatial_scatter_image(self, adata_hne: AnnData):
        pl.spatial_scatter(adata_hne, na_color="lightgrey")
        self.compare("SpatialStatic_spatial_scatter_image", tolerance=70)

    def test_plot_spatial_scatter_noimage(self, adata_hne: AnnData):
        pl.spatial_scatter(adata_hne, shape=None, na_color="lightgrey")

    def test_plot_spatial_scatter_group_outline(self, adata_hne: AnnData):
        pl.spatial_scatter(adata_hne, shape="circle", color="cluster", groups="Cortex_1", outline=True)

    def test_plot_spatial_scatter_title_single(self, adata_hne_concat: AnnData):
        pl.spatial_scatter(
            adata_hne_concat,
            shape="hex",
            library_key="library_id",
            library_id=["V2_Adult_Mouse_Brain"],
            color=["Sox17", "cluster"],
            title="Visium test",
        )

    def test_plot_spatial_scatter_crop_graph(self, adata_hne_concat: AnnData):
        pl.spatial_scatter(
            adata_hne_concat,
            shape="square",
            library_key="library_id",
            size=[0.3, 0.3],
            color=["Sox17", "cluster"],
            connectivity_key="spatial_connectivities",
            edges_width=5,
            title=None,
            outline=True,
            library_first=False,
            outline_width=(0.05, 0.05),
            crop_coord=[(0, 0, 300, 300), (0, 0, 300, 300)],
            scalebar_dx=2.0,
            scalebar_kwargs={"scale_loc": "bottom", "location": "lower right"},
        )

    def test_plot_spatial_scatter_crop_noorigin(self, adata_hne_concat: AnnData):
        pl.spatial_scatter(
            adata_hne_concat,
            shape="circle",
            library_key="library_id",
            color=["Sox17", "cluster"],
            outline_width=(0.05, 0.05),
            crop_coord=[(300, 300, 5000, 5000), (3000, 3000, 5000, 5000)],
            scalebar_dx=2.0,
            scalebar_kwargs={"scale_loc": "bottom", "location": "lower right"},
        )

    def test_plot_spatial_scatter_group_multi(self, adata_hne: AnnData):
        spatial_neighbors(adata_hne)
        pl.spatial_scatter(
            adata_hne,
            shape="circle",
            color=["Sox9", "cluster", "leiden"],
            groups=["Cortex_1", "Cortex_3", "3"],
            crop_coord=[(0, 0, 500, 500)],
            connectivity_key="spatial_connectivities",
        )

    def test_plot_spatial_scatter_group(self, adata_hne_concat: AnnData):
        pl.spatial_scatter(
            adata_hne_concat,
            cmap="inferno",
            shape="hex",
            library_key="library_id",
            library_id=["V1_Adult_Mouse_Brain", "V2_Adult_Mouse_Brain"],
            size=[1, 1.25],
            color=["Sox17", "cluster"],
            edges_width=5,
            title=None,
            outline=True,
            outline_width=(0.05, 0.05),
            scalebar_dx=2.0,
            scalebar_kwargs={"scale_loc": "bottom", "location": "lower right"},
        )

    def test_plot_spatial_scatter_nospatial(self, adata_hne_concat: AnnData):
        adata = adata_hne_concat.copy()
        spatial_neighbors(adata)
        adata.uns.pop("spatial")
        pl.spatial_scatter(
            adata_hne_concat,
            shape=None,
            library_key="library_id",
            library_id=["V1_Adult_Mouse_Brain", "V2_Adult_Mouse_Brain"],
            connectivity_key="spatial_connectivities",
            edges_width=3,
            size=[1.0, 50],
            color="cluster",
        )

    def test_plot_spatial_scatter_axfig(self, adata_hne: AnnData):
        fig, ax = plt.subplots(1, 2, figsize=(3, 3), dpi=40)
        pl.spatial_scatter(
            adata_hne,
            shape="square",
            color=["Sox17", "cluster"],
            fig=fig,
            ax=ax,
        )

    @pytest.mark.skipif(platform.system() == "Darwin", reason="Fails on macOS 3.8 CI")
    def test_plot_spatial_scatter_novisium(self, adata_mibitof: AnnData):
        spatial_neighbors(adata_mibitof, coord_type="generic", radius=50)
        pl.spatial_scatter(
            adata_mibitof,
            library_key="library_id",
            library_id=["point8"],
            na_color="lightgrey",
            connectivity_key="spatial_connectivities",
            edges_width=0.5,
        )

    def test_plot_spatial_segment(self, adata_mibitof: AnnData):
        pl.spatial_segment(
            adata_mibitof,
            seg_cell_id="cell_id",
            library_key="library_id",
            na_color="lightgrey",
        )

    def test_tol_plot_spatial_segment_group(self, adata_mibitof: AnnData):
        pl.spatial_segment(
            adata_mibitof,
            color=["Cluster"],
            groups=["Fibroblast", "Endothelial"],
            library_key="library_id",
            seg_cell_id="cell_id",
            img=False,
            seg=True,
            figsize=(5, 5),
            legend_na=False,
            scalebar_dx=2.0,
            scalebar_kwargs={"scale_loc": "bottom", "location": "lower right"},
        )
        self.compare("SpatialStatic_spatial_segment_group", tolerance=60)

    def test_plot_spatial_segment_crop(self, adata_mibitof: AnnData):
        pl.spatial_segment(
            adata_mibitof,
            color=["Cluster", "cell_size"],
            groups=["Fibroblast", "Endothelial"],
            library_key="library_id",
            seg_cell_id="cell_id",
            img=True,
            seg=True,
            seg_outline=True,
            seg_contourpx=15,
            figsize=(5, 5),
            cmap="magma",
            vmin=500,
            crop_coord=[(100, 100, 500, 500), (100, 100, 500, 500), (100, 100, 500, 500)],
            img_alpha=0.5,
        )

    def test_plot_spatial_scatter_categorical_alpha(self, adata_hne: AnnData):
        pl.spatial_scatter(adata_hne, shape="circle", color="cluster", alpha=0)

    def test_tol_plot_spatial_scatter_non_unique_colors(self, adata_hne: AnnData):
        adata_hne.uns["cluster_colors"] = ["#000000"] * len(adata_hne.uns["cluster_colors"])
        pl.spatial_scatter(adata_hne, color="cluster", legend_loc=None)
        self.compare("SpatialStatic_spatial_scatter_non_unique_colors", tolerance=70)

    def test_tol_plot_palette_listed_cmap(self, adata_hne: AnnData):
        del adata_hne.uns["cluster_colors"]
        palette = plt.get_cmap("Set3")
        assert isinstance(palette, ListedColormap)
        pl.spatial_scatter(adata_hne, color="cluster", palette=palette, legend_loc=None)
        self.compare("SpatialStatic_palette_listed_cmap", tolerance=70)


class TestSpatialStaticUtils:
    @staticmethod
    def _create_anndata(shape: str | None, library_id: str | Sequence[str] | None, library_key: str | None):
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
    @pytest.mark.parametrize("library_key", [None, "library_id"])
    def test_get_library_id(self, shape, library_id, library_key):
        adata = TestSpatialStaticUtils._create_anndata(shape, library_id, library_key)
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
                    with pytest.raises(KeyError, match="library_id"):
                        _get_libid(adata)
            else:
                assert library_id == _get_libid(adata)
        else:
            if library_id is None:
                with pytest.raises(KeyError, match=Key.uns.spatial):
                    _get_libid(adata)
            else:
                assert library_id == _get_libid(adata)
