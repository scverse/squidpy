import platform
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from anndata import AnnData
from matplotlib.testing.compare import compare_images
from scanpy import settings as s
from scipy.sparse import issparse
from squidpy.im import ImageContainer

from tests.conftest import ACTUAL, DPI, EXPECTED, TOL, PlotTester, PlotTesterMeta


@pytest.mark.skipif(platform.system() == "Linux", reason="Fails on ubuntu")
@pytest.mark.qt()
class TestNapari(PlotTester, metaclass=PlotTesterMeta):
    def test_add_same_layer(self, adata: AnnData, napari_cont: ImageContainer, capsys):
        from napari.layers import Points

        s.logfile = sys.stderr
        s.verbosity = 4

        viewer = napari_cont.interactive(adata)
        cnt = viewer._controller

        data = np.random.normal(size=adata.n_obs)
        cnt.add_points(data, layer_name="layer1")
        cnt.add_points(np.random.normal(size=adata.n_obs), layer_name="layer1")

        err = capsys.readouterr().err

        assert "Layer `layer1` is already loaded" in err
        assert len(viewer._controller.view.layers) == 2
        assert viewer._controller.view.layernames == {"V1_Adult_Mouse_Brain", "layer1"}
        assert isinstance(viewer._controller.view.layers["layer1"], Points)
        np.testing.assert_array_equal(viewer._controller.view.layers["layer1"].metadata["data"], data)

    def test_add_not_categorical_series(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata)
        cnt = viewer._controller

        with pytest.raises(TypeError, match=r"Expected a `categorical` type,.*"):
            cnt.add_points(adata.obs["in_tissue"].astype(int), layer_name="layer1")

    def test_plot_simple_canvas(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata)

        viewer.screenshot(dpi=DPI)

    def test_plot_symbol(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata, symbol="square")
        cnt = viewer._controller

        cnt.add_points(adata.obs_vector(adata.var_names[42]), layer_name="foo")
        viewer.screenshot(dpi=DPI)

    def test_plot_gene_X(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata)
        cnt = viewer._controller

        cnt.add_points(adata.obs_vector(adata.var_names[42]), layer_name="foo")
        viewer.screenshot(dpi=DPI)

    def test_plot_obs_continuous(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata)
        cnt = viewer._controller

        cnt.add_points(np.random.RandomState(42).normal(size=adata.n_obs), layer_name="quux")
        viewer.screenshot(dpi=DPI)

    def test_plot_obs_categorical(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata)
        cnt = viewer._controller

        cnt.add_points(adata.obs["leiden"], key="leiden", layer_name="quas")
        viewer.screenshot(dpi=DPI)

    def test_plot_cont_cmap(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata, cmap="inferno")
        cnt = viewer._controller

        cnt.add_points(adata.obs_vector(adata.var_names[42]), layer_name="wex")
        viewer.screenshot(dpi=DPI)

    def test_plot_cat_cmap(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata, palette="Set3")
        cnt = viewer._controller

        cnt.add_points(adata.obs["leiden"].astype("category"), key="in_tissue", layer_name="exort")
        viewer.screenshot(dpi=DPI)

    def test_plot_blending(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata, blending="additive")
        cnt = viewer._controller

        for gene in adata.var_names[42:46]:
            data = adata.obs_vector(gene)
            if issparse(data):  # ALayer handles sparsity, here we have to do it ourselves
                data = data.X
            cnt.add_points(data, layer_name=gene)

        viewer.screenshot(dpi=DPI)

    def test_plot_crop_center(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.crop_corner(0, 0, size=500).interactive(adata)
        bdata = viewer.adata
        cnt = viewer._controller

        cnt.add_points(bdata.obs_vector(bdata.var_names[42]), layer_name="foo")

        viewer.screenshot(dpi=DPI)

    @pytest.mark.skip(reason="Soon to be deprecated.")
    def test_plot_crop_corner(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.crop_center(500, 500, radius=250).interactive(adata)
        bdata = viewer.adata
        cnt = viewer._controller

        cnt.add_points(bdata.obs_vector(bdata.var_names[42]), layer_name="foo")

        viewer.screenshot(dpi=DPI)

    def test_plot_scalefactor(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        scale = 2
        napari_cont.data.attrs["scale"] = scale

        viewer = napari_cont.interactive(adata)
        cnt = viewer._controller
        model = cnt._model

        data = np.random.normal(size=adata.n_obs)
        cnt.add_points(data, layer_name="layer1")

        # ignore z-dim
        np.testing.assert_allclose(adata.obsm["spatial"][:, ::-1] * scale, model.coordinates[:, 1:])

        viewer.screenshot(dpi=DPI)

    @pytest.mark.skip(reason="Soon to be deprecated.")
    @pytest.mark.parametrize("size", [(800, 600), (600, 800), (800, 800)])
    @pytest.mark.parametrize("x", [-200, 200])
    @pytest.mark.parametrize("y", [-200, 200])
    def test_corner_corner_cases(
        self, qtbot, adata: AnnData, napari_cont: ImageContainer, y: int, x: int, size: tuple[int, int]
    ):
        viewer = napari_cont.crop_corner(y, x, size=size).interactive(adata)
        bdata = viewer.adata
        cnt = viewer._controller

        cnt.add_points(bdata.obs_vector(bdata.var_names[42]), layer_name="foo")

        basename = f"{self.__class__.__name__[4:]}_corner_case_{y}_{x}_{'_'.join(map(str, size))}.png"
        viewer.screenshot(dpi=DPI)
        plt.savefig(ACTUAL / basename, dpi=DPI)
        plt.close()

        res = compare_images(str(EXPECTED / basename), str(ACTUAL / basename), 2 * TOL)

        assert res is None, res
