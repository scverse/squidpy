import sys
import pytest

from scanpy import settings as s
from anndata import AnnData

from scipy.sparse import issparse
import numpy as np

from squidpy.im import ImageContainer
from tests.conftest import DPI, PlotTester, PlotTesterMeta
from squidpy.constants._pkg_constants import Key


@pytest.mark.qt
class TestNapari(PlotTester, metaclass=PlotTesterMeta):
    def test_no_valid_libraries(self, qtbot, adata: AnnData, cont: ImageContainer):
        with pytest.raises(ValueError, match=r"Unable to find any valid datasets\..*"):
            # in adata, there's 'V1_Adult_Mouse_Brain', in cont it's named as 'image'
            _ = cont.interactive(adata)

    def test_add_same_layer(self, qtbot, adata: AnnData, napari_cont: ImageContainer, capsys):
        from napari.layers import Points

        s.logfile = sys.stderr
        s.verbosity = 4

        viewer = napari_cont.interactive(adata)
        cnt = viewer._controller

        data = np.random.normal(size=adata.n_obs)
        cnt.add_points(data, layer_name="layer1")
        cnt.add_points(np.random.normal(size=adata.n_obs), layer_name="layer1")

        err = capsys.readouterr().err

        assert "Point layer `layer1` is already loaded" in err
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
        viewer = napari_cont.interactive(adata, cont_cmap="inferno")
        cnt = viewer._controller

        cnt.add_points(adata.obs_vector(adata.var_names[42]), layer_name="wex")
        viewer.screenshot(dpi=DPI)

    def test_plot_cat_cmap(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata, cat_cmap="Set3")
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

    def test_plot_add_image(self, qtbot, adata: AnnData, napari_cont: ImageContainer, capsys):
        from napari.layers import Image

        s.logfile = sys.stderr
        s.verbosity = 4

        viewer = napari_cont.interactive(adata)
        cnt = viewer._controller
        img = np.zeros((*napari_cont.shape, napari_cont.nchannels), dtype=np.float32)
        img[..., 0] = 1.0  # all red image

        napari_cont.add_img(img, img_id="foobar")

        cnt.add_image("foobar")
        err = capsys.readouterr().err

        assert "Unable to load spot diameter from `adata" in err
        assert viewer._controller.view.layernames == {"V1_Adult_Mouse_Brain"}

        # dummy value, doesn't matter until we start drawing points
        adata.uns[Key.uns.spatial]["foobar"] = {"scalefactors": {"spot_diameter_fullres": 0}}
        cnt.add_image("foobar")

        assert viewer._controller.view.layernames == {"V1_Adult_Mouse_Brain", "foobar"}
        assert isinstance(viewer._controller.view.layers["foobar"], Image)

        viewer.screenshot(dpi=DPI)
