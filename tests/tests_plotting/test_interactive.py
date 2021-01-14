import pytest

from anndata import AnnData

from squidpy.im import ImageContainer
from tests.conftest import DPI, PlotTester, PlotTesterMeta


@pytest.mark.qt
class TestNapari(PlotTester, metaclass=PlotTesterMeta):
    def test_no_valid_libraries(self, qtbot, adata: AnnData, cont: ImageContainer):
        with pytest.raises(ValueError, match=r"Unable to find any valid libraries\..*"):
            _ = cont.interactive(adata)

    def test_plot_simple_canvas(self, qtbot, adata: AnnData, napari_cont: ImageContainer):
        viewer = napari_cont.interactive(adata)

        assert viewer._controller.view.layernames == {"V1_Adult_Mouse_Brain"}

        viewer.screenshot(dpi=DPI)

    def _test_plot_gene_X(self):
        pass

    def _test_plot_gene_layer(self):
        pass

    def _test_plot_gene_raw(self):
        pass

    def _test_plot_obs_continuous(self):
        pass

    def _test_plot_obs_categorical(self):
        pass

    def _test_plot_obsm(self):
        pass

    def _test_plot_cmap(self):
        pass

    def _test_add_shapes(self):
        # may not be feasible
        pass

    def _test_plot_add_image(self):
        pass
