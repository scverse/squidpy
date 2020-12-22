from conftest import PlotTester, PlotTesterMeta
import pytest

from anndata import AnnData

import matplotlib.pyplot as plt

from squidpy.im import ImageContainer
from squidpy.pl import interactive


@pytest.mark.qt
class TestNapari(PlotTester, metaclass=PlotTesterMeta):
    @pytest.mark.parametrize(
        ("gene", "cluster", "obs_cont"),
        [
            ("Shoc2", "leiden", "leiden_cont"),
        ],
    )
    @pytest.mark.usefixtures("_test_napari")
    @pytest.mark.skip("FIXME: layers is empty")
    def test_plot_viewer_canvas(self, qtbot, adata: AnnData, cont: ImageContainer, gene, cluster, obs_cont):

        adata.obs[obs_cont] = adata.obs[cluster].values.astype(int)

        ad2nap = interactive(adata, cont)
        viewer = ad2nap.viewer

        assert viewer.layers[0].name == "Image"
        assert viewer.layers["Image"].data.shape == cont.shape[::-1] + (3,)

        img = ad2nap.screenshot()

        plt.imshow(img)
        # TODO:
        # assert ad2nap.get_layer(cluster).shape == cont.shape[::-1] + (4,)
        # assert ad2nap.get_layer(obs_cont).shape == cont.shape[::-1]
        # assert ad2nap.get_layer(gene).shape == cont.shape[::-1]
