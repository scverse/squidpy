import pytest
from napari.conftest import make_test_viewer  # noqa: F401

from anndata import AnnData

from squidpy.im import ImageContainer
from squidpy.pl import interactive


@pytest.fixture(autouse=True, scope="session")
def load_napari_conftest(pytestconfig):
    from napari import conftest

    pytestconfig.pluginmanager.register(conftest, "napari-conftest")


@pytest.mark.qt
@pytest.mark.parametrize(
    "gene, cluster, obs_cont",
    [
        ("Shoc2", "leiden", "leiden_cont"),
    ],
)
def test_viewer(make_test_viewer, adata: AnnData, cont: ImageContainer, gene, cluster, obs_cont):  # noqa: F811

    adata.obs[obs_cont] = adata.obs[cluster].values.astype(int)

    ad2nap = interactive(adata, cont)

    assert ad2nap.viewer.layers[0].name == "Image"
    assert ad2nap.viewer.layers["Image"].data.shape == cont.shape[::-1] + (3,)

    assert ad2nap.get_layer(cluster).shape == cont.shape[::-1] + (4,)
    assert ad2nap.get_layer(obs_cont).shape == cont.shape[::-1]
    assert ad2nap.get_layer(gene).shape == cont.shape[::-1]
