from pathlib import Path

import napari
from qtpy.QtCore import QTimer

from anndata import AnnData

from squidpy import plotting
from squidpy.image.object import ImageContainer

HERE: Path = Path(__file__).parents[1]
IMG_PATH = HERE / "_data/test_img.jpg"


def test_napari(adata: AnnData):

    library_id = list(adata.uns["spatial"].keys())
    cont = ImageContainer(str(IMG_PATH), img_id=library_id[0])

    gene = adata.var_names[1]
    cluster = "leiden"
    obs_continuous = "leiden_continuous"
    adata.obs[obs_continuous] = adata.obs[cluster].values.astype(int)

    with napari.gui_qt() as app:
        viewer = plotting.interactive(adata, cont, [cluster, gene, obs_continuous], with_qt=False)
        time_in_msec = 1000
        QTimer().singleShot(time_in_msec, app.quit)

    viewer.close()

    assert len(viewer.layers[cluster].data) == adata.shape[0]
    assert viewer.layers[cluster].data[0].shape == (6, 2)
    assert viewer.layers[obs_continuous].data[0].shape == (6, 2)
    assert viewer.layers[gene].data[0].shape == (6, 2)
