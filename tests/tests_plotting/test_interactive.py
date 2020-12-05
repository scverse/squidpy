from pathlib import Path

import napari
from qtpy.QtCore import QTimer

from anndata import AnnData

from squidpy.im import ImageContainer
from squidpy.pl import interactive

HERE: Path = Path(__file__).parents[1]
IMG_PATH = HERE / "_data/test_img.jpg"


def test_napari(adata: AnnData):

    library_id = list(adata.uns["spatial"].keys())
    img = ImageContainer(str(IMG_PATH), img_id=library_id[0])

    gene = adata.var_names[1]
    cluster = "leiden"
    obs_continuous = "leiden_continuous"
    adata.obs[obs_continuous] = adata.obs[cluster].values.astype(int)

    with napari.gui_qt() as app:
        ad2nap = interactive(adata, img)
        time_in_msec = 1000
        QTimer().singleShot(time_in_msec, app.quit)

    ad2nap.viewer.close()

    assert ad2nap.get_layer(cluster).shape == img.shape[::-1] + (4,)
    assert ad2nap.get_layer(obs_continuous).shape == img.shape[::-1]
    assert ad2nap.get_layer(gene).shape == img.shape[::-1]
