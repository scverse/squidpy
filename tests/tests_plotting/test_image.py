from anndata import AnnData

from squidpy import plotting
from squidpy.image.object import ImageContainer


def test_napari(adata: AnnData):

    img_path = "./../_data/test_data.h5ad"
    cont = ImageContainer(img_path)

    gene = adata.var_names[1]
    cluster = "leiden"

    viewer = plotting.interactive(adata, cont, [cluster, gene])

    assert len(viewer.layers[cluster].data) == adata.shape[0]
    assert viewer.layers[cluster].data[0].shape == (6, 2)
    assert viewer.layers[gene].data[0].shape == (6, 2)
