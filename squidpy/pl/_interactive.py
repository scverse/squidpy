import napari
from magicgui import magicgui
from PyQt5.QtWidgets import QListWidget

import anndata as ad
from scanpy import logging as logg

import numpy as np
from scipy.sparse import csr_matrix
from pandas.api.types import is_categorical_dtype

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from skimage.draw import disk

from squidpy.im.object import ImageContainer


class AnnData2Napari:
    """
    Class to interact with Napari.

    Parameters
    ----------
    adata
        the adata object

    """

    def __init__(
        self,
        adata: ad.AnnData,
        img: ImageContainer,
        obsm_key: str = "spatial",
        library_id: str = None,
        palette: str = "Set1",
    ) -> None:

        self.genes = adata.var_names.sort_values().values
        self.obs = adata.obs.columns.sort_values().values

        self.coords = adata.obsm[obsm_key]
        self.palette = palette

        self.adata = adata
        self.image = self._get_image(img, library_id)
        self.img_shape = img.shape[0:2]

        self.mask_lst = _get_mask(self.adata, self.coords)

    def _get_image(self, img: ImageContainer, library_id: str = None):

        if library_id is None:
            library_id = list(img.data.keys())[0]

        image = img.data[library_id].transpose("y", "x", ...).values

        return image

    def _get_gene(self, name):

        idx = np.where(name == self.adata.var_names)[0][0]

        if isinstance(self.adata.X, csr_matrix):
            vec = self.adata.X[:, idx].todense()
        else:
            vec = self.adata.X[:, idx]

        vec = (vec - vec.min()) / (vec.max() - vec.min())
        vec = np.array(vec).squeeze()
        return vec

    def _get_obs(self, name):

        if is_categorical_dtype(self.adata.obs[name]):
            vec = _get_col_categorical(self.adata, name, self.palette)
        else:
            vec = self.adata.obs[name].values
            vec = (vec - vec.min()) / (vec.max() - vec.min())

        return vec

    def _get_layer(self, name):

        if name in self.genes:
            vec = self._get_gene(name)
        elif name in self.obs:
            vec = self._get_obs(name)

        if len(vec.shape) > 1:
            _layer = np.zeros(self.img_shape + (4,))
        else:
            _layer = np.zeros(self.img_shape)

        for i, c in enumerate(self.mask_lst):
            rr, cc = c
            _layer[rr, cc] = vec[i]

        return _layer

    def _get_widget(
        self,
        obj_lst: list,
        title: str = None,
    ):

        list_widget = QListWidget()
        list_widget.setWindowTitle(title)
        for i in obj_lst:
            list_widget.addItem(i)
        return list_widget

    def open_napari(
        self,
    ):
        """Launch Napari."""
        self.viewer = napari.view_image(self.image, rgb=True)

        gene_widget = self._get_widget(self.genes, title="Genes")
        obs_widget = self._get_widget(self.obs, title="Obs")

        @magicgui(call_button="get obs")
        def get_obs_layer() -> None:

            name = obs_widget.currentItem().text()

            _layer = self._get_layer(name)
            logg.warning(f"Loading `{name}` layer")

            if len(_layer.shape) > 2:
                self.viewer.add_image(_layer, name=name, rgb=True)
            else:
                self.viewer.add_image(_layer, name=name, colormap="magma", blending="additive")
            return

        @magicgui(call_button="get gene")
        def get_gene_layer() -> None:

            name = gene_widget.currentItem().text()

            _layer = self._get_layer(name)
            logg.warning(f"Loading `{name}` layer")

            self.viewer.add_image(_layer, name=name, colormap="magma", blending="additive")
            return

        gene_exec = get_gene_layer.Gui()
        gene_widget.itemChanged.connect(gene_exec)

        obs_exec = get_obs_layer.Gui()
        obs_widget.itemChanged.connect(obs_exec)

        self.viewer.window.add_dock_widget(
            [gene_widget, gene_exec, obs_widget, obs_exec],
            area="right",
            name="genes",
        )

        return


def _get_mask(adata, coords):
    mask_lst = []

    spot_radius = _get_radius(adata)
    for i in np.arange(coords.shape[0]):
        y, x = coords[i, :]
        rr, cc = disk((x, y), spot_radius)
        mask_lst.append((rr, cc))

    return mask_lst


def _get_radius(adata):
    library_id = list(adata.uns["spatial"].keys())
    spot_radius = round(adata.uns["spatial"][library_id[0]]["scalefactors"]["spot_diameter_fullres"]) * 0.5
    return spot_radius


def _get_col_categorical(adata, c, palette):

    colors_key = f"{c}_colors"
    if colors_key in adata.uns.keys():
        cols = [to_rgba(i) for i in adata.uns[colors_key]]
    else:
        ncat = adata.obs[c].cat.categories.shape[0]
        cmap = plt.get_cmap(palette, ncat)
        cols = cmap(np.arange(ncat))

    cols_dict = {k: v for v, k in zip(cols, adata.obs[c].cat.categories)}
    cols = np.array([cols_dict[k] for k in adata.obs[c]])

    return cols
