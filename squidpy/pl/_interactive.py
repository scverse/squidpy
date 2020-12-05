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

from squidpy._docs import d
from squidpy.im.object import ImageContainer
from squidpy.constants._pkg_constants import Key


@d.get_full_description(base="AD2NAP")
@d.get_sections(base="AD2NAP", sections=["Parameters", "Returns"])
@d.dedent
class AnnData2Napari:
    """
    Explore AnnData with Napari.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    obsm
        Key in :attr:`anndata.AnnData.obsm` to spatial coordinates.
    library_id
        library id in :attr:`anndata.AnnData.uns.keys()`
    palette
        palette to use for categorical data, if missing in :attr:`anndata.AnnData.uns`

    Returns
    -------
    None
        A Napari instance is launched in the current session.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        img: ImageContainer,
        obsm: str = "spatial",
        library_id: str = None,
        palette: str = "Set1",
    ):

        self.genes_sorted = adata.var_names.sort_values().values
        self.obs_sorted = adata.obs.columns.sort_values().values
        self.coords = adata.obsm[obsm]
        self.palette = palette
        self.adata = adata

        if library_id is None:
            library_id = list(adata.uns["spatial"].keys())[0]

        self.image = img.data[library_id].transpose("y", "x", ...).values

        spot_radius = round(adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"]) * 0.5
        self.mask_lst = self._get_mask(spot_radius)

    def _get_mask(self, spot_radius):
        mask_lst = []
        for i in np.arange(self.coords.shape[0]):
            y, x = self.coords[i, :]
            rr, cc = disk((x, y), spot_radius)
            mask_lst.append((rr, cc))

        return mask_lst

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

    def get_layer(self, name):
        """Get layer from name."""
        if name in self.genes_sorted:
            vec = self._get_gene(name)
        elif name in self.obs_sorted:
            vec = self._get_obs(name)

        if len(vec.shape) > 1:
            _layer = np.zeros(self.image.shape[0:2] + (4,))
        else:
            _layer = np.zeros(self.image.shape[0:2])

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

        gene_widget = self._get_widget(self.genes_sorted, title="Genes")
        obs_widget = self._get_widget(self.obs_sorted, title="Obs")

        @magicgui(call_button="get obs")
        def get_obs_layer() -> None:

            name = obs_widget.currentItem().text()

            _layer = self.get_layer(name)
            logg.warning(f"Loading `{name}` layer")

            if len(_layer.shape) > 2:
                self.viewer.add_image(_layer, name=name, rgb=True)
            else:
                self.viewer.add_image(_layer, name=name, colormap="magma", blending="additive")
            return

        @magicgui(call_button="get gene")
        def get_gene_layer() -> None:

            name = gene_widget.currentItem().text()

            _layer = self.get_layer(name)
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

        return self


@d.dedent
def interactive(
    adata: ad.AnnData,
    img: ImageContainer,
    obsm: str = Key.obsm.spatial,
    library_id: str = None,
    palette: str = "Set1",
    **kwargs,
) -> None:
    """
    %(ADATA2NAPARI.full_desc)s .

    Parameters
    ----------
    %(ADATA2NAPARI.parameters)s

    Returns
    -------
    %(ADATA2NAPARI.returns)s
    """
    return AnnData2Napari(adata, img, obsm, library_id, palette, **kwargs).open_napari()


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
