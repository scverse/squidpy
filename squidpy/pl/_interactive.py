from typing import List, Union, Optional, Sequence
from functools import partial

import napari
from cycler import Cycler
from magicgui import magicgui
from PyQt5.QtWidgets import QListWidget

from scanpy import logging as logg
from anndata import AnnData
from scanpy.plotting._utils import (
    _set_colors_for_categorical_obs,
    _set_default_colors_for_categorical_obs,
)

import numpy as np
from scipy.sparse import spmatrix
from pandas.api.types import is_object_dtype, is_categorical_dtype

from matplotlib.colors import to_rgba

from skimage.draw import disk

from squidpy._docs import d
from squidpy.im.object import ImageContainer
from squidpy.constants._pkg_constants import Key


class AnnData2Napari:
    """
    Explore AnnData with Napari.

    napari is launched with AnnData2Napari.open_napari()
    """

    def __init__(
        self,
        adata: AnnData,
        img: ImageContainer,
        obsm: str = Key.obsm.spatial,
        palette: Union[str, Sequence[str], Cycler] = None,
        library_id: Optional[str] = None,
    ):

        self._genes_sorted = adata.var_names.sort_values().values
        self._obs_sorted = adata.obs.columns.sort_values().values
        self._coords = adata.obsm[obsm]
        self._palette = palette
        self._adata = adata

        if library_id is None:
            library_id = list(adata.uns[obsm].keys())[0]

        library_id_img = list(img.data.keys())[0]

        self._image = img.data[library_id_img].transpose("y", "x", ...).values

        spot_radius = round(adata.uns[obsm][library_id]["scalefactors"]["spot_diameter_fullres"]) * 0.5
        self._masks = self._get_mask(spot_radius)

    @property
    def masks(self) -> np.ndarray:
        """Get spot masks as list."""
        return self._masks

    def _get_mask(self, spot_radius):

        dsk = partial(disk, radius=spot_radius)
        vfunc = np.vectorize(lambda y, x: dsk((x, y)), otypes=[tuple])
        mask_lst = vfunc(self._coords[:, 0], self._coords[:, 1])

        return mask_lst

    def _get_gene(self, name):

        idx = np.where(name == self._adata.var_names)[0]
        if not len(idx):
            raise ValueError(f"{name} not present in `adata.var_names`")
        else:
            idx = idx[0]

        if isinstance(self._adata.X, spmatrix):
            vec = self._adata.X[:, idx].todense()
        else:
            vec = self._adata.X[:, idx]

        vec = (vec - vec.min()) / (vec.max() - vec.min())
        vec = np.array(vec).squeeze()
        return vec

    def _get_obs(self, name):

        ser = self._adata.obs[name].copy()
        if is_categorical_dtype(ser):
            vec = _get_col_categorical(self._adata, name, self._palette)
        elif is_object_dtype(ser) and not is_categorical_dtype(ser):
            ser = ser.astype("category")
            vec = _get_col_categorical(self._adata, name, self._palette)
        else:
            vec = ser.values
            vec = (vec - vec.min()) / (vec.max() - vec.min())

        return vec

    def get_layer(self, name):
        """Get layer from name."""
        if name in self._genes_sorted:
            vec = self._get_gene(name)
        elif name in self._obs_sorted:
            vec = self._get_obs(name)
        else:
            raise KeyError(f"{name} not present in either `var_names` or `obs`")
        if vec.ndim > 1:
            _layer = np.zeros(self._image.shape[0:2] + (4,))
        else:
            _layer = np.zeros(self._image.shape[0:2])

        for arr, v in zip(self._masks, vec):
            rr, cc = arr
            _layer[rr, cc] = v

        return _layer

    def _get_widget(
        self,
        obj_lst: List[str],
        title: Optional[str] = None,
    ):

        list_widget = QListWidget()
        list_widget.setWindowTitle(title)
        for i in obj_lst:
            list_widget.addItem(i)
        return list_widget

    def open_napari(self, **kwargs):
        """Launch Napari."""
        self.viewer = napari.view_image(self._image, rgb=True, **kwargs)

        gene_widget = self._get_widget(self._genes_sorted, title="Genes")
        obs_widget = self._get_widget(self._obs_sorted, title="Observations")

        @magicgui(call_button="Select observation")
        def get_obs_layer() -> None:

            name = obs_widget.currentItem().text()

            _layer = self.get_layer(name)
            logg.warning(f"Loading `{name}` layer")

            if _layer.ndim > 2:
                self.viewer.add_image(
                    _layer,
                    name=name,
                    rgb=True,
                )
            else:
                self.viewer.add_image(_layer, name=name, colormap="magma", blending="additive")
            return

        @magicgui(call_button="Select gene")
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
    adata: AnnData,
    img: ImageContainer,
    obsm: str = Key.obsm.spatial,
    palette: Union[str, Sequence[str], Cycler] = None,
    library_id: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Explore AnnData with Napari.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    obsm
        Key in :attr:`anndata.AnnData.obsm` to spatial coordinates.
    library_id
        library id in :attr:`anndata.AnnData.uns`.
    palette
        Palette should be either a valid :func:`~matplotlib.pyplot.colormaps` string,
        a sequence of colors (in a format that can be understood by matplotlib,
        eg. RGB, RGBS, hex, or a cycler object with key='color'.

    Returns
    -------
    None
        A Napari instance is launched in the current session.
    """
    return AnnData2Napari(adata, img, obsm, library_id, palette, **kwargs).open_napari(**kwargs)


def _get_col_categorical(adata, c, palette):
    colors_key = f"{c}_colors"
    if colors_key not in adata.uns.keys():
        if palette:
            _set_colors_for_categorical_obs(adata, c, palette)
        else:
            _set_default_colors_for_categorical_obs(adata, c)
    cols = [to_rgba(i) for i in adata.uns[colors_key]]

    cols_dict = {k: v for v, k in zip(cols, adata.obs[c].cat.categories)}
    cols = np.array([cols_dict[k] for k in adata.obs[c]])

    return cols
