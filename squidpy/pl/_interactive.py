from typing import List, Union, Callable, Optional, Sequence
from functools import partial, lru_cache

try:
    from functools import cached_property
except ImportError:  # <3.8

    def cached_property(fn: Callable) -> Callable:  # noqa: D103
        return lru_cache(maxsize=1)(property(fn))


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
from pandas.api.types import (
    infer_dtype,
    is_object_dtype,
    is_string_dtype,
    is_numeric_dtype,
    is_categorical_dtype,
)

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
        palette: Optional[Union[str, Sequence[str], Cycler]] = None,
        library_id: Optional[str] = None,
    ):

        self._genes_sorted = adata.var_names.sort_values().values
        self._obs_sorted = adata.obs.columns.sort_values().values
        self._coords = adata.obsm[obsm][:, ::-1]
        self._palette = palette
        self._adata = adata
        self._viewer = None

        if library_id is None:
            library_id = list(adata.uns[obsm].keys())[0]

        library_id_img = list(img.data.keys())[0]

        self._image = img.data[library_id_img].transpose("y", "x", ...).values
        self._spot_radius = round(adata.uns[obsm][library_id]["scalefactors"]["spot_diameter_fullres"]) * 0.5

    @cached_property
    def masks(self) -> np.ndarray:
        """Get spot masks."""
        # n x s x 2
        return np.apply_along_axis(partial(disk, radius=self._spot_radius), 1, self._coords[:, ::-1])

    @property
    def viewer(self):
        """:mod:`napari` viewer."""
        return self._viewer

    def _get_gene(self, name: str) -> np.ndarray:

        idx = np.where(name == self._adata.var_names)[0]
        if not len(idx):
            raise KeyError(f"Name `{name}` not present in `adata.var_names`.")
        idx = idx[0]

        # TODO: use_get_var
        if isinstance(self._adata.X, spmatrix):
            vec = self._adata.X[:, idx].todense()
        else:
            vec = self._adata.X[:, idx]

        vec = (vec - vec.min()) / (vec.max() - vec.min())
        vec = np.array(vec).squeeze()
        return vec

    def _get_obs(self, name):
        ser = self._adata.obs[name]
        if is_categorical_dtype(ser) or is_object_dtype(ser) or is_string_dtype(ser):
            return _get_col_categorical(self._adata, name, self._palette)

        if is_numeric_dtype(ser):
            vec = ser.values
            return (vec - vec.min()) / (vec.max() - vec.min())

        raise TypeError(f"Invalid column type `{infer_dtype(ser)}` for `adata.obs[{name!r}]`.")

    @lru_cache(maxsize=256)
    def _get_layer(self, name: str) -> np.ndarray:
        """Get layer from name."""
        if name in self._genes_sorted:
            vec = self._get_gene(name)
        elif name in self._obs_sorted:
            vec = self._get_obs(name)
        else:
            raise KeyError(f"`{name}` is not present in either `adata.var_names` or `adata.obs`.")
        return vec

    def _get_widget(
        self,
        obj_lst: List[str],
        title: Optional[str] = None,
    ) -> QListWidget:

        list_widget = QListWidget()
        list_widget.setWindowTitle(title)
        for i in obj_lst:
            list_widget.addItem(i)
        return list_widget

    def open_napari(self, **kwargs) -> "AnnData2Napari":
        """
        Launch :mod:`napari`.

        Parameters
        ----------
        kwargs
            Keyword arguments for :func:`napari.view_image`.

        Returns
        -------
        TODO.
            TODO.
        """
        self._viewer = napari.view_image(self._image, rgb=True, **kwargs)

        gene_widget = self._get_widget(self._genes_sorted, title="Genes")
        obs_widget = self._get_widget(self._obs_sorted, title="Observations")

        @magicgui(call_button="Select observation")
        def get_obs_layer() -> None:

            name = obs_widget.currentItem().text()

            _layer = self._get_layer(name)
            if isinstance(_layer[0], tuple):
                face_color = list(map(list, _layer))
                properties = None
            else:
                face_color = "val"
                properties = {"val": _layer}

            logg.info(f"Loading `{name}` layer")
            self._viewer.add_points(
                self._coords,
                size=self._spot_radius,
                face_color=face_color,
                properties=properties,
                name=name,
                face_colormap="magma",
                blending="additive",
            )

        @magicgui(call_button="Select gene")
        def get_gene_layer() -> None:

            name = gene_widget.currentItem().text()

            _layer = self._get_layer(name)
            logg.info(f"Loading `{name}` layer")
            properties = {"val": _layer}
            self._viewer.add_points(
                self._coords,
                size=self._spot_radius,
                face_color="val",
                properties=properties,
                name=name,
                face_colormap="magma",
                blending="additive",
            )

        gene_exec = get_gene_layer.Gui()
        gene_widget.itemChanged.connect(gene_exec)

        obs_exec = get_obs_layer.Gui()
        obs_widget.itemChanged.connect(obs_exec)

        self._viewer.window.add_dock_widget(
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
    Explore :mod:`anndata` with :mod:`napari`.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    obsm
        Key in :attr:`anndata.AnnData.obsm` to spatial coordinates.
    library_id
        Library id in :attr:`anndata.AnnData.uns`.
    palette
        Palette should be either a valid :func:`~matplotlib.pyplot.colormaps` string,
        a sequence of colors (in a format that can be understood by :mod:`matplotlib`,
        eg. RGB, RGBS, hex, or a cycler object with key='color'.

    Returns
    -------
    TODO
        TODO.
    """
    return AnnData2Napari(adata, img, obsm, library_id, palette).open_napari(**kwargs)


def _get_col_categorical(adata, c, palette):
    colors_key = f"{c}_colors"
    if colors_key not in adata.uns.keys():
        if palette is not None:
            _set_colors_for_categorical_obs(adata, c, palette)
        else:
            _set_default_colors_for_categorical_obs(adata, c)
    cols = [to_rgba(i) for i in adata.uns[colors_key]]

    col_dict = dict(zip(adata.obs[c].cat.categories, cols))
    return adata.obs[c].astype(str).apply(lambda r: col_dict[r]).to_numpy()
