from typing import Tuple, Union, Literal, Optional, Sequence
from pathlib import Path

import napari
from cycler import Cycler
from magicgui import magicgui
from napari.layers import Points, Shapes
from PyQt5.QtWidgets import QVBoxLayout

from scanpy import logging as logg
from anndata import AnnData
from scanpy.plotting._utils import (
    _set_colors_for_categorical_obs,
    _set_default_colors_for_categorical_obs,
)

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from pandas.api.types import (
    infer_dtype,
    is_object_dtype,
    is_string_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_categorical_dtype,
)

from matplotlib.colors import Colormap, to_rgb

from squidpy._docs import d
from squidpy.im.object import ImageContainer
from squidpy.pl._utils import _min_max_norm, _points_inside_triangles
from squidpy.pl._widgets import ListWidget, ColorBarWidget2, DoubleRangeSlider
from squidpy.constants._pkg_constants import Key


class AnnData2Napari:
    """
    Explore AnnData with Napari.

    napari is launched with AnnData2Napari.open_napari()
    """

    TEXT_SIZE: int = 24
    TEXT_COLOR: str = "white"

    # TODO: paletter -> point_cmap
    # TODO: image cmap if not able to change it?
    # TODO: enabling layers (through some QT widget)
    def __init__(
        self,
        adata: AnnData,
        img: ImageContainer,
        obsm: str = Key.obsm.spatial,
        palette: Union[str, Sequence[str], Cycler] = None,
        color_map: Union[Colormap, str, None] = "viridis",
        library_id: Optional[str] = None,
        key_added: Optional[str] = "selected",
        blending: Optional[str] = "opaque",
    ):
        self._adata = adata
        self._viewer = None
        self._coords = adata.obsm[obsm][:, ::-1]
        self._palette = palette
        self._cmap = color_map
        self._key_added = key_added
        self._layer_blending = blending

        # TODO: empty check
        if library_id is None:
            library_id = list(adata.uns[obsm].keys())[0]

        # TODO: empty check
        library_id_img = list(img.data.keys())[0]

        # TODO: image name for napari layer
        self._image = img.data[library_id_img].transpose("y", "x", ...).values
        # TODO: previously was round(val / 2), tough this visually matches the dot sizes in images
        self._spot_radius = adata.uns[obsm][library_id]["scalefactors"]["spot_diameter_fullres"]

    @property
    @d.dedent
    def adata(self) -> AnnData:
        """%(adata)s"""  # noqa: D400
        return self._adata

    @property
    def viewer(self) -> Optional[napari.Viewer]:
        """:mod:`napari` viewer."""
        return self._viewer

    def _get_gene(self, name: str) -> np.ndarray:
        # TODO: enable raw? binned colormap?
        # Use: adata.obs_vector once layers/raw implemented
        # will need to infer if we're on counts + use binned colormap + disable percentile?
        idx = np.where(name == self.adata.var_names)[0]
        if len(idx):
            return _min_max_norm(self.adata.X[:, idx[0]])

        raise KeyError(f"Name `{name}` not present in `adata.var_names`.")

    def _get_obs(self, name: str) -> np.ndarray:
        ser = self.adata.obs[name]
        if is_categorical_dtype(ser) or is_object_dtype(ser) or is_string_dtype(ser):
            return _get_col_categorical(self.adata, name, self._palette)
        if is_integer_dtype(ser) and ser.nunique() <= 2:  # most likely a boolean
            self.adata.obs[name] = self.adata.obs[name].astype(bool).astype("category")
            return _get_col_categorical(self.adata, name, self._palette)

        if is_numeric_dtype(ser):
            return _min_max_norm(ser.values)

        raise TypeError(f"Invalid column type `{infer_dtype(ser)}` for `adata.obs[{name!r}]`.")

    def _get_layer(self, name: str) -> np.ndarray:
        """Get layer from name."""
        if name in self.adata.var_names:
            return self._get_gene(name)
        if name in self.adata.obs.columns:
            return self._get_obs(name)

        raise KeyError(f"`{name}` is not present in either `adata.var_names` or `adata.obs`.")

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

        def point_closest_to_med(needles: pd.Series, haystack: pd.Series):
            # return needles.iloc[np.argmin(np.sum((needles - haystack) ** 2))]
            ix = np.argmin(np.sum((needles - haystack) ** 2))
            res = [False] * len(needles)
            res[ix] = True
            return res

        @magicgui(call_button="Select observation")
        def get_obs_layer(items=None) -> None:
            # TODO: async?
            layer = None
            for item in obs_widget.selectedItems() if items is None else items:
                name = item if isinstance(item, str) else item.text()
                if name in self.viewer.layers:
                    logg.warning(f"Selected layer `{name}` is already loaded")
                    continue
                _layer = self._get_layer(name)

                # TODO: be more robust when determining categorical (refactor the whole FN)
                # TODO: constant ("value")
                face_color = _layer if isinstance(_layer[0], np.ndarray) else "value"
                is_categorical = not isinstance(face_color, str)
                properties, text = {"value": _layer}, None

                if is_categorical:
                    df = pd.DataFrame(self._coords)
                    df[name] = self.adata.obs[name].values
                    df = df.groupby(name)[[0, 1]].apply(lambda g: list(np.median(g.values, axis=0)))
                    df = pd.DataFrame(r for r in df)

                    kdtree = KDTree(self._coords)
                    clusters = np.full(
                        (
                            len(
                                self._coords,
                            )
                        ),
                        fill_value="",
                    )
                    clusters[kdtree.query(df.values)[1]] = df.index
                    properties["cluster"] = clusters

                    text = {
                        "text": "{cluster}",
                        "size": self.TEXT_SIZE,
                        "color": self.TEXT_COLOR,
                        "anchor": "center",
                        "blending": "opaque",
                    }

                logg.info(f"Loading `{name}` layer")
                # TODO: disable already added points?
                layer = self.viewer.add_points(
                    self._coords,
                    name=name,
                    size=self._spot_radius,
                    face_color=face_color,
                    edge_width=1,
                    text=text,
                    blending=self._layer_blending,
                    properties=properties,
                )

                layer.editable = False
                layer.selected = False
                self._hide_point_controls(layer)

                # if it's categorical, remove the slider from bottom and add labels
                if is_categorical:
                    layer.events.select.connect(lambda e: slider.setVisible(False))

            if layer is not None:
                layer.selected = True

        @magicgui(call_button="Select gene")
        def get_gene_layer(items=None) -> None:
            # TODO: async?
            layer = None
            for item in gene_widget.selectedItems() if items is None else items:
                name = item if isinstance(item, str) else item.text()
                if name in self.viewer.layers:
                    logg.warning(f"Selected layer `{name}` is already loaded")
                    continue
                _layer = self._get_layer(name)

                logg.info(f"Loading `{name}` layer")
                # TODO: disable already added points?
                layer = self.viewer.add_points(
                    self._coords,
                    name=name,
                    size=self._spot_radius,
                    face_color="value",
                    face_colormap=self._cmap,
                    edge_width=1,
                    blending=self._layer_blending,
                    properties={"value": _layer},
                    # percentile metadata
                    metadata={"min": 0, "max": 100, "data": _layer},
                )
                layer.editable = False
                layer.selected = False
                layer.events.select.connect(selected_handler)
                self._hide_point_controls(layer)

            if layer is not None:
                layer.selected = True

        def selected_handler(event) -> None:
            source: Points = event.source
            if source.selected:
                # restore slider to the selected's ranges
                # TODO: constants
                val = (source.metadata["min"], source.metadata["max"])
                slider.setValue(val)
                cbw.setClim((val[0] / 100, val[1] / 100))
                cbw.update_color()

                slider.setVisible(True)

        @magicgui(
            auto_call=True,
            labels=False,  # TODO: setVisible(False) doesn't remove the label
            percentile={
                "widget_type": DoubleRangeSlider,  # TODO: maybe use QHRangeSlider from napari
                "minimum": 0,
                "maximum": 100,
                "value": (0, 100),
                "visible": False,
            },
        )
        # TODO: generalize? i.e. user function?
        def clip(percentile: Tuple[float, float] = (0, 100)) -> None:
            # TODO: async?
            for layer in self.viewer.layers:
                # multiple can be selected
                if isinstance(layer, Points) and layer.selected:
                    v = layer.metadata["data"]

                    clipped = np.clip(v, *np.percentile(v, percentile))
                    # save the percentile
                    layer.metadata = {**layer.metadata, "min": percentile[0], "max": percentile[1]}
                    # TODO: constant
                    layer.face_color = "value"
                    layer.properties = {"value": clipped}
                    layer._update_thumbnail()  # can't find another way to force it
                    layer.refresh_colors()

                    cbw.setClim((np.min(clipped), np.max(clipped)))

        def export(viewer: napari.Viewer) -> None:
            # TODO: async?
            for layer in viewer.layers:
                if isinstance(layer, Shapes) and layer.selected:
                    if not len(layer.data):
                        logg.warning(f"Shape layer `{layer.name}` has no visible shapes")
                        continue
                    shape_list = layer._data_view
                    triangles = shape_list._mesh.vertices[shape_list._mesh.displayed_triangles]

                    logg.info(f"Adding `adata.obs[{layer.name!r}]`\n       `adata.uns[{layer.name}!r]['meshes']`")

                    key = f"{layer.name}_{self._key_added}"
                    self.adata.obs[key] = pd.Categorical(_points_inside_triangles(self._coords, triangles))
                    self.adata.uns[key] = {"meshes": layer.data.copy()}

                    # handles uniqueness + sorting + non iterable
                    obs_widget.addItems(key)
                    # update already present layer
                    if key in viewer.layers:
                        layer = viewer.layers[key]
                        layer.face_color = _get_col_categorical(self.adata, key)
                        layer._update_thumbnail()
                        layer.refresh_colors()

        with napari.gui_qt():
            self._viewer = napari.view_image(self._image, **kwargs)
            # TODO: use hidden + hide also cbar
            self.viewer.layers[0].events.select.connect(lambda e: slider.setVisible(False))
            self.viewer.bind_key("Shift-E", export)

            # Select genes widget
            gene_widget = ListWidget(self.adata.var_names, title="Genes")
            gene_btn = get_gene_layer.Gui()
            gene_widget.enter_pressed.connect(gene_btn)
            gene_widget.doubleClicked.connect(lambda item: get_gene_layer(items=(item.data(),)))

            # Select observations widget
            obs_widget = ListWidget(self.adata.obs.columns, title="Observations")
            obs_btn = get_obs_layer.Gui()
            obs_widget.enter_pressed.connect(obs_btn)
            obs_widget.doubleClicked.connect(lambda item: get_obs_layer(items=(item.data(),)))

            layout = QVBoxLayout()

            cgui = clip.Gui()
            # TODO: enable slider for obs
            slider: DoubleRangeSlider = cgui.get_widget("percentile")

            cbw = ColorBarWidget2(self._cmap)
            cbw.setLayout(layout)
            layout.addWidget(cgui)

            # ideally, we would inject this to `Points` widget group, but it would be very hacky/brittle
            self.viewer.window.add_dock_widget([cgui, cbw], area="left", name="Percentile")

            # TODO: see if we can disallow deleting the image layer (e.g. by consuming deleting event on that layer)
            self._viewer.window.add_dock_widget(
                # TODO: the btns are a bit redundant, since pressing ENTER works
                # maybe we can remove them and add instead QLabels on top
                [gene_widget, gene_btn, obs_widget, obs_btn],
                area="right",
                name="genes",
            )

            return self

    def _hide_point_controls(self, layer: Points) -> None:
        # TODO: move this up
        to_hide = {
            "symbol:": "symbolComboBox",
            "point size:": "sizeSlider",
            "face color:": "faceColorEdit",
            "edge color:": "edgeColorEdit",
            "n-dim:": "ndimCheckBox",
        }
        points_controls = self.viewer.window.qt_viewer.controls.widgets[layer]

        from qtpy.QtWidgets import QLabel, QGridLayout

        gl: QGridLayout = points_controls.grid_layout

        labels = {}
        for i in range(gl.count()):
            item = gl.itemAt(i).widget()
            if isinstance(item, QLabel):
                labels[item.text()] = item

        for key, attr in to_hide.items():
            attr = getattr(points_controls, attr, None)
            if key in labels and attr is not None:
                attr.setHidden(True)
                labels[key].setHidden(True)

    def screenshot(self, path: Optional[Union[str, Path]] = None) -> Optional[np.ndarray]:
        """
        Take a screenshot.

        Parameters
        ----------
        path
            If `None`, don't save the screenshot.

        Returns
        -------
        :class:`numpy.ndarray`
            The screenshot.
        """
        if self.viewer is None:
            raise RuntimeError("No viewer is initialized.")
        return self.viewer.screenshot(path, canvas_only=True)


@d.dedent
def interactive(
    adata: AnnData,
    img: ImageContainer,
    obsm: str = Key.obsm.spatial,
    # TODO: make sure we're passing correct pallette
    # TODO: handle None palette?
    palette: Union[str, Sequence[str], Cycler] = None,
    color_map: Optional[Union[Colormap, str]] = "viridis",
    library_id: Optional[str] = None,
    key_added: Optional[str] = "selected",
    blending: Literal["translucent", "opaque", "additive"] = "opaque",
    **kwargs,
) -> AnnData2Napari:
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
    # TODO: only HVG subset
    return AnnData2Napari(
        adata,
        img=img,
        obsm=obsm,
        library_id=library_id,
        palette=palette,
        color_map=color_map,
        key_added=key_added,
        blending=blending,
    ).open_napari(**kwargs)


def _get_col_categorical(adata: AnnData, c: str, _palette=None) -> np.ndarray:
    # TODO: nice-to-have enable colorbar in Qt
    colors_key = f"{c}_colors"
    if colors_key not in adata.uns.keys():
        # TODO: this needs a categorical palette, not continuous
        if _palette is not None:
            _set_colors_for_categorical_obs(adata, c, _palette)
        else:
            _set_default_colors_for_categorical_obs(adata, c)
    cols = [to_rgb(i) for i in adata.uns[colors_key]]

    col_dict = dict(zip(adata.obs[c].cat.categories, cols))
    return np.array([col_dict[v] for v in adata.obs[c]])
