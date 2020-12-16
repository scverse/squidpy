from typing import Dict, Union, Literal, Optional, Sequence
from pathlib import Path

import napari
from cycler import Cycler
from napari.layers import Points, Shapes
from PyQt5.QtWidgets import (
    QLabel,
    QWidget,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
)

from scanpy import logging as logg
from anndata import AnnData
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from pandas.api.types import infer_dtype, is_categorical_dtype

from matplotlib.colors import Colormap, to_rgb

from squidpy._docs import d
from squidpy.im.object import ImageContainer
from squidpy.pl._utils import ALayer, _points_inside_triangles
from squidpy.pl._widgets import CBarWidget, AListWidget, RangeSlider, ObsmIndexWidget
from squidpy.constants._pkg_constants import Key


class AnnData2Napari:
    """
    Explore AnnData with Napari.

    :class:`napari.Viewer` is launched with :meth:`open_napari`.
    """

    TEXT_SIZE: int = 24

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
        # TODO: widget for ImageContainerLayer
        self._image = img.data[library_id_img].transpose("y", "x", ...).values
        self._spot_radius = adata.uns[obsm][library_id]["scalefactors"]["spot_diameter_fullres"]

        # TODO:
        # other idea is to have this as a context manager for ImageContainer (which would need to save adata object)
        # example usage:
        # ic = ImageContainer(adata, ...)
        # with ic.interactive(...) as interactive:  # here we build this object
        #     interactive.open(...)
        #     print(interactive.viewer)
        #     interactive.screenshot()
        # current problem is that we don't clean up .viewer even after the session has been closed
        # with CTX manager, we could easily do it

        # UI
        # TODO: make local?
        self._colorbar = None

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

        def export(viewer: napari.Viewer) -> None:
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
                    # TODO: use layer.name...
                    if key in viewer.layers:
                        layer = viewer.layers[key]
                        layer.face_color = _get_categorical(self.adata, key)
                        layer._update_thumbnail()
                        layer.refresh_colors()

        # TODO: separate GUI initialization from showing, i.e. initialize all req widgets in a separate
        # TODO: method called from init, then in this function just open napari
        with napari.gui_qt():
            self._viewer = napari.view_image(self._image, **kwargs)
            self.viewer.bind_key("Shift-E", export)

            alayer = ALayer(self.adata)
            parent = self.viewer.window._qt_window

            # gene
            var_lab = QLabel("Genes[default]:")
            var_lab.setToolTip("Select gene expression")
            var_widget = AListWidget(alayer, attr="var", controller=self)

            # obs
            obs_label = QLabel("Observations:")
            obs_label.setToolTip("TODO")
            obs_widget = AListWidget(alayer, attr="obs", controller=self)

            # obsm
            obsm_label = QLabel("Obsm:", parent=parent)
            obsm_label.setToolTip("TODO")
            obsm_widget = AListWidget(alayer, attr="obsm", controller=self, multiselect=False, parent=parent)
            obsm_index_widget = ObsmIndexWidget(alayer, parent=parent)
            obsm_index_widget.setToolTip("Select dimension.")
            obsm_index_widget.currentTextChanged.connect(obsm_widget.setIndex)
            obsm_widget.itemClicked.connect(obsm_index_widget.addItems)

            # layer selection
            layer_label = QLabel("Layers:", parent=parent)
            layer_widget = QComboBox(parent=parent)
            layer_widget.addItem("default", None)
            layer_widget.addItems(self.adata.layers.keys())
            layer_widget.currentTextChanged.connect(var_widget.setLayer)
            layer_widget.currentTextChanged.connect(lambda text: var_lab.setText(f"Genes[{text}]:"))
            layer_widget.setCurrentIndex(0)

            # raw selection
            raw_widget = QWidget()
            raw_layout = QHBoxLayout()
            raw_label = QLabel("Raw:")
            raw_label.setToolTip("Access the .raw attribute.")
            raw = QCheckBox(parent=parent)
            raw.setChecked(False)
            raw.stateChanged.connect(layer_widget.setDisabled)
            raw.stateChanged.connect(lambda state: var_widget.setRaw(state))
            raw.stateChanged.connect(lambda state: var_lab.setText("Genes[raw]:" if state else "Genes:"))
            raw_layout.addWidget(raw_label)
            raw_layout.addWidget(raw)
            raw_layout.addStretch()
            raw_widget.setLayout(raw_layout)

            self._colorbar = CBarWidget(self._cmap, height=50)

            self.viewer.window.add_dock_widget(self._colorbar, area="left", name="Percentile")
            self._viewer.window.add_dock_widget(
                [
                    layer_label,
                    layer_widget,
                    raw_widget,
                    var_lab,
                    var_widget,
                    obs_label,
                    obs_widget,
                    obsm_label,
                    obsm_widget,
                    obsm_index_widget,
                ],
                area="right",
                name="genes",
            )

            return self

    def _get_label_positions(self, vec: pd.Series, col_dict: dict) -> Dict[str, np.ndarray]:
        # TODO: do something more clever
        df = pd.DataFrame(self._coords)
        df["clusters"] = vec.values
        df = df.groupby("clusters")[[0, 1]].apply(lambda g: list(np.median(g.values, axis=0)))
        df = pd.DataFrame((r for r in df), index=df.index)

        kdtree = KDTree(self._coords)
        clusters = np.full(
            (
                len(
                    self._coords,
                )
            ),
            fill_value="",
            dtype=object,
        )
        # index consists of the categories that need not be string
        clusters[kdtree.query(df.values)[1]] = df.index.astype(str)
        colors = np.array([col_dict[v] if v != "" else (0, 0, 0) for v in vec])

        return {"clusters": clusters, "colors": colors}

    def _add_points(self, vec: Union[np.ndarray, pd.Series], key: str, layer_name: str) -> None:
        def move_to_front(_):
            if not layer.visible:
                return
            try:
                index = self.viewer.layers.index(layer)
            except ValueError:
                return

            self.viewer.layers.move(index, -1)

        if layer_name in (_lay.name for _lay in self.viewer.layers):
            logg.warning(f"Selected layer `{layer_name}` is already loaded")
            return

        if isinstance(vec, pd.Series):
            if not is_categorical_dtype(vec):
                raise TypeError(f"Expected a `categorical` type, found `{infer_dtype(vec)}`.")

            face_color, col_dict = _get_categorical(self.adata, key=key, palette=self._palette, vec=vec)
            is_categorical = True
            properties, metadata = self._get_label_positions(vec, col_dict), None

            text = {
                "text": "{clusters}",
                "size": self.TEXT_SIZE,
                "color": "white",  # properties["colors"],
                "anchor": "center",
                "blending": "translucent",
            }
        else:
            is_categorical, text, face_color = False, None, "value"
            properties = {"value": vec}
            metadata = {"perc": (0, 100), "data": vec, "minmax": (np.min(vec), np.max(vec))}

        logg.info(f"Loading `{layer_name}` layer")
        layer: Points = self.viewer.add_points(
            self._coords,
            name=layer_name,
            size=self._spot_radius,
            face_color=face_color,
            edge_width=1,
            text=text,
            blending=self._layer_blending,
            properties=properties,
            metadata=metadata,
        )
        # https://github.com/napari/napari/issues/2019
        # TODO: uncomment the 2 lines below once a solution is found for the contrast
        # we could use the selected points where the cluster labels are position as a black BG
        # layer._text._color = properties["colors"]
        # layer._text.events.color()

        self._hide_point_controls(layer, is_categorical=is_categorical)

        layer.editable = False
        # QoL change: selected layer is brought to the top
        # TODO: only for opaque blending?
        layer.events.select.connect(move_to_front)

    def _hide_point_controls(self, layer: Points, is_categorical: bool):
        # TODO: constants
        to_hide = {
            "symbol:": "symbolComboBox",
            "point size:": "sizeSlider",
            "face color:": "faceColorEdit",
            "edge color:": "edgeColorEdit",
            "n-dim:": "ndimCheckBox",
        }
        points_controls = self.viewer.window.qt_viewer.controls.widgets[layer]

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

        if not is_categorical:
            idx = gl.indexOf(attr)
            row, *_ = gl.getItemPosition(idx)

            slider = RangeSlider(
                layer=layer,
                colorbar=self._colorbar,
                initial_values=(0, 100),
                data_range=(0, 100),
                step_size=0.01,
                collapsible=False,
            )
            slider.valuesChanged.emit((0, 100))

            gl.replaceWidget(labels[key], QLabel("percentile:"))
            gl.replaceWidget(attr, slider)

            # TODO: try also adding the new cbar? fixed canvas is problem (need to look into vispy)
            # TODO: otherwise just use global - local would be nicer since for raw, we could change the format
            # TODO: from float to int (currently, it would be too painful)
            # gl.removeWidget(attr)
            # gl.removeWidget(labels[key])
            # gl.addWidget(self._colorbar, row, 0)
            # gl.setRowStretch(row, 1)

    @property
    @d.dedent
    def adata(self) -> AnnData:
        """%(adata)s"""  # noqa: D400
        return self._adata

    @property
    def viewer(self) -> Optional[napari.Viewer]:
        """:mod:`napari` viewer."""
        return self._viewer

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
    # TODO: deprecate in favor of ImageContainer.interactive()
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


def _get_categorical(
    adata: AnnData, key: str, palette: Optional[str] = None, vec: Optional[pd.Series] = None
) -> np.ndarray:
    if vec is not None:
        # TODO: do we really want to do this? alt. is to create dummy column and delete artefacts from the adata object
        if not is_categorical_dtype(vec):
            raise TypeError(f"Expected a `categorical` type, found `{infer_dtype(vec)}`.")
        adata.obs[key] = vec.values

    add_colors_for_categorical_sample_annotation(
        adata, key=key, force_update_colors=palette is not None, palette=palette
    )
    cols = [to_rgb(i) for i in adata.uns[f"{key}_colors"]]

    col_dict = dict(zip(adata.obs[key].cat.categories, cols))
    return np.array([col_dict[v] for v in adata.obs[key]]), col_dict
