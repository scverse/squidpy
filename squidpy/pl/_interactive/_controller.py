from typing import Any, Dict, Union, Optional
from pathlib import Path

from scanpy import logging as logg
from anndata import AnnData

from pandas.core.dtypes.common import is_categorical_dtype
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QLabel, QGridLayout

from napari import Viewer
from napari.layers import Points, Shapes

from skimage import img_as_float

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy._docs import d
from squidpy._utils import singledispatchmethod
from squidpy.pl._utils import _points_inside_triangles
from squidpy.pl._interactive._view import ImageView
from squidpy.pl._interactive._model import ImageModel
from squidpy.pl._interactive._utils import _get_categorical, _position_cluster_labels
from squidpy.pl._interactive._widgets import RangeSlider

# label string: attribute name
_WIDGETS_TO_HIDE = {
    "symbol:": "symbolComboBox",
    "point size:": "sizeSlider",
    "face color:": "faceColorEdit",
    "edge color:": "edgeColorEdit",
    "n-dim:": "ndimCheckBox",
}


@d.dedent
class ImageController:
    """
    Controller class.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    """

    def __init__(self, adata: AnnData, img: ImageContainer, **kwargs: Any):
        self._model = ImageModel(adata=adata, container=img, **kwargs)
        self._view = ImageView(model=self.model, controller=self)

        self.view._init_UI()

    def add_image(self, layer: str) -> bool:
        """
        Add a new :mod:`napari` image layer.

        Parameters
        ----------
        layer
            Layer in the underlying's :class:`ImageContainer` which contains the image.

        Returns
        -------
        `True` if the layer has been added, otherwise `False`.
        """
        if layer in self.view.layernames:
            self._handle_already_present(layer)
            return False

        img: np.ndarray = self.model.container.data[layer].transpose("y", "x", ...).values
        if img.shape[-1] > 4:
            logg.warning(f"Unable to show image of shape `{img.shape}`")
            return False

        logg.info(f"Creating image `{layer}` layer")
        self.view.viewer.add_image(
            img_as_float(img),
            name=layer,
            rgb=True,
            colormap=self.model.cmap,
            blending=self.model.blending,
        )

        return True

    def add_points(self, vec: Union[np.ndarray, pd.Series], layer_name: str, key: Optional[str] = None) -> bool:
        """
        Add a new :mod:`napari` points layer.

        Parameters
        ----------
        vec
            Values to plot. If :class:`pandas.Series`, it is expected to be categorical.
        layer_name
            Name of the layer to add.
        key
            Key from :attr:`anndata.AnnData.obs` from where the data was taken from.
            Only used when ``vec`` is :class:`pandas.Series`.

        Returns
        -------
        `True` if the layer has been added, otherwise `False`.
        """
        if layer_name in self.view.layernames:
            self._handle_already_present(layer_name)
            return False

        logg.info(f"Creating point `{layer_name}` layer")
        properties = self._get_points_properties(vec, key=key)
        layer: Points = self.view.viewer.add_points(
            self.model.coordinates,
            name=layer_name,
            size=self.model.spot_diameter,
            opacity=1,
            edge_width=1,
            blending=self.model.blending,
            face_colormap=self.model.cmap,
            edge_colormap=self.model.cmap,
            symbol=self.model.symbol.v,
            **properties,
        )
        # https://github.com/napari/napari/issues/2019
        # TODO: uncomment the 2 lines below once a solution is found for contrasting colors
        # we could use the selected points where the cluster labels are position as a black BG
        # layer._text._color = properties["colors"]
        # layer._text.events.color()
        self._hide_points_controls(layer, is_categorical=is_categorical_dtype(vec))

        layer.editable = False
        layer.events.select.connect(self._move_layer_to_front)

        return True

    def export(self, _: Viewer) -> None:
        """Export shapes into :class:`AnnData` object."""
        for layer in self.view.layers:
            if not isinstance(layer, Shapes) or not layer.selected:
                continue
            if not len(layer.data):
                logg.warning(f"Shape layer `{layer.name}` has no visible shapes")
                continue

            key = f"{layer.name}_{self.model.key_added}"

            logg.info(f"Adding `adata.obs[{key!r}]`\n       `adata.uns[{key!r}]['meshes']`")
            self._save_shapes(layer, key=key)
            self._update_obs_items(key)

    def show(self, restore: bool = False) -> None:
        """
        Launch the :class:`napari.Viewer`.

        Parameters
        ----------
        restore
            Whether to reinitialize the GUI after it has been destroyed.

        Returns
        -------
        Nothing, just launches the viewer.
        """
        try:
            self.view.viewer.show()
        except RuntimeError:
            if restore:
                self.view._init_UI()
                self.view.viewer.show()
            else:
                logg.error("The viewer has already been closed. Try specifying `restore=True`")

    @d.get_full_description(base="cont_close")
    def close(self) -> None:
        """Close the :class:`napari.Viewer` or do nothing, if it's already closed."""
        try:
            self.view.viewer.close()
        except RuntimeError:
            pass

    def screenshot(self, path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Take a screenshot of the viewer's canvas.

        Parameters
        ----------
        path
            Path where to save the screenshot. If `None`, don't save it.

        Returns
        -------
        Screenshot as an RGB array of shape ``(height, width, 3)``.
        """
        return np.asarray(self.view.viewer.screenshot(path, canvas_only=True))

    def _handle_already_present(self, layer_name: str) -> None:
        logg.warning(f"Layer `{layer_name}` is already loaded")
        self.view.layers.unselect_all()
        self.view.layers[layer_name].selected = True

    def _move_layer_to_front(self, event: Any) -> None:
        layer = event.source
        if not layer.visible:
            return

        try:
            index = self.view.layers.index(layer)
        except ValueError:
            return

        self.view.layers.move(index, -1)

    def _save_shapes(self, layer: Shapes, key: str) -> None:
        shape_list = layer._data_view
        triangles = shape_list._mesh.vertices[shape_list._mesh.displayed_triangles]

        self.model.adata.obs[key] = pd.Categorical(_points_inside_triangles(self.model.coordinates, triangles))
        self.model.adata.uns[key] = {"meshes": layer.data.copy()}

    def _update_obs_items(self, key: str) -> None:
        self.view._obs_widget.addItems(key)
        if key in self.view.layernames:
            # update already present layer
            layer = self.view.layers[key]
            layer.face_color = _get_categorical(self.model.adata, key)
            layer._update_thumbnail()
            layer.refresh_colors()

    @singledispatchmethod
    def _get_points_properties(self, vec: Union[np.ndarray, pd.Series], **_: Any) -> Dict[str, Any]:
        raise NotImplementedError(type(vec))

    @_get_points_properties.register(np.ndarray)
    def _(self, vec: np.ndarray, **_) -> Dict[str, Any]:
        return {
            "text": None,
            "face_color": "value",
            "properties": {"value": vec},
            "metadata": {"perc": (0, 100), "data": vec, "minmax": (np.nanmin(vec), np.nanmax(vec))},
        }

    @_get_points_properties.register(pd.Series)  # type: ignore[no-redef]
    def _(self, vec: pd.Series, key: str) -> Dict[str, Any]:
        face_color = _get_categorical(self.model.adata, key=key, palette=self.model.palette, vec=vec)
        return {
            "text": {"text": "{clusters}", "size": 24, "color": "white", "anchor": "center"},
            "face_color": face_color,
            "properties": _position_cluster_labels(self.model.coordinates, vec, face_color),
            "metadata": None,
        }

    def _hide_points_controls(self, layer: Points, is_categorical: bool) -> None:
        try:
            # shouldn't happen
            points_controls = self.view.viewer.window.qt_viewer.controls.widgets[layer]
        except KeyError:
            return

        gl: QGridLayout = points_controls.grid_layout

        labels = {}
        for i in range(gl.count()):
            item = gl.itemAt(i).widget()
            if isinstance(item, QLabel):
                labels[item.text()] = item

        label_key, widget = "", None
        # remove all widgets which can modify the layer
        for label_key, widget_name in _WIDGETS_TO_HIDE.items():
            widget = getattr(points_controls, widget_name, None)
            if label_key in labels and widget is not None:
                widget.setHidden(True)
                labels[label_key].setHidden(True)

        if not is_categorical:  # add the slider
            idx = gl.indexOf(widget)
            row, *_ = gl.getItemPosition(idx)

            slider = RangeSlider(
                layer=layer,
                colorbar=self.view._colorbar,
                initial_values=(0, 100),
                data_range=(0, 100),
                step_size=0.01,
                collapsible=False,
            )
            slider.valuesChanged.emit((0, 100))

            gl.replaceWidget(labels[label_key], QLabel("percentile:"))
            gl.replaceWidget(widget, slider)

    @property
    def view(self) -> ImageView:
        """View managed by this controller."""
        return self._view

    @property
    def model(self) -> ImageModel:
        """Model managed by this controller."""
        return self._model
