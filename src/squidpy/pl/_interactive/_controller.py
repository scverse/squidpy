from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from napari import Viewer
from napari.layers import Points, Shapes
from pandas import CategoricalDtype
from pandas.core.dtypes.common import is_categorical_dtype
from PyQt5.QtWidgets import QGridLayout, QLabel, QWidget
from scanpy import logging as logg

from squidpy._docs import d
from squidpy._utils import NDArrayA, singledispatchmethod
from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy.pl._interactive._model import ImageModel
from squidpy.pl._interactive._utils import (
    _display_channelwise,
    _get_categorical,
    _position_cluster_labels,
)
from squidpy.pl._interactive._view import ImageView
from squidpy.pl._interactive._widgets import RangeSlider  # type: ignore[attr-defined]
from squidpy.pl._utils import _points_inside_triangles

__all__ = ["ImageController"]

# label string: attribute name
_WIDGETS_TO_HIDE = {
    "symbol:": "symbolComboBox",
    "point size:": "sizeSlider",
    "face color:": "faceColorEdit",
    "edge color:": "edgeColorEdit",
    "out of slice:": "outOfSliceCheckBox",
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

        if self.model.container.data[layer].attrs.get("segmentation", False):
            return self.add_labels(layer)

        img: xr.DataArray = self.model.container.data[layer].transpose("z", "y", "x", ...)
        multiscale = np.prod(img.shape[1:3]) > (2**16) ** 2
        n_channels = img.shape[-1]

        rgb = img.attrs.get("rgb", None)
        if n_channels == 1:
            rgb, colormap = False, "gray"
        else:
            colormap = self.model.cmap

        if rgb is None:
            logg.debug("Automatically determining whether image is an RGB image")
            rgb = not _display_channelwise(img.data)

        if rgb:
            contrast_limits = None
        else:
            img = img.transpose(..., "z", "y", "x")  # channels first
            contrast_limits = float(img.min()), float(img.max())

        logg.info(f"Creating image `{layer}` layer")
        self.view.viewer.add_image(
            img.data,
            name=layer,
            rgb=rgb,
            colormap=colormap,
            blending=self.model.blending,
            multiscale=multiscale,
            contrast_limits=contrast_limits,
        )

        return True

    def add_labels(self, layer: str) -> bool:
        """
        Add a new :mod:`napari` labels layer.

        Parameters
        ----------
        layer
            Layer in the underlying's :class:`ImageContainer` which contains the labels image.

        Returns
        -------
        `True` if the layer has been added, otherwise `False`.
        """
        # beware `update_library` in view.py - needs to be in this order
        img: xr.DataArray = self.model.container.data[layer].transpose(..., "z", "y", "x")
        if img.ndim != 4:
            logg.warning(f"Unable to show image of shape `{img.shape}`, too many dimensions")
            return False

        if img.shape[0] != 1:
            logg.warning(f"Unable to create labels layer of shape `{img.shape}`, too many channels `{img.shape[0]}`")
            return False

        if not np.issubdtype(img.dtype, np.integer):
            # could also return to `add_images` and render it as image
            logg.warning(f"Expected label image to be a subtype of `numpy.integer`, found `{img.dtype}`")
            return False

        logg.info(f"Creating label `{layer}` layer")
        self.view.viewer.add_labels(
            img.data,
            name=layer,
            multiscale=np.prod(img.shape[-2:]) > (2**16) ** 2,
        )

        return True

    def add_points(self, vec: NDArrayA | pd.Series, layer_name: str, key: str | None = None) -> bool:
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
            blending=self.model.blending,
            face_colormap=self.model.cmap,
            edge_colormap=self.model.cmap,
            symbol=self.model.symbol.v,
            **properties,
        )
        # TODO(michalk8): add contrasting fg/bg color once https://github.com/napari/napari/issues/2019 is done
        self._hide_points_controls(layer, is_categorical=isinstance(vec.dtype, CategoricalDtype))
        layer.editable = False

        return True

    def export(self, _: Viewer) -> None:
        """Export shapes into :class:`AnnData` object."""
        for layer in self.view.layers:
            if not isinstance(layer, Shapes) or layer not in self.view.viewer.layers.selection:
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

    def screenshot(self, path: str | Path | None = None, canvas_only: bool = True) -> NDArrayA:
        """
        Take a screenshot of the viewer's canvas.

        Parameters
        ----------
        path
            Path where to save the screenshot. If `None`, don't save it.
        canvas_only
            Whether to show only the canvas or also the widgets.

        Returns
        -------
        Screenshot as an RGB array of shape ``(height, width, 3)``.
        """
        return np.asarray(self.view.viewer.screenshot(path, canvas_only=canvas_only))

    def _handle_already_present(self, layer_name: str) -> None:
        logg.debug(f"Layer `{layer_name}` is already loaded")
        self.view.viewer.layers.selection.select_only(self.view.layers[layer_name])

    def _save_shapes(self, layer: Shapes, key: str) -> None:
        shape_list = layer._data_view
        triangles = shape_list._mesh.vertices[shape_list._mesh.displayed_triangles]

        # TODO(michalk8): account for current Z-dim?
        points_mask: NDArrayA = _points_inside_triangles(self.model.coordinates[:, 1:], triangles)

        self.model.adata.obs[key] = pd.Categorical(points_mask)
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
    def _get_points_properties(self, vec: NDArrayA | pd.Series, **_: Any) -> dict[str, Any]:
        raise NotImplementedError(type(vec))

    @_get_points_properties.register(np.ndarray)
    def _(self, vec: NDArrayA, **_: Any) -> dict[str, Any]:
        return {
            "text": None,
            "face_color": "value",
            "properties": {"value": vec},
            "metadata": {"perc": (0, 100), "data": vec, "minmax": (np.nanmin(vec), np.nanmax(vec))},
        }

    @_get_points_properties.register(pd.Series)
    def _(self, vec: pd.Series, key: str) -> dict[str, Any]:
        face_color = _get_categorical(self.model.adata, key=key, palette=self.model.palette, vec=vec)
        return {
            "text": {"text": "{clusters}", "size": 24, "color": "white", "anchor": "center"},
            "face_color": face_color,
            "properties": _position_cluster_labels(self.model.coordinates, vec, face_color),
            "metadata": None,
        }

    def _hide_points_controls(self, layer: Points, is_categorical: bool) -> None:
        try:
            # TODO(michalk8): find a better way: https://github.com/napari/napari/issues/3066
            points_controls = self.view.viewer.window._qt_viewer.controls.widgets[layer]
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

        if TYPE_CHECKING:
            assert isinstance(widget, QWidget)

        if not is_categorical:  # add the slider
            if widget is None:
                logg.warning("Unable to set the percentile slider")
                return
            idx = gl.indexOf(widget)
            row, *_ = gl.getItemPosition(idx)

            slider = RangeSlider(
                layer=layer,
                colorbar=self.view._colorbar,
            )
            slider.valueChanged.emit((0, 100))
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
