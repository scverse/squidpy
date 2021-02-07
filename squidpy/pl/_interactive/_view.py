from typing import FrozenSet

from PyQt5.QtWidgets import QLabel, QWidget, QComboBox, QHBoxLayout

import napari

from squidpy.pl._interactive._model import ImageModel
from squidpy.pl._interactive._widgets import (
    CBarWidget,
    AListWidget,
    ObsmIndexWidget,
    TwoStateCheckBox,
    LibraryListWidget,
)

__all__ = ["ImageView"]


class ImageView:
    """
    View class which initializes :class:`napari.Viewer`.

    Parameters
    ----------
    model
        Model for this view.
    controller
        Controller for this view.
    """

    def __init__(self, model: ImageModel, controller: "ImageController"):  # type: ignore[name-defined] # noqa: F821
        self._model = model
        self._controller = controller

    def _init_UI(self) -> None:
        self._viewer = napari.Viewer(title="Squidpy", show=False)
        self.viewer.bind_key("Shift-E", self.controller.export)
        parent = self.viewer.window._qt_window

        # image
        image_lab = QLabel("Images:")
        image_lab.setToolTip("Keys in `ImageContainer`' containing the image data for this library.")
        image_widget = LibraryListWidget(self.controller, multiselect=False, unique=True)
        image_widget.setMaximumHeight(100)
        image_widget.addItems(tuple(self.model.container))
        image_widget.setCurrentItem(image_widget.item(0))

        # gene
        var_lab = QLabel("Genes:", parent=parent)
        var_lab.setToolTip("Gene names from `adata.var_names` or `adata.raw.var_names`.")
        var_widget = AListWidget(self.controller, self.model.alayer, attr="var", parent=parent)

        # obs
        obs_label = QLabel("Observations:", parent=parent)
        obs_label.setToolTip("Keys in `adata.obs` containing cell observations.")
        self._obs_widget = AListWidget(self.controller, self.model.alayer, attr="obs", parent=parent)

        # obsm
        obsm_label = QLabel("Obsm:", parent=parent)
        obsm_label.setToolTip("Keys in `adata.obsm` containing bases information.")
        obsm_widget = AListWidget(self.controller, self.model.alayer, attr="obsm", multiselect=False, parent=parent)
        obsm_index_widget = ObsmIndexWidget(self.model.alayer, parent=parent)
        obsm_index_widget.setToolTip("Select the baes dimension.")
        obsm_index_widget.currentTextChanged.connect(obsm_widget.setIndex)
        obsm_widget.itemClicked.connect(obsm_index_widget.addItems)

        # layer selection
        layer_label = QLabel("Layers:", parent=parent)
        layer_label.setToolTip("Keys in `adata.layers` used when visualizing gene expression.")
        layer_widget = QComboBox(parent=parent)
        layer_widget.addItem("default", None)
        layer_widget.addItems(self.model.adata.layers.keys())
        layer_widget.currentTextChanged.connect(var_widget.setLayer)
        layer_widget.setCurrentText("default")

        # raw selection
        raw_cbox = TwoStateCheckBox(parent=parent)
        raw_cbox.setDisabled(self.model.adata.raw is None)
        raw_cbox.checkChanged.connect(layer_widget.setDisabled)
        raw_cbox.checkChanged.connect(var_widget.setRaw)
        raw_layout = QHBoxLayout()
        raw_label = QLabel("Raw:", parent=parent)
        raw_label.setToolTip("Whether to access `adata.raw.X` or `adata.X` when visualizing gene expression.")
        raw_layout.addWidget(raw_label)
        raw_layout.addWidget(raw_cbox)
        raw_layout.addStretch()
        raw_widget = QWidget(parent=parent)
        raw_widget.setLayout(raw_layout)

        widgets = (
            image_lab,
            image_widget,
            layer_label,
            layer_widget,
            raw_widget,
            var_lab,
            var_widget,
            obs_label,
            self._obs_widget,  # needed for controller to add mask
            obsm_label,
            obsm_widget,
            obsm_index_widget,
        )
        self._colorbar = CBarWidget(self.model.cmap, parent=parent)

        self.viewer.window.add_dock_widget(self._colorbar, area="left", name="percentile")
        self.viewer.window.add_dock_widget(widgets, area="right", name="genes")

    @property
    def layers(self) -> napari.components.layerlist.LayerList:
        """List of layers of :attr:`napari.Viewer.layers`."""
        return self.viewer.layers

    @property
    def layernames(self) -> FrozenSet[str]:
        """Names of :attr:`napari.Viewer.layers`."""
        return frozenset(layer.name for layer in self.layers)

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    @property
    def model(self) -> ImageModel:
        """Model for this view."""
        return self._model

    @property
    def controller(self) -> "ImageController":  # type: ignore[name-defined] # noqa: F821
        """Controller for this view."""  # noqa: D401
        return self._controller
