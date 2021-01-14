from typing import FrozenSet

from PyQt5.QtWidgets import QLabel, QWidget, QComboBox, QHBoxLayout

import napari

from squidpy.pl.interactive._model import ImageModel
from squidpy.pl.interactive._widgets import (
    CBarWidget,
    AListWidget,
    ObsmIndexWidget,
    TwoStateCheckBox,
    LibraryListWidget,
)


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

        # TODO: there's got to be some better way
        lib_haystack = set(self.model.adata.uns[self.model.spatial_key].keys())
        lib_ixs = [ix for ix in self.model.container.data.keys() if ix in lib_haystack]

        # TODO: reintroduce tooltips?
        # library
        lib_lab = QLabel("Library:")
        lib_widget = LibraryListWidget(self.controller, multiselect=False, unique=True)
        lib_widget.setMaximumHeight(100)
        lib_widget.addItems(lib_ixs)
        lib_widget.setCurrentItem(lib_widget.item(0))

        # gene
        var_lab = QLabel("Genes:", parent=parent)
        var_widget = AListWidget(self.controller, self.model.alayer, attr="var", parent=parent)

        # obs
        obs_label = QLabel("Observations:", parent=parent)
        self._obs_widget = AListWidget(self.controller, self.model.alayer, attr="obs", parent=parent)

        # obsm
        obsm_label = QLabel("Obsm:", parent=parent)
        obsm_widget = AListWidget(self.controller, self.model.alayer, attr="obsm", multiselect=False, parent=parent)
        obsm_index_widget = ObsmIndexWidget(self.model.alayer, parent=parent)
        obsm_index_widget.setToolTip("Select dimension.")
        obsm_index_widget.currentTextChanged.connect(obsm_widget.setIndex)
        obsm_widget.itemClicked.connect(obsm_index_widget.addItems)

        # layer selection
        layer_label = QLabel("Layers:", parent=parent)
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
        raw_layout.addWidget(QLabel("Raw:", parent=parent))
        raw_layout.addWidget(raw_cbox)
        raw_layout.addStretch()
        raw_widget = QWidget(parent=parent)
        raw_widget.setLayout(raw_layout)

        widgets = (
            lib_lab,
            lib_widget,
            layer_label,
            layer_widget,
            raw_widget,
            var_lab,
            var_widget,
            obs_label,
            self._obs_widget,
            obsm_label,
            obsm_widget,
            obsm_index_widget,
        )
        self._colorbar = CBarWidget(self.model.cont_cmap, parent=parent)

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
