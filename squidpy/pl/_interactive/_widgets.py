from abc import abstractmethod
from vispy import scene
from typing import Any, Tuple, Union, Iterable, Optional
from vispy.scene.widgets import ColorBarWidget
from vispy.color.colormap import Colormap, MatplotlibColormap

import numpy as np

from PyQt5 import QtCore, QtWidgets

from napari.layers import Points
from napari._qt.widgets.qt_range_slider import QHRangeSlider

from squidpy.pl._utils import ALayer

__all__ = ["TwoStateCheckBox", "AListWidget", "CBarWidget", "RangeSlider"]


# TODO: should inherit from ABC, but MC conflict (need to see how it's done for Qt)
class ListWidget(QtWidgets.QListWidget):
    indexChanged = QtCore.pyqtSignal(object)
    enterPressed = QtCore.pyqtSignal(object)

    def __init__(self, controller: Any, unique: bool = True, multiselect: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        if multiselect:
            self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        else:
            self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self._index = 0
        self._unique = unique
        self._controller = controller

        self.itemDoubleClicked.connect(lambda item: self._onAction((item.text(),)))
        self.enterPressed.connect(self._onAction)
        self.indexChanged.connect(self._onAction)

    @abstractmethod
    def setIndex(self, index: int) -> None:
        pass

    def getIndex(self) -> int:
        return self._index

    @abstractmethod
    def _onAction(self, items: Iterable[str]) -> None:
        pass

    def addItems(self, labels: Union[str, Iterable[str]]) -> None:
        if isinstance(labels, str):
            labels = (labels,)
        labels = tuple(labels)

        if self._unique:
            labels = tuple(label for label in labels if self.findItems(label, QtCore.Qt.MatchExactly) is not None)

        if len(labels):
            super().addItems(labels)
            self.sortItems(QtCore.Qt.AscendingOrder)

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() == QtCore.Qt.Key_Return:
            event.accept()
            self.enterPressed.emit(tuple(s.text() for s in self.selectedItems()))
        else:
            super().keyPressEvent(event)


class LibraryListWidget(ListWidget):
    def __init__(self, controller: Any, **kwargs: Any):
        super().__init__(controller, **kwargs)

        self.currentTextChanged.connect(self._onAction)

    def setIndex(self, index: int) -> None:
        # not used
        if index == self._index:
            return

        self._index = index
        self.indexChanged.emit(tuple(s.text() for s in self.selectedItems()))

    def _onAction(self, items: Union[str, Iterable[str]]) -> None:
        if isinstance(items, str):
            items = (items,)

        for item in items:
            if self._controller.add_image(item):
                # only add 1 item
                break


class TwoStateCheckBox(QtWidgets.QCheckBox):
    checkChanged = QtCore.pyqtSignal(bool)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.setTristate(False)
        self.setChecked(False)
        self.stateChanged.connect(self._onStateChanged)

    def _onStateChanged(self, state: QtCore.Qt.CheckState) -> None:
        self.checkChanged.emit(state == QtCore.Qt.Checked)


class AListWidget(ListWidget):
    rawChanged = QtCore.pyqtSignal()
    layerChanged = QtCore.pyqtSignal()

    def __init__(self, controller: Any, alayer: ALayer, attr: str, **kwargs: Any):
        if attr not in ALayer.VALID_ATTRIBUTES:
            raise ValueError(f"Invalid attribute `{attr}`. Valid options are: `{list(ALayer.VALID_ATTRIBUTES)}`.")
        super().__init__(controller, **kwargs)

        self._adata_layer = alayer

        self._attr = attr
        self._getter = getattr(self._adata_layer, f"get_{attr}")

        self.rawChanged.connect(self._onChange)
        self.layerChanged.connect(self._onChange)

        self._onChange()

    def _onChange(self) -> None:
        self.clear()
        self.addItems(self._adata_layer.get_items(self._attr))

    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):
            vec, name = self._getter(item, index=self.getIndex())
            self._controller.add_points(vec, key=item, layer_name=name)

    def setRaw(self, is_raw: bool) -> None:
        if is_raw == self.getRaw():
            return

        self._adata_layer.raw = is_raw
        self.rawChanged.emit()

    def getRaw(self) -> bool:
        return self._adata_layer.raw

    def setIndex(self, index: Union[str, int]) -> None:
        if isinstance(index, str):
            index = 0 if index == "" else int(index, base=10)
        if index == self._index:
            return

        self._index = index
        if self._attr == "obsm":
            self.indexChanged.emit(tuple(s.text() for s in self.selectedItems()))

    def getIndex(self) -> int:
        return self._index

    def setLayer(self, layer: Optional[str]) -> None:
        if layer in ("default", "None"):
            layer = None
        if layer == self.getLayer():
            return

        self._adata_layer.layer = layer
        self.layerChanged.emit()

    def getLayer(self) -> Optional[str]:
        return self._adata_layer.layer


class ObsmIndexWidget(QtWidgets.QComboBox):
    def __init__(self, alayer: ALayer, max_visible: int = 6, **kwargs: Any):
        super().__init__(**kwargs)

        self._adata = alayer.adata
        self.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMaxVisibleItems(max_visible)
        self.setStyleSheet("combobox-popup: 0;")

    def addItems(self, texts: Union[QtWidgets.QListWidgetItem, int, Iterable[str]]) -> None:
        if isinstance(texts, QtWidgets.QListWidgetItem):
            try:
                texts = self._adata.obsm[texts.text()].shape[1]
            except (KeyError, IndexError):
                texts = 0
        if isinstance(texts, int):
            texts = tuple(str(i) for i in range(texts))

        self.clear()
        super().addItems(texts)


class CBarWidget(QtWidgets.QWidget):
    FORMAT = "{0:0.2f}"

    cmapChanged = QtCore.pyqtSignal(str)
    climChanged = QtCore.pyqtSignal((float, float))

    def __init__(
        self,
        cmap: Union[str, Colormap],
        label: Optional[str] = None,
        width: Optional[int] = 250,
        height: Optional[int] = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._cmap = cmap
        self._clim = (0.0, 1.0)
        self._oclim = self._clim

        self._width = width
        self._height = height
        self._label = label

        self.__init_UI()

    def __init_UI(self) -> None:
        self.setFixedWidth(self._width)
        self.setFixedHeight(self._height)

        # use napari's BG color for dark mode
        self._canvas = scene.SceneCanvas(
            size=(self._width, self._height), bgcolor="#262930", parent=self, decorate=False, resizable=False, dpi=150
        )
        self._colorbar = ColorBarWidget(
            self._create_colormap(self.getCmap()),
            orientation="top",
            label=self._label,
            label_color="white",
            clim=self.getClim(),
            border_width=1.0,
            border_color="black",
            padding=(0.33, 0.167),
            axis_ratio=0.05,
            innterpolation="linear",
        )

        self._canvas.central_widget.add_widget(self._colorbar)

        self.climChanged.connect(self.onClimChanged)
        self.cmapChanged.connect(self.onCmapChanged)

    def _create_colormap(self, cmap: str) -> Colormap:
        ominn, omaxx = self.getOclim()
        delta = omaxx - ominn + 1e-12

        minn, maxx = self.getClim()
        minn = (minn - ominn) / delta
        maxx = (maxx - ominn) / delta

        assert 0 <= minn <= 1, f"Expected `min` to be in `[0, 1]`, found `{minn}`"
        assert 0 <= maxx <= 1, f"Expected `maxx` to be in `[0, 1]`, found `{maxx}`"

        cm = MatplotlibColormap(cmap)

        return Colormap(cm[np.linspace(minn, maxx, len(cm.colors))], interpolation="linear")

    def setCmap(self, cmap: str) -> None:
        if self._cmap == cmap:
            return

        self._cmap = cmap
        self.cmapChanged.emit(cmap)

    def getCmap(self) -> str:
        return self._cmap

    def onCmapChanged(self, value: str) -> None:
        # this does not trigger update for some reason...
        self._colorbar.cmap = self._create_colormap(value)
        self._colorbar._colorbar._update()

    def setClim(self, value: Tuple[float, float]) -> None:
        if value == self._clim:
            return

        self._clim = value
        self.climChanged.emit(*value)

    def getClim(self) -> Tuple[float, float]:
        return self._clim

    def getOclim(self) -> Tuple[float, float]:
        return self._oclim

    def setOclim(self, value: Tuple[float, float]) -> None:
        # original color limit used for 0-1 normalization
        self._oclim = value

    def onClimChanged(self, minn: float, maxx: float) -> None:
        # ticks are not working with vispy's colorbar
        self._colorbar.cmap = self._create_colormap(self.getCmap())
        self._colorbar.clim = (self.FORMAT.format(minn), self.FORMAT.format(maxx))

    def getCanvas(self) -> scene.SceneCanvas:
        return self._canvas

    def getColorBar(self) -> ColorBarWidget:
        return self._colorbar

    def setLayout(self, layout: QtWidgets.QLayout) -> None:
        layout.addWidget(self.getCanvas().native)
        super().setLayout(layout)

    def update_color(self) -> None:
        # when changing selected layers that have the same limit
        # could also trigger it as self._colorbar.clim = self.getClim()
        # but the above option also updates geometry
        # cbarwidget->cbar->cbarvisual
        self._colorbar._colorbar._colorbar._update()


class RangeSlider(QHRangeSlider):
    def __init__(self, *args: Any, layer: Points, colorbar: CBarWidget, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._layer = layer
        self._colorbar = colorbar

        self._layer.events.select.connect(self._onLayerSelected)
        self.valuesChanged.connect(self._onValueChange)

    # TODO: use constants
    def _onValueChange(self, percentile: Tuple[float, float]) -> None:
        v = self._layer.metadata["data"]
        clipped = np.clip(v, *np.percentile(v, percentile))

        self._layer.metadata = {**self._layer.metadata, "perc": percentile}
        self._layer.face_color = "value"
        self._layer.properties = {"value": clipped}
        self._layer._update_thumbnail()  # can't find another way to force it
        self._layer.refresh_colors()

        self._onLayerSelected()

    def _onLayerSelected(self, _event: Optional[QtCore.QEvent] = None) -> None:
        source: Points = self._layer
        self._colorbar.setOclim(source.metadata["minmax"])
        self._colorbar.setClim((np.min(source.properties["value"]), np.max(source.properties["value"])))
        self._colorbar.update_color()
