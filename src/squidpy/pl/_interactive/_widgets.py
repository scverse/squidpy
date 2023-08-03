# type: ignore
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
from deprecated import deprecated
from napari.layers import Points
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from scanpy import logging as logg
from superqt import QRangeSlider
from vispy import scene
from vispy.color.colormap import Colormap, MatplotlibColormap
from vispy.scene.widgets import ColorBarWidget

from squidpy.pl._utils import ALayer

__all__ = ["TwoStateCheckBox", "AListWidget", "CBarWidget", "RangeSlider", "ObsmIndexWidget", "LibraryListWidget"]


class ListWidget(QtWidgets.QListWidget):
    indexChanged = QtCore.pyqtSignal(object)
    enterPressed = QtCore.pyqtSignal(object)

    def __init__(self, controller: Any, unique: bool = True, multiselect: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        if multiselect:
            self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        else:
            self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self._index: int | str = 0
        self._unique = unique
        self._controller = controller

        self.itemDoubleClicked.connect(lambda item: self._onAction((item.text(),)))
        self.enterPressed.connect(self._onAction)
        self.indexChanged.connect(self._onAction)

    @abstractmethod
    def setIndex(self, index: int | str) -> None:
        pass

    def getIndex(self) -> int | str:
        return self._index

    @abstractmethod
    def _onAction(self, items: Iterable[str]) -> None:
        pass

    def addItems(self, labels: str | Iterable[str]) -> None:
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

    def setIndex(self, index: int | str) -> None:
        # not used
        if index == self._index:
            return

        self._index = index
        self.indexChanged.emit(tuple(s.text() for s in self.selectedItems()))

    def _onAction(self, items: str | Iterable[str]) -> None:
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
    libraryChanged = QtCore.pyqtSignal()

    def __init__(self, controller: Any, alayer: ALayer, attr: str, **kwargs: Any):
        if attr not in ALayer.VALID_ATTRIBUTES:
            raise ValueError(f"Invalid attribute `{attr}`. Valid options are `{sorted(ALayer.VALID_ATTRIBUTES)}`.")
        super().__init__(controller, **kwargs)

        self._alayer = alayer

        self._attr = attr
        self._getter = getattr(self._alayer, f"get_{attr}")

        self.rawChanged.connect(self._onChange)
        self.layerChanged.connect(self._onChange)
        self.libraryChanged.connect(self._onChange)

        self._onChange()

    def _onChange(self) -> None:
        self.clear()
        self.addItems(self._alayer.get_items(self._attr))

    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):
            try:
                vec, name = self._getter(item, index=self.getIndex())
            except Exception as e:  # noqa: BLE001
                logg.error(e)
                continue
            self._controller.add_points(vec, key=item, layer_name=name)

    def setRaw(self, is_raw: bool) -> None:
        if is_raw == self.getRaw():
            return

        self._alayer.raw = is_raw
        self.rawChanged.emit()

    def getRaw(self) -> bool:
        return self._alayer.raw

    def setIndex(self, index: str | int) -> None:
        if isinstance(index, str):
            if index == "":
                index = 0
            elif self._attr != "obsm":
                index = int(index, base=10)
            # for obsm, we convert index to int if needed (if not a DataFrame) in the ALayer
        if index == self._index:
            return

        self._index = index
        if self._attr == "obsm":
            self.indexChanged.emit(tuple(s.text() for s in self.selectedItems()))

    def getIndex(self) -> int | str:
        return self._index

    def setLayer(self, layer: str | None) -> None:
        if layer in ("default", "None"):
            layer = None
        if layer == self.getLayer():
            return

        self._alayer.layer = layer
        self.layerChanged.emit()

    def getLayer(self) -> str | None:
        return self._alayer.layer

    def setLibraryId(self, library_id: str) -> None:
        if library_id == self.getLibraryId():
            return

        self._alayer.library_id = library_id
        self.libraryChanged.emit()

    def getLibraryId(self) -> str:
        return self._alayer.library_id


class ObsmIndexWidget(QtWidgets.QComboBox):
    def __init__(self, alayer: ALayer, max_visible: int = 6, **kwargs: Any):
        super().__init__(**kwargs)

        self._alayer = alayer
        self.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMaxVisibleItems(max_visible)
        self.setStyleSheet("combobox-popup: 0;")

    def addItems(self, texts: QtWidgets.QListWidgetItem | int | Iterable[str]) -> None:
        if isinstance(texts, QtWidgets.QListWidgetItem):
            try:
                key = texts.text()
                if isinstance(self._alayer.adata.obsm[key], pd.DataFrame):
                    texts = sorted(self._alayer.adata.obsm[key].select_dtypes(include=[np.number, "category"]).columns)
                elif hasattr(self._alayer.adata.obsm[key], "shape"):
                    texts = self._alayer.adata.obsm[key].shape[1]
                else:
                    texts = np.asarray(self._alayer.adata.obsm[key]).shape[1]
            except (KeyError, IndexError):
                texts = 0
        if isinstance(texts, int):
            texts = tuple(str(i) for i in range(texts))

        self.clear()
        super().addItems(tuple(texts))


class CBarWidget(QtWidgets.QWidget):
    FORMAT = "{0:0.2f}"

    cmapChanged = QtCore.pyqtSignal(str)
    climChanged = QtCore.pyqtSignal((float, float))

    def __init__(
        self,
        cmap: str | Colormap,
        label: str | None = None,
        width: int | None = 250,
        height: int | None = 50,
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

    def setClim(self, value: tuple[float, float]) -> None:
        if value == self._clim:
            return

        self._clim = value
        self.climChanged.emit(*value)

    def getClim(self) -> tuple[float, float]:
        return self._clim

    def getOclim(self) -> tuple[float, float]:
        return self._oclim

    def setOclim(self, value: tuple[float, float]) -> None:
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


@deprecated
class RangeSlider(QRangeSlider):
    def __init__(self, *args: Any, layer: Points, colorbar: CBarWidget, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._layer = layer
        self._colorbar = colorbar
        self.setValue((0, 100))
        self.setSliderPosition((0, 100))
        self.setSingleStep(0.01)
        self.setOrientation(Qt.Horizontal)

        self.valueChanged.connect(self._onValueChange)

    def _onValueChange(self, percentile: tuple[float, float]) -> None:
        # TODO(michalk8): use constants
        v = self._layer.metadata["data"]
        clipped = np.clip(v, *np.percentile(v, percentile))

        self._layer.metadata = {**self._layer.metadata, "perc": percentile}
        self._layer.face_color = "value"
        self._layer.properties = {"value": clipped}
        self._layer._update_thumbnail()  # can't find another way to force it
        self._layer.refresh_colors()

        self._colorbar.setOclim(self._layer.metadata["minmax"])
        self._colorbar.setClim((np.min(self._layer.properties["value"]), np.max(self._layer.properties["value"])))
        self._colorbar.update_color()
