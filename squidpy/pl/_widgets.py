# /***************************************************************************
# Name                 : RangeSlider
# Description          : A slider for ranges
# Date                 : Jun 20, 2012
# copyright            : (C) 2012 by Giuseppe Sucameli
# email                : brush.tyler@gmail.com
#
# the code is based on RangeSlider by phil
# (see https://svn.enthought.com/enthought/browser/TraitsBackendQt/trunk/enthought/traits/ui/qt4/extra/range_slider.py)
# licensed under GPLv2
# ***************************************************************************/

# /**************************************************************************
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU General Public License as published by  *
# *   the Free Software Foundation; either version 2 of the License, or     *
# *   (at your option) any later version.                                   *
# *                                                                         *
# ***************************************************************************/
from typing import Tuple, Union, Iterable, Optional

from PyQt5 import QtGui, QtCore, QtWidgets
from vispy import scene
from vispy.scene import widgets
from vispy.color.colormap import Colormap, MatplotlibColormap

import numpy as np

from squidpy.pl._utils import ALayer


class RangeSlider(QtWidgets.QSlider):
    """
    A slider for ranges.

    This class provides a dual-slider for ranges, where there is a defined
    maximum and minimum, as is a normal slider, but instead of having a
    single slider value, there are 2 slider values.

    This class emits the same signals as the QSlider base class, with the
    exception of valueChanged.
    In addition, two new signals are emitted to catch the movement of
    each handle, lowValueChanged(int) and highValueChanged(int).
    """

    highValueChanged = QtCore.pyqtSignal(int)
    lowValueChanged = QtCore.pyqtSignal(int)

    def __init__(self, **kwargs):
        super().__init__(QtCore.Qt.Horizontal, **kwargs)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QtWidgets.QStyle.SC_None
        self.hover_control = QtWidgets.QStyle.SC_None
        self.click_offset = 0

        # -1 for the low, 1 for the high, 0 for both
        self.active_slider = 0

        self.valueChanged.connect(self.lowValueChanged)
        self.valueChanged.connect(self.highValueChanged)

    def lowValue(self):
        return self._low

    def setLowValue(self, low):
        low = max(low, self.minimum())
        if low == self._low:
            return

        self._low = low
        self.update()

        if self.hasTracking():
            self.lowValueChanged.emit(self._low)
            self.valueChanged.emit(self._low)

    def highValue(self):
        return self._high

    def setHighValue(self, high):
        high = min(high, self.maximum())
        if high == self._high:
            return

        self._high = high
        self.update()

        if self.hasTracking():
            self.highValueChanged.emit(self._high)
            self.valueChanged.emit(self._high)

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp

        painter = QtGui.QPainter(self)
        style = QtWidgets.QApplication.style()

        # draw groove
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        opt.siderValue = 0
        opt.sliderPosition = 0
        opt.subControls = QtWidgets.QStyle.SC_SliderGroove
        if self.tickPosition() != self.NoTicks:
            opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks
        style.drawComplexControl(QtWidgets.QStyle.CC_Slider, opt, painter, self)
        groove = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, self)

        # drawSpan
        # opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        opt.subControls = QtWidgets.QStyle.SC_SliderGroove
        # if self.tickPosition() != self.NoTicks:
        #    opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks
        opt.siderValue = 0
        opt.sliderPosition = self._low
        low_rect = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)
        opt.sliderPosition = self._high
        high_rect = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)

        low_pos = self.__pick(low_rect.center())
        high_pos = self.__pick(high_rect.center())

        min_pos = min(low_pos, high_pos)
        max_pos = max(low_pos, high_pos)

        c = QtCore.QRect(low_rect.center(), high_rect.center()).center()
        if opt.orientation == QtCore.Qt.Horizontal:
            span_rect = QtCore.QRect(QtCore.QPoint(min_pos, c.y() - 2), QtCore.QPoint(max_pos, c.y() + 1))
        else:
            span_rect = QtCore.QRect(QtCore.QPoint(c.x() - 2, min_pos), QtCore.QPoint(c.x() + 1, max_pos))

        # self.initStyleOption(opt)
        if opt.orientation == QtCore.Qt.Horizontal:
            groove.adjust(0, 0, -1, 0)
        else:
            groove.adjust(0, 0, 0, -1)

        if True:  # self.isEnabled():
            highlight = self.palette().color(QtGui.QPalette.Highlight)
            painter.setBrush(QtGui.QBrush(highlight))
            painter.setPen(QtGui.QPen(highlight, 0))
            # painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Dark), 0))
            """
            if opt.orientation == QtCore.Qt.Horizontal:
                self.setupPainter(painter, opt.orientation, groove.center().x(),
                                  groove.top(), groove.center().x(), groove.bottom())
            else:
                self.setupPainter(painter, opt.orientation, groove.left(), groove.center().y(),
                                  groove.right(), groove.center().y())
            """
            # spanRect =
            painter.drawRect(span_rect.intersected(groove))
            # painter.drawRect(groove)

        for i, value in enumerate([self._low, self._high]):
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QtWidgets.QStyle.SC_SliderHandle  # | QtWidgets.QStyle.SC_SliderGroove
            else:
                opt.subControls = QtWidgets.QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value
            style.drawComplexControl(QtWidgets.QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):
        event.accept()

        style = QtWidgets.QApplication.style()
        button = event.button()

        # In a normal slider control, when the user clicks on a point in the
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts

        if button:
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = 0
            mid = (self.maximum() - self.minimum()) / 2.0

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)

                if hit == style.SC_SliderHandle:
                    self.pressed_control = hit

                    # if both handles are close together, avoid locks
                    # choosing the one with more empty space near it
                    if self._low + 2 >= self._high:
                        self.active_slider = 1 if self._high < mid else -1
                    else:
                        self.active_slider = -1 if i == 0 else 1
                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break

            if self.active_slider == 0:
                self.pressed_control = QtWidgets.QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
                self.setSliderDown(True)
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if self.pressed_control != QtWidgets.QStyle.SC_SliderHandle:
            event.ignore()
            return

        self.setSliderDown(False)
        return QtWidgets.QSlider.mouseReleaseEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.pressed_control != QtWidgets.QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)

        old_click_offset = self.click_offset
        self.click_offset = new_pos

        if self.active_slider == 0:
            offset = new_pos - old_click_offset
            new_low = self._low + offset
            new_high = self._high + offset

            if new_low < self.minimum():
                diff = self.minimum() - new_low
                new_low += diff
                new_high += diff
            if new_high > self.maximum():
                diff = self.maximum() - new_high
                new_low += diff
                new_high += diff

            self.setLowValue(new_low)
            self.setHighValue(new_high)

        elif self.active_slider < 0:
            if new_pos > self._high:
                new_pos = self._high
            self.setLowValue(new_pos)

        else:
            if new_pos < self._low:
                new_pos = self._low
            self.setHighValue(new_pos)

    def __pick(self, pt):
        return pt.x() if self.orientation() == QtCore.Qt.Horizontal else pt.y()

    def __pixelPosToRangeValue(self, pos):
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QtWidgets.QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            handle_length = sr.width()
            slider_min = gr.x() + handle_length / 2
            slider_max = gr.right() - handle_length / 2 + 1
        else:
            handle_length = sr.height()
            slider_min = gr.y() + handle_length / 2
            slider_max = gr.bottom() - handle_length / 2 + 1

        return self.minimum() + style.sliderValueFromPosition(
            0, self.maximum() - self.minimum(), pos - slider_min, slider_max - slider_min, opt.upsideDown
        )

    def setValue(self, value: Tuple[float, float]):
        self.setLowValue(value[0])
        self.setHighValue(value[1])

    def value(self) -> Tuple[int, int]:
        return self.lowValue(), self.highValue()


class DoubleRangeSlider(RangeSlider):
    PRECISION = 1000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def value(self) -> Tuple[float, float]:
        return super().lowValue() / self.PRECISION, super().highValue() / self.PRECISION

    def setValue(self, value: Tuple[float, float]):
        # should only access this, not setLowValue
        self.setLowValue(value[0] * self.PRECISION)
        self.setHighValue(value[1] * self.PRECISION)

    def setMinimum(self, a0: int) -> None:
        super().setMinimum(a0 * self.PRECISION)

    def setMaximum(self, a0: int) -> None:
        super().setMaximum(a0 * self.PRECISION)


class AListWidget(QtWidgets.QListWidget):
    rawChanged = QtCore.pyqtSignal()
    layerChanged = QtCore.pyqtSignal()
    indexChanged = QtCore.pyqtSignal(object)
    enterPressed = QtCore.pyqtSignal(object)

    def __init__(self, alayer: ALayer, attr: str, controller, unique: bool = True, multiselect: bool = True, **kwargs):
        if attr not in ALayer.VALID_ATTRIBUTES:
            raise ValueError(f"Invalid attribute `{attr}`. Valid options are: `{list(ALayer.VALID_ATTRIBUTES)}`.")
        super().__init__(**kwargs)

        if multiselect:
            self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self._adata_layer = alayer
        self._controller = controller

        self._index = 0
        self._attr = attr
        self._getter = getattr(self._adata_layer, f"get_{attr}")
        self._unique = unique

        self.rawChanged.connect(self._onChange)
        self.layerChanged.connect(self._onChange)

        self.itemDoubleClicked.connect(lambda item: self._onAction((item.text(),)))
        self.enterPressed.connect(self._onAction)
        self.indexChanged.connect(self._onAction)

        self._onChange()

    def _onChange(self):
        self.clear()
        self.addItems(self._adata_layer.get_items(self._attr))

    def _onAction(self, items: Iterable[str]):
        for item in sorted(set(items)):
            vec, name = self._getter(item, index=self.getIndex())
            self._controller._add_points(vec, key=item, layer_name=name)

    def setRaw(self, is_raw: bool):
        if is_raw == self.getRaw():
            return

        self._adata_layer.raw = is_raw
        self.rawChanged.emit()

    def getRaw(self) -> bool:
        return self._adata_layer.raw

    def setIndex(self, index: Union[str, int]):
        if isinstance(index, str):
            index = 0 if index == "" else int(index, base=10)
        if index == self._index:
            return

        self._index = index
        if self._attr == "obsm":
            self.indexChanged.emit(tuple(s.text() for s in self.selectedItems()))

    def getIndex(self):
        return self._index

    def setLayer(self, layer: Optional[str]):
        if layer == "None":
            layer = None
        if layer == self.getLayer():
            return

        self._adata_layer.layer = layer
        self.layerChanged.emit()

    def getLayer(self) -> Optional[str]:
        return self._adata_layer.layer

    def addItems(self, labels: Union[str, Iterable[str]]) -> None:
        if isinstance(labels, str) or not isinstance(labels, Iterable):
            labels = (labels,)
        if self._unique:
            labels = tuple(label for label in labels if self.findItems(label, QtCore.Qt.MatchExactly) is not None)
        if len(labels):
            super().addItems(labels)
            self.sortItems(QtCore.Qt.AscendingOrder)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            event.accept()
            self.enterPressed.emit(tuple(s.text() for s in self.selectedItems()))
        else:
            super().keyPressEvent(event)


class ObsmIndexWidget(QtWidgets.QComboBox):
    def __init__(self, alayer: ALayer, max_visible: int = 6, **kwargs):
        super().__init__(**kwargs)

        self._adata = alayer.adata
        self.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMaxVisibleItems(max_visible)
        self.setStyleSheet("combobox-popup: 0;")

    def _onSelectionChanged(self, texts: Union[QtWidgets.QListWidgetItem, int, Iterable[str]]) -> None:
        pass

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
    FORMAT = "{:.02f}"

    cmapChanged = QtCore.pyqtSignal(str)
    climChanged = QtCore.pyqtSignal((float, float))

    def __init__(
        self, cmap: str, label: Optional[str] = None, width: Optional[int] = 300, height: Optional[int] = 50, **kwargs
    ):
        super().__init__(**kwargs)

        self._cmap = cmap or ""
        self._clim = (0, 1)
        self._oclim = self._clim

        self._width = width
        self._height = height
        self._label = label

        self.__initUI()

    def __initUI(self):
        self.setFixedWidth(self._width)
        self.setFixedHeight(self._height)

        # cheat a litte - bgcol is napari's bgcolor - hope nobody uses the lightmode
        self._canvas = scene.SceneCanvas(size=(self._width, self._height), bgcolor="#262930")
        # TODO: place the labels more nicely (+ ticks)
        self._colorbar = widgets.ColorBarWidget(
            self._create_colormap(self.getCmap()),
            orientation="top",
            label=self._label,
            label_color="white",
            clim=self.getClim(),
            border_width=1.0,
            border_color="black",
            padding=(0.15, 0.5),
            axis_ratio=0.25,
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
        self._colorbar.cmap = self._create_colormap(self.getCmap())
        self._colorbar.clim = (self.FORMAT.format(minn), self.FORMAT.format(maxx))

    def getCanvas(self) -> scene.SceneCanvas:
        return self._canvas

    def getColorBar(self):
        return self._colorbar

    def setLayout(self, layout) -> None:
        layout.addWidget(self.getCanvas().native)
        super().setLayout(layout)

    def update_color(self) -> None:
        # when changing selected layers that have the same limit
        # could also trigger it as self._colorbar.clim = self.getClim()
        # but the above option also updates geometry
        # cbarwidget->cbar->cbarvisual
        self._colorbar._colorbar._colorbar._update()
