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
from typing import Tuple, Iterable, Optional

from PyQt5 import QtGui, QtCore, QtWidgets


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


class ListWidget(QtWidgets.QListWidget):
    enter_pressed = QtCore.pyqtSignal()

    def __init__(self, items: Iterable[str], title: Optional[str] = None):
        super().__init__()
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.addItems(items)
        self.sortItems(QtCore.Qt.AscendingOrder)

        self.setWindowTitle(title)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            event.accept()
            self.enter_pressed.emit()
        else:
            super().keyPressEvent(event)
