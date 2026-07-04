#!/usr/bin/env python3
"""Left sidebar: tower-height sliders (drive forward kinematics) and
effector-position sliders (drive inverse kinematics).
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSlider,
    QVBoxLayout,
    QWidget,
)

_SLIDER_STEPS_PER_MM = 10  # QSlider is integer-only; scale for 0.1mm resolution


class SliderSpinRow(QWidget):
    """A slider + spinbox pair kept in sync, emitting one valueChanged per
    user-driven edit (from either widget). set_value() updates both without
    emitting, so programmatic writes never re-trigger this row's signal.
    """

    valueChanged = Signal(float)

    def __init__(self, minimum, maximum, initial, parent=None):
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(round(minimum * _SLIDER_STEPS_PER_MM))
        self.slider.setMaximum(round(maximum * _SLIDER_STEPS_PER_MM))

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(2)
        self.spinbox.setRange(minimum, maximum)
        self.spinbox.setSingleStep(1.0)

        self.set_value(initial)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spinbox)

    def value(self):
        return self.spinbox.value()

    def set_value(self, value):
        value = max(self._minimum, min(self._maximum, value))
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(round(value * _SLIDER_STEPS_PER_MM))
        self.spinbox.setValue(value)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)

    def _on_slider_changed(self, raw):
        value = raw / _SLIDER_STEPS_PER_MM
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(value)

    def _on_spinbox_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(round(value * _SLIDER_STEPS_PER_MM))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)


class SidebarPanel(QWidget):
    """Tower-height (FK input) and effector-position (IK input) controls."""

    towerHeightsChanged = Signal(float, float, float)
    effectorPositionChanged = Signal(float, float, float)

    def __init__(self, axis_limits, parent=None):
        super().__init__(parent)
        z_min = axis_limits["z_min_mm"]
        z_max = axis_limits["z_max_mm"]
        r = axis_limits["printable_radius_mm"]

        layout = QVBoxLayout(self)

        tower_box = QGroupBox("Tower Heights (Forward Kinematics)")
        tower_form = QFormLayout(tower_box)
        self._tower_rows = [
            SliderSpinRow(z_min, z_max, z_min, self) for _ in range(3)
        ]
        for i, row in enumerate(self._tower_rows):
            tower_form.addRow(f"Axis {i + 1}", row)
            row.valueChanged.connect(self._emit_tower_heights)
        layout.addWidget(tower_box)

        effector_box = QGroupBox("Effector Position (Inverse Kinematics)")
        effector_form = QFormLayout(effector_box)
        self._effector_rows = [
            SliderSpinRow(-r, r, 0.0, self),
            SliderSpinRow(-r, r, 0.0, self),
            SliderSpinRow(z_min, z_max, z_min, self),
        ]
        for label, row in zip(("X", "Y", "Z"), self._effector_rows):
            effector_form.addRow(label, row)
            row.valueChanged.connect(self._emit_effector_position)
        layout.addWidget(effector_box)

        layout.addStretch(1)

    def _emit_tower_heights(self, _value):
        self.towerHeightsChanged.emit(*(row.value() for row in self._tower_rows))

    def _emit_effector_position(self, _value):
        self.effectorPositionChanged.emit(*(row.value() for row in self._effector_rows))

    def set_tower_heights(self, h0, h1, h2):
        for row, value in zip(self._tower_rows, (h0, h1, h2)):
            row.set_value(value)

    def set_effector_position(self, x, y, z):
        for row, value in zip(self._effector_rows, (x, y, z)):
            row.set_value(value)
