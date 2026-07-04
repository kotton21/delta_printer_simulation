#!/usr/bin/env python3
"""Left sidebar for the conical delta tab: tower rail-position sliders
(drive forward kinematics) and effector-position sliders (drive inverse
kinematics). Reuses the generic SliderSpinRow widget from widgets/sidebar.py
-- only the labels/ranges/signal names differ from the linear delta's
SidebarPanel, since "rail position along a tilted rail" replaces "tower
height" here.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout, QWidget

from widgets.sidebar import SliderSpinRow


class ConicalSidebarPanel(QWidget):
    """Rail-position (FK input) and effector-position (IK input) controls
    for the conical delta.
    """

    railSChanged = Signal(float, float, float)
    effectorPositionChanged = Signal(float, float, float)

    def __init__(self, rail_length_mm, workspace_hint, parent=None):
        super().__init__(parent)
        z_min = workspace_hint["z_min_mm"]
        z_max = workspace_hint["z_max_mm"]
        r = workspace_hint["printable_radius_mm"]

        layout = QVBoxLayout(self)

        rail_box = QGroupBox("Tower Rail Position (Forward Kinematics)")
        rail_form = QFormLayout(rail_box)
        self._rail_rows = [
            SliderSpinRow(0.0, rail_length_mm, 0.0, self) for _ in range(3)
        ]
        for i, row in enumerate(self._rail_rows):
            rail_form.addRow(f"Rail S (Tower {i + 1})", row)
            row.valueChanged.connect(self._emit_rail_s)
        layout.addWidget(rail_box)

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

    def _emit_rail_s(self, _value):
        self.railSChanged.emit(*(row.value() for row in self._rail_rows))

    def _emit_effector_position(self, _value):
        self.effectorPositionChanged.emit(*(row.value() for row in self._effector_rows))

    def set_rail_s(self, s0, s1, s2):
        for row, value in zip(self._rail_rows, (s0, s1, s2)):
            row.set_value(value)

    def set_effector_position(self, x, y, z):
        for row, value in zip(self._effector_rows, (x, y, z)):
            row.set_value(value)
