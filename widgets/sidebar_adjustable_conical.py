#!/usr/bin/env python3
"""Left sidebar for the adjustable-rod conical delta tab: tower rail-position
sliders paired with an INDEPENDENT arm-length slider per tower (together
drive forward kinematics), and effector-position sliders (drive inverse
kinematics). Reuses the generic SliderSpinRow widget from widgets/sidebar.py,
same as ConicalSidebarPanel -- the difference is one Arm Length row per
tower rather than a single shared one, since each tower's rod length is its
own independent input to both FK and IK (see
adjustable_conical_delta_kinematics.py's module docstring for why rod length
belongs on the FK side, decoupled per tower).
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout, QWidget

from widgets.sidebar import SliderSpinRow


class AdjustableConicalSidebarPanel(QWidget):
    """Rail-position + per-tower arm-length (FK input) and effector-position
    (IK input) controls for the adjustable-rod conical delta.
    """

    # s0, s1, s2, rod_length0, rod_length1, rod_length2
    forwardInputsChanged = Signal(float, float, float, float, float, float)
    effectorPositionChanged = Signal(float, float, float)

    def __init__(self, rail_length_mm, rod_length_range, workspace_hint, parent=None):
        super().__init__(parent)
        z_min = workspace_hint["z_min_mm"]
        z_max = workspace_hint["z_max_mm"]
        r = workspace_hint["printable_radius_mm"]

        layout = QVBoxLayout(self)

        rail_box = QGroupBox("Tower Rail Position + Arm Length (Forward Kinematics)")
        rail_form = QFormLayout(rail_box)
        self._rail_rows = []
        self._arm_length_rows = []
        for i in range(3):
            rail_row = SliderSpinRow(0.0, rail_length_mm, 0.0, self)
            rail_form.addRow(f"Rail S (Tower {i + 1})", rail_row)
            rail_row.valueChanged.connect(self._emit_forward_inputs)
            self._rail_rows.append(rail_row)

            arm_row = SliderSpinRow(
                rod_length_range["min_mm"], rod_length_range["max_mm"], rod_length_range["default_mm"], self,
            )
            rail_form.addRow(f"Arm Length (Tower {i + 1})", arm_row)
            arm_row.valueChanged.connect(self._emit_forward_inputs)
            self._arm_length_rows.append(arm_row)

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

    def _emit_forward_inputs(self, _value):
        self.forwardInputsChanged.emit(
            *(row.value() for row in self._rail_rows),
            *(row.value() for row in self._arm_length_rows),
        )

    def _emit_effector_position(self, _value):
        self.effectorPositionChanged.emit(*(row.value() for row in self._effector_rows))

    def set_rail_s(self, s0, s1, s2):
        for row, value in zip(self._rail_rows, (s0, s1, s2)):
            row.set_value(value)

    def set_arm_lengths(self, r0, r1, r2):
        for row, value in zip(self._arm_length_rows, (r0, r1, r2)):
            row.set_value(value)

    def arm_lengths(self):
        return [row.value() for row in self._arm_length_rows]

    def set_effector_position(self, x, y, z):
        for row, value in zip(self._effector_rows, (x, y, z)):
            row.set_value(value)
