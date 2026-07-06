#!/usr/bin/env python3
"""Generic "Robot Parameters" panel: a form of labeled spin boxes plus an
Apply button, used by both the linear- and conical-delta tabs to edit the
physical geometry (radius, tower height, rod length, axis/workspace limits,
...) that is baked into the pickled kinematics model.

Unlike the FK/IK sliders in widgets/sidebar.py, changing one of these values
does not just move a pose -- it changes constants that the kinematics
expressions were symbolically derived from, so editing them requires
re-deriving (and re-pickling) the whole model. This widget only collects the
new values and reports them via `applyRequested`; the owning tab is
responsible for rebuilding the model/runtime and swapping the rest of the UI.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RobotParamsPanel(QWidget):
    """`fields` is an ordered list of dicts, each with keys:
    key (dict key used in the emitted values dict), label (display text),
    value (initial value), minimum, maximum, decimals (optional, default 2),
    step (optional, default 1.0), suffix (optional, default " mm").
    """

    applyRequested = Signal(dict)

    def __init__(self, fields, title="Robot Parameters", parent=None):
        super().__init__(parent)

        box = QGroupBox(title)
        form = QFormLayout(box)

        self._spinboxes = {}
        for field in fields:
            spin = QDoubleSpinBox()
            spin.setDecimals(field.get("decimals", 2))
            spin.setRange(field["minimum"], field["maximum"])
            spin.setSingleStep(field.get("step", 1.0))
            spin.setSuffix(field.get("suffix", " mm"))
            spin.setValue(field["value"])
            form.addRow(field["label"], spin)
            self._spinboxes[field["key"]] = spin

        self._apply_button = QPushButton("Apply && Rebuild Model")
        self._apply_button.clicked.connect(self._on_apply_clicked)
        form.addRow(self._apply_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        layout.addStretch(1)

    def values(self):
        return {key: spin.value() for key, spin in self._spinboxes.items()}

    def set_values(self, values):
        for key, value in values.items():
            if key in self._spinboxes:
                self._spinboxes[key].setValue(value)

    def _on_apply_clicked(self):
        self.applyRequested.emit(self.values())
