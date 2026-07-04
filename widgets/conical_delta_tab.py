#!/usr/bin/env python3
"""Conical delta tab: wires the conical sidebar's rail-position /
effector-position sliders to ConicalDeltaKinematicsRuntime's forward/inverse
kinematics and the 2D visualizer.

Unlike the linear delta tab, unreachable poses are common here (the rod is
short relative to the tower height/base radius, so a large fraction of the
effector-position slider range is not actually reachable) -- so, per
explicit design requirement, this tab shows a persistent inline "Unreachable"
label instead of relying solely on the shared window status bar, and simply
skips the draw call on failure (the visualizer keeps showing the last valid
pose) rather than drawing an invalid one.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from render_geometry import build_render_geometry_conical
from widgets.sidebar_conical import ConicalSidebarPanel
from widgets.view_2d_conical import ConicalTwoDViewWidget

DEFAULT_EFFECTOR_POINT = (0.0, 0.0, 100.0)


class ConicalDeltaTab(QWidget):
    """Tab content for the conical ("teepee") linear delta."""

    statusMessage = Signal(str)  # empty string means "clear"

    def __init__(self, model, runtime, parent=None):
        super().__init__(parent)

        self._model = model
        self._runtime = runtime

        self._unreachable_label = QLabel()
        self._unreachable_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        self._unreachable_label.setVisible(False)

        self._sidebar = ConicalSidebarPanel(model["rail_length_mm"], model["workspace_hint"])
        self._visualizer = ConicalTwoDViewWidget(model)

        row = QHBoxLayout()
        row.addWidget(self._sidebar, stretch=0)
        row.addWidget(self._visualizer, stretch=1)

        outer = QVBoxLayout(self)
        outer.addWidget(self._unreachable_label)
        outer.addLayout(row)

        self._sidebar.railSChanged.connect(self._on_rail_s_changed)
        self._sidebar.effectorPositionChanged.connect(self._on_effector_position_changed)

        self._initialize_default_pose()

    def _show_unreachable(self, exc):
        self._unreachable_label.setText(f"Unreachable: {exc}")
        self._unreachable_label.setVisible(True)
        self.statusMessage.emit(f"Unreachable pose: {exc}")

    def _clear_unreachable(self):
        self._unreachable_label.setVisible(False)
        self.statusMessage.emit("")

    def _initialize_default_pose(self):
        x, y, z = DEFAULT_EFFECTOR_POINT
        rail_s = self._runtime.inverse_kinematics((x, y, z))
        self._sidebar.set_rail_s(*rail_s)
        self._sidebar.set_effector_position(x, y, z)
        geom = build_render_geometry_conical(self._model, rail_s, (x, y, z))
        self._visualizer.update_pose(geom)
        self._clear_unreachable()

    def _on_rail_s_changed(self, s0, s1, s2):
        try:
            x, y, z = self._runtime.forward_kinematics([s0, s1, s2])
        except ValueError as exc:
            self._show_unreachable(exc)
            return
        self._clear_unreachable()
        self._sidebar.set_effector_position(x, y, z)
        geom = build_render_geometry_conical(self._model, [s0, s1, s2], (x, y, z))
        self._visualizer.update_pose(geom)

    def _on_effector_position_changed(self, x, y, z):
        try:
            s0, s1, s2 = self._runtime.inverse_kinematics((x, y, z))
        except ValueError as exc:
            self._show_unreachable(exc)
            return
        self._clear_unreachable()
        self._sidebar.set_rail_s(s0, s1, s2)
        geom = build_render_geometry_conical(self._model, [s0, s1, s2], (x, y, z))
        self._visualizer.update_pose(geom)
