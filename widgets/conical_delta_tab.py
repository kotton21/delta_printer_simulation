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

Also hosts a RobotParamsPanel (stacked below the sidebar's sliders, in a
left-hand column next to the visualizer) exposing every geometry/
workspace-hint value in the pickled model (base_radius_mm, tower_height_mm,
rod_length_mm, z_min_mm, z_max_mm, printable_radius_mm). Editing these and
clicking Apply re-derives the kinematics symbolically (see
conical_delta_kinematics.build_kinematics_model), re-pickles the model to
disk so the change persists across restarts, and rebuilds the
sidebar/visualizer (their ranges and drawn tower geometry are fixed at
construction time from the model, so they can't just be patched in place).
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

import conical_delta_kinematics as cdk
from conical_delta_runtime import ConicalDeltaKinematicsRuntime
from render_geometry import build_render_geometry_conical
from widgets.robot_params import RobotParamsPanel
from widgets.sidebar_conical import ConicalSidebarPanel
from widgets.view_2d_conical import ConicalTwoDViewWidget

DEFAULT_EFFECTOR_POINT = (0.0, 0.0, 100.0)


def _param_fields(model):
    geometry = model["geometry"]
    workspace_hint = model["workspace_hint"]
    return [
        {"key": "base_radius_mm", "label": "Base Radius", "value": geometry["base_radius_mm"],
         "minimum": 10.0, "maximum": 2000.0},
        {"key": "tower_height_mm", "label": "Tower Height", "value": geometry["tower_height_mm"],
         "minimum": 10.0, "maximum": 5000.0},
        {"key": "rod_length_mm", "label": "Rod Length", "value": geometry["rod_length_mm"],
         "minimum": 10.0, "maximum": 2000.0},
        {"key": "z_min_mm", "label": "Z Min (hint)", "value": workspace_hint["z_min_mm"],
         "minimum": -1000.0, "maximum": 1000.0},
        {"key": "z_max_mm", "label": "Z Max (hint)", "value": workspace_hint["z_max_mm"],
         "minimum": 10.0, "maximum": 5000.0},
        {"key": "printable_radius_mm", "label": "Printable Radius (hint)",
         "value": workspace_hint["printable_radius_mm"], "minimum": 10.0, "maximum": 2000.0},
    ]


class ConicalDeltaTab(QWidget):
    """Tab content for the conical ("teepee") linear delta."""

    statusMessage = Signal(str)  # empty string means "clear"

    def __init__(self, model, runtime, parent=None):
        super().__init__(parent)

        self._model = model
        self._runtime = runtime

        self._params_panel = RobotParamsPanel(_param_fields(self._model))
        self._params_panel.applyRequested.connect(self._on_apply_params)

        self._unreachable_label = QLabel()
        self._unreachable_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        self._unreachable_label.setVisible(False)

        # Left column: rail/effector sliders on top, robot-parameters panel
        # below. Only the sidebar/visualizer are recreated on Apply (their
        # ranges and drawn tower geometry are fixed at construction from the
        # model), so the params panel is added once and the sidebar is always
        # (re-)inserted above it via insertWidget(0, ...).
        self._left_layout = QVBoxLayout()
        self._sidebar = None
        self._visualizer = None

        self._row_layout = QHBoxLayout()
        self._row_layout.addLayout(self._left_layout, stretch=0)

        self._build_sidebar_and_visualizer()
        self._left_layout.addWidget(self._params_panel)

        outer = QVBoxLayout(self)
        outer.addWidget(self._unreachable_label)
        outer.addLayout(self._row_layout)

        self._initialize_default_pose()

    def _build_sidebar_and_visualizer(self):
        self._sidebar = ConicalSidebarPanel(self._model["rail_length_mm"], self._model["workspace_hint"])
        self._visualizer = ConicalTwoDViewWidget(self._model)

        self._sidebar.railSChanged.connect(self._on_rail_s_changed)
        self._sidebar.effectorPositionChanged.connect(self._on_effector_position_changed)

        self._left_layout.insertWidget(0, self._sidebar)
        self._row_layout.addWidget(self._visualizer, stretch=1)

    def _rebuild_body(self):
        self._left_layout.removeWidget(self._sidebar)
        self._sidebar.deleteLater()
        self._row_layout.removeWidget(self._visualizer)
        self._visualizer.deleteLater()
        self._build_sidebar_and_visualizer()

    def _on_apply_params(self, values):
        geometry = {
            "base_radius_mm": values["base_radius_mm"],
            "tower_height_mm": values["tower_height_mm"],
            "rod_length_mm": values["rod_length_mm"],
        }
        workspace_hint = {
            "z_min_mm": values["z_min_mm"],
            "z_max_mm": values["z_max_mm"],
            "printable_radius_mm": values["printable_radius_mm"],
        }

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            new_model = cdk.build_kinematics_model(geometry=geometry, workspace_hint=workspace_hint, verbose=False)
            cdk.save_model(new_model, cdk.DEFAULT_MODEL_FILE)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.statusMessage.emit(f"Failed to rebuild model: {exc}")
            return
        QApplication.restoreOverrideCursor()

        self._model = new_model
        self._runtime = ConicalDeltaKinematicsRuntime.from_model(new_model)
        self._rebuild_body()
        self._initialize_default_pose()
        self.statusMessage.emit(f"Robot parameters updated and saved to {cdk.DEFAULT_MODEL_FILE}.")

    def _show_unreachable(self, exc):
        self._unreachable_label.setText(f"Unreachable: {exc}")
        self._unreachable_label.setVisible(True)
        self.statusMessage.emit(f"Unreachable pose: {exc}")

    def _clear_unreachable(self):
        self._unreachable_label.setVisible(False)
        self.statusMessage.emit("")

    def _initialize_default_pose(self):
        hint = self._model["workspace_hint"]
        x, y, z = DEFAULT_EFFECTOR_POINT
        z = min(max(z, hint["z_min_mm"]), hint["z_max_mm"])
        try:
            rail_s = self._runtime.inverse_kinematics((x, y, z))
        except ValueError as exc:
            self._show_unreachable(exc)
            return
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
