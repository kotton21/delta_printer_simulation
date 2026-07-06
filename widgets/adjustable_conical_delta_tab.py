#!/usr/bin/env python3
"""Adjustable-rod conical delta tab: wires the AdjustableConicalSidebarPanel's
rail-position + per-tower arm-length sliders (forward kinematics) and
effector-position sliders (inverse kinematics) to
AdjustableConicalDeltaKinematicsRuntime, and the shared conical 2D visualizer
(identical rendering to the fixed-rod conical delta -- rod length only
changes which rail_s a given effector point maps to, not how a (base, dir,
rail_s) triple is drawn).

Redundancy handling (see adjustable_conical_delta_kinematics.py's module
docstring for the underlying math): with each tower's rod length free and
independent, a single effector target has, per tower, a whole interval of
workable rod lengths rather than one, and the three towers' intervals never
interact. Per the requirement that arm length should only move when it has
to -- and only the arm(s) that actually need it -- effector-slider (IK)
moves first try the CURRENT three arm lengths; only the towers whose
current arm length falls outside that tower's own reachable interval get
nudged (to the nearest edge of that interval, the minimal change that
restores reachability) via resolve_rod_lengths; any tower already fine is
left untouched. Rail-position + arm-length slider moves (FK) are simpler:
forward kinematics is just evaluated at whatever the six sliders currently
say, no resolution needed.

Also hosts a RobotParamsPanel exposing base_radius_mm, tower_height_mm, the
arm-length range (min/default/max, shared across all three towers as the
allowed slider range, though each tower's current value is independent),
and the workspace-hint values. Editing these and clicking Apply re-derives
the kinematics symbolically (each tower's rod length stays its own free
symbol -- see build_kinematics_model), re-pickles the model, and rebuilds
the sidebar/visualizer, exactly as ConicalDeltaTab does for the fixed-rod
model.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

import adjustable_conical_delta_kinematics as acdk
from adjustable_conical_delta_runtime import AdjustableConicalDeltaKinematicsRuntime
from render_geometry import build_render_geometry_conical
from widgets.robot_params import RobotParamsPanel
from widgets.sidebar_adjustable_conical import AdjustableConicalSidebarPanel
from widgets.view_2d_conical import ConicalTwoDViewWidget

DEFAULT_EFFECTOR_POINT = (0.0, 0.0, 100.0)


def _param_fields(model):
    geometry = model["geometry"]
    rod_range = model["rod_length_range_mm"]
    workspace_hint = model["workspace_hint"]
    return [
        {"key": "base_radius_mm", "label": "Base Radius", "value": geometry["base_radius_mm"],
         "minimum": 10.0, "maximum": 2000.0},
        {"key": "tower_height_mm", "label": "Tower Height", "value": geometry["tower_height_mm"],
         "minimum": 10.0, "maximum": 5000.0},
        {"key": "rod_length_min_mm", "label": "Arm Length Min", "value": rod_range["min_mm"],
         "minimum": 10.0, "maximum": 2000.0},
        {"key": "rod_length_default_mm", "label": "Arm Length Default", "value": rod_range["default_mm"],
         "minimum": 10.0, "maximum": 2000.0},
        {"key": "rod_length_max_mm", "label": "Arm Length Max", "value": rod_range["max_mm"],
         "minimum": 10.0, "maximum": 2000.0},
        {"key": "z_min_mm", "label": "Z Min (hint)", "value": workspace_hint["z_min_mm"],
         "minimum": -1000.0, "maximum": 1000.0},
        {"key": "z_max_mm", "label": "Z Max (hint)", "value": workspace_hint["z_max_mm"],
         "minimum": 10.0, "maximum": 5000.0},
        {"key": "printable_radius_mm", "label": "Printable Radius (hint)",
         "value": workspace_hint["printable_radius_mm"], "minimum": 10.0, "maximum": 2000.0},
    ]


class AdjustableConicalDeltaTab(QWidget):
    """Tab content for the conical ("teepee") linear delta with an
    independently runtime-adjustable rod (arm) length per tower.
    """

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
        self._sidebar = AdjustableConicalSidebarPanel(
            self._model["rail_length_mm"], self._model["rod_length_range_mm"], self._model["workspace_hint"],
        )
        self._visualizer = ConicalTwoDViewWidget(self._model)

        self._sidebar.forwardInputsChanged.connect(self._on_forward_inputs_changed)
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
        }
        rod_length_range = {
            "min_mm": values["rod_length_min_mm"],
            "max_mm": values["rod_length_max_mm"],
            "default_mm": values["rod_length_default_mm"],
        }
        workspace_hint = {
            "z_min_mm": values["z_min_mm"],
            "z_max_mm": values["z_max_mm"],
            "printable_radius_mm": values["printable_radius_mm"],
        }

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            new_model = acdk.build_kinematics_model(
                geometry=geometry, rod_length_range=rod_length_range,
                workspace_hint=workspace_hint, verbose=False,
            )
            acdk.save_model(new_model, acdk.DEFAULT_MODEL_FILE)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.statusMessage.emit(f"Failed to rebuild model: {exc}")
            return
        QApplication.restoreOverrideCursor()

        self._model = new_model
        self._runtime = AdjustableConicalDeltaKinematicsRuntime.from_model(new_model)
        self._rebuild_body()
        self._initialize_default_pose()
        self.statusMessage.emit(f"Robot parameters updated and saved to {acdk.DEFAULT_MODEL_FILE}.")

    def _show_unreachable(self, exc):
        self._unreachable_label.setText(f"Unreachable: {exc}")
        self._unreachable_label.setVisible(True)
        self.statusMessage.emit(f"Unreachable pose: {exc}")

    def _clear_unreachable(self):
        self._unreachable_label.setVisible(False)
        self.statusMessage.emit("")

    def _rod_length_limits(self):
        rod_range = self._model["rod_length_range_mm"]
        return rod_range["min_mm"], rod_range["max_mm"]

    def _initialize_default_pose(self):
        hint = self._model["workspace_hint"]
        default_rod = self._model["rod_length_range_mm"]["default_mm"]
        rod_lengths = [default_rod] * acdk.ARM_COUNT
        x, y, z = DEFAULT_EFFECTOR_POINT
        z = min(max(z, hint["z_min_mm"]), hint["z_max_mm"])
        rod_min, rod_max = self._rod_length_limits()
        resolved = self._runtime.resolve_rod_lengths((x, y, z), rod_lengths, rod_min, rod_max)
        if resolved is None:
            self._show_unreachable(ValueError(f"no arm length in [{rod_min}, {rod_max}] reaches {(x, y, z)}"))
            return
        try:
            rail_s = self._runtime.inverse_kinematics((x, y, z), resolved)
        except ValueError as exc:
            self._show_unreachable(exc)
            return
        self._sidebar.set_rail_s(*rail_s)
        self._sidebar.set_arm_lengths(*resolved)
        self._sidebar.set_effector_position(x, y, z)
        geom = build_render_geometry_conical(self._model, rail_s, (x, y, z))
        self._visualizer.update_pose(geom)
        self._clear_unreachable()

    def _on_forward_inputs_changed(self, s0, s1, s2, r0, r1, r2):
        rail_s = [s0, s1, s2]
        rod_lengths = [r0, r1, r2]
        try:
            x, y, z = self._runtime.forward_kinematics(rail_s, rod_lengths)
        except ValueError as exc:
            self._show_unreachable(exc)
            return
        self._clear_unreachable()
        self._sidebar.set_effector_position(x, y, z)
        geom = build_render_geometry_conical(self._model, rail_s, (x, y, z))
        self._visualizer.update_pose(geom)

    def _on_effector_position_changed(self, x, y, z):
        rod_lengths = self._sidebar.arm_lengths()
        rod_min, rod_max = self._rod_length_limits()
        resolved = self._runtime.resolve_rod_lengths((x, y, z), rod_lengths, rod_min, rod_max)
        if resolved is None:
            self._show_unreachable(ValueError(f"no arm length in [{rod_min}, {rod_max}] reaches {(x, y, z)}"))
            return

        try:
            rail_s = self._runtime.inverse_kinematics((x, y, z), resolved)
        except ValueError as exc:
            self._show_unreachable(exc)
            return

        self._clear_unreachable()
        if resolved != rod_lengths:
            self._sidebar.set_arm_lengths(*resolved)
        self._sidebar.set_rail_s(*rail_s)
        geom = build_render_geometry_conical(self._model, rail_s, (x, y, z))
        self._visualizer.update_pose(geom)
