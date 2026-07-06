#!/usr/bin/env python3
"""Linear delta tab: wires the sidebar's tower-height / effector-position
sliders to DeltaKinematicsRuntime's forward/inverse kinematics and the 2D
visualizer, guarding against feedback loops via the sidebar's
signal-blocked setters. Lifted unchanged from the original single-schema
MainWindow so its behavior is identical to before the GUI grew tabs.

Also hosts a RobotParamsPanel (stacked below the sidebar's sliders, in a
left-hand column next to the visualizer) exposing every geometry/axis-limit
value in the pickled model (delta_radius_mm, rod_length_mm, z_min_mm,
z_max_mm, printable_radius_mm). Editing these and clicking Apply re-derives
the kinematics symbolically (see
linear_delta_kinematics.build_kinematics_model), re-pickles the model to
disk so the change persists across restarts, and rebuilds the
sidebar/visualizer (their ranges and drawn tower geometry are fixed at
construction time from the model, so they can't just be patched in place).

The tower-height sliders' upper bound is NOT axis_limits["z_max_mm"] (that's
the max *effector* Z, not the max carriage travel) -- it's
linear_delta_kinematics.compute_max_carriage_heights(model), the actual
highest a carriage can sit while still reaching some point in the
documented workspace.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget

import linear_delta_kinematics as ldk
from linear_delta_runtime import DeltaKinematicsRuntime
from render_geometry import build_render_geometry
from widgets.robot_params import RobotParamsPanel
from widgets.sidebar import SidebarPanel
from widgets.view_2d import TwoDViewWidget

DEFAULT_EFFECTOR_POINT = (0.0, 0.0, 200.0)


def _param_fields(model):
    geometry = model["geometry"]
    axis_limits = model["axis_limits"]
    return [
        {"key": "delta_radius_mm", "label": "Delta Radius", "value": geometry["delta_radius_mm"],
         "minimum": 10.0, "maximum": 2000.0},
        {"key": "rod_length_mm", "label": "Rod Length", "value": geometry["rod_length_mm"],
         "minimum": 10.0, "maximum": 2000.0},
        {"key": "z_min_mm", "label": "Z Min", "value": axis_limits["z_min_mm"],
         "minimum": -1000.0, "maximum": 1000.0},
        {"key": "z_max_mm", "label": "Z Max", "value": axis_limits["z_max_mm"],
         "minimum": 10.0, "maximum": 5000.0},
        {"key": "printable_radius_mm", "label": "Printable Radius", "value": axis_limits["printable_radius_mm"],
         "minimum": 10.0, "maximum": 2000.0},
    ]


class LinearDeltaTab(QWidget):
    """Tab content for the vertical-rail Kossel K280-style linear delta."""

    statusMessage = Signal(str)  # empty string means "clear"

    def __init__(self, model, runtime, parent=None):
        super().__init__(parent)

        self._model = model
        self._runtime = runtime

        self._params_panel = RobotParamsPanel(_param_fields(self._model))
        self._params_panel.applyRequested.connect(self._on_apply_params)

        # Left column: tower/effector sliders on top, robot-parameters panel
        # below. Only the sidebar/visualizer are recreated on Apply (their
        # ranges and drawn tower geometry are fixed at construction from the
        # model), so the params panel is added once and the sidebar is always
        # (re-)inserted above it via insertWidget(0, ...).
        self._left_layout = QVBoxLayout()
        self._sidebar = None
        self._visualizer = None

        self._outer_layout = QHBoxLayout(self)
        self._outer_layout.addLayout(self._left_layout, stretch=0)

        self._build_sidebar_and_visualizer()
        self._left_layout.addWidget(self._params_panel)

        self._initialize_default_pose()

    def _build_sidebar_and_visualizer(self):
        tower_height_max = max(ldk.compute_max_carriage_heights(self._model))
        self._sidebar = SidebarPanel(self._model["axis_limits"], tower_height_max)
        self._visualizer = TwoDViewWidget(self._model)

        self._sidebar.towerHeightsChanged.connect(self._on_tower_heights_changed)
        self._sidebar.effectorPositionChanged.connect(self._on_effector_position_changed)

        self._left_layout.insertWidget(0, self._sidebar)
        self._outer_layout.addWidget(self._visualizer, stretch=1)

    def _rebuild_body(self):
        self._left_layout.removeWidget(self._sidebar)
        self._sidebar.deleteLater()
        self._outer_layout.removeWidget(self._visualizer)
        self._visualizer.deleteLater()
        self._build_sidebar_and_visualizer()

    def _on_apply_params(self, values):
        geometry = {
            "delta_radius_mm": values["delta_radius_mm"],
            "rod_length_mm": values["rod_length_mm"],
        }
        axis_limits = {
            "z_min_mm": values["z_min_mm"],
            "z_max_mm": values["z_max_mm"],
            "printable_radius_mm": values["printable_radius_mm"],
        }

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            new_model = ldk.build_kinematics_model(geometry=geometry, axis_limits=axis_limits, verbose=False)
            ldk.save_model(new_model, ldk.DEFAULT_MODEL_FILE)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.statusMessage.emit(f"Failed to rebuild model: {exc}")
            return
        QApplication.restoreOverrideCursor()

        self._model = new_model
        self._runtime = DeltaKinematicsRuntime.from_model(new_model)
        self._rebuild_body()
        self._initialize_default_pose()
        self.statusMessage.emit(f"Robot parameters updated and saved to {ldk.DEFAULT_MODEL_FILE}.")

    def _initialize_default_pose(self):
        limits = self._model["axis_limits"]
        x, y, z = DEFAULT_EFFECTOR_POINT
        z = min(max(z, limits["z_min_mm"]), limits["z_max_mm"])
        try:
            heights = self._runtime.inverse_kinematics((x, y, z))
        except ValueError as exc:
            self.statusMessage.emit(f"Unreachable default pose: {exc}")
            return
        self.statusMessage.emit("")
        self._sidebar.set_tower_heights(*heights)
        self._sidebar.set_effector_position(x, y, z)
        geom = build_render_geometry(self._model, heights, (x, y, z))
        self._visualizer.update_pose(geom)

    def _on_tower_heights_changed(self, h0, h1, h2):
        try:
            x, y, z = self._runtime.forward_kinematics([h0, h1, h2])
        except ValueError as exc:
            self.statusMessage.emit(f"Unreachable pose: {exc}")
            return
        self.statusMessage.emit("")
        self._sidebar.set_effector_position(x, y, z)
        geom = build_render_geometry(self._model, [h0, h1, h2], (x, y, z))
        self._visualizer.update_pose(geom)

    def _on_effector_position_changed(self, x, y, z):
        try:
            h0, h1, h2 = self._runtime.inverse_kinematics((x, y, z))
        except ValueError as exc:
            self.statusMessage.emit(f"Unreachable pose: {exc}")
            return
        self.statusMessage.emit("")
        self._sidebar.set_tower_heights(h0, h1, h2)
        geom = build_render_geometry(self._model, [h0, h1, h2], (x, y, z))
        self._visualizer.update_pose(geom)
