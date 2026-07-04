#!/usr/bin/env python3
"""Main window: wires the sidebar's tower-height / effector-position
sliders to DeltaKinematicsRuntime's forward/inverse kinematics and the 2D
visualizer, guarding against feedback loops via the sidebar's
signal-blocked setters.
"""
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from render_geometry import build_render_geometry
from widgets.sidebar import SidebarPanel
from widgets.view_2d import TwoDViewWidget

DEFAULT_EFFECTOR_POINT = (0.0, 0.0, 200.0)


class MainWindow(QMainWindow):
    def __init__(self, model, runtime, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Delta Robot Kinematics Visualizer")

        self._model = model
        self._runtime = runtime

        self._sidebar = SidebarPanel(model["axis_limits"])
        self._visualizer = TwoDViewWidget(model)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self._sidebar, stretch=0)
        layout.addWidget(self._visualizer, stretch=1)
        self.setCentralWidget(central)

        self._sidebar.towerHeightsChanged.connect(self._on_tower_heights_changed)
        self._sidebar.effectorPositionChanged.connect(self._on_effector_position_changed)

        self._initialize_default_pose()

    def _initialize_default_pose(self):
        x, y, z = DEFAULT_EFFECTOR_POINT
        heights = self._runtime.inverse_kinematics((x, y, z))
        self._sidebar.set_tower_heights(*heights)
        self._sidebar.set_effector_position(x, y, z)
        geom = build_render_geometry(self._model, heights, (x, y, z))
        self._visualizer.update_pose(geom)

    def _on_tower_heights_changed(self, h0, h1, h2):
        try:
            x, y, z = self._runtime.forward_kinematics([h0, h1, h2])
        except ValueError as exc:
            self.statusBar().showMessage(f"Unreachable pose: {exc}")
            return
        self.statusBar().clearMessage()
        self._sidebar.set_effector_position(x, y, z)
        geom = build_render_geometry(self._model, [h0, h1, h2], (x, y, z))
        self._visualizer.update_pose(geom)

    def _on_effector_position_changed(self, x, y, z):
        try:
            h0, h1, h2 = self._runtime.inverse_kinematics((x, y, z))
        except ValueError as exc:
            self.statusBar().showMessage(f"Unreachable pose: {exc}")
            return
        self.statusBar().clearMessage()
        self._sidebar.set_tower_heights(h0, h1, h2)
        geom = build_render_geometry(self._model, [h0, h1, h2], (x, y, z))
        self._visualizer.update_pose(geom)
