#!/usr/bin/env python3
"""Main window: hosts one tab per delta schema. Each tab owns its own
model/runtime pair and wiring (see widgets/linear_delta_tab.py and
widgets/conical_delta_tab.py); this window just arranges them in a
QTabWidget and forwards each tab's status messages to the shared status
bar.
"""
from PySide6.QtWidgets import QMainWindow, QTabWidget

from widgets.conical_delta_tab import ConicalDeltaTab
from widgets.linear_delta_tab import LinearDeltaTab


class MainWindow(QMainWindow):
    def __init__(self, linear_model, linear_runtime, conical_model, conical_runtime, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Delta Robot Kinematics Visualizer")

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        linear_tab = LinearDeltaTab(linear_model, linear_runtime)
        conical_tab = ConicalDeltaTab(conical_model, conical_runtime)

        linear_tab.statusMessage.connect(self._on_status_message)
        conical_tab.statusMessage.connect(self._on_status_message)

        tabs.addTab(linear_tab, "Linear Delta (K280)")
        tabs.addTab(conical_tab, "Conical Delta")

    def _on_status_message(self, message):
        if message:
            self.statusBar().showMessage(message)
        else:
            self.statusBar().clearMessage()
