#!/usr/bin/env python3
import sys

from PySide6.QtWidgets import QApplication

from linear_delta_kinematics import DEFAULT_MODEL_FILE, load_model
from linear_delta_runtime import DeltaKinematicsRuntime
from main_window import MainWindow


def main(argv):
    app = QApplication(argv)

    model = load_model(DEFAULT_MODEL_FILE)
    runtime = DeltaKinematicsRuntime.from_model(model)

    window = MainWindow(model, runtime)
    window.resize(1200, 700)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
