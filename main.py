#!/usr/bin/env python3
import sys

from PySide6.QtWidgets import QApplication

import conical_delta_kinematics
import linear_delta_kinematics
from conical_delta_runtime import ConicalDeltaKinematicsRuntime
from linear_delta_runtime import DeltaKinematicsRuntime
from main_window import MainWindow


def main(argv):
    app = QApplication(argv)

    linear_model = linear_delta_kinematics.load_model(linear_delta_kinematics.DEFAULT_MODEL_FILE)
    linear_runtime = DeltaKinematicsRuntime.from_model(linear_model)

    conical_model = conical_delta_kinematics.load_model(conical_delta_kinematics.DEFAULT_MODEL_FILE)
    conical_runtime = ConicalDeltaKinematicsRuntime.from_model(conical_model)

    window = MainWindow(linear_model, linear_runtime, conical_model, conical_runtime)
    window.resize(1200, 700)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
