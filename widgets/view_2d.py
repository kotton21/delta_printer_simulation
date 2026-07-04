#!/usr/bin/env python3
"""2D front/side/top views of the delta robot pose, rendered with
pyqtgraph. Static geometry (tower rails, tower-base triangle) is drawn
once; per-pose items are updated in place via setData().
"""
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

# (title, horizontal-axis index, vertical-axis index); axis index: 0=x, 1=y, 2=z
_VIEWS = [
    ("Front (X-Z)", 0, 2),
    ("Side (Y-Z)", 1, 2),
    ("Top (X-Y)", 0, 1),
]

_RAIL_PEN = pg.mkPen(color=(120, 120, 120), width=1, style=Qt.PenStyle.DashLine)
_TRIANGLE_PEN = pg.mkPen(color=(120, 120, 120), width=1)
_ROD_PEN = pg.mkPen(color=(200, 80, 40), width=2)


class TwoDViewWidget(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        axis_limits = model["axis_limits"]
        tower_positions = model["tower_positions"]

        self._glw = pg.GraphicsLayoutWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._glw)

        horiz = max([abs(c) for pos in tower_positions for c in pos] + [axis_limits["printable_radius_mm"]])
        pad_h = horiz * 0.15
        z_lo, z_hi = axis_limits["z_min_mm"], axis_limits["z_max_mm"]
        pad_v = (z_hi - z_lo) * 0.1
        ranges = {
            0: (-horiz - pad_h, horiz + pad_h),
            1: (-horiz - pad_h, horiz + pad_h),
            2: (z_lo - pad_v, z_hi + pad_v),
        }

        self._views = []

        for title, ia, ib in _VIEWS:
            plot = self._glw.addPlot(title=title)
            plot.setAspectLocked(True)
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.setXRange(*ranges[ia], padding=0)
            plot.setYRange(*ranges[ib], padding=0)

            if ib == 2:
                for pos in tower_positions:
                    coord = pos[ia]
                    plot.addItem(pg.PlotCurveItem(x=[coord, coord], y=[z_lo, z_hi], pen=_RAIL_PEN))
            else:
                xs = [pos[0] for pos in tower_positions] + [tower_positions[0][0]]
                ys = [pos[1] for pos in tower_positions] + [tower_positions[0][1]]
                plot.addItem(pg.PlotCurveItem(x=xs, y=ys, pen=_TRIANGLE_PEN))

            carriage_item = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 120, 220), pen=None)
            rod_item = pg.PlotCurveItem(pen=_ROD_PEN, connect="pairs")
            effector_item = pg.ScatterPlotItem(size=14, symbol="star", brush=pg.mkBrush(230, 30, 30), pen=None)
            for item in (carriage_item, rod_item, effector_item):
                plot.addItem(item)

            self._views.append({"ia": ia, "ib": ib, "carriage": carriage_item, "rods": rod_item, "effector": effector_item})

    def update_pose(self, render_geometry):
        towers = render_geometry["towers"]
        effector = render_geometry["effector"]

        for view in self._views:
            ia, ib = view["ia"], view["ib"]

            carriage_x = [t["carriage"][ia] for t in towers]
            carriage_y = [t["carriage"][ib] for t in towers]
            view["carriage"].setData(x=carriage_x, y=carriage_y)

            rod_x, rod_y = [], []
            for t in towers:
                rod_x.extend([t["carriage"][ia], effector[ia]])
                rod_y.extend([t["carriage"][ib], effector[ib]])
            view["rods"].setData(x=rod_x, y=rod_y)

            view["effector"].setData(x=[effector[ia]], y=[effector[ib]])
