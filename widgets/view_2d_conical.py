#!/usr/bin/env python3
"""2D front/side/top views of the conical delta robot pose, rendered with
pyqtgraph. Static geometry (tilted tower rails converging to the shared
apex, tower-base triangle) is drawn once; per-pose items are updated in
place via setData(). Structurally the same as widgets/view_2d.py's
TwoDViewWidget, except each rail is drawn as the actual tilted segment
from a tower's base anchor to the shared apex, not a vertical line.
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


class ConicalTwoDViewWidget(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        tower_bases = model["tower_bases_mm"]
        apex = model["visualization"]["apex_mm"]
        tower_height = model["geometry"]["tower_height_mm"]
        base_radius = model["geometry"]["base_radius_mm"]

        self._glw = pg.GraphicsLayoutWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._glw)

        horiz = max([abs(c) for pos in tower_bases for c in pos[:2]] + [base_radius])
        pad_h = horiz * 0.15
        z_lo, z_hi = 0.0, tower_height
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
                for base in tower_bases:
                    plot.addItem(pg.PlotCurveItem(
                        x=[base[ia], apex[ia]], y=[base[2], apex[2]], pen=_RAIL_PEN,
                    ))
            else:
                xs = [pos[0] for pos in tower_bases] + [tower_bases[0][0]]
                ys = [pos[1] for pos in tower_bases] + [tower_bases[0][1]]
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
