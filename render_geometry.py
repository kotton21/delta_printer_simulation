#!/usr/bin/env python3
"""Repackages kinematics runtime output into a plain dict for the GUI's
visualizer widgets. No kinematics math lives here -- just geometry already
produced by DeltaKinematicsRuntime and the static tower_positions/axis_limits
already present in the pickled model.
"""


def build_render_geometry(model, heights, effector_point):
    """Return {"towers": [{"base": (x,y,0), "carriage": (x,y,h)}, ...],
    "effector": (x,y,z), "axis_limits": {...}} for the given tower heights
    and the effector point already computed by forward/inverse kinematics.
    """
    towers = [
        {"base": (tx, ty, 0.0), "carriage": (tx, ty, h)}
        for (tx, ty), h in zip(model["tower_positions"], heights)
    ]
    return {
        "towers": towers,
        "effector": tuple(effector_point),
        "axis_limits": model["axis_limits"],
    }
