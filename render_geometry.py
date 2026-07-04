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


def build_render_geometry_conical(model, rail_s_values, effector_point):
    """Conical-delta counterpart to build_render_geometry: each tower's
    carriage sits at `base + s*dir` along its tilted rail (not directly
    above a fixed (x,y) the way the linear model's carriage is), so the
    carriage position must be reconstructed from tower_bases_mm/tower_dirs
    rather than just paired with a height.

    Returns {"towers": [{"base": (x,y,0), "carriage": (x,y,z)}, ...],
    "effector": (x,y,z)}.
    """
    towers = [
        {"base": base, "carriage": (base[0] + s * d[0], base[1] + s * d[1], base[2] + s * d[2])}
        for base, d, s in zip(model["tower_bases_mm"], model["tower_dirs"], rail_s_values)
    ]
    return {
        "towers": towers,
        "effector": tuple(effector_point),
    }
