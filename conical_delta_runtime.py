#!/usr/bin/env python3
"""High-frequency runtime for the conical ("teepee") delta kinematics.

conical_delta_kinematics.py derives the model (sympy) and pickles it.
This module loads that pickle once and lambdifies its IK/FK expressions
once at construction time, so repeated calls pay no sympy overhead --
suitable for a real-time control loop calling inverse_kinematics /
forward_kinematics many times per second.
"""
import math
import sys
import time

import sympy as sp

from conical_delta_kinematics import DEFAULT_MODEL_FILE, load_model, validate_rail_limits


class ConicalDeltaKinematicsRuntime:
    """Loads a pickled conical-delta kinematics model once and exposes
    cheap, repeated-call inverse/forward kinematics.
    """

    def __init__(self, pickle_path=DEFAULT_MODEL_FILE, *, validate=True):
        self._init_from_model(load_model(pickle_path), validate=validate)

    @classmethod
    def from_model(cls, model, *, validate=True):
        self = cls.__new__(cls)
        self._init_from_model(model, validate=validate)
        return self

    def _init_from_model(self, model, *, validate):
        self._model = model
        self._validate_default = validate
        self._tower_bases = model["tower_bases_mm"]
        self._tower_dirs = model["tower_dirs"]
        self._rail_length = model["rail_length_mm"]

        x, y, z = model["symbols"]["x"], model["symbols"]["y"], model["symbols"]["z"]
        self._ik_fns = [sp.lambdify((x, y, z), expr, "math") for expr in model["ik_exprs"]]
        # cse=True: the FK expressions run into the thousands of ops (the
        # tilted-rail trilateration doesn't cancel as cleanly as the
        # linear model's vertical-rail case) -- without common
        # subexpression elimination this lambdified function is ~50x
        # slower per call.
        self._fk_fn = sp.lambdify(model["symbols"]["rail_s"], model["fk_exprs"], "math", cse=True)

    def inverse_kinematics(self, point, *, validate=None):
        """Return [s0, s1, s2] (mm, distance from each tower's base
        anchor) for the given (x, y, z) effector point.
        """
        px, py, pz = point
        try:
            rail_s = [fn(px, py, pz) for fn in self._ik_fns]
        except ValueError as exc:
            raise ValueError(f"point unreachable ({exc})") from exc

        if validate if validate is not None else self._validate_default:
            validate_rail_limits(self._model, point, rail_s)
        return rail_s

    def forward_kinematics(self, rail_s_values):
        """Return the effector (x, y, z) for the given carriage rail positions."""
        return self._fk_fn(*rail_s_values)

    def carriage_positions(self, point, *, validate=None):
        """Same per-tower report shape as compute_carriage_positions:
        {"tower", "carriage_rail_s_mm", "carriage_joint", "effector_joint"}.
        """
        rail_s = self.inverse_kinematics(point, validate=validate)
        px, py, pz = point
        return [
            {
                "tower": i,
                "carriage_rail_s_mm": s,
                "carriage_joint": (bx + s * dx, by + s * dy, bz + s * dz),
                "effector_joint": (px, py, pz),
            }
            for i, (s, (bx, by, bz), (dx, dy, dz))
            in enumerate(zip(rail_s, self._tower_bases, self._tower_dirs))
        ]


def _benchmark(runtime, n=100_000):
    point = (30.0, -20.0, 80.0)
    rail_s = runtime.inverse_kinematics(point)  # warmup

    t0 = time.perf_counter()
    for _ in range(n):
        runtime.inverse_kinematics(point)
    t1 = time.perf_counter()
    print(f"inverse_kinematics (validate=True):  {n/(t1-t0):>12,.0f} calls/sec")

    t0 = time.perf_counter()
    for _ in range(n):
        runtime.inverse_kinematics(point, validate=False)
    t1 = time.perf_counter()
    print(f"inverse_kinematics (validate=False): {n/(t1-t0):>12,.0f} calls/sec")

    t0 = time.perf_counter()
    for _ in range(n):
        runtime.forward_kinematics(rail_s)
    t1 = time.perf_counter()
    print(f"forward_kinematics:                  {n/(t1-t0):>12,.0f} calls/sec")

    t0 = time.perf_counter()
    for _ in range(n):
        runtime.carriage_positions(point)
    t1 = time.perf_counter()
    print(f"carriage_positions:                  {n/(t1-t0):>12,.0f} calls/sec")


def main(argv):
    pickle_path = argv[1] if len(argv) > 1 else DEFAULT_MODEL_FILE

    t0 = time.perf_counter()
    runtime = ConicalDeltaKinematicsRuntime(pickle_path)
    t1 = time.perf_counter()
    print(f"loaded + lambdified {pickle_path} in {(t1-t0)*1e3:.2f} ms\n")

    _benchmark(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
