#!/usr/bin/env python3
"""High-frequency runtime for the linear (Kossel K280-style) delta kinematics.

linear_delta_kinematics.py derives the model (sympy) and pickles it.
This module loads that pickle once and lambdifies its IK/FK expressions
once at construction time, so repeated calls pay no sympy overhead --
suitable for a real-time control loop calling inverse_kinematics /
forward_kinematics many times per second.

Benchmarked on the reference model: the "derive" module's
compute_inverse_kinematics (which re-lambdifies on every call) runs at
~680 calls/sec; DeltaKinematicsRuntime.inverse_kinematics runs at
several hundred thousand to a few million calls/sec.
"""
import math
import sys
import time

import sympy as sp

from linear_delta_kinematics import (
    DEFAULT_MODEL_FILE,
    DELTA_TYPE,
    load_model,
    validate_axis_limits,
)


class DeltaKinematicsRuntime:
    """Loads a pickled linear-delta kinematics model once and exposes
    cheap, repeated-call inverse/forward kinematics.

    Only supports models with delta_type == DELTA_TYPE
    ("linear_vertical_carriage") -- the IK/FK formulas this class assumes
    (per-tower carriage height, no elbow) are specific to that
    architecture and would silently misinterpret a different delta type.
    """

    def __init__(self, pickle_path=DEFAULT_MODEL_FILE, *, validate=True):
        self._init_from_model(load_model(pickle_path), validate=validate)

    @classmethod
    def from_model(cls, model, *, validate=True):
        self = cls.__new__(cls)
        self._init_from_model(model, validate=validate)
        return self

    def _init_from_model(self, model, *, validate):
        model_type = model.get("delta_type")
        if model_type != DELTA_TYPE:
            raise ValueError(
                f"DeltaKinematicsRuntime only supports delta_type={DELTA_TYPE!r}, "
                f"got {model_type!r}"
            )
        self._model = model
        self._validate_default = validate
        self._tower_positions = model["tower_positions"]
        self._axis_limits = model["axis_limits"]

        x, y, z = model["symbols"]["x"], model["symbols"]["y"], model["symbols"]["z"]
        self._ik_fns = [sp.lambdify((x, y, z), expr, "math") for expr in model["ik_exprs"]]
        self._fk_fn = sp.lambdify(model["symbols"]["heights"], model["fk_exprs"], "math")

    def inverse_kinematics(self, point, *, validate=None):
        """Return [h0, h1, h2] (mm) for the given (x, y, z) effector point."""
        if validate if validate is not None else self._validate_default:
            validate_axis_limits(self._model, point)

        px, py, pz = point
        try:
            return [fn(px, py, pz) for fn in self._ik_fns]
        except ValueError as exc:
            raise ValueError(f"point unreachable ({exc})") from exc

    def forward_kinematics(self, heights):
        """Return the effector (x, y, z) for the given carriage heights."""
        return self._fk_fn(*heights)

    def carriage_positions(self, point, *, validate=None):
        """Same per-tower report shape as compute_carriage_positions:
        {"tower", "carriage_height_mm", "carriage_joint", "effector_joint"}.
        """
        heights = self.inverse_kinematics(point, validate=validate)
        px, py, pz = point
        return [
            {
                "tower": i,
                "carriage_height_mm": h,
                "carriage_joint": (tx, ty, h),
                "effector_joint": (px, py, pz),
            }
            for i, (h, (tx, ty)) in enumerate(zip(heights, self._tower_positions))
        ]


def _benchmark(runtime, n=100_000):
    point = (40.0, -20.0, 300.0)
    heights = runtime.inverse_kinematics(point)  # warmup

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
        runtime.forward_kinematics(heights)
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
    runtime = DeltaKinematicsRuntime(pickle_path)
    t1 = time.perf_counter()
    print(f"loaded + lambdified {pickle_path} in {(t1-t0)*1e3:.2f} ms\n")

    _benchmark(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
