#!/usr/bin/env python3
"""High-frequency runtime for the adjustable-rod conical delta kinematics.

adjustable_conical_delta_kinematics.py derives the model (sympy, with each
tower's rod length left as its own free symbol) and pickles it. This module
loads that pickle once and lambdifies its IK/FK expressions once at
construction time -- now with each tower's rod length as an extra ordinary
function argument alongside x/y/z (IK, one rod length per tower's own
expression) or s0/s1/s2 (FK, all three rod lengths), so varying any rod
length at call time costs nothing extra: no re-lambdifying, no re-deriving.
"""
import math
import sys
import time

import sympy as sp

from adjustable_conical_delta_kinematics import (
    ARM_COUNT,
    DEFAULT_MODEL_FILE,
    load_model,
    reachable_rod_length_bounds,
    resolve_rod_lengths,
    validate_rail_limits,
)


class AdjustableConicalDeltaKinematicsRuntime:
    """Loads a pickled adjustable-rod conical-delta kinematics model once
    and exposes cheap, repeated-call inverse/forward kinematics plus the
    per-tower rod-length redundancy resolver.
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
        rod_length_syms = model["symbols"]["rod_lengths"]
        self._ik_fns = [
            sp.lambdify((x, y, z, rod_length_syms[i]), expr, "math")
            for i, expr in enumerate(model["ik_exprs"])
        ]
        # cse=True: same reasoning as ConicalDeltaKinematicsRuntime -- the
        # tilted-rail trilateration doesn't cancel cleanly, so this is
        # ~50x slower per call without common subexpression elimination.
        fk_args = model["symbols"]["rail_s"] + rod_length_syms
        self._fk_fn = sp.lambdify(fk_args, model["fk_exprs"], "math", cse=True)

    def inverse_kinematics(self, point, rod_lengths, *, validate=None):
        """Return [s0, s1, s2] (mm) for the given (x, y, z) effector point,
        using each tower's own entry in `rod_lengths` (length ARM_COUNT).
        """
        px, py, pz = point
        try:
            rail_s = [fn(px, py, pz, rl) for fn, rl in zip(self._ik_fns, rod_lengths)]
        except ValueError as exc:
            raise ValueError(f"point unreachable ({exc})") from exc

        if validate if validate is not None else self._validate_default:
            validate_rail_limits(self._model, point, rail_s)
        return rail_s

    def forward_kinematics(self, rail_s_values, rod_lengths):
        """Return the effector (x, y, z) for the given carriage rail
        positions and per-tower rod lengths.
        """
        return self._fk_fn(*rail_s_values, *rod_lengths)

    def reachable_rod_length_bounds(self, point, rod_length_min, rod_length_max):
        """See adjustable_conical_delta_kinematics.reachable_rod_length_bounds.
        Returns a list of ARM_COUNT (lo, hi)-or-None entries, one per tower.
        """
        return reachable_rod_length_bounds(self._model, point, rod_length_min, rod_length_max)

    def resolve_rod_lengths(self, point, current_rod_lengths, rod_length_min, rod_length_max):
        """See adjustable_conical_delta_kinematics.resolve_rod_lengths."""
        return resolve_rod_lengths(self._model, point, current_rod_lengths, rod_length_min, rod_length_max)

    def carriage_positions(self, point, rod_lengths, *, validate=None):
        """Same per-tower report shape as compute_carriage_positions:
        {"tower", "carriage_rail_s_mm", "carriage_joint", "effector_joint"}.
        """
        rail_s = self.inverse_kinematics(point, rod_lengths, validate=validate)
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
    rod_lengths = [runtime._model["rod_length_range_mm"]["default_mm"]] * ARM_COUNT
    rail_s = runtime.inverse_kinematics(point, rod_lengths)  # warmup

    t0 = time.perf_counter()
    for _ in range(n):
        runtime.inverse_kinematics(point, rod_lengths)
    t1 = time.perf_counter()
    print(f"inverse_kinematics (validate=True):  {n/(t1-t0):>12,.0f} calls/sec")

    t0 = time.perf_counter()
    for _ in range(n):
        runtime.inverse_kinematics(point, rod_lengths, validate=False)
    t1 = time.perf_counter()
    print(f"inverse_kinematics (validate=False): {n/(t1-t0):>12,.0f} calls/sec")

    t0 = time.perf_counter()
    for _ in range(n):
        runtime.forward_kinematics(rail_s, rod_lengths)
    t1 = time.perf_counter()
    print(f"forward_kinematics:                  {n/(t1-t0):>12,.0f} calls/sec")

    t0 = time.perf_counter()
    for _ in range(n):
        runtime.carriage_positions(point, rod_lengths)
    t1 = time.perf_counter()
    print(f"carriage_positions:                  {n/(t1-t0):>12,.0f} calls/sec")

    rod_min = runtime._model["rod_length_range_mm"]["min_mm"]
    rod_max = runtime._model["rod_length_range_mm"]["max_mm"]
    t0 = time.perf_counter()
    for _ in range(n):
        runtime.resolve_rod_lengths(point, rod_lengths, rod_min, rod_max)
    t1 = time.perf_counter()
    print(f"resolve_rod_lengths:                  {n/(t1-t0):>12,.0f} calls/sec")


def main(argv):
    pickle_path = argv[1] if len(argv) > 1 else DEFAULT_MODEL_FILE

    t0 = time.perf_counter()
    runtime = AdjustableConicalDeltaKinematicsRuntime(pickle_path)
    t1 = time.perf_counter()
    print(f"loaded + lambdified {pickle_path} in {(t1-t0)*1e3:.2f} ms\n")

    _benchmark(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
