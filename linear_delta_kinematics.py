#!/usr/bin/env python3
"""Symbolic kinematic model of a linear (Kossel-style) 3-tower delta robot.

Each of the 3 towers has a carriage that slides vertically along a
fixed-radius rail; a single rigid diagonal rod (fixed length, no
intermediate elbow) connects each carriage directly to the moving
effector platform. Inverse kinematics solves for each tower's carriage
HEIGHT, not an angle -- unlike a rotary delta.

Only the rigid-link geometry is modeled: belts, motors, gearboxes, and
any transmission ratio between a motor shaft and carriage position are
out of scope.

Units: millimeters. Coordinate convention: origin at the print-bed
center, +Z points up toward the tower tops (matches Marlin's
Z_MIN_POS=0-at-the-bed convention) -- carriage height and target Z
share this frame.

Default geometry and axis limits mirror the Kossel K280 (HE3D), taken
from its published Marlin firmware Configuration.h (DELTA_RADIUS,
DELTA_DIAGONAL_ROD, DELTA_PRINTABLE_RADIUS, DELTA_HEIGHT/Z_MAX_POS,
Z_MIN_POS). The K280 config models a single combined delta radius
rather than separate carriage/effector radii, so this module does the
same -- the "effector joint" reported per tower is simply the target
point itself, with no additional radial offset.

Usage:
    Derive the forward/inverse kinematics with sympy and pickle them:
        python3 linear_delta_kinematics.py

    Load a pickled model and solve inverse kinematics for a target
    effector position, printing the resulting carriage positions:
        python3 linear_delta_kinematics.py linear_delta_kinematics.pkl X Y Z
"""
import math
import pickle
import sys

import sympy as sp

GEOMETRY_MM = {
    "delta_radius_mm": 190.0,  # DELTA_RADIUS
    "rod_length_mm": 340.0,    # DELTA_DIAGONAL_ROD
}

AXIS_LIMITS_MM = {
    "z_min_mm": 0.0,                # Z_MIN_POS -- the bed
    "z_max_mm": 555.0,              # Z_MAX_POS / DELTA_HEIGHT
    "printable_radius_mm": 135.0,   # DELTA_PRINTABLE_RADIUS
}

ARM_COUNT = 3
DEFAULT_MODEL_FILE = "linear_delta_kinematics.pkl"

# Identifies the kinematic architecture this model implements: vertical
# towers with carriages that slide straight up/down, connected to the
# effector by fixed-length rods with no intermediate elbow (the
# Kossel/RepRap-style linear delta). Stored in the pickled model so
# downstream consumers (a runtime, a visualizer) can tell what they're
# looking at once other delta types (e.g. rotary, or other linear
# variants) exist alongside this one.
DELTA_TYPE = "linear_vertical_carriage"


def _tower_angles():
    return [sp.Rational(2 * i, ARM_COUNT) * sp.pi for i in range(ARM_COUNT)]


def _forward_kinematics_exprs(delta_radius, rod_length, phis, heights):
    """Closed-form effector position (x, y, z) as a function of the three
    carriage heights, derived by trilateration: each carriage-to-effector
    distance defines a sphere of radius `rod_length` around that
    carriage's (fully known) position; subtracting sphere equations
    pairwise eliminates the quadratic term and leaves two linear
    equations in (x, y, z), which combined with one original sphere
    equation give a quadratic in z. The "-sqrt" branch is the
    effector-below-carriages solution (verified numerically).
    """
    x, y, z = sp.symbols("x y z", real=True)
    Q = sp.Matrix([x, y, z])

    eqs = []
    for phi, h in zip(phis, heights):
        C = sp.Matrix([delta_radius*sp.cos(phi), delta_radius*sp.sin(phi), h])
        D = sp.expand(-C)
        eqs.append(sp.expand(Q.dot(Q) + 2*Q.dot(D) + D.dot(D) - rod_length**2))

    lin_eqs = [sp.expand(eqs[0] - eqs[i]) for i in range(1, len(eqs))]
    sol_xy = sp.solve(lin_eqs, [x, y], dict=True)[0]

    quad_in_z = sp.expand(eqs[0].subs(sol_xy))
    poly_z = sp.Poly(quad_in_z, z)
    a_z, b_z, c_z = poly_z.all_coeffs()
    disc_z = b_z**2 - 4*a_z*c_z
    z_expr = (-b_z - sp.sqrt(disc_z)) / (2*a_z)
    x_expr = sol_xy[x].subs(z, z_expr)
    y_expr = sol_xy[y].subs(z, z_expr)
    return x_expr, y_expr, z_expr


def _visualization_metadata(geometry, axis_limits, phis, tower_positions):
    """Descriptive metadata for a GUI that only unpickles the model dict
    (no access to this module's docstring/source). Purely descriptive --
    no new physical constants, only labels and a convenience bundling of
    `phis`/`tower_positions` already present elsewhere in the model.
    """
    return {
        "units": "mm",
        "coordinate_convention": (
            "origin at print-bed center; +Z points up toward tower tops; "
            "matches Marlin Z_MIN_POS=0-at-bed convention"
        ),
        "source_machine": "Kossel K280 (HE3D) Marlin Configuration.h",
        "effector_representation": "point",  # no separate effector radius in this model
        "towers": [
            {
                "index": i,
                "azimuth_deg": math.degrees(phi),
                "position_mm": pos,
            }
            for i, (phi, pos) in enumerate(zip(phis, tower_positions))
        ],
    }


def build_kinematics_model(geometry=GEOMETRY_MM, axis_limits=AXIS_LIMITS_MM, verbose=True):
    """Derive forward- and inverse-kinematics expressions with sympy.

    Returns a dict of picklable sympy expressions (plus plain-Python
    geometry/layout/limit data) describing a linear delta robot with
    `ARM_COUNT` towers spaced evenly around the base.
    """
    def log(msg):
        if verbose:
            print(msg)

    log("Building linear delta kinematics model")
    log(f"  delta type: {DELTA_TYPE}")
    log(f"  geometry: delta_radius={geometry['delta_radius_mm']} mm, "
        f"rod_length={geometry['rod_length_mm']} mm")
    log(f"  axis limits: z in [{axis_limits['z_min_mm']}, {axis_limits['z_max_mm']}] mm, "
        f"printable_radius={axis_limits['printable_radius_mm']} mm")

    delta_radius = sp.nsimplify(geometry["delta_radius_mm"])
    rod_length = sp.nsimplify(geometry["rod_length_mm"])

    x, y, z = sp.symbols("x y z", real=True)
    heights = sp.symbols(f"h0:{ARM_COUNT}", real=True)
    phis = _tower_angles()

    log(f"  deriving inverse kinematics for {ARM_COUNT} towers "
        f"(solving the rod-length constraint for carriage height)...")
    ik_exprs = []
    tower_positions = []
    for i, (phi, h) in enumerate(zip(phis, heights)):
        dx = delta_radius*sp.cos(phi) - x
        dy = delta_radius*sp.sin(phi) - y
        disc = rod_length**2 - dx**2 - dy**2
        ik_exprs.append(z + sp.sqrt(disc))  # carriage sits above the effector
        pos = (float(delta_radius*sp.cos(phi)), float(delta_radius*sp.sin(phi)))
        tower_positions.append(pos)
        log(f"    tower {i}: azimuth={math.degrees(float(phi)):6.1f} deg, "
            f"position=({pos[0]:.3f}, {pos[1]:.3f}) mm")

    log("  deriving forward kinematics (trilateration of the 3 carriage spheres)...")
    fk_exprs = _forward_kinematics_exprs(delta_radius, rod_length, phis, heights)
    phis_rad = [float(p) for p in phis]

    log("  kinematics derivation complete")

    return {
        "schema_version": 1,
        "delta_type": DELTA_TYPE,
        "geometry": dict(geometry),
        "axis_limits": dict(axis_limits),
        "arm_count": ARM_COUNT,
        "phis": phis_rad,
        "symbols": {"x": x, "y": y, "z": z, "heights": heights},
        "ik_exprs": ik_exprs,                # h_i(x, y, z), one per tower
        "tower_positions": tower_positions,  # fixed (x, y) of each tower's rail
        "fk_exprs": fk_exprs,                # (x, y, z)(h0, h1, h2)
        "visualization": _visualization_metadata(geometry, axis_limits, phis_rad, tower_positions),
    }


def save_model(model, filename):
    with open(filename, "wb") as fh:
        pickle.dump(model, fh)


def load_model(filename):
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def validate_axis_limits(model, point):
    """Raise ValueError if `point` falls outside the robot's documented
    Z travel and printable-radius limits.
    """
    limits = model["axis_limits"]
    px, py, pz = point

    if not (limits["z_min_mm"] <= pz <= limits["z_max_mm"]):
        raise ValueError(
            f"z={pz} mm is outside axis limits "
            f"[{limits['z_min_mm']}, {limits['z_max_mm']}] mm"
        )

    horiz = math.hypot(px, py)
    if horiz > limits["printable_radius_mm"]:
        raise ValueError(
            f"horizontal distance {horiz:.3f} mm exceeds printable radius "
            f"{limits['printable_radius_mm']} mm"
        )


def compute_max_carriage_heights(model):
    """Return the maximum reachable carriage height (mm) for each tower,
    given the model's rod length and axis limits.

    From ik_exprs, h_i = z + sqrt(rod_length**2 - dx**2 - dy**2), where
    (dx, dy) is the horizontal offset between the effector and tower i's
    own (x, y) rail position. For fixed offset this is increasing in z,
    and for fixed z it's decreasing in the offset, so it's maximized by
    driving z to z_max and the offset to the smallest value reachable
    within the printable radius: zero if the tower's own position already
    lies within that radius, otherwise the distance from the tower to the
    printable-radius circle's edge.
    """
    axis_limits = model["axis_limits"]
    rod_length = model["geometry"]["rod_length_mm"]
    printable_radius = axis_limits["printable_radius_mm"]
    z_max = axis_limits["z_max_mm"]

    heights = []
    for tx, ty in model["tower_positions"]:
        tower_radius = math.hypot(tx, ty)
        min_offset = max(0.0, tower_radius - printable_radius)
        heights.append(z_max + math.sqrt(max(rod_length**2 - min_offset**2, 0.0)))
    return heights


def compute_inverse_kinematics(model, point):
    """Return the carriage height (mm) for each tower that reaches the
    given effector position, or raise ValueError if the point is outside
    the robot's documented axis limits or geometric workspace.
    """
    validate_axis_limits(model, point)

    x, y, z = model["symbols"]["x"], model["symbols"]["y"], model["symbols"]["z"]
    px, py, pz = point

    heights = []
    for i, ik_expr in enumerate(model["ik_exprs"]):
        ik_fn = sp.lambdify((x, y, z), ik_expr, "math")
        try:
            h = ik_fn(px, py, pz)
        except ValueError as exc:
            raise ValueError(f"tower {i}: point unreachable ({exc})") from exc
        heights.append(h)
    return heights


def compute_forward_kinematics(model, heights):
    """Return the effector (x, y, z) for the given carriage heights."""
    fk_fn = sp.lambdify(model["symbols"]["heights"], model["fk_exprs"], "math")
    return fk_fn(*heights)


def compute_carriage_positions(model, point):
    """Compute inverse kinematics for `point` and return, per tower, the
    carriage height plus the 3D position of the carriage joint and the
    effector joint (the target point itself -- this model uses a single
    combined `delta_radius_mm`, per the K280's own firmware convention,
    so there is no separate effector-side radial offset).
    """
    heights = compute_inverse_kinematics(model, point)
    px, py, pz = point

    towers = []
    for i, h in enumerate(heights):
        tx, ty = model["tower_positions"][i]
        towers.append({
            "tower": i,
            "carriage_height_mm": h,
            "carriage_joint": (tx, ty, h),
            "effector_joint": (px, py, pz),
        })
    return towers


_VALIDATION_SAMPLE_POINTS = [
    (0.0, 0.0, 0.0),
    (40.0, -20.0, 300.0),
    (0.0, 0.0, 554.9),
    (100.0, 50.0, 10.0),
    (-80.0, 80.0, 500.0),
]


def _run_validation_tests(model, verbose=True, tol=1e-6):
    """Run a battery of correctness checks against a freshly built model,
    printing each individual test's result. Raises AssertionError (after
    printing every test, not just the first failure) if anything fails.
    """
    def log(msg):
        if verbose:
            print(msg)

    rod_length = model["geometry"]["rod_length_mm"]
    limits = model["axis_limits"]
    failures = []

    log("\nRunning validation tests")

    log("  [1] FK(IK(point)) round trip")
    for pt in _VALIDATION_SAMPLE_POINTS:
        heights = compute_inverse_kinematics(model, pt)
        back = compute_forward_kinematics(model, heights)
        err = math.dist(pt, back)
        ok = err < tol
        log(f"      {'PASS' if ok else 'FAIL'}  point={pt}  round-trip error={err:.3e} mm")
        if not ok:
            failures.append(f"round trip failed for {pt}: error {err:.3e} mm")

    log("  [2] rod-length preservation (carriage-to-effector distance == rod_length)")
    for pt in _VALIDATION_SAMPLE_POINTS:
        for tower in compute_carriage_positions(model, pt):
            d = math.dist(tower["carriage_joint"], tower["effector_joint"])
            ok = abs(d - rod_length) < tol
            log(f"      {'PASS' if ok else 'FAIL'}  point={pt} tower={tower['tower']}  "
                f"rod length={d:.6f} mm (expected {rod_length} mm)")
            if not ok:
                failures.append(f"rod length mismatch for {pt} tower {tower['tower']}: {d} mm")

    log("  [3] symmetric case: (0, 0, z) gives equal carriage heights across all towers")
    heights = compute_inverse_kinematics(model, (0.0, 0.0, 300.0))
    ok = max(heights) - min(heights) < tol
    log(f"      {'PASS' if ok else 'FAIL'}  heights={heights}")
    if not ok:
        failures.append(f"symmetric case failed: heights {heights}")

    log("  [4] axis-limit rejection paths")
    rejection_cases = [
        ("z below z_min", (0.0, 0.0, limits["z_min_mm"] - 5.0)),
        ("z above z_max", (0.0, 0.0, limits["z_max_mm"] + 5.0)),
        ("xy outside printable radius", (limits["printable_radius_mm"] + 15.0, 0.0, 300.0)),
    ]
    for label, pt in rejection_cases:
        try:
            compute_inverse_kinematics(model, pt)
            log(f"      FAIL  {label}: point={pt} (no error raised)")
            failures.append(f"expected rejection for {label} at {pt}, none raised")
        except ValueError as exc:
            log(f"      PASS  {label}: point={pt} -> {exc}")

    if failures:
        raise AssertionError("validation failed:\n" + "\n".join(f"  - {f}" for f in failures))

    total = len(_VALIDATION_SAMPLE_POINTS) * (1 + ARM_COUNT) + 1 + len(rejection_cases)
    log(f"\nAll validation tests passed ({total} checks).")


def _print_report(point, towers):
    px, py, pz = point
    print(f"Target effector position: ({px:.6f}, {py:.6f}, {pz:.6f}) mm\n")
    for tower in towers:
        cx, cy, cz = tower["carriage_joint"]
        ex, ey, ez = tower["effector_joint"]
        print(f"Tower {tower['tower']}:")
        print(f"  carriage height : {tower['carriage_height_mm']:.4f} mm")
        print(f"  carriage joint  : ({cx:.6f}, {cy:.6f}, {cz:.6f}) mm")
        print(f"  effector joint  : ({ex:.6f}, {ey:.6f}, {ez:.6f}) mm")
        print()


def main(argv):
    if len(argv) == 1:
        model = build_kinematics_model()
        _run_validation_tests(model)
        print(f"\nSaving kinematics model to {DEFAULT_MODEL_FILE} ...")
        save_model(model, DEFAULT_MODEL_FILE)
        print("Saved.")
        return 0

    if len(argv) != 5:
        print(f"usage: {argv[0]} [kinematics.pkl x y z]", file=sys.stderr)
        return 2

    model_file = argv[1]
    try:
        point = tuple(float(v) for v in argv[2:5])
    except ValueError:
        print("x, y, z must be numbers", file=sys.stderr)
        return 2

    print(f"Loading kinematics model from {model_file} ...")
    model = load_model(model_file)
    print(f"Computing inverse kinematics for target {point} ...\n")
    try:
        towers = compute_carriage_positions(model, point)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_report(point, towers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
