#!/usr/bin/env python3
"""Symbolic kinematic model of a conical ("teepee") 3-tower linear delta robot.

Unlike linear_delta_kinematics.py's vertical-rail Kossel K280 model, each of
this robot's 3 towers is a straight rail that leans INWARD: it runs from a
fixed anchor point on a base circle up to a single APEX point shared by all
three towers on the central axis, like the ribs of a teepee/cone. A carriage
slides along that tilted rail; a rigid diagonal rod (fixed length, no
intermediate elbow) connects each carriage directly to the moving effector
platform, exactly as in the linear model.

Because the rail is tilted, the carriage's actuator coordinate is NOT a
height -- it's a scalar distance traveled along the rail from its base
anchor, called `rail_s` here (bounded in [0, rail_length], where
rail_length is the fixed base-anchor-to-apex distance, identical for all
three towers by symmetry). Inverse kinematics solves for each tower's
`rail_s`, not an angle or a height.

This model is a strict generalization of the linear model's math: setting
each tower's rail direction to (0, 0, 1) and its base anchor to
(delta_radius*cos(phi), delta_radius*sin(phi), 0) recovers the linear
model's IK/FK formulas exactly.

Only the rigid-link geometry is modeled: belts, motors, gearboxes, and any
transmission ratio between a motor shaft and carriage position are out of
scope.

Units: millimeters. Coordinate convention: origin at the base-circle
center (z=0 plane); +Z points up toward the shared apex.

This is a synthetic/custom geometry (not derived from a real machine's
firmware config, unlike the linear model's K280 provenance): 3 towers,
120 degrees apart, 1.2 m tall, 0.4 m base radius, 0.45 m rods.

Usage:
    Derive the forward/inverse kinematics with sympy and pickle them:
        python3 conical_delta_kinematics.py

    Load a pickled model and solve inverse kinematics for a target
    effector position, printing the resulting carriage positions:
        python3 conical_delta_kinematics.py conical_delta_kinematics.pkl X Y Z
"""
import math
import pickle
import sys

import sympy as sp

GEOMETRY_MM = {
    "base_radius_mm": 400.0,    # radius of each tower's rail-base anchor circle, at z=0
    "tower_height_mm": 1200.0,  # z of the shared apex above the base plane
    "rod_length_mm": 450.0,     # fixed length of each diagonal rod
}

# Advisory only -- NOT a gating check (unlike the linear model's
# axis_limits, which does gate before solving IK). For a tilted rail
# there is no equally simple closed-form pre-check on (x, y, z) alone, so
# the real gate is validate_rail_limits, applied AFTER computing candidate
# rail_s values. This dict exists purely to seed GUI slider ranges / plot
# bounds with numbers roughly matching the actual reachable envelope.
WORKSPACE_HINT_MM = {
    "z_min_mm": 0.0,
    "z_max_mm": 206.0,              # ~ sqrt(rod_length^2 - base_radius^2): max on-axis z with all carriages at rail base
    "printable_radius_mm": 120.0,   # rough horizontal reach at z=0 before a tower's own disc goes negative
}

ARM_COUNT = 3
DEFAULT_MODEL_FILE = "conical_delta_kinematics.pkl"


def _tower_angles():
    return [sp.Rational(2 * i, ARM_COUNT) * sp.pi for i in range(ARM_COUNT)]


def _tower_geometry(base_radius, tower_height, phis):
    """Return (bases, apex, dirs, rail_length) for the conical rail
    geometry: `bases[i]` is tower i's fixed rail-anchor point at z=0;
    `apex` is the single point shared by all three rails at
    z=tower_height; `dirs[i]` is the fixed unit vector from bases[i]
    toward apex; `rail_length` is the common rail length (identical for
    all towers by symmetry, since every base point is the same distance
    from the on-axis apex).

    `base_radius`/`tower_height` are plain sp.Float values (see
    build_kinematics_model for why), so `rail_length` and each `dirs[i]`
    collapse to plain floats too -- no exact irrational cone-geometry
    term is carried through the derivation, only the exact sqrt(3) that
    already comes from the 120-degree tower angles via cos/sin.
    """
    apex = sp.Matrix([0, 0, tower_height])
    bases = [sp.Matrix([base_radius * sp.cos(phi), base_radius * sp.sin(phi), 0]) for phi in phis]
    rail_length = sp.sqrt(base_radius**2 + tower_height**2)
    dirs = [(apex - b) / rail_length for b in bases]
    return bases, apex, dirs, rail_length


def _forward_kinematics_exprs(bases, dirs, rail_s_symbols, rod_length):
    """Closed-form effector position (x, y, z) as a function of the three
    carriages' rail_s values, derived by trilateration: each carriage's
    position `base_i + s_i*dir_i` is a fully known-once-s_i point, so its
    distance to the effector defines a sphere of radius `rod_length`
    around that point; subtracting sphere equations pairwise eliminates
    the quadratic term and leaves two linear equations in (x, y, z),
    which combined with one original sphere equation give a quadratic in
    z. The "-sqrt" branch is the effector-below-carriages solution
    (verified numerically) -- identical reasoning to the linear model's
    _forward_kinematics_exprs, just with a general per-tower center
    `base_i + s_i*dir_i` in place of `(delta_radius*cos(phi),
    delta_radius*sin(phi), h)`.
    """
    x, y, z = sp.symbols("x y z", real=True)
    Q = sp.Matrix([x, y, z])

    centers = [b + s * d for b, s, d in zip(bases, rail_s_symbols, dirs)]

    eqs = []
    for C in centers:
        D = sp.expand(-C)
        eqs.append(sp.expand(Q.dot(Q) + 2 * Q.dot(D) + D.dot(D) - rod_length**2))

    lin_eqs = [sp.expand(eqs[0] - eqs[i]) for i in range(1, len(eqs))]
    sol_xy = sp.solve(lin_eqs, [x, y], dict=True)[0]

    quad_in_z = sp.expand(eqs[0].subs(sol_xy))
    poly_z = sp.Poly(quad_in_z, z)
    a_z, b_z, c_z = poly_z.all_coeffs()
    disc_z = b_z**2 - 4 * a_z * c_z
    z_expr = (-b_z - sp.sqrt(disc_z)) / (2 * a_z)
    x_expr = sol_xy[x].subs(z, z_expr)
    y_expr = sol_xy[y].subs(z, z_expr)
    return x_expr, y_expr, z_expr


def _visualization_metadata(geometry, workspace_hint, phis, bases, apex, dirs, rail_length):
    """Descriptive metadata for a GUI that only unpickles the model dict
    (no access to this module's docstring/source). Purely descriptive --
    no new physical constants, only labels and a convenience bundling of
    values already present elsewhere in the model.
    """
    return {
        "units": "mm",
        "coordinate_convention": (
            "origin at base-circle center (z=0 plane); +Z points up toward "
            "the shared apex; carriages ride straight rails from each "
            "tower's base anchor to the apex, not vertically"
        ),
        "source_machine": "custom conical/teepee linear delta (synthetic geometry, not a real machine)",
        "effector_representation": "point",
        "geometry_shape": "conical",
        "apex_mm": tuple(float(c) for c in apex),
        "rail_length_mm": float(rail_length),
        "tower_height_mm": geometry["tower_height_mm"],
        "base_radius_mm": geometry["base_radius_mm"],
        "towers": [
            {
                "index": i,
                "azimuth_deg": math.degrees(phi),
                "base_position_mm": tuple(float(c) for c in base),
                "dir": tuple(float(c) for c in d),
            }
            for i, (phi, base, d) in enumerate(zip(phis, bases, dirs))
        ],
    }


def build_kinematics_model(geometry=GEOMETRY_MM, workspace_hint=WORKSPACE_HINT_MM, verbose=True):
    """Derive forward- and inverse-kinematics expressions with sympy.

    Returns a dict of picklable sympy expressions (plus plain-Python
    geometry/layout/hint data) describing a conical delta robot with
    `ARM_COUNT` towers spaced evenly around the base, each leaning inward
    to a shared apex.
    """
    def log(msg):
        if verbose:
            print(msg)

    log("Building conical delta kinematics model")
    log(f"  geometry: base_radius={geometry['base_radius_mm']} mm, "
        f"tower_height={geometry['tower_height_mm']} mm, "
        f"rod_length={geometry['rod_length_mm']} mm")

    # Deliberately NOT sp.nsimplify'd (unlike the linear model's
    # delta_radius/rod_length): the cone geometry introduces its own
    # irrational term (rail_length = sqrt(base_radius^2+tower_height^2)),
    # which combined with the exact sqrt(3) from the 120-degree tower
    # angles produces unsimplifiable cross terms under sp.expand()/solve()
    # and makes the derivation extremely slow. Keeping these as plain
    # sp.Float lets sympy auto-collapse each tower's numeric geometry
    # (base position, direction vector, rail_length) to a plain float as
    # soon as it's combined with the exact trig values, leaving only x,
    # y, z, s0, s1, s2 as symbols and one final symbolic sqrt per
    # expression (the rod-length constraint) -- the same "one shared
    # sqrt at the end" shape the linear model already relies on.
    base_radius = sp.Float(geometry["base_radius_mm"])
    tower_height = sp.Float(geometry["tower_height_mm"])
    rod_length = sp.Float(geometry["rod_length_mm"])

    x, y, z = sp.symbols("x y z", real=True)
    rail_s = sp.symbols(f"s0:{ARM_COUNT}", real=True)
    phis = _tower_angles()

    bases, apex, dirs, rail_length = _tower_geometry(base_radius, tower_height, phis)

    log(f"  deriving inverse kinematics for {ARM_COUNT} towers "
        f"(solving the rod-length constraint for carriage rail position)...")
    ik_exprs = []
    for i, (base, d) in enumerate(zip(bases, dirs)):
        Q = sp.Matrix([x, y, z])
        R = Q - base
        d_expr = d.dot(R)
        perp_sq = R.dot(R) - d_expr**2
        disc = rod_length**2 - perp_sq
        ik_exprs.append(sp.expand(d_expr) + sp.sqrt(sp.expand(disc)))
        log(f"    tower {i}: azimuth={math.degrees(float(phis[i])):6.1f} deg, "
            f"base=({float(base[0]):.3f}, {float(base[1]):.3f}, {float(base[2]):.3f}) mm")

    log("  deriving forward kinematics (trilateration of the 3 carriage spheres)...")
    fk_exprs = _forward_kinematics_exprs(bases, dirs, rail_s, rod_length)
    phis_rad = [float(p) for p in phis]

    tower_bases_mm = [tuple(float(c) for c in base) for base in bases]
    tower_dirs = [tuple(float(c) for c in d) for d in dirs]
    rail_length_mm = float(rail_length)

    log("  kinematics derivation complete")

    return {
        "geometry": dict(geometry),
        "workspace_hint": dict(workspace_hint),
        "arm_count": ARM_COUNT,
        "phis": phis_rad,
        "symbols": {"x": x, "y": y, "z": z, "rail_s": rail_s},
        "ik_exprs": ik_exprs,                    # s_i(x, y, z), one per tower
        "tower_bases_mm": tower_bases_mm,        # fixed (x, y, z) rail-base anchor of each tower
        "tower_dirs": tower_dirs,                # fixed unit vector from base anchor toward apex
        "rail_length_mm": rail_length_mm,        # common rail length (base anchor to apex)
        "fk_exprs": fk_exprs,                    # (x, y, z)(s0, s1, s2)
        "schema_version": 1,
        "visualization": _visualization_metadata(
            geometry, workspace_hint, phis_rad, tower_bases_mm, tuple(float(c) for c in apex),
            tower_dirs, rail_length_mm,
        ),
    }


def save_model(model, filename):
    with open(filename, "wb") as fh:
        pickle.dump(model, fh)


def load_model(filename):
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def validate_rail_limits(model, point, rail_s_values):
    """Raise ValueError if any tower's computed rail_s falls outside
    [0, rail_length] -- i.e. the carriage would have to sit below its
    base anchor or above the shared apex, which is not physically
    realizable. Unlike the linear model's validate_axis_limits (which
    gates on (x, y, z) BEFORE solving IK), this is called AFTER computing
    candidate rail_s values, because there is no simple closed-form
    pre-check on (x, y, z) alone for a tilted-rail geometry.
    """
    rail_length = model["rail_length_mm"]
    px, py, pz = point

    for i, s in enumerate(rail_s_values):
        if not (0.0 <= s <= rail_length):
            raise ValueError(
                f"tower {i}: point {(px, py, pz)} needs rail_s={s:.3f} mm, "
                f"outside [0, {rail_length:.3f}] mm (carriage would sit "
                f"below its base anchor or above the shared apex)"
            )


def compute_inverse_kinematics(model, point):
    """Return the carriage rail position (mm, distance from base anchor)
    for each tower that reaches the given effector position, or raise
    ValueError if the point is outside the robot's geometric workspace
    (too far from a rail line for the rod to bridge, or requiring a
    carriage position beyond the physical rail's [0, rail_length] span).
    """
    x, y, z = model["symbols"]["x"], model["symbols"]["y"], model["symbols"]["z"]
    px, py, pz = point

    rail_s_values = []
    for i, ik_expr in enumerate(model["ik_exprs"]):
        ik_fn = sp.lambdify((x, y, z), ik_expr, "math")
        try:
            s = ik_fn(px, py, pz)
        except ValueError as exc:
            raise ValueError(f"tower {i}: point unreachable ({exc})") from exc
        rail_s_values.append(s)

    validate_rail_limits(model, point, rail_s_values)
    return rail_s_values


def compute_forward_kinematics(model, rail_s_values):
    """Return the effector (x, y, z) for the given carriage rail positions.

    cse=True matters here: unlike the linear model's compact FK
    expressions, this model's trilateration solve produces x/y
    expressions with thousands of ops (the tilted-rail centers don't
    cancel as cleanly as the vertical-rail case), so common-subexpression
    elimination is needed to keep per-call evaluation fast.
    """
    fk_fn = sp.lambdify(model["symbols"]["rail_s"], model["fk_exprs"], "math", cse=True)
    return fk_fn(*rail_s_values)


def compute_carriage_positions(model, point):
    """Compute inverse kinematics for `point` and return, per tower, the
    carriage rail position plus the 3D position of the carriage joint and
    the effector joint (the target point itself).
    """
    rail_s_values = compute_inverse_kinematics(model, point)
    px, py, pz = point

    towers = []
    for i, s in enumerate(rail_s_values):
        bx, by, bz = model["tower_bases_mm"][i]
        dx, dy, dz = model["tower_dirs"][i]
        towers.append({
            "tower": i,
            "carriage_rail_s_mm": s,
            "carriage_joint": (bx + s * dx, by + s * dy, bz + s * dz),
            "effector_joint": (px, py, pz),
        })
    return towers


# Sample points chosen well within the small reachable envelope (rod_length
# 450mm is short relative to the 1200mm tower height / 400mm base radius:
# max on-axis z with all carriages at rail base is
# sqrt(450^2-400^2) ~= 206 mm) and clear of the apex singularity (as all
# rail_s -> rail_length, the 3 carriage centers converge toward the shared
# apex and trilateration degrades numerically) -- deliberately NOT scaled
# to the linear model's ~555mm-tall K280 envelope.
_VALIDATION_SAMPLE_POINTS = [
    (0.0, 0.0, 0.0),
    (30.0, -20.0, 80.0),
    (0.0, 0.0, 150.0),
    (60.0, 30.0, 20.0),
    (-40.0, 40.0, 50.0),
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
    rail_length = model["rail_length_mm"]
    failures = []

    log("\nRunning validation tests")

    log("  [1] FK(IK(point)) round trip")
    for pt in _VALIDATION_SAMPLE_POINTS:
        rail_s = compute_inverse_kinematics(model, pt)
        back = compute_forward_kinematics(model, rail_s)
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

    log("  [3] symmetric case: (0, 0, z) gives equal rail_s across all towers")
    rail_s = compute_inverse_kinematics(model, (0.0, 0.0, 100.0))
    ok = max(rail_s) - min(rail_s) < tol
    log(f"      {'PASS' if ok else 'FAIL'}  rail_s={rail_s}")
    if not ok:
        failures.append(f"symmetric case failed: rail_s {rail_s}")

    log("  [4] rejection paths (domain error, then rail-bounds error)")
    rejection_cases = [
        ("too far from every rail line for the rod to bridge", (500.0, 500.0, 500.0)),
        ("rail_s would exceed rail_length (too high on-axis)", (0.0, 0.0, 1000.0)),
        ("rail_s would go negative (behind the base anchors)", (0.0, 0.0, -215.0)),
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
    log(f"\nAll validation tests passed ({total} checks). rail_length={rail_length:.3f} mm")


def _print_report(point, towers):
    px, py, pz = point
    print(f"Target effector position: ({px:.6f}, {py:.6f}, {pz:.6f}) mm\n")
    for tower in towers:
        cx, cy, cz = tower["carriage_joint"]
        ex, ey, ez = tower["effector_joint"]
        print(f"Tower {tower['tower']}:")
        print(f"  carriage rail position : {tower['carriage_rail_s_mm']:.4f} mm")
        print(f"  carriage joint         : ({cx:.6f}, {cy:.6f}, {cz:.6f}) mm")
        print(f"  effector joint         : ({ex:.6f}, {ey:.6f}, {ez:.6f}) mm")
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
