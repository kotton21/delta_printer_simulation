#!/usr/bin/env python3
"""Symbolic kinematic model of a conical ("teepee") 3-tower linear delta
robot whose rod (arm) length is an INDEPENDENT runtime-adjustable parameter
per tower, instead of one fixed constant shared by all three and baked into
the model.

Geometry is identical to conical_delta_kinematics.py: 3 towers, evenly
spaced, each a straight rail leaning inward from a fixed base anchor to a
single shared apex; a carriage slides along the rail (actuator coordinate
`rail_s`, distance from the base anchor); a rigid rod connects the carriage
to the effector.

The one difference: each tower's rod length is kept as its OWN free sympy
symbol (`rod_length0`, `rod_length1`, `rod_length2`) all the way through
derivation, rather than substituted with a fixed numeric value (or sharing
one symbol across all three towers). Both the inverse-kinematics expressions
(tower i is a function of x, y, z, rod_length_i -- only its own rod length,
never the other towers') and the forward-kinematics expressions (function of
s0, s1, s2, rod_length0, rod_length1, rod_length2) accept rod length as an
ordinary lambdified argument, so a GUI can vary any subset of the three rod
lengths every frame at effectively zero cost -- no re-derivation, no
re-pickling.

Why this is a real (not fake) extra degree of freedom, and why it is
per-tower independent: for a fixed effector target (x, y, z), each tower's
IK equation `rail_s_i = d_i + sqrt(rod_length_i^2 - perp_i^2)` involves only
that tower's own rod_length_i (d_i, perp_i depend only on geometry and the
target, never on any rod length). So whether tower i's rail_s stays within
its valid [0, rail_length] span depends only on rod_length_i, completely
decoupled from what rod_length_j (j != i) is set to. That's the redundancy:
with 3 rails, a 3-DOF effector position is normally fully determined by
fixed geometry, but letting each rod length vary independently turns it into
a 6-input system with a 3-parameter family (one free parameter per tower) of
valid rail_s solutions per target.

Because of that redundancy, IK does not require touching any rod length in
ordinary operation -- pick one default rod length for all three towers once
(see ROD_LENGTH_RANGE_MM) and most targets are reachable without ever
changing it. Only when a target's required rail_s for a particular tower
would fall outside that tower's [0, rail_length] span at its current rod
length does that tower's rod length need to move, and
`reachable_rod_length_bounds` below computes -- per tower, in closed form,
no search -- the exact interval of that tower's rod length values (if any)
that make a given target reachable via that tower, so a caller can nudge
just the tower(s) that need it, by the minimum amount necessary, leaving the
others untouched.

Units: millimeters. Coordinate convention: origin at the base-circle
center (z=0 plane); +Z points up toward the shared apex.

Usage:
    Derive the forward/inverse kinematics with sympy and pickle them:
        python3 adjustable_conical_delta_kinematics.py

    Load a pickled model and solve inverse kinematics for a target
    effector position at given per-tower rod lengths, printing the
    resulting carriage positions:
        python3 adjustable_conical_delta_kinematics.py adjustable_conical_delta_kinematics.pkl X Y Z ROD0 ROD1 ROD2
"""
import math
import pickle
import sys

import sympy as sp

GEOMETRY_MM = {
    "base_radius_mm": 400.0,    # radius of each tower's rail-base anchor circle, at z=0
    "tower_height_mm": 1200.0,  # z of the shared apex above the base plane
}

# Unlike conical_delta_kinematics.py's single fixed rod_length_mm, this
# model treats each tower's rod length as an independent live parameter:
# "default_mm" seeds all three GUI sliders / the initial pose,
# "min_mm"/"max_mm" bound how far the redundancy resolver (and the user)
# may push any one of them. All three towers share the same allowed
# range, but each tower's current value is tracked separately.
ROD_LENGTH_RANGE_MM = {
    "min_mm": 200.0,
    "max_mm": 700.0,
    "default_mm": 450.0,
}

# Advisory only, evaluated at the default rod length -- see the matching
# comment in conical_delta_kinematics.py. Seeds GUI slider ranges / plot
# bounds, not a gating check.
WORKSPACE_HINT_MM = {
    "z_min_mm": 0.0,
    "z_max_mm": 206.0,              # ~ sqrt(default_rod_length^2 - base_radius^2)
    "printable_radius_mm": 120.0,
}

ARM_COUNT = 3
DEFAULT_MODEL_FILE = "adjustable_conical_delta_kinematics.pkl"


def _tower_angles():
    return [sp.Rational(2 * i, ARM_COUNT) * sp.pi for i in range(ARM_COUNT)]


def _tower_geometry(base_radius, tower_height, phis):
    """Return (bases, apex, dirs, rail_length); identical in shape to
    conical_delta_kinematics._tower_geometry -- rod length plays no part in
    the rail geometry itself, only in how far along a rail the rod's other
    end (the carriage) needs to sit.
    """
    apex = sp.Matrix([0, 0, tower_height])
    bases = [sp.Matrix([base_radius * sp.cos(phi), base_radius * sp.sin(phi), 0]) for phi in phis]
    rail_length = sp.sqrt(base_radius**2 + tower_height**2)
    dirs = [(apex - b) / rail_length for b in bases]
    return bases, apex, dirs, rail_length


def _forward_kinematics_exprs(bases, dirs, rail_s_symbols, rod_length_symbols):
    """Same trilateration approach as conical_delta_kinematics's version,
    with each tower's `rod_length_symbols[i]` a free symbol instead of one
    plain float shared by all three. Because the three rod lengths are now
    independent symbols, subtracting eqs pairwise no longer cancels the
    rod-length term (eqs[0] carries -rod_length0**2, eqs[i] carries
    -rod_length_i**2), so the linear x/y solve now carries a
    `rod_length_i**2 - rod_length0**2` constant term absent from the
    single-shared-rod-length model -- still linear in x, y, so `sp.solve`
    handles it the same way. The final substitution back into eqs[0] (to
    get z) reintroduces rod_length0 the same way as before.
    """
    x, y, z = sp.symbols("x y z", real=True)
    Q = sp.Matrix([x, y, z])

    centers = [b + s * d for b, s, d in zip(bases, rail_s_symbols, dirs)]

    eqs = []
    for C, rod_length in zip(centers, rod_length_symbols):
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


def _visualization_metadata(geometry, rod_length_range, workspace_hint, phis, bases, apex, dirs, rail_length):
    return {
        "units": "mm",
        "coordinate_convention": (
            "origin at base-circle center (z=0 plane); +Z points up toward "
            "the shared apex; carriages ride straight rails from each "
            "tower's base anchor to the apex, not vertically"
        ),
        "source_machine": "custom conical/teepee linear delta with independent per-arm adjustable rod length (synthetic geometry)",
        "effector_representation": "point",
        "geometry_shape": "conical",
        "apex_mm": tuple(float(c) for c in apex),
        "rail_length_mm": float(rail_length),
        "tower_height_mm": geometry["tower_height_mm"],
        "base_radius_mm": geometry["base_radius_mm"],
        "rod_length_range_mm": dict(rod_length_range),
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


def build_kinematics_model(geometry=GEOMETRY_MM, rod_length_range=ROD_LENGTH_RANGE_MM,
                            workspace_hint=WORKSPACE_HINT_MM, verbose=True):
    """Derive forward- and inverse-kinematics expressions with sympy,
    keeping each tower's rod length as its own free symbol (see module
    docstring).
    """
    def log(msg):
        if verbose:
            print(msg)

    log("Building adjustable-rod conical delta kinematics model")
    log(f"  geometry: base_radius={geometry['base_radius_mm']} mm, "
        f"tower_height={geometry['tower_height_mm']} mm")
    log(f"  rod_length range: [{rod_length_range['min_mm']}, {rod_length_range['max_mm']}] mm, "
        f"default={rod_length_range['default_mm']} mm (kept symbolic per tower, not baked in)")

    base_radius = sp.Float(geometry["base_radius_mm"])
    tower_height = sp.Float(geometry["tower_height_mm"])
    rod_lengths = sp.symbols(f"rod_length0:{ARM_COUNT}", real=True, positive=True)

    x, y, z = sp.symbols("x y z", real=True)
    rail_s = sp.symbols(f"s0:{ARM_COUNT}", real=True)
    phis = _tower_angles()

    bases, apex, dirs, rail_length = _tower_geometry(base_radius, tower_height, phis)

    log(f"  deriving inverse kinematics for {ARM_COUNT} towers "
        f"(each tower's rod_length left symbolic and independent)...")
    ik_exprs = []
    for i, (base, d, rod_length) in enumerate(zip(bases, dirs, rod_lengths)):
        Q = sp.Matrix([x, y, z])
        R = Q - base
        d_expr = d.dot(R)
        perp_sq = R.dot(R) - d_expr**2
        disc = rod_length**2 - sp.expand(perp_sq)
        ik_exprs.append(sp.expand(d_expr) + sp.sqrt(disc))
        log(f"    tower {i}: azimuth={math.degrees(float(phis[i])):6.1f} deg, "
            f"base=({float(base[0]):.3f}, {float(base[1]):.3f}, {float(base[2]):.3f}) mm")

    log("  deriving forward kinematics (trilateration of the 3 carriage spheres)...")
    fk_exprs = _forward_kinematics_exprs(bases, dirs, rail_s, rod_lengths)
    phis_rad = [float(p) for p in phis]

    tower_bases_mm = [tuple(float(c) for c in base) for base in bases]
    tower_dirs = [tuple(float(c) for c in d) for d in dirs]
    rail_length_mm = float(rail_length)

    log("  kinematics derivation complete")

    return {
        "geometry": dict(geometry),
        "rod_length_range_mm": dict(rod_length_range),
        "workspace_hint": dict(workspace_hint),
        "arm_count": ARM_COUNT,
        "phis": phis_rad,
        "symbols": {"x": x, "y": y, "z": z, "rail_s": rail_s, "rod_lengths": rod_lengths},
        "ik_exprs": ik_exprs,                    # s_i(x, y, z, rod_length_i), one per tower
        "tower_bases_mm": tower_bases_mm,        # fixed (x, y, z) rail-base anchor of each tower
        "tower_dirs": tower_dirs,                # fixed unit vector from base anchor toward apex
        "rail_length_mm": rail_length_mm,        # common rail length (base anchor to apex)
        "fk_exprs": fk_exprs,                    # (x, y, z)(s0, s1, s2, rod_length0, rod_length1, rod_length2)
        "schema_version": 1,
        "visualization": _visualization_metadata(
            geometry, rod_length_range, workspace_hint, phis_rad, tower_bases_mm,
            tuple(float(c) for c in apex), tower_dirs, rail_length_mm,
        ),
    }


def save_model(model, filename):
    with open(filename, "wb") as fh:
        pickle.dump(model, fh)


def load_model(filename):
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def validate_rail_limits(model, point, rail_s_values):
    """Same gate as conical_delta_kinematics.validate_rail_limits: raise
    ValueError if any tower's rail_s falls outside [0, rail_length].
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


def reachable_rod_length_bounds(model, point, rod_length_min, rod_length_max):
    """Return a list of ARM_COUNT entries, each either a closed interval
    (lo, hi) -- clipped to [rod_length_min, rod_length_max] -- of rod
    lengths for which THAT tower alone can reach `point` (i.e. its rail_s
    lands in [0, rail_length]), or None if no rod length in that range
    lets that tower reach it. Towers are independent: tower i's interval
    depends only on tower i's own geometry and rod length, never on any
    other tower's rod length (see module docstring), so there is no
    cross-tower intersection here, unlike a shared-rod-length model.

    This is a closed-form solve, not a search: for a fixed point, a
    tower's rail_s is a strictly increasing function of its own rod
    length (since `d_i` and `perp_i` below depend only on geometry + the
    point), so the set of rod lengths keeping that tower's rail_s in
    [0, rail_length] is itself a single interval [lo_i, hi_i]. Derivation:
    rail_s_i(rod_length_i) = d_i + sqrt(rod_length_i^2 - perp_i), so
    rail_s_i in [0, L]  <=>  sqrt(rod_length_i^2 - perp_i) in [-d_i, L - d_i].
    The upper bound gives hi_i = sqrt(perp_i + (L - d_i)^2) (or no solution
    at all if L < d_i, i.e. the point is beyond the apex along this tower's
    rail regardless of rod length). The lower bound is only binding when
    d_i < 0, giving lo_i = sqrt(perp_i + d_i^2) = |R_i| (distance from this
    tower's base anchor to the point); otherwise lo_i = sqrt(perp_i), the
    domain floor below which the sqrt itself is undefined.
    """
    rail_length = model["rail_length_mm"]
    px, py, pz = point
    bounds = []

    for base, d in zip(model["tower_bases_mm"], model["tower_dirs"]):
        rx, ry, rz = px - base[0], py - base[1], pz - base[2]
        r_dot_r = rx * rx + ry * ry + rz * rz
        d_i = rx * d[0] + ry * d[1] + rz * d[2]
        perp_sq = max(0.0, r_dot_r - d_i**2)

        if rail_length < d_i:
            bounds.append(None)  # unreachable via this tower at any rod length
            continue

        tower_hi = math.sqrt(perp_sq + (rail_length - d_i) ** 2)
        tower_lo = math.sqrt(r_dot_r) if d_i < 0 else math.sqrt(perp_sq)

        lo = max(rod_length_min, tower_lo)
        hi = min(rod_length_max, tower_hi)
        bounds.append((lo, hi) if lo <= hi else None)

    return bounds


# reachable_rod_length_bounds is exact for the closed-form per-tower
# interval, but the lambdified ik_exprs it's checked against accumulate
# floating-point error along a different computation path, so a rod
# length sitting exactly on a returned boundary can land a hair outside
# [0, rail_length] when actually evaluated. Back off from each boundary
# by this margin before using it.
_ROD_LENGTH_BOUNDS_EPSILON_MM = 1e-4


def resolve_rod_lengths(model, point, current_rod_lengths, rod_length_min, rod_length_max):
    """Return a list of ARM_COUNT rod lengths to actually use for `point`:
    each tower's `current_rod_lengths[i]` unchanged if it already reaches
    the point via that tower, otherwise the closest value within that
    tower's reachable interval (i.e. the minimal nudge that restores
    reachability for that tower only -- other towers are left untouched),
    or None (for the whole result) if some tower has no rod length in
    [rod_length_min, rod_length_max] that reaches the point at all.
    """
    bounds = reachable_rod_length_bounds(model, point, rod_length_min, rod_length_max)
    if any(b is None for b in bounds):
        return None

    resolved = []
    for current, (lo, hi) in zip(current_rod_lengths, bounds):
        margin = min(_ROD_LENGTH_BOUNDS_EPSILON_MM, (hi - lo) / 2)
        lo, hi = lo + margin, hi - margin
        if lo <= current <= hi:
            resolved.append(current)
        else:
            resolved.append(min(max(current, lo), hi))
    return resolved


def compute_inverse_kinematics(model, point, rod_lengths):
    """Return the carriage rail position (mm) for each tower that reaches
    `point` with that tower's entry in `rod_lengths`, or raise ValueError
    if unreachable.
    """
    x, y, z = model["symbols"]["x"], model["symbols"]["y"], model["symbols"]["z"]
    rod_length_syms = model["symbols"]["rod_lengths"]
    px, py, pz = point

    rail_s_values = []
    for i, (ik_expr, rod_length) in enumerate(zip(model["ik_exprs"], rod_lengths)):
        ik_fn = sp.lambdify((x, y, z, rod_length_syms[i]), ik_expr, "math")
        try:
            s = ik_fn(px, py, pz, rod_length)
        except ValueError as exc:
            raise ValueError(f"tower {i}: point unreachable ({exc})") from exc
        rail_s_values.append(s)

    validate_rail_limits(model, point, rail_s_values)
    return rail_s_values


def compute_forward_kinematics(model, rail_s_values, rod_lengths):
    """Return the effector (x, y, z) for the given carriage rail positions
    and per-tower rod lengths.
    """
    args = model["symbols"]["rail_s"] + model["symbols"]["rod_lengths"]
    fk_fn = sp.lambdify(args, model["fk_exprs"], "math", cse=True)
    return fk_fn(*rail_s_values, *rod_lengths)


def compute_carriage_positions(model, point, rod_lengths):
    """Compute inverse kinematics for `point` at the given per-tower rod
    lengths and return, per tower, the carriage rail position plus the 3D
    position of the carriage joint and the effector joint.
    """
    rail_s_values = compute_inverse_kinematics(model, point, rod_lengths)
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


_VALIDATION_SAMPLE_POINTS = [
    (0.0, 0.0, 0.0),
    (30.0, -20.0, 80.0),
    (0.0, 0.0, 150.0),
    (60.0, 30.0, 20.0),
    (-40.0, 40.0, 50.0),
]


def _run_validation_tests(model, verbose=True, tol=1e-6):
    def log(msg):
        if verbose:
            print(msg)

    default_rod = model["rod_length_range_mm"]["default_mm"]
    default_rods = [default_rod] * ARM_COUNT
    rod_min = model["rod_length_range_mm"]["min_mm"]
    rod_max = model["rod_length_range_mm"]["max_mm"]
    rail_length = model["rail_length_mm"]
    failures = []

    log("\nRunning validation tests")

    log("  [1] FK(IK(point, default_rods), default_rods) round trip")
    for pt in _VALIDATION_SAMPLE_POINTS:
        rail_s = compute_inverse_kinematics(model, pt, default_rods)
        back = compute_forward_kinematics(model, rail_s, default_rods)
        err = math.dist(pt, back)
        ok = err < tol
        log(f"      {'PASS' if ok else 'FAIL'}  point={pt}  round-trip error={err:.3e} mm")
        if not ok:
            failures.append(f"round trip failed for {pt}: error {err:.3e} mm")

    log("  [2] rod-length preservation (carriage-to-effector distance == that tower's rod_length)")
    for pt in _VALIDATION_SAMPLE_POINTS:
        for tower in compute_carriage_positions(model, pt, default_rods):
            d = math.dist(tower["carriage_joint"], tower["effector_joint"])
            ok = abs(d - default_rod) < tol
            log(f"      {'PASS' if ok else 'FAIL'}  point={pt} tower={tower['tower']}  "
                f"rod length={d:.6f} mm (expected {default_rod} mm)")
            if not ok:
                failures.append(f"rod length mismatch for {pt} tower {tower['tower']}: {d} mm")

    log("  [3] symmetric case: (0, 0, z) with equal rod lengths gives equal rail_s across all towers")
    rail_s = compute_inverse_kinematics(model, (0.0, 0.0, 100.0), default_rods)
    ok = max(rail_s) - min(rail_s) < tol
    log(f"      {'PASS' if ok else 'FAIL'}  rail_s={rail_s}")
    if not ok:
        failures.append(f"symmetric case failed: rail_s {rail_s}")

    log("  [4] rejection at default rod lengths (domain error, then rail-bounds error)")
    rejection_cases = [
        ("too far from every rail line for the rod to bridge", (500.0, 500.0, 500.0)),
        ("rail_s would exceed rail_length (too high on-axis)", (0.0, 0.0, 1000.0)),
        ("rail_s would go negative (behind the base anchors)", (0.0, 0.0, -215.0)),
    ]
    for label, pt in rejection_cases:
        try:
            compute_inverse_kinematics(model, pt, default_rods)
            log(f"      FAIL  {label}: point={pt} (no error raised)")
            failures.append(f"expected rejection for {label} at {pt}, none raised")
        except ValueError as exc:
            log(f"      PASS  {label}: point={pt} -> {exc}")

    log("  [5] redundancy: a point unreachable at default rod lengths becomes reachable "
        "after resolve_rod_lengths adjusts all three towers together")
    # On-axis, with the IK branch this model picks (see
    # reachable_rod_length_bounds' docstring), the *shorter* the rod the
    # *higher* the on-axis point it can reach at this geometry (e.g.
    # rod=300mm reaches on-axis z~900mm, rod=450mm only ~750mm, rod=700mm
    # only ~500mm) -- this is the concrete "shorten the arm to reach
    # higher" redundancy case the model exists for. z=780 sits just above
    # the default rod's on-axis reach but within a shorter rod's reach.
    # By symmetry an on-axis point needs the same adjustment on all 3 towers.
    pt = (0.0, 0.0, 780.0)
    unreachable_at_default = False
    try:
        compute_inverse_kinematics(model, pt, default_rods)
    except ValueError:
        unreachable_at_default = True
    log(f"      {'PASS' if unreachable_at_default else 'FAIL'}  point={pt} unreachable at "
        f"default rod_lengths={default_rods} mm")
    if not unreachable_at_default:
        failures.append(f"expected {pt} to be unreachable at default rod lengths")

    resolved = resolve_rod_lengths(model, pt, default_rods, rod_min, rod_max)
    resolved_ok = resolved is not None and all(r < default_rod for r in resolved)
    log(f"      {'PASS' if resolved_ok else 'FAIL'}  resolve_rod_lengths -> {resolved} "
        f"(expected all shorter than default {default_rod})")
    if not resolved_ok:
        failures.append(f"expected resolve_rod_lengths to shorten all rods for {pt}, got {resolved}")
    else:
        try:
            rail_s = compute_inverse_kinematics(model, pt, resolved)
            log(f"      PASS  point={pt} reachable at resolved rod_lengths={[f'{r:.3f}' for r in resolved]} mm, "
                f"rail_s={[f'{s:.3f}' for s in rail_s]}")
        except ValueError as exc:
            log(f"      FAIL  point={pt} still unreachable at resolved rod_lengths={resolved}: {exc}")
            failures.append(f"resolved rod lengths {resolved} still unreachable for {pt}: {exc}")

    log("  [6] per-tower independence: adjusting only the towers that need it, leaving others untouched")
    # An off-axis point that only tower 0 (azimuth 0 deg, base at +X)
    # struggles to reach at the default rod length should only move
    # tower 0's resolved rod length; the other two, already fine at the
    # default, should come back unchanged.
    bounds = reachable_rod_length_bounds(model, pt, rod_min, rod_max)
    only_one_needs_change = bounds[0] is not None and not (bounds[0][0] <= default_rod <= bounds[0][1])
    log(f"      per-tower bounds at {pt}: {bounds}")
    resolved2 = resolve_rod_lengths(model, pt, default_rods, rod_min, rod_max)
    independence_ok = resolved2 is not None and all(
        (resolved2[i] == default_rod) == (bounds[i] is not None and bounds[i][0] <= default_rod <= bounds[i][1])
        for i in range(ARM_COUNT)
    )
    log(f"      {'PASS' if independence_ok else 'FAIL'}  resolve_rod_lengths only moved towers "
        f"outside their own reachable interval: {resolved2}")
    if not independence_ok:
        failures.append(f"resolve_rod_lengths changed a tower's rod length that didn't need it: {resolved2}")

    if failures:
        raise AssertionError("validation failed:\n" + "\n".join(f"  - {f}" for f in failures))

    total = len(_VALIDATION_SAMPLE_POINTS) * (1 + ARM_COUNT) + 1 + len(rejection_cases) + 4
    log(f"\nAll validation tests passed ({total} checks). rail_length={rail_length:.3f} mm")


def _print_report(point, rod_lengths, towers):
    px, py, pz = point
    print(f"Target effector position: ({px:.6f}, {py:.6f}, {pz:.6f}) mm, "
          f"rod_lengths={[f'{r:.3f}' for r in rod_lengths]} mm\n")
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

    if len(argv) != 8:
        print(f"usage: {argv[0]} [kinematics.pkl x y z rod0 rod1 rod2]", file=sys.stderr)
        return 2

    model_file = argv[1]
    try:
        point = tuple(float(v) for v in argv[2:5])
        rod_lengths = [float(v) for v in argv[5:8]]
    except ValueError:
        print("x, y, z, rod0, rod1, rod2 must be numbers", file=sys.stderr)
        return 2

    print(f"Loading kinematics model from {model_file} ...")
    model = load_model(model_file)
    print(f"Computing inverse kinematics for target {point} at rod_lengths={rod_lengths} ...\n")
    try:
        towers = compute_carriage_positions(model, point, rod_lengths)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_report(point, rod_lengths, towers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
