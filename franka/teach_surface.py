#!/usr/bin/env python3
"""Teach surface points: YOU choose XY by hand, the ROBOT measures Z objectively.

WHY THIS REPLACES record_corners.py
-----------------------------------
record_corners.py recorded the kissed pose and stopped there. The kiss was then the
ONLY depth reference, and it was not good enough: the plane built from the last set of
kissed corners sat +0.863 mm above the true contact surface (spread +0.05 to +1.31 mm
across 9 probed nodes), which silently destroyed the two shallowest levels of a
648-press campaign -- they recorded 0.007 N and 0.003 N, i.e. no contact at all.

The split of labour here:
  * YOU set XY and kiss the surface -- eyes and judgement, which a human is good at,
    and the kiss gives a good starting estimate of Z;
  * the ROBOT then measures Z objectively from that estimate, using the load cell.
    A kiss has no force feedback, so its error is unknowable; a force-referenced
    descent's error is measurable and small.

⚠ THE TWO STAGES NEED DIFFERENT CONTROLLERS, WHICH IS WHY THIS IS NOT ONE PASS.
Hand-guiding requires franka_control ALONE -- the Cartesian servo holds position and
fights you. The fine Z descent requires the servo. Rather than restart controllers at
every point, do all the XY first, then all the Z:

  STAGE 1  --teach-xy    franka_control only.  You hand-guide and KISS. Nothing moves.
  STAGE 2  --probe-z     servo reach launch + HEX21. Per point, the robot:
                           1. rises 2 mm above your kissed pose,
                           2. zeroes the F/T over 2 s, out of contact,
                           3. descends slowly until it sees contact,
                           4. crawls on and extrapolates the surface. Hands off.

⚠ STAGE 2 NEEDS THE HEX21 MOVED TO TACTILE (same as surface mapping, HANDOFF §3.2).
It reads force locally to find contact. Move it back to the workstation afterwards.

HOW Z IS DETERMINED -- extrapolation, not a threshold
-----------------------------------------------------
A force threshold always reports the surface too deep: by the time force is
detectable you are already pressing. Near zero the response is ~0.43 N/mm, so even a
generous 0.05 N trigger sits ~0.12 mm in.

Instead we descend PAST first contact, collect several (z, force) pairs in the light-touch
region, fit a line, and extrapolate to force = 0. That is the free surface, with the
threshold bias removed rather than merely made small.

⚠ Impedance sag does not corrupt this. Sag measures 0.218*F - 0.053 mm, so at the
sub-newton forces used here it is under 0.1 mm -- and we record the ACHIEVED ee_z from
franka_states, never the commanded z, so it cancels regardless.

FREE BONUS: the slope of that same fit IS the local stiffness, in N/mm. Stiffness
varies 9.4x across this elastomer, and knowing it per point is what makes
force-targeted presses possible instead of depth-targeted ones.

Output: surface_points.json -- no timestamp or parameters in the filename, and
snake_case keys with units as a suffix, per the dataset conventions in HANDOFF §16.3.

USAGE
    # STAGE 1 -- franka_control ONLY, servo stopped, user-stop released
    python3 teach_surface.py --teach-xy
    python3 teach_surface.py --teach-xy --labels center,bottom-left,top-left,top-right,bottom-right

    # STAGE 2 -- stop franka_control, bring up the servo reach launch, HEX21 on tactile
    python3 teach_surface.py --probe-z
    python3 teach_surface.py --probe-z --dry-run     # print the plan, no motion
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from franka_msgs.msg import FrankaState

OUT_PATH = Path(__file__).with_name("surface_points.json")
SCHEMA_VERSION = 1
DEFAULT_LABELS = ["center", "bottom-left", "top-left", "top-right", "bottom-right"]

# --- stage 1 guards (ported from record_one_point.py, which record_corners.py lacked) --
DQ_MAX_RAD_S = 0.004      # reject a snapshot while the arm is still settling
FK_TOL_MM = 1.0           # FK(joints) vs measured O_T_EE -- catches a changed tool
MAX_TILT_DEG = 5.0        # ⚠ reject a kiss tilted more than this off vertical. Stage 2
                          # LEVELS the flange to straight-down before descending, and
                          # levelling rotates about the flange origin -- so the physical
                          # cone tip, which sits some unknown length L below the flange,
                          # swings sideways by L*sin(tilt). Keeping the kiss near level
                          # keeps that shift small. At 5 deg and L = 40 mm it is ~3.5 mm;
                          # at 1 deg, ~0.7 mm. Hold it as upright as you can.

# --- stage 2 descent ---
HOVER_MM = 2.0            # rise this far ABOVE the kissed pose before zeroing
SEARCH_BELOW_MM = 2.0     # ...then be willing to search this far BELOW it
# ⚠ Total travel is HOVER_MM + SEARCH_BELOW_MM. An earlier version capped the descent
# at 2.5 mm while hovering 4.0 mm up, so it could not physically reach the surface and
# failed at every point. Caught in simulation before it ever ran. Keep the two coupled.
COARSE_STEP_MM = 0.25     # fast search for first contact
FINE_STEP_MM = 0.20       # ⚠ do NOT reduce this below ~0.15 mm. The joint-impedance
                          # controller tracks a commanded pose to only ~0.5 mm
                          # (HANDOFF §4.4), so commands finer than that do not move the
                          # arm -- it holds, then slips. Measured on the 2026-08-08 run
                          # with a 0.05 mm step: 16 consecutive commands moved z by
                          # 0.028 mm total, then one step jumped 0.50 mm. The (z, force)
                          # data arrives in clumps instead of a sweep, and top-right
                          # fitted a line through only TWO distinct z values (R2 0.35).
BACKOFF_MM = 0.60         # after coarse contact, retreat this far and come back finely
SETTLE_S = 0.7            # > controller min_duration (0.5 s) so each step settles
SAMPLE_S = 0.25           # force averaging per step
SETTLE_BEFORE_ZERO_S = 1.5  # ⚠ dwell after arriving at hover, BEFORE zeroing the F/T.
                          # The impedance controller reaches "stopped" with a residual
                          # steady-state torque in whatever direction it last travelled.
                          # That load lands in the baseline, and a shifted baseline moves
                          # the extrapolated surface without changing the fitted slope --
                          # which is exactly the signature measured on 2026-08-08: rows
                          # swept +u read 0.26 mm higher than rows swept -u (r = +0.75 and
                          # +0.81 on two independent runs) while stiffness was unaffected.
BASELINE_S = 2.0          # ⚠ the F/T zero. Averaged over 2 s at the raised pose, out of
                          # contact, immediately before each descent. This is a NUMERICAL
                          # zero of the recorded stream -- the HEX21 has no tare command
                          # and the driver is read-only, so nothing is reset in hardware.
                          # 2 s at ~1 kHz is ~2000 samples, which takes the 0.116 N
                          # per-sample noise down to ~0.003 N on the baseline.
FIT_FORCE_MAX_N = 0.60    # collect fine points up to here, then stop and extrapolate.
                          # Raised from 0.40: at ~0.4 N/mm that spans ~1.5 mm of contact,
                          # so a 0.20 mm step yields ~7 genuinely distinct z levels
                          # rather than the 2-4 the old settings produced.
FIT_FORCE_MIN_N = 0.06    # ignore samples below this -- they are noise, not contact
MIN_DISTINCT_Z = 4        # quality gate: a line needs more than a couple of clusters
MIN_FIT_R2 = 0.90         # quality gate: below this the surface_z is not trustworthy
APPROACH_MM = 12.0        # travel height between points


def read_state(timeout=5.0):
    st = rospy.wait_for_message("/franka_state_controller/franka_states",
                                FrankaState, timeout=timeout)
    T = st.O_T_EE                       # column-major 4x4; translation at 12,13,14
    xyz = [float(T[12]), float(T[13]), float(T[14])]
    return st, [float(a) for a in st.q], xyz


def quat_from_O_T_EE(T):
    """Rotation part of the column-major 4x4 -> quaternion [x, y, z, w].

    Delegates to home_and_level.R_to_quat rather than re-deriving it. There were
    already two quaternion conversions in this repo (home_and_level and
    servo_client); a third hand-rolled one would eventually disagree with them, and
    an orientation that is silently wrong by a few degrees commands a tilted descent
    into the silicone.
    """
    from home_and_level import R_to_quat
    R = np.array([[T[0], T[4], T[8]],
                  [T[1], T[5], T[9]],
                  [T[2], T[6], T[10]]], float)
    return [float(v) for v in R_to_quat(R)]


def build_reach_points(named_xyz, center_xyz=None):
    """Build the dict map_surface.preflight_reach needs. It REQUIRES a 'center' key.

    ⚠ map_surface.py:353 does points["center"]["xyz"] unconditionally, to form the
    home -> approach target. So passing a SUBSET of labels (--labels top-right) or a
    numbered grid raised `KeyError: 'center'` before any motion -- harmless but it
    aborts the run. Always supply one: the real taught centre if we have it, otherwise
    the centroid of whatever points we are about to visit.
    """
    out = {k: {"xyz": [float(v) for v in xyz]} for k, xyz in named_xyz.items()}
    if "center" not in out:
        if center_xyz is not None:
            out["center"] = {"xyz": [float(v) for v in center_xyz]}
        else:
            a = np.array(list(named_xyz.values()), float)
            out["center"] = {"xyz": [float(v) for v in a.mean(axis=0)]}
    return out


def load_points():
    if not OUT_PATH.exists():
        sys.exit("[ERROR] %s not found -- run --teach-xy first." % OUT_PATH.name)
    return json.loads(OUT_PATH.read_text())


def save(data):
    OUT_PATH.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------- stage 1
def teach_xy(labels):
    import panda_fk as pfk
    rospy.init_node("teach_surface_xy", anonymous=True, disable_signals=True)

    print("=" * 70)
    print(" STAGE 1 of 2 -- YOU set XY, the robot will measure Z later")
    print("=" * 70)
    from home_and_level import tilt_deg
    print("  Hand-guide the tool to KISS each point -- just touching, no indent.")
    print("  Hold it as UPRIGHT as you can: tilt is rejected above %.1f deg, because"
          % MAX_TILT_DEG)
    print("  stage 2 levels the flange and that swings the tip sideways.")
    print("  Your kiss is the starting estimate; stage 2 rises 2 mm, zeroes the load")
    print("  cell over 2 s, and descends to measure the surface properly from there.")
    print("  franka_control must be the ONLY controller up (the servo fights you).\n")

    points = {}
    for label in labels:
        while True:
            input("  >> KISS the [%s] point (touch, no indent), then Enter... " % label)
            try:
                st, joints, xyz = read_state()
            except Exception as e:
                print("     [ERROR] could not read franka_states: %s" % e)
                continue

            # GUARD 1: the arm must be still. After hand-guiding it keeps drifting for
            # a moment, and a snapshot taken then bakes that drift into the datum.
            dq = max(abs(v) for v in st.dq)
            if dq > DQ_MAX_RAD_S:
                print("     [REJECTED] still moving: max|dq| = %.5f rad/s (need < %.3f)."
                      % (dq, DQ_MAX_RAD_S))
                print("                Let go, wait a second, press Enter again.")
                continue

            # GUARD 2: FK of the joints must reproduce the measured pose. A gap means
            # the tool/EE transform changed and every derived target would be biased.
            try:
                xyz_fk, _ = pfk.pose_of(joints)
                d_mm = float(np.abs(np.array(xyz_fk) - np.array(xyz)).max() * 1e3)
                if d_mm > FK_TOL_MM:
                    print("     [REJECTED] FK disagrees with the measured pose by %.3f mm"
                          % d_mm)
                    print("                (limit %.1f mm). The tool transform may have"
                          % FK_TOL_MM)
                    print("                changed -- fix that before teaching anything.")
                    continue
            except Exception as e:
                print("     [warn] FK cross-check unavailable (%s) -- continuing." % e)
                d_mm = None

            quat = quat_from_O_T_EE(st.O_T_EE)

            # GUARD 3: the flange must already be close to straight-down, because
            # stage 2 will level it exactly and that rotation moves the physical tip.
            tilt = float(tilt_deg(quat))
            if tilt > MAX_TILT_DEG:
                print("     [REJECTED] flange is %.2f deg off vertical (limit %.1f)."
                      % (tilt, MAX_TILT_DEG))
                print("                Straighten the tool so it points at the floor,")
                print("                re-kiss, and press Enter again.")
                continue
            print("     tilt off vertical: %.2f deg  (stage 2 will level the last bit)"
                  % tilt)

            points[label] = {
                "label": label,
                "joints_rad": [round(v, 6) for v in joints],
                "xy_m": [round(xyz[0], 6), round(xyz[1], 6)],
                "kissed_z_m": round(xyz[2], 6),
                "quat_xyzw": [round(v, 6) for v in quat],
                "dq_max_rad_s": round(dq, 6),
                "tilt_deg_at_kiss": round(tilt, 3),
                "fk_vs_measured_mm": None if d_mm is None else round(d_mm, 4),
                "robot_mode": int(st.robot_mode),
                # filled in by stage 2:
                "surface_z_m": None, "stiffness_n_per_mm": None,
                "surface_fit_r2": None, "surface_fit_n_points": None,
                "surface_fit_distinct_z": None, "surface_quality_ok": None,
                "probe_steps": None,
            }
            print("     recorded [%s]  xy = %.5f, %.5f   kissed z = %.5f  (dq %.5f)"
                  % (label, xyz[0], xyz[1], xyz[2], dq))
            break

    save({"schema_version": SCHEMA_VERSION,
          "recorded_at": datetime.now().isoformat(timespec="seconds"),
          "frame": "panda_link0", "stage": "xy",
          "note": "XY, orientation and a kissed Z set by hand; surface_z_m and "
                  "stiffness_n_per_mm measured objectively by --probe-z.",
          "points": points})
    print("\n" + "=" * 70)
    print(" Saved %d points -> %s   (stage = xy, Z not yet measured)"
          % (len(points), OUT_PATH.name))
    print("=" * 70)
    print(" NEXT: stop franka_control, move the HEX21 to TACTILE, bring up")
    print("       cartesian_pose_servo_reach.launch, then:")
    print("           python3 teach_surface.py --probe-z")


# ---------------------------------------------------------------- stage 2
def probe_one(servo, ft, x, y, z_hover, z_held, quat, args, label="",
              approach_z=None):
    """Coarse-find contact, back off, fine-step, extrapolate force to zero.

    Returns (surface_z, stiffness_n_per_mm, r2, rows).

    Two phases because one is too slow: a pure 0.05 mm crawl over the whole
    hover+search range is ~110 steps at ~1 s each. Coarse-then-fine finds contact in
    ~20 steps and only crawls the last millimetre, where the data actually matters.
    """
    import map_surface as m
    floor_z = z_held - args.search_below_mm / 1000.0     # absolute hard limit

    # ⚠ APPROACH EVERY POINT THE SAME WAY: traverse at a safe height, then come STRAIGHT
    # DOWN. A single diagonal move leaves a residual lateral load whose direction depends
    # on where the arm came from, and that load biases the F/T baseline -- measured as a
    # 0.26 mm offset between rows swept +u and -u. Splitting the move means the last
    # motion before the zero is always vertical, so whatever residual remains is the
    # same at every point and cancels out of relative depths.
    if approach_z is not None:
        m._gross_move(servo, x, y, approach_z, quat, name="%s: traverse" % label)
    m._gross_move(servo, x, y, z_hover, quat, name="%s: -> hover" % label)

    settle = getattr(args, "settle_before_zero_s", SETTLE_BEFORE_ZERO_S)
    if settle > 0:
        print("    settling %.1f s before zeroing ..." % settle, flush=True)
        time.sleep(settle)
    base, base_sd, base_n = ft.read_window(BASELINE_S)
    print("    baseline Fz = %+.4f N (sd %.4f, n=%d)   floor z = %.5f"
          % (base, base_sd, base_n, floor_z), flush=True)

    rows = []

    def step_to(z, tag):
        servo.send_target(x, y, z, quat)
        time.sleep(args.settle_s)
        pos, _ = servo.current_pose()
        fz, sd, n = ft.read_window(SAMPLE_S)
        F = abs(fz - base)
        rows.append({"phase": tag, "z_cmd_m": round(z, 6),
                     "z_achieved_m": round(pos[2], 6),
                     "force_n": round(F, 4), "force_sd_n": round(sd, 4)})
        return pos[2], F

    try:
        # ---- phase 1: coarse search for first contact ----
        z, contact_z = z_hover, None
        while z > floor_z:
            z = max(floor_z, z - args.coarse_step_mm / 1000.0)
            zach, F = step_to(z, "coarse")
            print("      coarse z %.5f  F %6.3f N" % (zach, F), flush=True)
            if F >= FIT_FORCE_MIN_N:
                contact_z = z
                print("      first contact near z %.5f" % zach, flush=True)
                break
        if contact_z is None:
            print("    [warn] no contact within %.1f mm below your kiss. Either the kiss "
                  "was well high,\n           or the HEX21 is not reading."
                  % args.search_below_mm)
            return None, None, None, rows, 0

        # ---- phase 2: back off above contact, then crawl down through it ----
        z = min(z_hover, contact_z + BACKOFF_MM / 1000.0)
        step_to(z, "backoff")
        while z > floor_z:
            z = max(floor_z, z - args.fine_step_mm / 1000.0)
            zach, F = step_to(z, "fine")
            print("      fine   z %.5f  F %6.3f N" % (zach, F), flush=True)
            if F >= args.fit_force_max_n:
                break
    finally:
        print("    retracting ...", flush=True)
        try:
            m._gross_move(servo, x, y, z_hover, quat, name="%s: retract" % label)
            if approach_z is not None:
                m._gross_move(servo, x, y, approach_z, quat,
                              name="%s: lift to traverse height" % label)
        except Exception as e:
            print("    !! retract failed (%s) -- CHECK THE ARM." % e)

    # Extrapolate the light-touch line to force = 0. FINE points only -- the coarse
    # pass is too sparse and its last step may already be well into contact. Uses the
    # ACHIEVED z, so impedance sag cancels.
    fine = [r for r in rows if r["phase"] == "fine"]
    z = np.array([r["z_achieved_m"] for r in fine])
    F = np.array([r["force_n"] for r in fine])
    use = (F >= FIT_FORCE_MIN_N) & (F <= args.fit_force_max_n * 1.05)
    if use.sum() < 3:
        print("    [ERROR] only %d usable fine contact points -- cannot fit." % use.sum())
        return None, None, None, rows

    # ⚠ QUALITY GATE. The arm holds then slips, so many samples can share one z. Count
    # DISTINCT z levels, not samples: a line through two clusters looks like a fit and
    # is not one. top-right on the 2026-08-08 run passed the old sample count with 9
    # points at 2 z values and returned R2 0.35, which would have silently corrupted a
    # corner of the surface plane.
    zc = np.sort(z[use])
    distinct = 1 + int((np.diff(zc) > (args.fine_step_mm / 1000.0) * 0.5).sum())
    slope, intercept = np.polyfit(z[use], F[use], 1)      # F = slope*z + intercept
    surface_z = float(-intercept / slope)                 # z where force crosses zero
    pred = slope * z[use] + intercept
    r2 = float(1 - ((F[use] - pred) ** 2).sum()
               / max(((F[use] - F[use].mean()) ** 2).sum(), 1e-12))
    stiffness = float(-slope / 1000.0)                    # N/mm (z in metres)
    if distinct < MIN_DISTINCT_Z or r2 < MIN_FIT_R2:
        print("    ⚠ LOW CONFIDENCE: %d distinct z levels (need %d), R2 %.4f (need %.2f)."
              % (distinct, MIN_DISTINCT_Z, r2, MIN_FIT_R2))
        print("      surface_z = %.6f is reported but flagged. Re-probe this point."
              % surface_z)
    else:
        print("    fit: %d distinct z levels, R2 %.4f" % (distinct, r2))
    return surface_z, stiffness, r2, rows, distinct


def probe_z(args):
    data = load_points()
    pts = data["points"]
    labels = [l for l in (args.labels.split(",") if args.labels else pts.keys()) if l in pts]
    if not labels:
        sys.exit("[ERROR] no matching points in %s" % OUT_PATH.name)

    print("=" * 70)
    print(" STAGE 2 of 2 -- the robot measures Z at %d taught XY position(s)" % len(labels))
    print("=" * 70)
    for l in labels:
        p = pts[l]
        print("  %-13s xy %.5f, %.5f   kissed z %.5f" % (l, p["xy_m"][0], p["xy_m"][1],
                                                         p["kissed_z_m"]))
    print("\n  per point: rise %.1f mm above your kiss, zero the F/T over %.1f s out of"
          % (args.hover_mm, BASELINE_S))
    print("  contact, then descend. Search up to %.1f mm below the kissed pose."
          % args.search_below_mm)
    print("  coarse %.2f mm steps to find contact, back off %.2f mm, then %.3f mm steps"
          % (args.coarse_step_mm, BACKOFF_MM, args.fine_step_mm))
    print("  up to %.2f N; fit the light-touch band and extrapolate to F = 0."
          % args.fit_force_max_n)
    print("  worst case ~%d steps/point (~%.0f s), hard floor %.1f mm below your pose."
          % (int((args.hover_mm + args.search_below_mm) / args.coarse_step_mm)
             + int((BACKOFF_MM + 0.6) / args.fine_step_mm),
             ((args.hover_mm + args.search_below_mm) / args.coarse_step_mm
              + (BACKOFF_MM + 0.6) / args.fine_step_mm) * (args.settle_s + SAMPLE_S),
             args.search_below_mm))
    print("=" * 70)
    print("  ⚠ HEX21 must be on TACTILE. Servo reach launch up. Hands off the arm.")
    print("  %s" % ("Flange is LEVELLED to straight-down before each descent."
                    if not args.keep_taught_orientation else
                    "⚠ NOT levelling -- descending at the taught tilt."))
    print("  ⚠ Keep a hand near the user-stop -- the servo does not stop on contact.")
    print("=" * 70)
    if args.dry_run:
        print("\n[DRY RUN] no ROS, no motion.")
        return

    rospy.init_node("teach_surface_z", disable_signals=True)
    import map_surface as m
    from franka_grid_logger import clear_reflex, warn_if_loaded
    from servo_client import CartesianServo
    from franka_surface_map import WittensteinFT
    from home_and_level import flat_down_quat, tilt_deg
    if not args.no_recovery:
        clear_reflex()
    warn_if_loaded()
    servo = CartesianServo()

    # Reach preflight over the real targets, BEFORE any motion.
    grid = [{"x": pts[l]["xy_m"][0], "y": pts[l]["xy_m"][1],
             "z_plane": pts[l]["kissed_z_m"]} for l in labels]
    saved = m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM
    m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = args.search_below_mm, APPROACH_MM
    try:
        named = {l: [pts[l]["xy_m"][0], pts[l]["xy_m"][1], pts[l]["kissed_z_m"]]
                 for l in labels}
        ctr = ([pts["center"]["xy_m"][0], pts["center"]["xy_m"][1],
                pts["center"]["kissed_z_m"]] if "center" in pts else None)
        m.preflight_reach(servo, build_reach_points(named, ctr), grid,
                          args.hover_mm / 1000.0)
    finally:
        m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = saved

    # One approach height for every point, clear of the HIGHEST surface, so a traverse
    # can never dip toward a high spot.
    approach_z = max(pts[l]["kissed_z_m"] for l in labels) + APPROACH_MM / 1000.0
    print("  traverse height   : %.5f m (%.1f mm above the highest kiss)"
          % (approach_z, APPROACH_MM))
    ok = 0
    with WittensteinFT(port=m.SERIAL_PORT) as ft:
        try:
            for l in labels:
                p = pts[l]
                x, y = p["xy_m"]
                quat = p["quat_xyzw"]
                z_held = p["kissed_z_m"]
                z_hover = z_held + args.hover_mm / 1000.0
                print("\n[%s]" % l, flush=True)
                # LEVEL THE FLANGE FIRST. The move up to the raised pose is commanded
                # with the levelled orientation, so by the time we zero the load cell
                # and start descending, the approach axis points at the floor and the
                # cone goes straight in. Yaw is preserved (flat_down_quat keeps the
                # heading), so only the tilt is removed.
                if not args.keep_taught_orientation:
                    quat_level = [float(v) for v in flat_down_quat(quat)]
                    before, after = tilt_deg(quat), tilt_deg(quat_level)
                    print("    levelling flange: %.2f deg -> %.2f deg off vertical"
                          % (before, after), flush=True)
                    p["quat_used_xyzw"] = [round(v, 6) for v in quat_level]
                    p["tilt_deg_before_level"] = round(float(before), 3)
                    p["tilt_deg_after_level"] = round(float(after), 3)
                    quat = quat_level
                else:
                    print("    keeping the taught orientation (%.2f deg off vertical)"
                          % tilt_deg(quat), flush=True)
                    p["quat_used_xyzw"] = [round(v, 6) for v in quat]
                out = probe_one(servo, ft, x, y, z_hover, z_held, quat, args, l,
                                approach_z=approach_z)
                sz, stiff, r2, rows = out[0], out[1], out[2], out[3]
                distinct = out[4] if len(out) > 4 else None
                p["probe_steps"] = rows
                if sz is None:
                    print("    -> FAILED, surface_z_m left null")
                    continue
                p["surface_z_m"] = round(sz, 6)
                p["stiffness_n_per_mm"] = round(stiff, 4)
                p["surface_fit_r2"] = round(r2, 4)
                p["surface_fit_distinct_z"] = distinct
                p["surface_quality_ok"] = bool(
                    distinct is not None and distinct >= MIN_DISTINCT_Z and r2 >= MIN_FIT_R2)
                p["surface_fit_n_points"] = int(sum(
                    1 for r in rows if FIT_FORCE_MIN_N <= r["force_n"] <= args.fit_force_max_n * 1.05))
                ok += 1
                print("    -> surface_z = %.6f m   (%+.3f mm vs your kiss; positive = "
                      "the real surface is BELOW it)"
                      % (sz, (p["kissed_z_m"] - sz) * 1e3))
                print("       local stiffness %.3f N/mm   fit R2 %.4f on %d points"
                      % (stiff, r2, p["surface_fit_n_points"]))
        except KeyboardInterrupt:
            print("\n[interrupted] keeping what was measured.")
        finally:
            data["stage"] = "complete" if ok == len(pts) else "partial"
            data["probed_at"] = datetime.now().isoformat(timespec="seconds")
            data["probe_params"] = {"coarse_step_mm": args.coarse_step_mm,
                                    "fine_step_mm": args.fine_step_mm,
                                    "hover_mm": args.hover_mm,
                                    "search_below_mm": args.search_below_mm,
                                    "fit_force_max_n": args.fit_force_max_n,
                                    "fit_force_min_n": FIT_FORCE_MIN_N,
                                    "method": "linear fit of the light-touch band, "
                                              "extrapolated to force = 0"}
            save(data)
            print("\nsaved %s  (%d/%d points measured)" % (OUT_PATH.name, ok, len(pts)))

    if ok:
        zs = [pts[l]["surface_z_m"] for l in pts if pts[l]["surface_z_m"] is not None]
        ks = [pts[l]["stiffness_n_per_mm"] for l in pts
              if pts[l]["stiffness_n_per_mm"] is not None]
        print("\n  surface tilt across the measured points : %.3f mm"
              % ((max(zs) - min(zs)) * 1e3))
        if len(ks) > 1:
            print("  local stiffness range                   : %.2f .. %.2f N/mm (%.1fx)"
                  % (min(ks), max(ks), max(ks) / max(min(ks), 1e-9)))
            print("  -> that spread is why presses must target FORCE, not depth.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--teach-xy", action="store_true",
                   help="STAGE 1: you hand-guide, we record XY + orientation only. "
                        "franka_control ONLY -- stop the servo first.")
    g.add_argument("--probe-z", action="store_true",
                   help="STAGE 2: robot descends and measures the surface Z at each "
                        "taught XY. Servo reach launch + HEX21 ON TACTILE.")
    ap.add_argument("--labels", default=None,
                    help="comma-separated point names (default: the 5 standard ones for "
                         "--teach-xy, or whatever is in the file for --probe-z)")
    ap.add_argument("--coarse-step-mm", type=float, default=COARSE_STEP_MM)
    ap.add_argument("--fine-step-mm", type=float, default=FINE_STEP_MM)
    ap.add_argument("--hover-mm", type=float, default=HOVER_MM)
    ap.add_argument("--search-below-mm", type=float, default=SEARCH_BELOW_MM)
    ap.add_argument("--settle-s", type=float, default=SETTLE_S)
    ap.add_argument("--settle-before-zero-s", type=float, default=SETTLE_BEFORE_ZERO_S,
                    help="dwell at hover before zeroing the F/T (default %.1f s). Lets the "
                         "residual load from the move decay so it does not bias the "
                         "baseline." % SETTLE_BEFORE_ZERO_S)
    ap.add_argument("--fit-force-max-n", type=float, default=FIT_FORCE_MAX_N)
    ap.add_argument("--keep-taught-orientation", action="store_true",
                    help="do NOT level the flange -- descend at whatever tilt you "
                         "kissed with. Default is to level to straight-down first.")
    ap.add_argument("--no-recovery", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.teach_xy:
        teach_xy((args.labels.split(",") if args.labels else DEFAULT_LABELS))
    else:
        probe_z(args)


if __name__ == "__main__":
    main()
