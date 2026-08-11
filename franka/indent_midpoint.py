#!/usr/bin/env python3
"""Indent N times at the single taught point, logging everything.

Invoked ON TACTILE by master_midpoint.py (so the camera + Wittenstein F/T are
recorded on the workstation over the same window).

TARGET CONSTRUCTION
-------------------
ONE point is taught by hand-positioning the tool exactly where it should indent --
that point IS the target, so nothing is interpolated. The servo controller takes
Cartesian targets only, so:

  XY + orientation  <- forward kinematics of the taught JOINTS via panda_fk
                       (validated to 0.001 mm against taught snapshots), NOT the
                       stored O_T_EE, so the geometry comes from the encoders. The
                       two are cross-checked and any gap is reported.
  Z (surface)       <- the taught (kissed) z. Each repeat dips INDENT_MM below it.
  orientation       <- the TAUGHT orientation, NOT re-levelled with flat_down_quat,
                       so contact geometry matches how the point was kissed.

Normally master_midpoint.py resolves the target on the workstation and passes it in
via --x/--y/--surface-z/--quat; this script then cross-checks it against its own
one_point.json so a stale file on either machine cannot pass unnoticed.

Reuses the tested motion/logging code: map_surface._gross_move (settles between
moves, §4.5 reflex avoidance) and preflight_reach (clamp check before any motion),
plus franka_grid_logger's Ctx / FrankaLogger (93-col schema) / hold / clear_reflex /
warn_if_loaded. Column layout is unchanged, so postprocess.py works as-is.

Does NOT read the F/T -- the HEX21 is on the WORKSTATION for the recording.

Prereq (on tactile), start from HOME with the REACH launch:
    source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
    roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
        robot_ip:=10.1.196.5 load_gripper:=false

Usage:  python3 indent_midpoint.py [out_csv] [--repeats 5] [--indent-mm 2] [--dry-run]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rospy
from franka_msgs.msg import FrankaState

import panda_fk as pfk
import map_surface as m
from servo_client import CartesianServo
from franka_grid_logger import (Ctx, FrankaLogger, clear_reflex, warn_if_loaded, hold,
                                TARE_S)

POINTS_FILE = Path(__file__).with_name("one_point.json")

INDENT_MM = 2.0        # dip this far BELOW the reference surface height
HOVER_MM  = 5.0        # travel / retract / tare at this height above the surface
DWELL_S   = 2.0        # hold the indent this long (the measurement window)
REPEATS   = 5
APPROACH_MM = 15.0     # from home, first go this far above the midpoint


def load_point(path):
    """Read one_point.json -> (data, label, point dict)."""
    d = json.loads(Path(path).read_text())
    if "point" not in d:
        sys.exit("[ERROR] %s has no 'point' key -- re-teach with record_one_point.py."
                 % path)
    return d, d.get("label", "indent_point"), d["point"]


def build_target(pt):
    """(x, y, surface_z, quat, diagnostics) -- see the module docstring.

    Geometry comes from FK of the taught JOINTS, not from the stored O_T_EE xyz.
    """
    q = pt["joints"]
    xyz_fk, quat = pfk.pose_of(q)
    x, y = float(xyz_fk[0]), float(xyz_fk[1])
    surface_z = float(xyz_fk[2])
    meas = pt.get("xyz_measured_O_T_EE") or pt.get("xyz")
    diag = {
        "joints": [float(v) for v in q],
        "xyz_from_joints_fk": [float(v) for v in xyz_fk],
        "xyz_measured_O_T_EE": [float(v) for v in meas],
        "fk_vs_measured_delta_mm": [float(v) for v in
                                    (np.array(xyz_fk) - np.array(meas, float)) * 1e3],
        "quat": [float(v) for v in quat],
    }
    return x, y, surface_z, quat, diag


def report(label, pt, x, y, surface_z, quat, diag, indent_mm, hover_mm, repeats):
    print("=" * 72)
    print(" INDENT AT THE TAUGHT POINT (%d repeats)" % repeats)
    print("=" * 72)
    print("  label                      : %s" % label)
    print("  joints                     : [%s]"
          % ", ".join("%+.4f" % v for v in diag["joints"]))
    print("  FK(joints)                 : [%s]   <- target"
          % ", ".join("%.5f" % v for v in diag["xyz_from_joints_fk"]))
    print("  measured O_T_EE            : [%s]"
          % ", ".join("%.5f" % v for v in diag["xyz_measured_O_T_EE"]))
    print("  FK - measured              : [%s] mm"
          % ", ".join("%+.4f" % v for v in diag["fk_vs_measured_delta_mm"]))
    print("  orientation (taught)       : [%s]" % ", ".join("%+.4f" % v for v in quat))
    print("\n  TARGET x, y                : %.5f, %.5f" % (x, y))
    print("  surface_z (kissed)         : %.5f m" % surface_z)
    print("  dip_z  = surface - %.1f mm  : %.5f m" % (indent_mm, surface_z - indent_mm / 1e3))
    print("  hover  = surface + %.1f mm  : %.5f m" % (hover_mm, surface_z + hover_mm / 1e3))
    print("  repeats                    : %d" % repeats)
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", nargs="?",
                    default=str(Path.home() / "E-BTS/recordings/midpoint_franka.csv"),
                    help="where to write the franka_states CSV")
    ap.add_argument("--points", default=str(POINTS_FILE))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--indent-mm", type=float, default=INDENT_MM)
    ap.add_argument("--hover-mm", type=float, default=HOVER_MM)
    ap.add_argument("--dwell-s", type=float, default=DWELL_S)
    ap.add_argument("--log-hz", type=float, default=200.0)
    ap.add_argument("--no-recovery", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the computed target and exit -- no ROS, no motion")
    ap.add_argument("--max-points", type=int, default=None,
                    help="alias for --repeats (orchestration compatibility)")
    # master_midpoint.py measures the midpoint on the workstation and passes it in
    # explicitly, so both machines act on the SAME target. When given, these override
    # the local derivation entirely (--quat takes 4 values, xyzw).
    ap.add_argument("--x", type=float, default=None)
    ap.add_argument("--y", type=float, default=None)
    ap.add_argument("--surface-z", type=float, default=None)
    ap.add_argument("--quat", type=float, nargs=4, default=None,
                    help="orientation as x y z w")
    args = ap.parse_args()

    repeats = args.max_points if args.max_points is not None else args.repeats
    if repeats < 1:
        sys.exit("[ERROR] repeats must be >= 1")

    explicit = (args.x is not None and args.y is not None
                and args.surface_z is not None and args.quat is not None)
    if explicit:
        # Target supplied by master_midpoint.py -- use it verbatim.
        x, y, surface_z, quat = args.x, args.y, args.surface_z, list(args.quat)
        print("Target supplied by the caller (measured on the workstation):")
        print("  x=%.5f  y=%.5f  surface_z=%.5f  quat=[%s]"
              % (x, y, surface_z, ", ".join("%+.4f" % v for v in quat)))
        print("  repeats=%d  indent=%.1f mm  hover=%.1f mm  dwell=%.1f s"
              % (repeats, args.indent_mm, args.hover_mm, args.dwell_s))
        # Cross-check against the local derivation when the taught file is present, so
        # a stale one_point.json on either machine cannot pass unnoticed.
        if Path(args.points).exists():
            _, _, lpt = load_point(args.points)
            lx, ly, lz, _, _ = build_target(lpt)
            dmm = np.array([lx - x, ly - y, lz - surface_z]) * 1e3
            print("  local derivation differs by [%s] mm"
                  % ", ".join("%+.3f" % v for v in dmm))
            if np.abs(dmm).max() > 0.05:
                print("  [WARN] the caller's target and this machine's one_point.json")
                print("         disagree by more than 0.05 mm -- one of them is stale.")
    else:
        if not Path(args.points).exists():
            sys.exit("[ERROR] %s not found -- run record_one_point.py first, or pass "
                     "--x/--y/--surface-z/--quat." % args.points)
        d, label, pt = load_point(args.points)
        x, y, surface_z, quat, diag = build_target(pt)
        report(label, pt, x, y, surface_z, quat, diag,
               args.indent_mm, args.hover_mm, repeats)

    if args.dry_run:
        print("\n[DRY RUN] no ROS, no motion.")
        return

    hover = args.hover_mm / 1000.0
    indent = args.indent_mm / 1000.0
    hover_z = surface_z + hover
    dip_z = surface_z - indent

    rospy.init_node("indent_midpoint", disable_signals=True)
    if not args.no_recovery:
        clear_reflex()
    warn_if_loaded()

    servo = CartesianServo()

    # Reach preflight BEFORE any motion. Feed map_surface's checker a synthetic
    # one-point "grid" and a synthetic "center" so it validates exactly our targets:
    # the approach hover, the travel hover and the deepest dip.
    pre_points = {"center": {"xyz": [x, y, surface_z]}}
    pre_grid = [{"x": x, "y": y, "z_plane": surface_z}]
    saved_depth, saved_approach = m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM
    m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = args.indent_mm, APPROACH_MM
    try:
        m.preflight_reach(servo, pre_points, pre_grid, hover)
    finally:
        m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = saved_depth, saved_approach

    # FK self-check against the LIVE arm: catches a changed tool/EE transform since
    # teaching, which would bias every joint-derived target.
    st = rospy.wait_for_message("/franka_state_controller/franka_states",
                                FrankaState, timeout=5.0)
    pfk.self_check(st.q, st.O_T_EE, tol_mm=1.0, label="live")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    logger = FrankaLogger(args.out_csv, ctx, log_hz=args.log_hz)   # log BEFORE homing
    print("Logging franka_states to %s | %d repeats | %s"
          % (args.out_csv, repeats, "full rate" if not args.log_hz else
             "%.0f Hz" % args.log_hz))

    try:
        ctx.set("home")
        # home -> hover above the midpoint, ONE settled min-jerk move, taught orientation
        m._gross_move(servo, x, y, surface_z + APPROACH_MM / 1000.0, quat,
                      name="home -> hover above midpoint")

        print("Indenting the midpoint %d times, %.1f mm below the reference surface ..."
              % (repeats, args.indent_mm))
        for i in range(repeats):
            # col carries the repeat index so postprocess labels the 5 pokes
            # distinctly; row stays 0. point_index is the repeat too.
            print("Repeat %d/%d  surface_z=%.5f  dip_z=%.5f" % (i + 1, repeats,
                                                                surface_z, dip_z))
            ctx.set("travel", idx=i, col=i, row=0,
                    target=(x, y, hover_z), surface_z=surface_z)
            m._gross_move(servo, x, y, hover_z, quat, name="travel to hover")

            # TARE: hold still, OUT OF CONTACT, so this indent gets its own F/T zero
            ctx.set("tare", target=(x, y, hover_z))
            hold(servo, x, y, hover_z, quat, TARE_S)

            ctx.set("dip", target=(x, y, dip_z))
            m._gross_move(servo, x, y, dip_z, quat, name="dip %.1fmm" % args.indent_mm)

            ctx.set("dwell", target=(x, y, dip_z))
            hold(servo, x, y, dip_z, quat, args.dwell_s)

            ctx.set("retract", target=(x, y, hover_z))
            m._gross_move(servo, x, y, hover_z, quat, name="retract")

        ctx.set("park")
        m._gross_move(servo, x, y, surface_z + APPROACH_MM / 1000.0, quat,
                      name="park above midpoint")
        ctx.set("done")
    finally:
        logger.close()
        print("Logged %d franka_states samples to %s" % (logger.count, args.out_csv))


if __name__ == "__main__":
    main()
