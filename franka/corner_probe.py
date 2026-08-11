#!/usr/bin/env python3
"""
Corner boundary probe for the silicon elastomer  (Franka + HEX21).

Double-checks the 5 points taught with record_corners.py (center + 4 corners),
kissing each. It drives to every point using the RECORDED JOINT STATES -- not
Cartesian XYZ -- so the arm reproduces the exact taught posture and does NOT do
weird elbow/joint reconfigurations (which happen when you hand MoveIt a bare
XYZ goal on a redundant 7-DOF arm).

At each taught point it lands at the kiss and reads the Wittenstein normal
force to confirm real contact (no indent). Between points it lifts straight up
a few mm so the tool never drags across the silicon.

FLOW
----
  move to center joint state (home) and take a no-contact baseline
  for each taught point (joints):
      move_to_joints(taught joints)      # exact posture, lands at the kiss
      read Fz -> contact?  (|dFz| above baseline)
      lift straight up a few mm to clear the surface
      refresh baseline at the lifted (no-contact) pose
  return to center
  print PASS/FAIL per point

Reflex note: loosen the Panda's Cartesian force collision thresholds first, or
a light kiss may trigger a protective stop.

Run on `tactile` with the clean MoveIt stack up (see franka_surface_map.py).
"""

import json
import time
from pathlib import Path

from franka_surface_map import WittensteinFT, FrankaArm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CORNERS_FILE = Path(__file__).with_name("corner_joints.json")

# Fallback home if the taught file has no "center" point.
CENTER_JOINTS = [-0.001887, 0.510936, -0.000861, -1.780176,
                 -0.005104, 2.277656, 0.77038]

SERIAL_PORT = "auto"       # auto-detect HEX21 by USB VID:PID

LIFT_MM = 8.0              # straight-up clearance between points (avoids drag)
BASELINE_S = 1.0           # no-contact force window (at the lifted pose)
SAMPLE_S = 0.35            # force window measured at the kiss
SETTLE_S = 0.3             # dwell after arriving before measuring
FORCE_SIGMAS = 2.0         # contact when |dFz| >= max(sigmas*std, min_delta)
FORCE_MIN_DELTA_N = 0.05   # N -- floor above sensor rest noise


def load_points():
    if not CORNERS_FILE.exists():
        raise SystemExit(
            f"[ERROR] {CORNERS_FILE.name} not found. Run record_corners.py first.")
    data = json.loads(CORNERS_FILE.read_text())
    return data.get("points") or data["corners"]   # "corners" = older files


def measure_baseline(ft):
    """No-contact baseline: return (mean, threshold)."""
    mean, std, n = ft.read_window(BASELINE_S)
    threshold = max(FORCE_SIGMAS * std, FORCE_MIN_DELTA_N)
    print(f"    baseline Fz={mean:+.4f} N std={std:.4f} N (n={n}) "
          f"-> contact when |dFz|>={threshold:.4f} N", flush=True)
    return mean, threshold


def main():
    print("=" * 64)
    print(" CORNER BOUNDARY PROBE  (drive by taught JOINT STATES, no indent)")
    print("=" * 64)

    points = load_points()
    print(f"Loaded {len(points)} taught points from {CORNERS_FILE.name}: "
          f"{list(points)}\n")

    home_joints = points.get("center", {}).get("joints", CENTER_JOINTS)
    lift = LIFT_MM / 1000.0

    arm = FrankaArm(vel_scale=0.1, acc_scale=0.1)

    print("Homing to center joint state ...", flush=True)
    arm.move_to_joints(home_joints)
    arm.lock_orientation()
    hx, hy, hz = arm.ee_xyz()

    results = []
    with WittensteinFT(port=SERIAL_PORT) as ft:
        # lift off the center and take the first no-contact baseline
        arm.move_to(hx, hy, hz + lift)
        baseline, threshold = measure_baseline(ft)

        for label, c in points.items():
            print(f"[{label}]  driving to taught joints ...", flush=True)

            # PRIMARY: reproduce the exact taught posture via joint states
            arm.move_to_joints(c["joints"])
            arm.lock_orientation()               # this point's own orientation
            x, y, z = arm.ee_xyz()
            time.sleep(SETTLE_S)

            mean_fz, _std, n = ft.read_window(SAMPLE_S)
            dfz = abs(mean_fz - baseline)
            contact = bool(n) and dfz >= threshold
            tag = "CONTACT" if contact else "no contact"
            print(f"    Fz={mean_fz:+.4f} N  |dFz|={dfz:.4f} N  -> {tag}",
                  flush=True)
            results.append((label, contact, mean_fz, dfz))

            # lift straight up to clear the surface, refresh baseline up there
            arm.move_to(x, y, z + lift)
            baseline, threshold = measure_baseline(ft)
            print("", flush=True)

    arm.move_to_joints(home_joints)              # home

    # ---- summary ----
    print("=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    all_ok = True
    for label, contact, fz, dfz in results:
        if contact:
            print(f"  {label:<13} PASS   Fz={fz:+.3f} N  (|dFz|={dfz:.3f} N)")
        else:
            print(f"  {label:<13} FAIL   no contact (|dFz|={dfz:.3f} N) -- re-teach")
            all_ok = False
    print("-" * 64)
    if all_ok:
        print("  All points made contact -> taught joint states verified.")
    else:
        print("  A point registered no contact -> re-teach it with record_corners.py.")
    print("=" * 64)


if __name__ == "__main__":
    main()
