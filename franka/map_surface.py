#!/usr/bin/env python3
"""
Surface mapping of the silicon elastomer  (Franka SERVO + HEX21).

Builds a height map z_touch(x, y) over the elastomer by probing a grid of XY
points and descending at each until the Wittenstein normal force first rises
(no indent -- it stops at first contact). Output: surface_map.csv (+ heatmap).

WHY THIS SERVO REWRITE EXISTS
-----------------------------
The old map was probed through MoveIt's trajectory controller, which undershot /
behaved differently against the elastomer than the Cartesian pose SERVO that now
runs the experiment. A height map is only useful if it is expressed in the frame
the experiment motion actually reaches, so we RE-PROBE with the SAME servo
(CartesianPoseServoController via servo_client.CartesianServo). The servo reaches
the exact commanded Z, so each touch height is read from the true EE pose at
first contact -- more accurate than before.

MOTION / SAFETY
---------------
* All motion is the incremental Cartesian servo (hard velocity/acceleration
  limits, straight-line integration toward each target -- no elbow-flip slam).
* Orientation is FLAT-DOWN (flange approach axis -> global -Z), preserving yaw,
  computed from the current pose -- leveling is automatic, no manual jog.
* The servo has no joint-homing, so START is reached by Cartesian xyz: from home
  the arm goes STRAIGHT to a hover above the taught CENTER point (corner_joints.json)
  in one settled move, then the raster begins at the bottom-left. That taught xyz
  is the FK of your recorded joint state (same FrankaState snapshot), so it homes
  to your recorded reference without hand-positioning.
* Every gross move WAITS until the arm has arrived AND stopped before the next
  target is sent -- issuing a target mid-motion makes the controller replan from a
  moving state, which triggers a joint-acceleration-discontinuity REFLEX.
* Each point is wrapped so ANY failure retracts the tool straight up. The
  descent can go at most MAX_DEPTH_MM below the corner plane (hard floor).

GRID
----
Bilinear interpolation of the 4 TAUGHT corner XYZ in corner_joints.json, inset a
few mm from the edges. Serpentine order so neighbours are adjacent.

PREREQS (on tactile)
--------------------
    source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
    # Start from HOME: use the reach launch (max_translation raised to 0.55 m so
    # the arm can travel home -> surface; the stock launch's 0.30 m clamp can't).
    roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
        robot_ip:=10.1.196.5 load_gripper:=false
    # HEX21 plugged into THIS machine (auto-detected by USB VID:PID 0483:5740).

Use --dry-run (no ROS/motion) to preview the grid, and --max-points N for a
cautious first run.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from home_and_level import flat_down_quat, tilt_deg

CORNERS_FILE = Path(__file__).with_name("corner_joints.json")
OUT_CSV = Path(__file__).with_name("surface_map.csv")
OUT_PNG = Path(__file__).with_name("surface_map.png")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COLS = 9                   # points along the 30 mm edge (BL->BR), ~3.75 mm
ROWS = 11                  # points along the 36 mm edge (BL->TL), ~3.6 mm
INSET_MM = 0.0             # 0 = probe corner-to-corner (cover the edges). Positive
                           # keeps probes that far IN from the taught edges.

SERIAL_PORT = "auto"
HOVER_MM = 3.0             # start each descent this far above the corner-plane
MAX_DEPTH_MM = 2.0         # HARD floor: never descend deeper than this below plane
CENTER_APPROACH_MM = 15.0  # from home, go straight to this height above the center
PROBE_STEP_MM = 0.4        # descend in these increments; read force after each

# WHY EACH MOVE MUST SETTLE BEFORE THE NEXT (avoids the reflex)
# ------------------------------------------------------------
# The servo replans a fresh min-jerk segment whenever a NEW target arrives, and
# that segment assumes ZERO initial velocity. If a new target is sent while the
# arm is still moving (e.g. move_to returned early on a loose tolerance), the
# commanded velocity jumps -> a joint-acceleration discontinuity -> the robot
# reflexes ("cartesian_motion_generator_joint_acceleration_discontinuity") and
# freezes. So every gross move WAITS until the arm has both arrived AND stopped
# before the next target is issued (see _gross_move).
#
# WHY WE DON'T USE move_to's TOLERANCE FOR PROBING
# The internal joint_impedance controller tracks a commanded pose to only ~0.5 mm,
# so move_to can never converge to a sub-mm tolerance. Probing instead paces the
# descent (send_target + settle) and records the TRUE EE pose at contact.
REACH_TOL_M = 0.003        # gross move counts as "arrived" within 3 mm
STOP_EPS_M = 0.0006        # < 0.6 mm between polls (~12 mm/s) counts as "stopped"
SETTLE_POLLS = 3           # need this many consecutive arrived+stopped polls
SETTLE_HOLD_S = 0.4        # then hold the target this long so cmd velocity is 0
# "Stuck" = the error stops DECREASING (a reflex-freeze), NOT low instantaneous
# speed -- a min-jerk move starts very gently, so early polls move <STOP_EPS but
# the error is still shrinking. Only abort if err makes no real progress for a
# while. PROGRESS_EPS must exceed the ~0.5 mm impedance jitter.
PROGRESS_EPS_M = 0.001     # err must drop this much to count as progress
NO_PROGRESS_S = 5.0        # err stuck (no progress) this long -> reflex/clamp abort
POLL_S = 0.05
TRAVEL_TIMEOUT_S = 45.0    # gross travel at 0.03 m/s is slow (a 0.4 m leg ~ 25 s)
PROBE_SETTLE_S = 0.6       # > controller min_duration so each probe step settles

# The servo controller clamps every target to a sphere of radius max_translation
# around its START pose (where the controller was launched). We pre-check that
# every target is inside that sphere and abort with guidance BEFORE moving, so a
# clamped target can never make the arm press the boundary until you e-stop.
REACH_MARGIN_M = 0.02      # stay this far inside the cap
CLAMP_RESID_M = 0.010      # a clamped gross move leaves a > 1 cm residual

BASELINE_S = 0.6
SAMPLE_S = 0.2
FORCE_SIGMAS = 2.0
FORCE_MIN_DELTA_N = 0.05


def load_points():
    if not CORNERS_FILE.exists():
        raise SystemExit(f"[ERROR] {CORNERS_FILE.name} not found. Teach corners first.")
    data = json.loads(CORNERS_FILE.read_text())
    return data.get("points") or data["corners"]


def build_grid(points, inset_mm=INSET_MM):
    """Bilinear grid over the 4 taught corners; serpentine order.

    inset_mm=0 spans corner-to-corner (grid corners land ON the taught corners,
    i.e. the elastomer edges). Positive insets keep probes that far inside.
    """
    BL = np.array(points["bottom-left"]["xyz"], float)
    TL = np.array(points["top-left"]["xyz"], float)
    TR = np.array(points["top-right"]["xyz"], float)
    BR = np.array(points["bottom-right"]["xyz"], float)

    x_len = np.linalg.norm(BR - BL)
    y_len = np.linalg.norm(TL - BL)
    fu = (inset_mm / 1000.0) / x_len if x_len > 0 else 0.0
    fv = (inset_mm / 1000.0) / y_len if y_len > 0 else 0.0
    us = np.linspace(fu, 1.0 - fu, COLS)
    vs = np.linspace(fv, 1.0 - fv, ROWS)

    def bilinear(u, v):
        return ((1 - u) * (1 - v) * BL + u * (1 - v) * BR +
                (1 - u) * v * TL + u * v * TR)

    grid = []
    for i, v in enumerate(vs):
        col_order = range(COLS) if i % 2 == 0 else range(COLS - 1, -1, -1)
        for j in col_order:
            p = bilinear(us[j], v)
            grid.append({"row": i, "col": j, "x": float(p[0]),
                         "y": float(p[1]), "z_plane": float(p[2])})
    return grid


def _count_ok(path):
    """How many 'ok' rows an existing map CSV has (0 if absent/unreadable)."""
    try:
        with open(path) as f:
            return sum(1 for r in csv.DictReader(f) if r.get("status") == "ok")
    except Exception:
        return 0


def save_csv(rows, path=None):
    """Write the map CSV, BACKING UP any existing file first.

    A partial/aborted run must never silently destroy a good map (it did once:
    an early reflex abort left rows==[] and overwrote a complete 99-point map with
    just a header). So we back up to <name>.prev.csv and warn loudly if the new
    data has fewer 'ok' points than what we are replacing.
    """
    path = Path(path) if path else OUT_CSV
    fields = ["point_id", "row", "col", "x", "y", "z_plane", "z_touch",
              "depth_from_plane_mm", "contact_fz", "status"]
    old_ok = _count_ok(path)
    new_ok = sum(1 for r in rows if r.get("status") == "ok")
    if path.exists():
        backup = path.with_suffix(".prev.csv")
        try:
            backup.write_bytes(path.read_bytes())
            print(f"[MAP] backed up previous map ({old_ok} ok) -> {backup.name}", flush=True)
        except Exception as e:
            print(f"[MAP] WARNING: could not back up {path.name}: {e}", flush=True)
    if old_ok and new_ok < old_ok:
        print("!" * 72, flush=True)
        print(f"[MAP] WARNING: writing {new_ok} ok points over a map that had {old_ok}.",
              flush=True)
        print(f"[MAP] The previous map is preserved in {path.with_suffix('.prev.csv').name}.",
              flush=True)
        print("!" * 72, flush=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[MAP] wrote {len(rows)} rows ({new_ok} ok) -> {path}", flush=True)


def save_heatmap(rows, path=None):
    path = Path(path) if path else OUT_PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[MAP] skipping heatmap ({e})", flush=True)
        return
    Z = np.full((ROWS, COLS), np.nan)
    for r in rows:
        if r["status"] == "ok":
            Z[r["row"], r["col"]] = r["z_touch"] * 1000.0
    plt.figure(figsize=(6, 7))
    im = plt.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, label="surface Z (mm, base frame)")
    plt.xlabel("grid col (30 mm edge)")
    plt.ylabel("grid row (36 mm edge)")
    plt.title("Elastomer surface height map (servo-probed)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[MAP] wrote heatmap -> {path}", flush=True)


def _probe_down(servo, ft, x, y, z_start, z_floor, step, quat, baseline, threshold):
    """Paced descent in `step` increments until |Fz - baseline| >= threshold.

    Each step COMMANDS the next depth (send_target, not move_to) and waits
    PROBE_SETTLE_S for the min-jerk segment to finish, then reads force. Returns
    (TRUE EE z at contact, mean_fz) or (None, None) at the floor with no contact.
    """
    z = z_start
    while z > z_floor + 1e-9:
        z = max(z - step, z_floor)
        servo.send_target(x, y, z, quat)     # controller latches + holds this target
        time.sleep(PROBE_SETTLE_S)           # let the min-jerk segment complete
        mean_fz, _s, n = ft.read_window(SAMPLE_S)
        pos, _ = servo.current_pose()         # true EE height right now
        contact = bool(n) and abs(mean_fz - baseline) >= threshold
        print(f"        z={pos[2]:.5f}  Fz={mean_fz:+.3f} (base {baseline:+.3f})"
              f"{'  <-- CONTACT' if contact else ''}", flush=True)
        if contact:
            return pos[2], mean_fz
    return None, None


def detect_touch(servo, ft, x, y, z_top, z_floor, quat):
    mean, std, _n = ft.read_window(BASELINE_S)
    threshold = max(FORCE_SIGMAS * std, FORCE_MIN_DELTA_N)
    step = PROBE_STEP_MM / 1000.0
    return _probe_down(servo, ft, x, y, z_top, z_floor, step, quat, mean, threshold)


def dry_run(grid, points):
    xs = [g["x"] for g in grid]
    ys = [g["y"] for g in grid]
    print(f"[DRY] {len(grid)} points | X {min(xs):.4f}..{max(xs):.4f} m "
          f"({(max(xs)-min(xs))*1000:.1f} mm) | Y {min(ys):.4f}..{max(ys):.4f} m "
          f"({(max(ys)-min(ys))*1000:.1f} mm)")
    for lbl in ("bottom-left", "bottom-right", "top-left", "top-right"):
        print(f"      taught {lbl:<12} = {points[lbl]['xyz']}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 7))
        plt.plot(xs, ys, "-o", ms=4, lw=0.6)      # line shows the serpentine path
        for lbl, m in (("bottom-left", "s"), ("bottom-right", "^"),
                       ("top-left", "v"), ("top-right", "D")):
            c = points[lbl]["xyz"]
            plt.scatter([c[0]], [c[1]], marker=m, s=80, label=lbl)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("base X (m)"); plt.ylabel("base Y (m)")
        plt.title(f"Planned probe grid + path ({COLS}x{ROWS})"); plt.legend(fontsize=7)
        plt.tight_layout()
        out = Path(__file__).with_name("surface_grid_preview.png")
        plt.savefig(out, dpi=180); plt.close()
        print(f"[DRY] grid preview -> {out}")
    except Exception as e:
        print(f"[DRY] (no plot: {e})")


def _gross_move(servo, x, y, z, quat, name="move", timeout=TRAVEL_TIMEOUT_S):
    """Command a gross Cartesian move and WAIT until the arm arrives AND stops.

    Re-publishes the target (so the controller holds it) and polls the true pose.
    Returns only once the EE is within REACH_TOL_M and moving < STOP_EPS_M/poll
    for SETTLE_POLLS in a row, then holds SETTLE_HOLD_S so the commanded velocity
    is zero before the caller issues the next target -- this is what prevents the
    joint-acceleration-discontinuity reflex at move-to-move transitions.

    Aborts if the arm stops far from the target (FROZEN_POLLS) -- that means it
    was clamped or reflex-aborted, not making progress.
    """
    print(f"[MAP] {name} -> ({x:.4f}, {y:.4f}, {z:.4f}) ...", flush=True)
    deadline = time.time() + timeout
    prev = None
    settled = 0
    best_err = float("inf")
    best_t = time.time()
    err = float("inf")
    while time.time() < deadline:
        servo.send_target(x, y, z, quat)
        time.sleep(POLL_S)
        pos, _ = servo.current_pose()
        err = max(abs(pos[0] - x), abs(pos[1] - y), abs(pos[2] - z))
        moved = max(abs(pos[i] - prev[i]) for i in range(3)) if prev else 1.0
        prev = pos
        # success: arrived AND stopped for a few polls in a row
        if err <= REACH_TOL_M and moved <= STOP_EPS_M:
            settled += 1
            if settled >= SETTLE_POLLS:
                break
        else:
            settled = 0
        # progress watchdog: a real move keeps shrinking err; a freeze does not
        if err < best_err - PROGRESS_EPS_M:
            best_err = err
            best_t = time.time()
        elif time.time() - best_t > NO_PROGRESS_S:
            break   # genuinely stuck -> err stays large -> raise below
    # hold the reached target so the commanded velocity damps fully to zero
    hold = time.time() + SETTLE_HOLD_S
    while time.time() < hold:
        servo.send_target(x, y, z, quat)
        time.sleep(POLL_S)
    if err > CLAMP_RESID_M:
        raise SystemExit(
            f"\n[ABORT] '{name}' did not reach its target (residual {err*1000:.1f} mm).\n"
            f"        The arm stopped short -- most likely a REFLEX abort (check the\n"
            f"        controller terminal for 'motion aborted by reflex') or a clamp.\n"
            f"        After a reflex you must relaunch the reach launch (or send an\n"
            f"        error-recovery) before the arm will move again. No further motion.")
    return err


def preflight_reach(servo, points, grid, hover):
    """Abort (before any motion) if any target is outside the servo's clamp sphere.

    Returns the start pose (pos, quat). Reads max_translation from the param
    server; the sphere is centred on the pose the controller started at, which is
    the arm's current pose if you launched the servo and run this right after.
    """
    import rospy
    pos0, q0 = servo.current_pose()
    cap = float(rospy.get_param("/cartesian_pose_servo_controller/max_translation", 0.30))

    cx, cy, cz = points["center"]["xyz"]
    targets = [(cx, cy, cz + CENTER_APPROACH_MM / 1000.0)]   # home -> center approach
    for g in grid:
        targets.append((g["x"], g["y"], g["z_plane"] + hover))
        targets.append((g["x"], g["y"], g["z_plane"] - MAX_DEPTH_MM / 1000.0))
    p0 = np.array(pos0, float)
    max_d = max(float(np.linalg.norm(np.array(t) - p0)) for t in targets)
    print(f"[MAP] start pose = [{pos0[0]:.4f}, {pos0[1]:.4f}, {pos0[2]:.4f}] m | "
          f"farthest target {max_d:.3f} m from start | servo cap {cap:.3f} m", flush=True)
    if max_d > cap - REACH_MARGIN_M:
        raise SystemExit(
            f"\n[ABORT] The farthest target is {max_d:.3f} m from the servo's start "
            f"pose, but the controller clamps every target to {cap:.3f} m from start.\n"
            f"        The arm cannot reach the surface from here.\n"
            f"        Fix: launch cartesian_pose_servo_reach.launch (max_translation\n"
            f"        0.55 m) from home. No motion done.")
    return pos0, q0


def approach_start(servo, points, level):
    """From HOME, go straight to a hover above the taught CENTER, in ONE settled
    move. No traverse-at-home-height, no separate big descent -- the single
    min-jerk segment starts from rest and _gross_move waits for it to finish, so
    it cannot trigger the transition reflex."""
    pos0, q0 = servo.current_pose()
    quat = flat_down_quat(q0) if level else list(q0)
    print(f"[MAP] current flange tilt from vertical: {tilt_deg(q0):.2f} deg", flush=True)
    cx, cy, cz = points["center"]["xyz"]
    _gross_move(servo, cx, cy, cz + CENTER_APPROACH_MM / 1000.0, quat,
                name="home -> hover above elastomer center")
    return quat, (cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute grid + preview only; no ROS, no motion.")
    ap.add_argument("--no-level", action="store_true",
                    help="Hold the current orientation instead of leveling flat-down.")
    ap.add_argument("--max-points", type=int, default=None,
                    help="Probe only the first N points (cautious first run).")
    ap.add_argument("--inset-mm", type=float, default=INSET_MM,
                    help="Keep probes this far in from the taught edges "
                         "(default 0 = probe corner-to-corner / cover the edges).")
    args = ap.parse_args()

    points = load_points()
    grid = build_grid(points, args.inset_mm)
    # A partial run writes to its OWN files, so it can never overwrite the real map.
    out_csv, out_png = OUT_CSV, OUT_PNG
    if args.max_points is not None:
        grid = grid[:args.max_points]
        out_csv = OUT_CSV.with_name("surface_map_partial.csv")
        out_png = OUT_PNG.with_name("surface_map_partial.png")
        print(f"[MAP] partial run ({args.max_points} pts) -> {out_csv.name} "
              f"(the full {OUT_CSV.name} will NOT be touched)", flush=True)

    if args.dry_run:
        dry_run(grid, points)
        return

    # imported here so --dry-run needs no ROS / running servo controller
    from servo_client import CartesianServo
    from franka_surface_map import WittensteinFT

    servo = CartesianServo()
    hover = HOVER_MM / 1000.0

    # Fail fast (before any motion) if the surface is outside the servo clamp.
    preflight_reach(servo, points, grid, hover)

    print("[MAP] home -> elastomer center via the servo (no manual jog) ...", flush=True)
    flat_quat, (cx, cy) = approach_start(servo, points, level=not args.no_level)

    print("[MAP] starting servo-probed raster from bottom-left ...", flush=True)
    rows = []
    try:
        with WittensteinFT(port=SERIAL_PORT) as ft:
            for pid, g in enumerate(grid):
                z_top = g["z_plane"] + hover
                z_floor = g["z_plane"] - MAX_DEPTH_MM / 1000.0
                tag = f"[{pid}] r{g['row']}c{g['col']}"
                row = {"point_id": pid, "row": g["row"], "col": g["col"],
                       "x": g["x"], "y": g["y"], "z_plane": g["z_plane"],
                       "z_touch": "", "depth_from_plane_mm": "",
                       "contact_fz": "", "status": ""}
                try:
                    # settled travel to this point's hover, then paced probe down,
                    # then ALWAYS retract straight up before moving on
                    _gross_move(servo, g["x"], g["y"], z_top, flat_quat, name=f"{tag} travel")
                    z_touch, fz = detect_touch(servo, ft, g["x"], g["y"], z_top, z_floor,
                                               flat_quat)
                    _gross_move(servo, g["x"], g["y"], z_top, flat_quat, name=f"{tag} retract")

                    if z_touch is None:
                        row["status"] = "no_contact"
                        print(f"{tag} NO CONTACT", flush=True)
                    else:
                        row["status"] = "ok"
                        row["z_touch"] = z_touch
                        row["depth_from_plane_mm"] = (g["z_plane"] - z_touch) * 1000.0
                        row["contact_fz"] = fz
                        print(f"{tag} z_touch={z_touch:.5f} "
                              f"({row['depth_from_plane_mm']:+.2f} mm)  Fz={fz:+.3f}",
                              flush=True)
                except Exception as e:
                    row["status"] = "error"
                    print(f"{tag} ERROR: {e} -- continuing to next point.", flush=True)
                rows.append(row)

        # park back over the taught center
        _gross_move(servo, cx, cy, points["center"]["xyz"][2] + CENTER_APPROACH_MM / 1000.0,
                    flat_quat, name="park above center")
    finally:
        # NEVER clobber a good map with nothing: an early abort (e.g. reflex) leaves
        # rows empty, and a --max-points run only has a few. Partial runs go to their
        # own file so the real surface_map.csv is untouched.
        if not rows:
            print("[MAP] no points probed -- leaving %s untouched." % OUT_CSV.name, flush=True)
        else:
            save_csv(rows, out_csv)
            save_heatmap(rows, out_png)
            ok = sum(1 for r in rows if r["status"] == "ok")
            print(f"[MAP] {ok}/{len(rows)} points mapped -> {out_csv.name}", flush=True)


if __name__ == "__main__":
    main()
