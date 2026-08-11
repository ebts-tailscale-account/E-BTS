#!/usr/bin/env python3
"""
Retry specific surface-map points that came back NO_CONTACT and MERGE the results
back into surface_map.csv (+ regenerate the heatmap).

Why this exists: the two right-edge CORNERS (point_id 8 = r0c8, 98 = r10c8) droop
more than the map's 2 mm depth floor, so the probe bottomed out before the force
rose enough. This re-probes just those points with a DEEPER floor. Everything
else (servo motion, settling, paced probe, force detection, reach preflight) is
imported from map_surface.py, so behaviour is identical -- only the floor changes.

Prereqs are the SAME as map_surface.py (on tactile):
    roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
        robot_ip:=10.1.196.5 load_gripper:=false
    # HEX21 plugged into tactile; arm at home (or anywhere -- it goes to center first)

Usage:
    python3 retry_points.py                 # retry every non-'ok' row in surface_map.csv, deeper
    python3 retry_points.py --points 8,98   # retry specific point_ids
    python3 retry_points.py --max-depth-mm 5 --points 8,98
"""

import argparse
import csv

import map_surface as m


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", default=None,
                    help="comma-separated point_ids to retry (default: all non-'ok' rows).")
    ap.add_argument("--max-depth-mm", type=float, default=4.0,
                    help="deeper hard floor for the retry (default 4; map used 2).")
    ap.add_argument("--inset-mm", type=float, default=m.INSET_MM,
                    help="must match the inset the original map was built with (default 0).")
    ap.add_argument("--no-level", action="store_true",
                    help="hold current orientation instead of leveling flat-down.")
    args = ap.parse_args()

    points = m.load_points()
    grid = m.build_grid(points, args.inset_mm)
    rows = read_rows(m.OUT_CSV)
    if len(grid) != len(rows):
        print(f"[WARN] grid ({len(grid)}) != surface_map.csv ({len(rows)}) length -- "
              f"make sure --inset-mm / ROWS / COLS match the original map.", flush=True)

    if args.points:
        want = sorted(int(x) for x in args.points.split(","))
    else:
        want = sorted(int(r["point_id"]) for r in rows if r["status"] != "ok")
    if not want:
        print("Nothing to retry (every row is 'ok').")
        return
    bad = [p for p in want if p < 0 or p >= len(grid)]
    if bad:
        raise SystemExit(f"[ERROR] point_id(s) out of range: {bad}")
    print(f"[RETRY] point_ids {want} with a -{args.max_depth_mm:.1f} mm floor "
          f"(map used -{m.MAX_DEPTH_MM:.1f} mm).", flush=True)

    from servo_client import CartesianServo
    from franka_surface_map import WittensteinFT
    servo = CartesianServo()

    # reach preflight against just the points we will visit
    m.preflight_reach(servo, points, [grid[i] for i in want], m.HOVER_MM / 1000.0)
    print("[RETRY] home -> elastomer center ...", flush=True)
    flat_quat, (cx, cy) = m.approach_start(servo, points, level=not args.no_level)

    hover = m.HOVER_MM / 1000.0
    depth = args.max_depth_mm / 1000.0
    updated = {}
    try:
        with WittensteinFT(port=m.SERIAL_PORT) as ft:
            for pid in want:
                g = grid[pid]
                z_top = g["z_plane"] + hover
                z_floor = g["z_plane"] - depth          # DEEPER than the map
                tag = f"[{pid}] r{g['row']}c{g['col']}"
                try:
                    m._gross_move(servo, g["x"], g["y"], z_top, flat_quat, name=f"{tag} travel")
                    z_touch, fz = m.detect_touch(servo, ft, g["x"], g["y"], z_top, z_floor, flat_quat)
                    m._gross_move(servo, g["x"], g["y"], z_top, flat_quat, name=f"{tag} retract")
                    if z_touch is None:
                        print(f"{tag} STILL NO CONTACT at -{args.max_depth_mm:.1f} mm", flush=True)
                        updated[pid] = None
                    else:
                        dmm = (g["z_plane"] - z_touch) * 1000.0
                        print(f"{tag} z_touch={z_touch:.5f} ({dmm:+.2f} mm)  Fz={fz:+.3f}", flush=True)
                        updated[pid] = (z_touch, dmm, fz)
                except Exception as e:
                    print(f"{tag} ERROR: {e}", flush=True)
            m._gross_move(servo, cx, cy, points["center"]["xyz"][2] + m.CENTER_APPROACH_MM / 1000.0,
                          flat_quat, name="park above center")
    finally:
        # merge the retried points back into the existing rows
        for r in rows:
            pid = int(r["point_id"])
            if pid not in updated:
                continue
            u = updated[pid]
            if u is None:
                r["status"], r["z_touch"], r["depth_from_plane_mm"], r["contact_fz"] = \
                    "no_contact", "", "", ""
            else:
                z_touch, dmm, fz = u
                r["status"], r["z_touch"], r["depth_from_plane_mm"], r["contact_fz"] = \
                    "ok", z_touch, dmm, fz
        # coerce types so save_csv / save_heatmap are happy, then write both
        typed = []
        for r in rows:
            t = dict(r)
            t["row"], t["col"] = int(float(r["row"])), int(float(r["col"]))
            if r["status"] == "ok" and r["z_touch"] not in ("", None):
                t["z_touch"] = float(r["z_touch"])
            typed.append(t)
        m.save_csv(typed)
        m.save_heatmap(typed)
        ok = sum(1 for r in typed if r["status"] == "ok")
        print(f"[RETRY] merged into {m.OUT_CSV.name}: {ok}/{len(typed)} points ok.", flush=True)


if __name__ == "__main__":
    main()
