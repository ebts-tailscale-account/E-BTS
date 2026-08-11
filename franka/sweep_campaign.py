#!/usr/bin/env python3
"""Grid x depth-ladder indentation campaign -- the ML training-data collector.

Invoked ON TACTILE by master_campaign.py (so the camera + Wittenstein F/T are
recorded on the workstation over the same window).

WHY THIS EXISTS (see HANDOFF.md section 14)
-------------------------------------------
franka_grid_logger.py's 99-point sweep is NOT a trainable dataset: every poke
commanded the SAME 2.0 mm, so force was never an independent variable. It varied
only through local stiffness, surface-map error and impedance sag -- a ~1 N window
that a robot-only model already predicts to 0.131 N RMSE with no camera at all.

This script fixes that by crossing a spatial grid with a DEPTH LADDER, so force
spans a real range at every location and is decoupled from position.

THREE DESIGN DECISIONS THAT MATTER
----------------------------------
1. SURFACE REFERENCE = the bilinear plane over the four HAND-KISSED corners
   (corner_joints.json), NOT surface_map.csv's z_touch.

   z_touch is force-probed and sits BELOW the true surface (HANDOFF section 12.5).
   The evidence is quantitative: fitting F = k*d^n to the dip ramps gives n = 0.72
   against run1's z_touch reference but n = 1.63 against mid5's hand-kissed
   reference. A sub-linear exponent is physically impossible for a cone indenter
   (Sneddon gives 2.0, Hertz sphere 1.5), so the 0.72 is the signature of a shifted
   depth origin, not real mechanics. Kissed corners are the trustworthy datum.
   --surface-map exists as an escape hatch but is NOT the default, deliberately.

2. DEPTH ORDER IS SHUFFLED WITHIN EACH LOCATION (seeded, reproducible).
   Sweeping the ladder monotonically would confound depth with viscoelastic creep
   and F/T drift. Shuffling per location spreads every depth level uniformly over
   the whole campaign while keeping travel serpentine, so the arm still walks the
   grid instead of teleporting (--order random forces full randomisation if you
   want it and can afford the extra travel).

3. THE 93-COLUMN LOG SCHEMA IS UNTOUCHED, so postprocess_indent.py works as-is.
   point_index is a globally unique indent counter (0..N-1) -- that is what
   postprocess_indent.py segments on. The depth level for each indent lives in the
   campaign_plan.csv sidecar, keyed by point_index. Adding a depth column to the
   log would have broken both post-processors for no gain.

Depth is a LABEL here, not a model input. Feeding indent depth to a force model
hands it 43% of the answer through a channel unrelated to the camera (section 14.1d).

Does NOT read the F/T -- the HEX21 stays on the WORKSTATION for the recording.

Prereq (on tactile), start from HOME with the REACH launch:
    source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
    roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
        robot_ip:=10.1.196.5 load_gripper:=false

Usage:
    python3 sweep_campaign.py --dry-run              # plan + geometry, no ROS
    python3 sweep_campaign.py --max-points 6         # cautious first run
    python3 sweep_campaign.py                        # the full campaign
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import rospy
from franka_msgs.msg import FrankaState

import panda_fk as pfk
import map_surface as m
from home_and_level import flat_down_quat, tilt_deg
from servo_client import CartesianServo
from franka_grid_logger import (Ctx, FrankaLogger, clear_reflex, warn_if_loaded, hold,
                                dip_to_depth, TARE_S)

CORNERS_FILE = Path(__file__).with_name("corner_joints.json")

# --- campaign geometry -------------------------------------------------------
# The elastomer is ~30 x 36 mm (HANDOFF section 5) and is STRAPPED AT THE SIDES. We
# stay inside a margin so the cone never indents near a strapped edge, where the
# support carries load directly (run1 measured 1.05-1.28 N there vs 0.6-0.8 N in the
# middle) and the deformation field is clipped by the boundary.
SPAN_U_MM = 24.0     # across the ~30 mm axis (Y in base frame, "col" direction)
SPAN_V_MM = 32.0     # across the ~36 mm axis (X in base frame, "row" direction)
PITCH_MM  = 3.0      # grid pitch. The marker pitch is 2.32 mm (section 14.3), so
                     # 3 mm slightly under-samples and 2 mm slightly over-samples.

# --- depth ladder ------------------------------------------------------------
# |Fz| from the MEASURED centre curve (depth_limit_probe, 2026-08-07, 0.4-4.5 mm):
#     F = 0.496*d + 0.090   R2 = 0.994   -- essentially LINEAR in depth
#   0.5 -> 0.34 N   1.0 -> 0.59 N   1.5 -> 0.83 N
#   2.0 -> 1.08 N   2.5 -> 1.33 N   3.0 -> 1.58 N
# Two things that measurement overturned, both worth not re-deriving:
#   * stiffness SOFTENS with depth (0.63 -> 0.37 N/mm, r = -0.826). The earlier
#     guess of membrane STIFFENING was wrong -- an unbacked sheet deflects bodily,
#     so incremental resistance drops rather than climbing.
#   * force is nowhere near limiting: 2.24 N at 4.5 mm, and ~3.7 N extrapolated at
#     8 mm is still under half the 8 N probe ceiling.
# Because F is linear in d, an evenly spaced DEPTH ladder gives an evenly spaced
# FORCE ladder -- which is exactly what a training set wants.
# The ladder can therefore go much deeper than this default once --map has set a
# per-location ceiling; the binding limit is OPTICAL (markers leaving frame), not
# force, and the hard stop is the side straps (~10 mm), not the material.
DEPTHS_MM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MAX_SAFE_DEPTH_MM = 4.0   # hard refusal above this without --i-have-run-the-probe.
                          # Raised 3.0 -> 4.0 on 2026-08-07: the 9-node --map probe HAS
                          # been run, and pilot_20260807_134855 then completed 648
                          # presses at 1.5-4.0 mm with zero incidents (max force 5.9 N,
                          # 74% of the 8 N probe ceiling). 4.0 mm is demonstrated, not
                          # extrapolated. Past it, the probe is still the gate.

HOVER_MM = 5.0       # travel / retract / tare at this height above the surface
DWELL_S  = 2.0       # hold the indent this long (the measurement window)
APPROACH_MM = 15.0   # from home, first go this far above the taught centre
SEED = 20260806


def load_points():
    if not CORNERS_FILE.exists():
        sys.exit("[ERROR] %s not found -- teach the corners first (record_corners.py)."
                 % CORNERS_FILE.name)
    data = json.loads(CORNERS_FILE.read_text())
    return data.get("points") or data["corners"]


def build_grid(points, span_u_mm, span_v_mm, pitch_mm, n_u=None, n_v=None,
               margins=None):
    """Bilinear grid over the 4 kissed corners, serpentine, centred on the block.

    u runs BL->BR (the ~30 mm axis, indexed by `col`);
    v runs BL->TL (the ~36 mm axis, indexed by `row`).
    Returns (grid, info) where info records the achieved spans and pitches -- the
    requested span rarely divides exactly by the pitch, so the ACHIEVED pitch is
    what gets reported and stored, never the requested one.

    n_u / n_v override the pitch-derived counts. depth_limit_probe.py uses
    n_u=n_v=3 to get centre + 4 corners + 4 edge midpoints on EXACTLY this inset
    rectangle -- sharing this function is what guarantees the depth-limit map and
    the campaign grid refer to the same piece of silicone.
    """
    BL = np.array(points["bottom-left"]["xyz"], float)
    TL = np.array(points["top-left"]["xyz"], float)
    TR = np.array(points["top-right"]["xyz"], float)
    BR = np.array(points["bottom-right"]["xyz"], float)

    u_len_mm = np.linalg.norm(BR - BL) * 1e3
    v_len_mm = np.linalg.norm(TL - BL) * 1e3
    for name, have, want in (("u", u_len_mm, span_u_mm), ("v", v_len_mm, span_v_mm)):
        if want > have:
            sys.exit("[ERROR] requested %s span %.1f mm exceeds the taught edge "
                     "%.1f mm." % (name, want, have))

    # PER-EDGE margins. The four taught corners are NOT equidistant from the physical
    # frame: on 2026-08-07 the indenter came close to a plastic wall on the v=1 side
    # while the other three edges were clear. A symmetric inset cannot express that,
    # so margins may be given per edge as (u_lo, u_hi, v_lo, v_hi) in mm.
    #
    # And the clearance a given edge needs SCALES WITH DEPTH: the indenter is a cone,
    # so its cross-section at the silicone surface grows as d*tan(theta). Measured
    # from the three wall-limited nodes, theta ~ 23-32 deg, i.e. clearance ~ 0.5*d.
    # A 2 mm margin therefore only supports ~4 mm of depth next to a wall.
    if margins is not None:
        mu_lo, mu_hi, mv_lo, mv_hi = [float(v) for v in margins]
        span_u_mm = u_len_mm - mu_lo - mu_hi
        span_v_mm = v_len_mm - mv_lo - mv_hi
        if span_u_mm <= 0 or span_v_mm <= 0:
            sys.exit("[ERROR] margins %s leave no area inside the %.1f x %.1f mm "
                     "taught quad." % (list(margins), u_len_mm, v_len_mm))
    else:
        mu_lo = mu_hi = (u_len_mm - span_u_mm) / 2.0
        mv_lo = mv_hi = (v_len_mm - span_v_mm) / 2.0

    n_u = int(n_u) if n_u else max(2, int(round(span_u_mm / pitch_mm)) + 1)
    n_v = int(n_v) if n_v else max(2, int(round(span_v_mm / pitch_mm)) + 1)
    us = np.linspace(mu_lo / u_len_mm, 1.0 - mu_hi / u_len_mm, n_u)
    vs = np.linspace(mv_lo / v_len_mm, 1.0 - mv_hi / v_len_mm, n_v)

    def bilinear(u, v):
        return ((1 - u) * (1 - v) * BL + u * (1 - v) * BR +
                (1 - u) * v * TL + u * v * TR)

    grid = []
    for i, v in enumerate(vs):
        cols = range(n_u) if i % 2 == 0 else range(n_u - 1, -1, -1)
        for j in cols:
            p = bilinear(us[j], v)
            grid.append({"row": i, "col": j, "x": float(p[0]),
                         "y": float(p[1]), "z_plane": float(p[2])})
    info = {
        "taught_u_len_mm": float(u_len_mm), "taught_v_len_mm": float(v_len_mm),
        "span_u_mm": float(span_u_mm), "span_v_mm": float(span_v_mm),
        "margin_u_mm": float(mu_lo), "margin_v_mm": float(mv_lo),
        "margin_u_lo_mm": float(mu_lo), "margin_u_hi_mm": float(mu_hi),
        "margin_v_lo_mm": float(mv_lo), "margin_v_hi_mm": float(mv_hi),
        "edge_xyz": {
            "u_lo (BL/TL side)": [float(v) for v in BL],
            "u_hi (BR/TR side)": [float(v) for v in BR],
            "v_lo (BL/BR side)": [float(v) for v in BL],
            "v_hi (TL/TR side)": [float(v) for v in TL],
        },
        "n_cols": n_u, "n_rows": n_v,
        "achieved_pitch_u_mm": float(span_u_mm / (n_u - 1)),
        "achieved_pitch_v_mm": float(span_v_mm / (n_v - 1)),
        "z_plane_range_mm": float((max(g["z_plane"] for g in grid) -
                                   min(g["z_plane"] for g in grid)) * 1e3),
    }
    return grid, info


def parse_margins(vals):
    """1, 2 or 4 numbers -> (u_lo, u_hi, v_lo, v_hi) in mm."""
    if vals is None:
        return None
    v = [float(x) for x in vals]
    if len(v) == 1:
        return (v[0], v[0], v[0], v[0])
    if len(v) == 2:
        return (v[0], v[0], v[1], v[1])
    if len(v) == 4:
        return tuple(v)
    sys.exit("[ERROR] --margin-mm takes 1, 2 or 4 values, got %d." % len(v))


def print_edges(info):
    """Show each edge's clearance NEXT TO the physical coordinate it corresponds to.

    Without this you cannot tell which number to change when the indenter nearly hits
    something -- the u/v naming is meaningless at the robot.
    """
    ex = info.get("edge_xyz", {})
    print("  edge clearances from the taught quad:")
    for key, mm in (("u_lo (BL/TL side)", info.get("margin_u_lo_mm")),
                    ("u_hi (BR/TR side)", info.get("margin_u_hi_mm")),
                    ("v_lo (BL/BR side)", info.get("margin_v_lo_mm")),
                    ("v_hi (TL/TR side)", info.get("margin_v_hi_mm"))):
        xyz = ex.get(key)
        loc = ("  at x %.4f, y %+.4f" % (xyz[0], xyz[1])) if xyz else ""
        print("     %-18s %5.2f mm%s" % (key, mm if mm is not None else float("nan"), loc))
    print("     (cone widens ~0.5*depth: %.1f mm clearance supports ~%.1f mm of indent)"
          % (min(v for v in (info.get("margin_u_lo_mm"), info.get("margin_u_hi_mm"),
                             info.get("margin_v_lo_mm"), info.get("margin_v_hi_mm"))
                 if v is not None),
             2.0 * min(v for v in (info.get("margin_u_lo_mm"), info.get("margin_u_hi_mm"),
                                   info.get("margin_v_lo_mm"), info.get("margin_v_hi_mm"))
                       if v is not None)))


def build_plan(grid, depths_mm, repeats, order, seed):
    """Cross the grid with the depth ladder -> the ordered list of indents.

    Every entry gets a unique point_index, which is the key the log and the
    campaign_plan.csv sidecar share.
    """
    rng = random.Random(seed)
    plan = []
    if order == "random":
        for g in grid:
            for d in depths_mm:
                for r in range(repeats):
                    plan.append((g, d, r))
        rng.shuffle(plan)
    else:  # serpentine locations, shuffled depth ladder within each location
        for g in grid:
            ladder = [(d, r) for d in depths_mm for r in range(repeats)]
            rng.shuffle(ladder)
            plan.extend((g, d, r) for d, r in ladder)

    return [{"point_index": i, "row": g["row"], "col": g["col"],
             "x": g["x"], "y": g["y"], "surface_z": g["z_plane"],
             "depth_mm": d, "repeat": r}
            for i, (g, d, r) in enumerate(plan)]


def save_plan(plan, info, path):
    fields = ["point_index", "row", "col", "x", "y", "surface_z", "depth_mm", "repeat"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in plan:
            w.writerow({k: p[k] for k in fields})
    Path(str(path) + ".json").write_text(json.dumps(info, indent=2))


def report(grid, info, plan, depths_mm, repeats, order, args):
    est_s = len(plan) * 8.0          # ~8 s/indent measured with depth correction on
    print("=" * 74)
    print(" INDENTATION CAMPAIGN  (grid x depth ladder)")
    print("=" * 74)
    print("  taught quad          : %.1f x %.1f mm (u x v)"
          % (info["taught_u_len_mm"], info["taught_v_len_mm"]))
    print("  explored span        : %.1f x %.1f mm"
          % (info["span_u_mm"], info["span_v_mm"]))
    print_edges(info)
    print("  grid                 : %d cols x %d rows = %d locations"
          % (info["n_cols"], info["n_rows"], len(grid)))
    print("  achieved pitch       : %.2f mm (u) x %.2f mm (v)   [marker pitch 2.32 mm]"
          % (info["achieved_pitch_u_mm"], info["achieved_pitch_v_mm"]))
    print("  surface tilt over it : %.2f mm (bilinear over the kissed corners)"
          % info["z_plane_range_mm"])
    print("  depth ladder         : %s mm" % ", ".join("%.1f" % d for d in depths_mm))
    print("  repeats per (loc,dep): %d" % repeats)
    print("  order                : %s" % order)
    print("  TOTAL INDENTS        : %d" % len(plan))
    print("  estimated robot time : %.0f min (%.1f h) at ~8 s/indent"
          % (est_s / 60.0, est_s / 3600.0))
    print("  estimated .raw size  : %.0f GB at 3.97 MB/s" % (est_s * 3.97e6 / 1e9))
    print("  hover / dwell / tare : %.1f mm / %.1f s / %.1f s per indent"
          % (args.hover_mm, args.dwell_s, args.tare_s))
    print("  tare                 : EVERY indent holds still out of contact at hover")
    print("                         for %.1f s BEFORE its dip (phase=\"tare\"), so each"
          % args.tare_s)
    print("                         one is zeroed on its OWN baseline, not a global one.")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", nargs="?",
                    default=str(Path.home() / "E-BTS/recordings/campaign_franka.csv"))
    ap.add_argument("--span-mm", type=float, nargs=2, default=[SPAN_U_MM, SPAN_V_MM],
                    metavar=("U", "V"),
                    help="explored span in mm: U across the ~30 mm axis, V across "
                         "the ~36 mm axis (default 24 32)")
    ap.add_argument("--margin-mm", type=float, nargs="+", default=None,
                    help="Clearance from the taught quad, in mm. 1 value = all four "
                         "edges; 2 = (u, v) symmetric per axis; 4 = (u_lo, u_hi, "
                         "v_lo, v_hi) per edge. Overrides --span-mm. NOTE the clearance "
                         "an edge needs SCALES WITH DEPTH -- the cone widens as "
                         "~0.5*depth, so 2 mm only supports ~4 mm of indent next to a "
                         "wall (measured 2026-08-07).")
    ap.add_argument("--pitch-mm", type=float, default=PITCH_MM)
    ap.add_argument("--depths-mm", type=float, nargs="+", default=DEPTHS_MM)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats of each (location, depth) pair")
    ap.add_argument("--order", choices=["serpentine", "random"], default="serpentine",
                    help="serpentine = walk the grid, shuffle the ladder per location "
                         "(default); random = fully shuffle every indent (more travel)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--hover-mm", type=float, default=HOVER_MM)
    ap.add_argument("--dwell-s", type=float, default=DWELL_S)
    ap.add_argument("--tare-s", type=float, default=TARE_S,
                    help="Hold still OUT OF CONTACT this long before EVERY dip "
                         "(default %.1f s). This window is what postprocess_indent.py "
                         "uses to zero that indent's F/T on its own baseline, so it "
                         "must be long enough to average the ~0.116 N Wittenstein "
                         "noise down: 1.0 s is ~1000 samples -> ~0.004 N." % TARE_S)
    ap.add_argument("--log-hz", type=float, default=200.0)
    ap.add_argument("--max-points", type=int, default=None,
                    help="run only the first N indents of the plan (cautious run)")
    ap.add_argument("--no-level", action="store_true")
    ap.add_argument("--no-recovery", action="store_true")
    ap.add_argument("--no-depth-correction", action="store_true",
                    help="skip dip_to_depth's closed loop (HANDOFF section 4.11)")
    ap.add_argument("--surface-map", default=None,
                    help="ESCAPE HATCH: take surface_z from a surface_map.csv "
                         "z_touch column instead of the kissed-corner plane. This "
                         "reference is known to sit below the true surface -- see "
                         "the module docstring. Not recommended.")
    ap.add_argument("--i-have-run-the-probe", action="store_true",
                    help="permit depths beyond %.1f mm (only after "
                         "depth_limit_probe.py)" % MAX_SAFE_DEPTH_MM)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and write the sidecar; no ROS, no motion")
    args = ap.parse_args()

    depths = sorted(args.depths_mm)
    if min(depths) <= 0:
        sys.exit("[ERROR] depths must be > 0 mm.")
    if max(depths) > MAX_SAFE_DEPTH_MM and not args.i_have_run_the_probe:
        sys.exit("[ERROR] deepest requested indent %.1f mm exceeds the %.1f mm "
                 "demonstrated limit.\n"
                 "        %.1f mm is vetted by 648 presses in pilot_20260807_134855 "
                 "(max force 5.9 N).\n        Beyond that the force is extrapolated: "
                 "the measured centre curve F = 0.451*d + 0.099\n        covers 0.4-10 mm "
                 "at ONE location and stiffness varies 9.4x across the block.\n"
                 "        Re-run depth_limit_probe.py --map at the deeper range, then "
                 "pass --i-have-run-the-probe."
                 % (max(depths), MAX_SAFE_DEPTH_MM, MAX_SAFE_DEPTH_MM))

    points = load_points()
    margins = parse_margins(args.margin_mm)
    grid, info = build_grid(points, args.span_mm[0], args.span_mm[1], args.pitch_mm,
                            margins=margins)

    if args.surface_map:
        smap = m.read_surface_map(args.surface_map) if hasattr(m, "read_surface_map") \
            else None
        if smap is None:
            from franka_grid_logger import read_surface_map as _rsm
            smap = _rsm(args.surface_map)
        print("[WARN] using surface_map z_touch as the datum -- known to sit BELOW "
              "the true surface (see the module docstring).")
        for g in grid:
            key = (g["col"], g["row"])
            if key in smap:
                g["z_plane"] = smap[key]
        info["surface_reference"] = "surface_map z_touch (%s)" % args.surface_map
    else:
        info["surface_reference"] = "bilinear plane over the 4 hand-kissed corners"

    plan = build_plan(grid, depths, args.repeats, args.order, args.seed)
    if args.max_points is not None:
        plan = plan[:args.max_points]
    info.update({"depths_mm": depths, "repeats": args.repeats, "order": args.order,
                 "seed": args.seed, "n_indents": len(plan),
                 "hover_mm": args.hover_mm, "dwell_s": args.dwell_s,
                 "tare_s": args.tare_s,
                 "depth_correction": not args.no_depth_correction})

    report(grid, info, plan, depths, args.repeats, args.order, args)

    # FIXED NAME, matching what master_campaign.py scp's back. The old derived name
    # (stem.replace("_franka","") + "_plan.csv") produced "franka_plan.csv" when the
    # out_csv was ".../franka.csv" -- so master looked for campaign_plan.csv and the
    # depth labels silently stayed on tactile. Never derive a filename that another
    # program has to guess: pin it.
    plan_path = Path(args.out_csv).with_name("campaign_plan.csv")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    save_plan(plan, info, plan_path)
    print("Plan written to %s (+ .json)" % plan_path)

    if args.dry_run:
        print("\n[DRY RUN] no ROS, no motion.")
        return

    hover = args.hover_mm / 1000.0

    rospy.init_node("sweep_campaign", disable_signals=True)
    if not args.no_recovery:
        clear_reflex()
    warn_if_loaded()

    servo = CartesianServo()

    # Reach preflight BEFORE any motion, over the true extremes of THIS plan.
    saved_depth, saved_approach = m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM
    m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = max(depths), APPROACH_MM
    try:
        m.preflight_reach(servo, points, grid, hover)
    finally:
        m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = saved_depth, saved_approach

    # FK self-check against the LIVE arm: catches a changed tool/EE transform since
    # the corners were taught, which would bias every target.
    st = rospy.wait_for_message("/franka_state_controller/franka_states",
                                FrankaState, timeout=5.0)
    pfk.self_check(st.q, st.O_T_EE, tol_mm=1.0, label="live")

    ctx = Ctx()
    logger = FrankaLogger(args.out_csv, ctx, log_hz=args.log_hz)  # log BEFORE homing
    print("Logging franka_states to %s | %d indents" % (args.out_csv, len(plan)))

    achieved, t_start = [], time.time()
    try:
        ctx.set("home")
        pos0, q0 = servo.current_pose()
        quat = flat_down_quat(q0) if not args.no_level else list(q0)
        print("[CAMPAIGN] flange tilt from vertical: %.2f deg" % tilt_deg(q0))
        cx, cy, cz = points["center"]["xyz"]
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat,
                      name="home -> hover above elastomer centre")

        for n, p in enumerate(plan):
            x, y, sz, d_mm = p["x"], p["y"], p["surface_z"], p["depth_mm"]
            hover_z = sz + hover
            el = time.time() - t_start
            eta = el / n * (len(plan) - n) if n else 0.0
            print("[%4d/%4d] r%-2d c%-2d  depth %.1f mm   elapsed %.0f min, ETA %.0f min"
                  % (n + 1, len(plan), p["row"], p["col"], d_mm, el / 60, eta / 60),
                  flush=True)

            ctx.set("travel", idx=p["point_index"], col=p["col"], row=p["row"],
                    target=(x, y, hover_z), surface_z=sz)
            m._gross_move(servo, x, y, hover_z, quat, name="travel")

            # TARE: hold still, OUT OF CONTACT -- this indent's own F/T zero AND the
            # undeformed reference frame the CNN/marker model differences against.
            ctx.set("tare", target=(x, y, hover_z))
            hold(servo, x, y, hover_z, quat, args.tare_s)

            ctx.set("dip", target=(x, y, sz - d_mm / 1000.0))
            got, cmd_z, iters = dip_to_depth(servo, x, y, sz, d_mm / 1000.0, quat,
                                             correct=not args.no_depth_correction)
            achieved.append((d_mm, got * 1e3))

            ctx.set("dwell", target=(x, y, cmd_z))
            hold(servo, x, y, cmd_z, quat, args.dwell_s)

            ctx.set("retract", target=(x, y, hover_z))
            m._gross_move(servo, x, y, hover_z, quat, name="retract")

        ctx.set("park")
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat,
                      name="park above centre")
        ctx.set("done")
    finally:
        logger.close()
        print("\nLogged %d franka_states samples to %s" % (logger.count, args.out_csv))
        if achieved:
            a = np.array(achieved)
            print("Depth accuracy per commanded level (commanded -> achieved):")
            for d in sorted(set(a[:, 0])):
                got = a[a[:, 0] == d, 1]
                print("  %.1f mm : mean %.3f  min %.3f  max %.3f  (n=%d)"
                      % (d, got.mean(), got.min(), got.max(), len(got)))
            print("Total wall time: %.1f min" % ((time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
