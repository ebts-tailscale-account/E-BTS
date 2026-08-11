#!/usr/bin/env python3
"""FORCE-LIMITED depth probe: how deep can we actually indent, and at what force?

Answers one question before the campaign ladder is allowed past 3 mm:
    what does F(depth) do at 4 mm and 5 mm?

HOW THE ELASTOMER IS MOUNTED (this drives everything below)
-----------------------------------------------------------
The silicone is ~5 mm thick and STRAPPED AT THE SIDES, with NOTHING UNDERNEATH.
It is an edge-supported sheet, not a layer bonded to a rigid backing. So:
  * there is NO substrate to bottom out on -- depth is not limited by thickness,
  * the practical limit is the STRAPS, estimated to give way around ~10 mm,
  * but the sheet still stiffens at large deflection, by MEMBRANE TENSION (it
    stretches) rather than by confinement.
Corroboration: run1's force map is stiff near the edges (col 0 ~ 1.05-1.28 N) and
soft in the middle (~0.6-0.8 N), which is the signature of edge support carrying
load directly near the straps.

WHY THIS IS NOT JUST "RUN sweep_campaign.py --depths-mm 4 5"
------------------------------------------------------------
Not because of a backing -- there isn't one. Because of EXTRAPOLATION.

Fitting F = k*d^n to mid5_center's dip ramps gives n = 1.63, k = 0.25 (between a
Hertz sphere at 1.5 and a Sneddon cone at 2.0 -- consistent with the rounded plastic
cone). Extrapolating that:  3 mm -> 1.5 N,  4 mm -> 2.5 N,  5 mm -> 3.6 N.

⚠ That fit is FIVE REPEATS AT ONE LOCATION (the centre) OVER 0.3-2.0 mm ONLY.
Quoting it at 4-5 mm extrapolates 2.5x beyond its data, into a regime where the
response changes character: local cone-contact mechanics gives way to whole-sheet
membrane stretching, which has a different exponent. The measured k also varies with
location -- run1 saw 0.45-1.46 N at nominally the same commanded depth.

We could not pin the transition from existing data either: the dip ramp reaches full
depth in ~1 s, so the shallow depth bands hold too few samples to fit a local
exponent (only 1.5-2.0 mm had enough, n = 1.58). Hence: MEASURE IT. It costs ten
minutes and it is the difference between a known and an assumed force range.

The servo is force-blind (HANDOFF section 4.11: "force-blind" means it never aborts
on contact), the only backstop is the ~20 N collision reflex, and the HEX21's own
range is not documented in this repo. So this script descends in SMALL STEPS and
reads the force BETWEEN every step, aborting on the first of:
    force ceiling (--max-force-n, default 8 N)
    stiffness blow-up (dF/dz over --max-stiffness-n-per-mm, default 15 N/mm --
        here that means membrane stiffening or a strap starting to load up, NOT
        backing contact)
    depth ceiling (--max-depth-mm)
and then retracts immediately.

FORCE FEEDBACK -- READ THIS
---------------------------
  --source hex21  (default, RECOMMENDED) needs the HEX21 MOVED TO TACTILE, exactly
                  as for surface mapping (HANDOFF section 3.2). It is the ground
                  truth and it is what the abort logic deserves.
  --source franka uses the robot's own O_F_ext_hat_K. That signal is BIASED by
                  +-2 N and pose-dependent (section 3.1), so it is a COARSE guard
                  only -- fine to catch a runaway, useless as a measurement. If you
                  use it, raise nothing and trust nothing below ~3 N.

This script does NOT record the camera. It is a mechanical characterisation, not a
dataset run -- so it needs no GUI, no master_*.py, no clock sync.

Prereq (on tactile), arm at HOME, REACH launch up:
    source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
    roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
        robot_ip:=10.1.196.5 load_gripper:=false

Usage:
    python3 depth_limit_probe.py --dry-run
    python3 depth_limit_probe.py                       # centre, to 5 mm or 8 N
    python3 depth_limit_probe.py --max-depth-mm 4      # stop at 4 mm
    python3 depth_limit_probe.py --source franka --max-force-n 5
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rospy

import map_surface as m
from home_and_level import flat_down_quat, tilt_deg

CORNERS_FILE = Path(__file__).with_name("corner_joints.json")
OUT_CSV = Path.home() / "E-BTS/depth_limit_probe.csv"

MAX_DEPTH_MM = 15.0         # COMMANDED ceiling. Raised 5.0 -> 7.0 -> 15.0 on
                            # 2026-08-07 at the operator\'s direction: both earlier
                            # ceilings stopped the probe while force was still tiny
                            # (2.24 N at 4.5 mm = 28% of the 8 N limit), so they were
                            # the binding constraint rather than anything physical.
                            # ⚠ MIND THE SAG: sag = 0.258*F - 0.148 mm (measured), so
                            # ~1.6 mm at 6.7 N. Commanding 15 ACHIEVES ~13.4; command
                            # ~16.6 to achieve 15.
PROMPT_FROM_MM = 3.0        # auto-descend (backstops live) to here, THEN start asking.
                            # Grounded in the 2026-08-07 centre probe, which showed no
                            # optical trouble to 4.5 mm. The centre is the WORST case
                            # optically -- it is the most compliant point, so it
                            # deforms most for a given depth; edge nodes go out of
                            # frame deeper, not shallower. Set 0 to prompt from step 1.
MAX_FORCE_N = 8.0           # abort the descent at this force
MAX_STIFFNESS_N_PER_MM = 15.0   # abort if the local slope blows up (membrane
                                # stiffening / a strap loading up)
STEP_MM = 0.25              # descend in these increments, reading force between each
SETTLE_S = 0.8              # > controller min_duration (0.5 s) so each step settles
SAMPLE_S = 0.25             # force averaging window per step
BASELINE_S = 1.0            # out-of-contact zero, taken at hover
STRAP_LIMIT_MM = 20.0       # hard refusal -- a typo guard (--max-depth-mm 150),
                            # NOT a physical model. ⚠ The operator originally
                            # estimated the side straps give way near 10 mm and then
                            # directed that the ceiling go to 15 mm; that judgement is
                            # theirs (they can see the rig) and this cap was raised to
                            # permit it. Override with --strap-limit-mm.
STRAP_ESTIMATE_MM = 10.0    # the operator\'s own earlier strap-failure estimate. Only
                            # used to print a reminder when the ceiling exceeds it.
HOVER_MM = 5.0
APPROACH_MM = 15.0


class FrankaWrench:
    """Coarse fallback force source: the robot's own external wrench.

    Deliberately given the same read_window() shape as WittensteinFT so the probe
    loop does not care which one it got. Biased +-2 N (HANDOFF section 3.1) -- this
    is a runaway guard, not an instrument.
    """

    def __init__(self):
        from franka_msgs.msg import FrankaState
        self._FrankaState = FrankaState

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read_window(self, seconds):
        vals, t0 = [], time.time()
        while time.time() - t0 < seconds:
            try:
                st = rospy.wait_for_message("/franka_state_controller/franka_states",
                                            self._FrankaState, timeout=1.0)
                vals.append(st.O_F_ext_hat_K[2])
            except Exception:
                break
        if not vals:
            return 0.0, 0.0, 0
        a = np.array(vals)
        return float(a.mean()), float(a.std()), len(a)


def open_source(kind):
    if kind == "franka":
        print("[PROBE] force source: FRANKA O_F_ext_hat_K  (BIASED +-2 N -- coarse "
              "guard only)")
        return FrankaWrench()
    from franka_surface_map import WittensteinFT
    print("[PROBE] force source: HEX21 (must be plugged into TACTILE for this run)")
    return WittensteinFT(port=m.SERIAL_PORT)


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


def build_probe_grid(points, span_u_mm, span_v_mm, margins=None):
    """Centre + 4 corners + 4 edge midpoints = a 3x3 lattice on the INSET rectangle.

    Reuses sweep_campaign.build_grid rather than re-deriving the bilinear-over-corners
    and margin maths. That is deliberate: if the probe and the campaign disagreed by
    even a millimetre about where the inset rectangle sits, the depth-limit map would
    be describing different silicone than the campaign indents, and nothing would
    warn you. One implementation, one rectangle.

    Imported lazily so --dry-run does not need the full campaign import chain.
    """
    from sweep_campaign import build_grid
    return build_grid(points, span_u_mm, span_v_mm, pitch_mm=None, n_u=3, n_v=3,
                      margins=margins)


def probe_point(servo, ft, x, y, surface_z, quat, args, label=""):
    """ONE descent. Returns (rows, reason, summary).

    This is the only place the descent exists -- --map loops over it rather than
    reimplementing it, so the abort logic and the guaranteed retract cannot drift
    apart between the single-point and map modes.

    TWO STOPPING AUTHORITIES, and they are not interchangeable:
      * THE OPERATOR owns the OPTICAL limit. Only a human watching the live event
        view can see the silicone leave frame or the marker pattern break down, and
        that is the limit that actually matters for a dataset -- an indent whose
        markers cannot be tracked is a useless training sample no matter how
        mechanically safe it was. In interactive mode you step down one increment at
        a time and say when to stop.
      * THE MACHINE owns the FORCE limit, always, in every mode. The force ceiling
        and the stiffness abort are NOT disabled by interactive mode. If you blink,
        they still fire.
    """
    hover_z = surface_z + args.hover_mm / 1000.0
    n_steps = int(np.ceil(args.max_depth_mm / args.step_mm))

    m._gross_move(servo, x, y, hover_z, quat, name="%s-> hover" % label)
    base, base_sd, base_n = ft.read_window(BASELINE_S)
    print("  baseline Fz = %+.4f N (sd %.4f, n=%d)" % (base, base_sd, base_n), flush=True)

    rows, reason, prev = [], "reached the depth ceiling", None
    stopped_by = "depth"
    step_idx, direction, coast = 0, "down", 0
    try:
        while step_idx < n_steps:
            step_idx += 1
            d_mm = min(step_idx * args.step_mm, args.max_depth_mm)
            servo.send_target(x, y, surface_z - d_mm / 1000.0, quat)
            time.sleep(args.settle_s)
            pos, _ = servo.current_pose()
            mean_fz, sd, n = ft.read_window(SAMPLE_S)
            F = abs(mean_fz - base)
            achieved_mm = (surface_z - pos[2]) * 1000.0

            k = None
            if direction == "down" and prev is not None and achieved_mm - prev[0] > 1e-3:
                k = (F - prev[1]) / (achieved_mm - prev[0])
            rows.append({"commanded_depth_mm": round(d_mm, 4),
                         "achieved_depth_mm": round(achieved_mm, 4),
                         "ee_z": round(pos[2], 6),
                         "Fz_raw_N": round(mean_fz, 4),
                         "Fz_tared_N": round(F, 4),
                         "Fz_sd_N": round(sd, 4), "n_samples": n,
                         "step_dir": direction,
                         "stiffness_N_per_mm": round(k, 3) if k is not None else ""})
            print("    cmd %.2f mm | got %.3f mm | F = %6.3f N%s%s"
                  % (d_mm, achieved_mm, F,
                     "  |  dF/dz = %6.2f N/mm" % k if k is not None else "",
                     "  [backed off]" if direction == "up" else ""), flush=True)

            # --- machine backstops: active in EVERY mode, interactive included ---
            if F >= args.max_force_n:
                reason = "FORCE CEILING: %.2f N >= %.2f N" % (F, args.max_force_n)
                stopped_by = "force"
                break
            if k is not None and k >= args.max_stiffness_n_per_mm:
                reason = ("STIFFNESS BLOW-UP: %.1f N/mm >= %.1f N/mm -- membrane "
                          "stiffening or a strap loading up"
                          % (k, args.max_stiffness_n_per_mm))
                stopped_by = "stiffness"
                break
            prev = (achieved_mm, F)
            direction = "down"

            # --- operator: the optical limit ---
            if args.hold_s > 0:
                time.sleep(args.hold_s)
            if args.interactive and achieved_mm < args.prompt_from_mm:
                # Auto-descend region: the backstops above still apply, we simply do
                # not stop to ask while the indent is far too shallow to be near any
                # optical limit.
                continue
            if coast > 0:
                # Operator asked to coast N steps without being asked. Backstops are
                # still live throughout -- coasting skips the PROMPT, not the checks.
                coast -= 1
                continue
            if args.interactive:
                try:
                    resp = input("    [Enter] +1 | <N> = +N steps | s = STOP HERE "
                                 "(optical limit) | b = back off | q = abort > "
                                 ).strip().lower()
                except EOFError:
                    resp = "s"
                if resp.isdigit() and int(resp) > 0:
                    # Coast N steps then ask again. This is what makes a 60-step
                    # ladder workable: skim the boring region, fine-step near the
                    # limit, and YOU choose the granularity live.
                    coast = int(resp)
                    print("    coasting %d steps (backstops still live) ..." % coast)
                    continue
                if resp.startswith("s"):
                    # "called the limit", not "optical limit" -- the operator may stop
                    # for markers leaving frame OR for plain caution about the rig.
                    # Both are valid and the log should not claim to know which.
                    reason = "OPERATOR: called the limit at %.2f mm" % achieved_mm
                    stopped_by = "operator"
                    break
                if resp.startswith("q"):
                    reason = "OPERATOR: aborted the run at %.2f mm" % achieved_mm
                    stopped_by = "operator-abort"
                    raise KeyboardInterrupt
                if resp.startswith("b"):
                    # step back UP one increment and re-measure there. Marked
                    # step_dir="up" and excluded from the k/n fit -- mixing loading
                    # and unloading points into one power-law fit would bias it.
                    step_idx = max(0, step_idx - 2)
                    direction = "up"
                    prev = None
    finally:
        # Retract FIRST, always -- before any file I/O, before any reporting, and
        # before moving on to the next point of a map run.
        print("  retracting ...", flush=True)
        try:
            m._gross_move(servo, x, y, hover_z, quat, name="%sretract" % label)
        except Exception as e:
            print("  !! retract failed (%s) -- CHECK THE ARM." % e)

    summary = summarise(rows, reason)
    summary["stopped_by"] = stopped_by
    return rows, reason, summary


def summarise(rows, reason):
    """Per-point summary: how deep we got, at what force, and the local F = k*d^n.

    The k/n fit uses DOWNWARD steps only -- a back-off point is on the unloading
    branch, and silicone is viscoelastic, so pooling both branches would bias the fit.
    """
    if not rows:
        return {"max_depth_mm": 0.0, "force_at_max_N": 0.0, "k": None, "n": None,
                "reason": reason, "n_steps": 0}
    down = [r for r in rows if r.get("step_dir", "down") == "down"]
    d = np.array([r["achieved_depth_mm"] for r in down]) if down else np.array([0.0])
    F = np.array([r["Fz_tared_N"] for r in down]) if down else np.array([0.0])
    k = n_fit = None
    good = (d > 0.3) & (F > 0.1)
    if good.sum() >= 4:
        n_fit, lk = np.polyfit(np.log(d[good]), np.log(F[good]), 1)
        k, n_fit = float(np.exp(lk)), float(n_fit)
    return {"max_depth_mm": float(np.max([r["achieved_depth_mm"] for r in rows])),
            "force_at_max_N": float(rows[-1]["Fz_tared_N"]),
            "k": k, "n": n_fit, "reason": reason, "n_steps": len(rows)}


def bilinear_on_3x3(vals, u, v):
    """Piecewise-bilinear interpolation of a 3x3 lattice at (u, v) in [0,1]^2.

    vals is indexed [row(v)][col(u)] with nodes at 0, 0.5, 1 on both axes.
    Piecewise-bilinear (not a biquadratic through all 9) on purpose: a quadratic
    through 3 nodes overshoots between them, and overshooting a DEPTH LIMIT upward
    is the one error mode that actually damages something.
    """
    u = min(max(u, 0.0), 1.0)
    v = min(max(v, 0.0), 1.0)
    j = 0 if u < 0.5 else 1
    i = 0 if v < 0.5 else 1
    fu = (u - 0.5 * j) / 0.5
    fv = (v - 0.5 * i) / 0.5
    return ((1 - fu) * (1 - fv) * vals[i][j] + fu * (1 - fv) * vals[i][j + 1] +
            (1 - fu) * fv * vals[i + 1][j] + fu * fv * vals[i + 1][j + 1])


def save_map(nodes, grid_info, args, out_csv):
    """Write the per-node results + a JSON sidecar carrying the interpolant."""
    fields = ["node", "row", "col", "u", "v", "x", "y", "surface_z",
              "max_depth_mm", "force_at_max_N", "k", "n", "n_steps",
              "stopped_by", "from_previous_run", "reused_node_moved_mm",
              "reason"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for nd in nodes:
            w.writerow({k: nd.get(k, "") for k in fields})
    side = {"span_mm": [args.span_mm[0], args.span_mm[1]],
            "grid": grid_info,
            "max_force_n": args.max_force_n,
            "max_depth_mm": args.max_depth_mm,
            "strap_limit_mm": STRAP_LIMIT_MM,
            "safety_factor": args.safety_factor,
            "interactive": args.interactive,
            "nodes": nodes,
            "interpolation": ("piecewise-bilinear on the 3x3 lattice of (u,v) in "
                              "[0,1]^2; u = BL->BR axis, v = BL->TL axis, both on "
                              "the INSET rectangle, not the taught quad"),
            "note": ("max_depth_mm is the deepest ACHIEVED depth before the abort "
                     "criterion fired -- it is a limit under THIS force ceiling, not "
                     "a material failure point.")}
    Path(str(out_csv) + ".json").write_text(json.dumps(side, indent=2))


def save_map_png(nodes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (no heatmap: %s)" % e)
        return
    D = np.full((3, 3), np.nan)
    K = np.full((3, 3), np.nan)
    for nd in nodes:
        D[nd["row"]][nd["col"]] = nd["max_depth_mm"]
        if nd.get("k") is not None:
            K[nd["row"]][nd["col"]] = nd["k"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, M, title, unit in ((axes[0], D, "Max indentation", "mm"),
                               (axes[1], K, "Local stiffness k (F = k*d^n)", "N/mm^n")):
        im = ax.imshow(M, origin="lower", cmap="viridis")
        for i in range(3):
            for j in range(3):
                if not np.isnan(M[i][j]):
                    ax.text(j, i, "%.2f" % M[i][j], ha="center", va="center",
                            color="w", fontsize=11)
        ax.set_title("%s (%s)" % (title, unit))
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["u=0", "u=0.5", "u=1"])
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["v=0", "v=0.5", "v=1"])
        fig.colorbar(im, ax=ax)
    fig.suptitle("Depth-limit map over the inset rectangle "
                 "(u = BL->BR / ~30 mm axis, v = BL->TL / ~36 mm axis)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("  heatmap -> %s" % path)


def report_map(nodes, args, out_csv):
    """Turn 9 probed nodes into a usable campaign recommendation."""
    D = [[None] * 3 for _ in range(3)]
    for nd in nodes:
        D[nd["row"]][nd["col"]] = nd["max_depth_mm"]
    depths = [nd["max_depth_mm"] for nd in nodes]
    worst = min(depths)
    safe = worst * args.safety_factor

    print("\n" + "=" * 74)
    print(" DEPTH-LIMIT MAP  (force ceiling %.1f N)" % args.max_force_n)
    print("=" * 74)
    print("  max indentation (mm), rows = v (BL->TL), cols = u (BL->BR):")
    for i in (2, 1, 0):
        print("     v=%.1f   " % (i * 0.5) + "  ".join("%6.2f" % D[i][j] for j in range(3)))
    print("              " + "  ".join("%6s" % ("u=%.1f" % (j * 0.5)) for j in range(3)))
    print()
    by = [nd.get("stopped_by", "?") for nd in nodes]
    n_op = sum(1 for b in by if b == "operator")
    print("  stopped by: %s"
          % ", ".join("%s x%d" % (t, by.count(t)) for t in sorted(set(by))))
    if n_op == len(nodes):
        print("     every node was set by the OPERATOR watching the live view --")
        print("     these ARE the optical limits, which is what we want.")
    elif n_op == 0:
        print("     [WARN] no node was operator-set, so this map is a MECHANICAL")
        print("     limit only. The optical limit may well be shallower. Re-run")
        print("     interactively from a terminal to capture it.")
    else:
        print("     [WARN] MIXED. Nodes stopped by a machine backstop reached their")
        print("     force/stiffness limit BEFORE you called an optical limit, so for")
        print("     those the optical limit is unknown and may be shallower.")
    ks = [nd["k"] for nd in nodes if nd.get("k") is not None]
    ns = [nd["n"] for nd in nodes if nd.get("n") is not None]
    if ks:
        print("  local law F = k*d^n :  k %.3f-%.3f (x%.1f spread),  n %.2f-%.2f"
              % (min(ks), max(ks), max(ks) / max(min(ks), 1e-9), min(ns), max(ns)))
    print("  deepest / shallowest node : %.2f / %.2f mm" % (max(depths), worst))
    for nd in nodes:
        if "FORCE" in nd["reason"] or "STIFFNESS" in nd["reason"]:
            print("    node %d (u=%.1f,v=%.1f) stopped early: %s"
                  % (nd["node"], nd["u"], nd["v"], nd["reason"]))
    print()
    print("  RECOMMENDED CAMPAIGN CEILING : %.2f mm" % safe)
    print("     = the SHALLOWEST node (%.2f mm) x safety factor %.2f."
          % (worst, args.safety_factor))
    print("     Using the shallowest rather than the interpolated per-location value")
    print("     is deliberate: between the 9 nodes the true limit can dip BELOW the")
    print("     interpolant, and overshooting a depth limit is the error that costs")
    print("     hardware. The per-location map in %s is there for" % Path(out_csv).name)
    print("     force-targeting (below), not for pushing individual points deeper.")
    if n_op == len(nodes) and args.safety_factor < 0.9:
        print("     NOTE: every node here was operator-set, so the 0.80 factor is")
        print("     stacked on top of your own judgement. --safety-factor 0.9 gives")
        print("     %.2f mm and is defensible when you called every limit yourself."
              % (worst * 0.9))
    print()
    print("  A LADDER THAT FITS: %s mm"
          % ", ".join("%.1f" % d for d in
                      np.round(np.linspace(safe / 6.0, safe, 6), 1)))
    print()
    if n_op:
        print("  CONFIRM IT ONCE, QUANTITATIVELY. Your eye set these limits on the live")
        print("  view; the tracker is the thing that actually has to cope. Record one")
        print("  indent at %.2f mm with master_midpoint.py, run marker_overlay.py" % safe)
        print("  --probe, and check roster n_kept holds up (204 markers at 2 mm today).")
        print("  If it collapses, come back down and re-map.")
    else:
        print("  THE OPTICAL LIMIT IS STILL UNMEASURED. Re-run interactively, or check")
        print("  it by hand: record a deep indent with master_midpoint.py, run")
        print("  marker_overlay.py --probe, and watch roster n_kept.")
    print("=" * 74)
    print(" Then: sweep_campaign.py --depths-mm <ladder> --i-have-run-the-probe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", nargs="?", default=None,
                    help="default: depth_limit_probe.csv, or depth_limit_map.csv "
                         "with --map")
    ap.add_argument("--map", action="store_true",
                    help="probe 9 points (centre + 4 corners + 4 edge midpoints) on "
                         "the inset rectangle and build an interpolated depth-limit "
                         "map, instead of a single point")
    ap.add_argument("--span-mm", type=float, nargs=2, default=[24.0, 32.0],
                    metavar=("U", "V"),
                    help="--map only: the inset rectangle to map, matching "
                         "sweep_campaign.py's --span-mm (default 24 32)")
    ap.add_argument("--resume", nargs="?", const="", default=None, metavar="MAP_CSV",
                    help="--map only: reuse nodes already measured in a previous map "
                         "CSV and probe ONLY the missing ones, then write one merged "
                         "map. Bare --resume uses the default output path. Node "
                         "geometry always comes from the freshly built grid, so a "
                         "stale file cannot inject stale coordinates.")
    ap.add_argument("--reprobe-nodes", default="", metavar="N,M",
                    help="--map --resume only: re-probe these nodes even though the "
                         "resume file already has them. Use after changing --margin-mm "
                         "or --span-mm, which MOVES the grid and invalidates the old "
                         "measurement at the moved nodes.")
    ap.add_argument("--skip-nodes", default="", metavar="N,M",
                    help="--map only: comma-separated node indices to skip entirely "
                         "(node 4 is the centre).")
    ap.add_argument("--margin-mm", type=float, nargs="+", default=None,
                    help="Clearance from the taught quad, in mm. 1 value = all four "
                         "edges; 2 = (u, v) symmetric per axis; 4 = (u_lo, u_hi, "
                         "v_lo, v_hi) per edge. Overrides --span-mm. NOTE the clearance "
                         "an edge needs SCALES WITH DEPTH -- the cone widens as "
                         "~0.5*depth, so 2 mm only supports ~4 mm of indent next to a "
                         "wall (measured 2026-08-07).")
    ap.add_argument("--safety-factor", type=float, default=0.8,
                    help="--map only: fraction of the shallowest node to recommend "
                         "as the campaign ceiling (default 0.8). Operator-set optical "
                         "limits already embody your judgement, so 0.9-1.0 is "
                         "defensible when every node was stopped by hand.")
    ap.add_argument("--interactive", dest="interactive", action="store_true",
                    default=None,
                    help="step down one increment at a time and ask before each "
                         "deeper step, so YOU set the optical limit while watching "
                         "the live event view. DEFAULT ON when stdin is a terminal.")
    ap.add_argument("--no-interactive", dest="interactive", action="store_false",
                    help="descend automatically; only the force/stiffness/depth "
                         "backstops stop it. The optical limit is then NOT captured.")
    ap.add_argument("--prompt-from-mm", type=float, default=PROMPT_FROM_MM,
                    help="interactive only: descend automatically (force/stiffness "
                         "backstops still live) until this ACHIEVED depth, then prompt "
                         "at every step. Default %.1f. Without this a 7 mm ladder at "
                         "0.25 mm steps is 28 prompts per node x 9 nodes. Set 0 to "
                         "prompt from the very first step." % PROMPT_FROM_MM)
    ap.add_argument("--hold-s", type=float, default=0.0,
                    help="pause this long at each step before continuing/prompting, "
                         "to give the live view time to settle (default 0)")
    ap.add_argument("--source", choices=["hex21", "franka"], default="hex21")
    ap.add_argument("--max-depth-mm", type=float, default=MAX_DEPTH_MM,
                    help="COMMANDED depth ceiling (default %.1f). Achieved depth is "
                         "~1.6 mm less at ~6.7 N because of impedance sag."
                         % MAX_DEPTH_MM)
    ap.add_argument("--strap-limit-mm", type=float, default=STRAP_LIMIT_MM,
                    help="hard refusal above this (default %.1f). A typo guard, not a "
                         "physical model." % STRAP_LIMIT_MM)
    ap.add_argument("--max-force-n", type=float, default=MAX_FORCE_N)
    ap.add_argument("--max-stiffness-n-per-mm", type=float,
                    default=MAX_STIFFNESS_N_PER_MM)
    ap.add_argument("--step-mm", type=float, default=STEP_MM)
    ap.add_argument("--settle-s", type=float, default=SETTLE_S)
    ap.add_argument("--hover-mm", type=float, default=HOVER_MM)
    ap.add_argument("--at", choices=["center", "bottom-left", "top-left",
                                     "top-right", "bottom-right"], default="center",
                    help="single-point mode: which taught point to probe (default the "
                         "centre, furthest from the stiff strapped edges)")
    ap.add_argument("--no-level", action="store_true")
    ap.add_argument("--no-recovery", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.max_depth_mm > args.strap_limit_mm:
        sys.exit("[ERROR] --max-depth-mm %.1f exceeds the hard cap of %.1f mm.\n"
                 "        Raise --strap-limit-mm only if you have decided the straps "
                 "take it."
                 % (args.max_depth_mm, args.strap_limit_mm))
    if not 0.1 < args.safety_factor <= 1.0:
        sys.exit("[ERROR] --safety-factor must be in (0.1, 1.0].")

    # Interactive is the DEFAULT: the optical limit is the real limit, and only a
    # human watching the live view can see it. Fall back automatically when there is
    # no terminal to prompt on (a piped/ssh-batch run would otherwise hang forever
    # holding the arm at depth).
    if args.interactive is None:
        args.interactive = sys.stdin.isatty()
        if not args.interactive:
            print("[PROBE] stdin is not a terminal -- falling back to automatic "
                  "descent.\n        The OPTICAL limit will NOT be captured; only the "
                  "force/stiffness\n        backstops apply. Run from a terminal for "
                  "operator-set limits.")

    if not CORNERS_FILE.exists():
        sys.exit("[ERROR] %s not found -- teach the corners first." % CORNERS_FILE.name)
    data = json.loads(CORNERS_FILE.read_text())
    pts = data.get("points") or data["corners"]

    out_csv = Path(args.out_csv) if args.out_csv else (
        Path.home() / "E-BTS" / ("depth_limit_map.csv" if args.map
                                 else "depth_limit_probe.csv"))

    # ---- build the target list -------------------------------------------------
    if args.map:
        grid, grid_info = build_probe_grid(pts, args.span_mm[0], args.span_mm[1],
                                           margins=parse_margins(args.margin_mm))
        targets = [{"node": i, "row": g["row"], "col": g["col"],
                    "u": g["col"] * 0.5, "v": g["row"] * 0.5,
                    "x": g["x"], "y": g["y"], "surface_z": g["z_plane"]}
                   for i, g in enumerate(grid)]
    else:
        if args.at not in pts:
            sys.exit("[ERROR] '%s' is not in %s (have: %s)."
                     % (args.at, CORNERS_FILE.name, ", ".join(sorted(pts))))
        px, py, sz = [float(v) for v in pts[args.at]["xyz"]]
        grid_info = None
        targets = [{"node": 0, "row": 0, "col": 0, "u": 0.5, "v": 0.5,
                    "x": px, "y": py, "surface_z": sz}]

    # ---- resume / skip -------------------------------------------------------
    all_targets = list(targets)      # unfiltered; the merge needs live geometry
    prior_nodes, prior_steps = {}, []
    if args.map and args.resume is not None:
        src = Path(args.resume) if args.resume else out_csv
        if not src.exists():
            sys.exit("[ERROR] --resume %s does not exist. Drop --resume for a fresh "
                     "map." % src)
        for row in csv.DictReader(open(src)):
            prior_nodes[int(row["node"])] = row
        st = src.with_name(src.stem + "_steps.csv")
        if st.exists():
            prior_steps = list(csv.DictReader(open(st)))
        print("[PROBE] --resume %s: reusing %d node(s) %s, %d prior step rows"
              % (src.name, len(prior_nodes), sorted(prior_nodes), len(prior_steps)))
    reprobe = {int(v) for v in args.reprobe_nodes.split(",") if v.strip()}
    if reprobe:
        dropped = sorted(reprobe & set(prior_nodes))
        for ni in dropped:
            prior_nodes.pop(ni)
        prior_steps = [r for r in prior_steps
                       if str(r.get("node", "")).strip() not in
                       {str(x) for x in dropped}]
        if dropped:
            print("[PROBE] --reprobe-nodes: discarding prior results for %s" % dropped)
    skip = {int(v) for v in args.skip_nodes.split(",") if v.strip()}
    if args.map and (prior_nodes or skip):
        todo = [t for t in targets
                if t["node"] not in prior_nodes and t["node"] not in skip]
        print("[PROBE] probing %d of %d nodes: %s"
              % (len(todo), len(targets), sorted(t["node"] for t in todo)))
        if not todo:
            sys.exit("[ERROR] every node is already covered -- nothing to probe.")
        targets = todo

    n_steps = int(np.ceil(args.max_depth_mm / args.step_mm))
    per_pt_s = n_steps * (args.settle_s + SAMPLE_S)
    print("=" * 74)
    print(" FORCE-LIMITED DEPTH PROBE%s" % ("  --  9-POINT MAP" if args.map else ""))
    print("=" * 74)
    if args.map:
        print("  inset rectangle      : %.1f x %.1f mm"
              % (grid_info["span_u_mm"], grid_info["span_v_mm"]))
        print_edges(grid_info)
        print("  probe points         : 3 x 3 = centre + 4 corners + 4 edge midpoints")
        print("  node spacing         : %.1f mm (u) x %.1f mm (v)"
              % (grid_info["achieved_pitch_u_mm"], grid_info["achieved_pitch_v_mm"]))
        for t in targets:
            print("      node %d  u=%.1f v=%.1f   [%.5f, %.5f]  surface_z %.5f"
                  % (t["node"], t["u"], t["v"], t["x"], t["y"], t["surface_z"]))
    else:
        t = targets[0]
        print("  point                : %s  [%.5f, %.5f]" % (args.at, t["x"], t["y"]))
        print("  surface_z (kissed)   : %.5f m" % t["surface_z"])
    print("  descend              : %.2f mm steps, settle %.1f s, to a max of %.1f mm"
          % (args.step_mm, args.settle_s, args.max_depth_mm))
    print("  ABORT on             : force > %.1f N, or dF/dz > %.1f N/mm, or depth "
          "> %.1f mm" % (args.max_force_n, args.max_stiffness_n_per_mm,
                         args.max_depth_mm))
    print("  worst case           : %d steps/point x %d point(s) ~ %.0f min"
          % (n_steps, len(targets), per_pt_s * len(targets) / 60.0))
    print("  force source         : %s" % args.source)
    if args.max_depth_mm > STRAP_ESTIMATE_MM:
        print("  ! ceiling %.1f mm is above your earlier ~%.0f mm strap estimate."
              % (args.max_depth_mm, STRAP_ESTIMATE_MM))
        print("    Proceeding as directed. The other guards: the %.1f N force ceiling"
              % args.max_force_n)
        print("    (~%.0f mm achieved on the measured curve) and you, watching."
              % ((args.max_force_n - 2.244) / 0.424 + 4.548))
        print("    ! SAG: commanding %.1f mm ACHIEVES ~%.1f mm (sag = 0.258*F - 0.148)."
              % (args.max_depth_mm,
                 args.max_depth_mm - max(0.0, 0.258 * (0.496 * args.max_depth_mm + 0.09) - 0.148)))
    if args.interactive:
        print("  STOPPING AUTHORITY   : YOU (optical) + machine backstops")
        if args.prompt_from_mm > 0:
            print("     auto-descending to %.1f mm (backstops LIVE), prompting from "
                  "there" % args.prompt_from_mm)
            print("     -> ~%d prompts per node instead of %d"
                  % (max(0, int((args.max_depth_mm - args.prompt_from_mm) / args.step_mm)),
                     int(args.max_depth_mm / args.step_mm)))
        print("     At any prompt, type a NUMBER to coast that many steps before being")
        print("     asked again -- skim the shallow region, fine-step near the limit.")
        print("     Watch the LIVE EVENT VIEW on the workstation. At each step:")
        print("       [Enter] go deeper   s = stop here   b = back off   q = abort")
        print("     Press 's' the moment the silicone leaves frame or the marker")
        print("     pattern breaks down -- that depth IS this node's limit.")
    else:
        print("  STOPPING AUTHORITY   : machine only (force/stiffness/depth)")
        print("     [WARN] the OPTICAL limit is not being captured in this mode.")
    print("  expected force, from the MEASURED centre curve (2026-08-07, 0.4-4.5 mm,")
    print("  F = 0.496*d + 0.090, R2 = 0.994 -- stiffness SOFTENS with depth,")
    print("  0.63 -> 0.37 N/mm, so the deep values are slight OVER-estimates):")
    for d in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
        if d <= args.max_depth_mm:
            # two estimates: the whole-range linear fit, and the deep-half slope
            # (0.424 N/mm) carried on from the last measured point. The softening
            # one is the better guess out here; the linear one brackets it high.
            lin = 0.496 * d + 0.090
            soft = 2.244 + 0.424 * (d - 4.548)
            print("      %4.1f mm -> %5.2f N (soft) / %5.2f N (lin)   %2.0f%% of %.1f N%s"
                  % (d, soft, lin, soft / args.max_force_n * 100, args.max_force_n,
                     "   <- extrapolated" if d > 4.5 else ""))
    print("=" * 74)
    print("  KEEP A HAND ON THE USER-STOP. The servo will not stop on contact.")
    print("=" * 74)

    if args.dry_run:
        print("\n[DRY RUN] no ROS, no motion.")
        return

    rospy.init_node("depth_limit_probe", disable_signals=True)
    from franka_grid_logger import clear_reflex, warn_if_loaded
    if not args.no_recovery:
        clear_reflex()
    warn_if_loaded()

    from servo_client import CartesianServo
    servo = CartesianServo()

    hover = args.hover_mm / 1000.0
    pre_grid = [{"x": t["x"], "y": t["y"], "z_plane": t["surface_z"]} for t in targets]
    saved_depth, saved_approach = m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM
    m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = args.max_depth_mm, APPROACH_MM
    try:
        m.preflight_reach(servo, pts, pre_grid, hover)
    finally:
        m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = saved_depth, saved_approach

    cx, cy, cz = [float(v) for v in pts["center"]["xyz"]]
    all_steps, nodes = [], []
    with open_source(args.source) as ft:
        pos0, q0 = servo.current_pose()
        quat = flat_down_quat(q0) if not args.no_level else list(q0)
        print("[PROBE] flange tilt from vertical: %.2f deg" % tilt_deg(q0))
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat,
                      name="home -> approach")
        try:
            for t in targets:
                print("\n[node %d/%d] u=%.1f v=%.1f  [%.5f, %.5f]"
                      % (t["node"] + 1, len(targets), t["u"], t["v"], t["x"], t["y"]),
                      flush=True)
                rows, reason, summ = probe_point(servo, ft, t["x"], t["y"],
                                                 t["surface_z"], quat, args,
                                                 label="n%d " % t["node"])
                for r in rows:
                    r2 = dict(r); r2.update(node=t["node"], u=t["u"], v=t["v"])
                    all_steps.append(r2)
                nd = dict(t); nd.update(summ); nodes.append(nd)
                print("  -> %.2f mm at %.2f N  (%s)"
                      % (summ["max_depth_mm"], summ["force_at_max_N"], reason))
        except KeyboardInterrupt:
            print("\n[PROBE] interrupted -- keeping what was measured.")
        finally:
            try:
                m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat,
                              name="park above centre")
            except Exception as e:
                print("[PROBE] !! park failed (%s) -- CHECK THE ARM." % e)

    if not all_steps:
        print("\nNo steps recorded.")
        return

    # Fold the reused nodes back in. Geometry comes from the LIVE grid (grid_full),
    # never from the prior file, so a stale CSV cannot smuggle in stale coordinates --
    # only the measurement itself (depth/force/k/n/reason) is carried over.
    if prior_nodes:
        by_node = {g["node"]: g for g in all_targets}
        for ni, row in sorted(prior_nodes.items()):
            g = by_node.get(ni)
            if g is None:
                continue
            nd = dict(g)
            for key, cast in (("max_depth_mm", float), ("force_at_max_N", float),
                              ("n_steps", int), ("stopped_by", str), ("reason", str)):
                if row.get(key) not in (None, ""):
                    nd[key] = cast(row[key])
            for key in ("k", "n"):
                nd[key] = float(row[key]) if row.get(key) not in (None, "") else None
            nd["from_previous_run"] = 1
            # ⚠ If the margins changed since the prior run, this node is no longer at
            # the same physical spot. Reusing it silently would attribute an old
            # measurement to new coordinates -- exactly the kind of error that leaves
            # no trace. Warn with the actual distance and let the operator judge.
            try:
                if row.get("x") and row.get("y"):
                    moved = np.hypot(float(row["x"]) - g["x"],
                                     float(row["y"]) - g["y"]) * 1e3
                    nd["reused_node_moved_mm"] = round(float(moved), 3)
                    if moved > 0.5:
                        print("  [WARN] reused node %d has MOVED %.2f mm since it was "
                              "measured" % (ni, moved))
                        print("         (grid geometry changed -- margins or span). Its "
                              "depth limit may no longer apply here; re-probe it with "
                              "--skip-nodes leaving %d out of the skip list." % ni)
            except (TypeError, ValueError):
                pass
            nodes.append(nd)
        for r in prior_steps:
            all_steps.append(r)
        nodes.sort(key=lambda z: z["node"])
        print("[PROBE] merged %d reused node(s) -> %d total" % (len(prior_nodes), len(nodes)))

    steps_csv = out_csv.with_name(out_csv.stem + "_steps.csv")
    # Union the columns: rows carried over from a previous run can have a different
    # (older) schema, and DictWriter would raise on the first unexpected key.
    fields = []
    for r in all_steps:
        for kk in r:
            if kk not in fields:
                fields.append(kk)
    with open(steps_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(all_steps)
    print("\nWrote %d steps to %s" % (len(all_steps), steps_csv))

    if args.map and len(nodes) == 9:
        save_map(nodes, grid_info, args, out_csv)
        print("Wrote the node map to %s (+ .json)" % out_csv)
        save_map_png(nodes, out_csv.with_suffix(".png"))
        report_map(nodes, args, out_csv)
    else:
        nd = nodes[0] if nodes else {}
        print("=" * 74)
        print(" RESULT: stopped because %s" % nd.get("reason", "?"))
        print("  deepest reached : %.3f mm  (strap limit ~%.0f mm)"
              % (nd.get("max_depth_mm", 0.0), STRAP_LIMIT_MM))
        print("  force there     : %.3f N" % nd.get("force_at_max_N", 0.0))
        if nd.get("k") is not None:
            print("  fitted F = %.3f * d^%.2f over the probed range" % (nd["k"], nd["n"]))
            print("  (mid5_center's 0.3-2.0 mm ramp gave k=0.25, n=1.63 -- a HIGHER n "
                  "here means\n   the sheet has moved into membrane stretching)")
        print("=" * 74)
        if args.map:
            print(" NOTE: --map wanted 9 nodes but only %d completed, so no map was "
                  "written." % len(nodes))
        print(" For a per-location limit across the whole elastomer, run with --map.")


if __name__ == "__main__":
    main()
