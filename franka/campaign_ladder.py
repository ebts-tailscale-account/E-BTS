#!/usr/bin/env python3
"""
campaign_ladder.py -- N random locations, each poked through the SAME depth ladder.

    python3 campaign_ladder.py --points 100 --seed 7 --dry-run
    python3 campaign_ladder.py --points 100 --seed 7

WHAT THIS IS FOR, AND WHY IT IS A THIRD SCRIPT
----------------------------------------------
It exists to measure ONE thing: how well the GUI's live contact-location estimate
(SOFTWARE_README §11b) agrees with where the robot actually poked, AND how that
agreement depends on indentation depth.

Neither sibling can answer that:

    campaign_planb.py    grid NODES x depth ladder. Every poke lands on one of the
                         99 mapped nodes, so it cannot separate localisation error
                         from "the estimator was tested on a lattice that happens
                         to align with the marker lattice".
    campaign_random.py   random (x, y) AND random depth, one poke per location.
                         Depth is explicitly "not a measured variable" there -- it
                         is only a knob to spread force. With one poke per point
                         and a continuous depth draw, error-versus-depth is a
                         scatter plot with n=1 per condition and no repeats.

This one crosses them: the SAMPLER is campaign_random's (continuous-random
locations, never snapped to a node, never extrapolating), and the DESIGN is
campaign_planb's randomised complete block (every location visited once at every
depth level). That gives, per depth level, N independent locations -- which is what
turns "the error was 0.3 mm" into "the error is 0.3 +- 0.05 mm at 1.0 mm and
0.18 +- 0.02 mm at 5.0 mm", i.e. an answer about the estimator rather than about
one poke.

Per the standing rule (HANDOFF §12) this is a SIBLING, not a flag: campaign_planb
and campaign_random are untouched. Everything that can move the robot -- the poke
cycle, the resume ledger, the reach preflight, the depth closed loop -- is imported
from them, so this file contains no new motion code at all. What is new is the
plan, and only the plan.

⚠ DEPTH IS THE DESIGN VARIABLE HERE, unlike in campaign_random.
The question is whether localisation degrades at shallow indentation, where the
divergence lobe is weak, and/or at deep indentation, where the elastomer shears
enough that the marker field's centre may no longer sit under the indenter axis.
So the ladder is commanded verbatim at every location (campaign_planb's
depth-targeted mode) and every location sees every rung.

⚠ THE DEPTH ORDER IS RANDOMISED PER LOCATION, AND THAT MATTERS MORE HERE THAN
ANYWHERE ELSE. A 1100-poke run takes ~2.5-3 h. The contact estimate is measured
against a marker baseline captured ONCE at the start, so anything that drifts over
those hours -- thermal, strap creep, illumination -- degrades it monotonically with
TIME. Walk the ladder in order and that drift lands entirely on the deep end and is
indistinguishable from a real depth effect. Randomised, it spreads across all rungs
and shows up as scatter instead of as bias. campaign_planb measured this effect
directly (corr(force, time) = +0.079 over 115 min); the same mechanism applies here
to localisation.

⚠ WHAT THIS RUN NEEDS THAT THE OTHERS DO NOT: the GUI's contact log.
The pairing this campaign exists to produce is

    contact.csv  (camera-estimated x, y)   vs   franka.csv  (ee_x, ee_y)

and `<run>_contact.csv` is only written if the Circle Tracking pane was open with a
VALID, UNLOADED baseline before the recording started. A run with a baseline taken
while something was already touching the pad produces a full, plausible, and wholly
wrong contact log. The checklist is printed at the top of every non-dry run; read
it rather than skipping it.

WHAT IT DOES NOT CHANGE
-----------------------
Surface height still comes from bilinear interpolation of surface_offset_map.csv,
the sampler still refuses cells with an unmeasured corner, and the depth ceiling
gate (DEMONSTRATED_DEPTH_MM) is the same one campaign_planb enforces.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import campaign_planb as pb                                              # noqa: E402
import campaign_random as cr                                             # noqa: E402
from campaign_planb import (APPROACH_MM, DEFAULT_EXPONENT,               # noqa: E402
                            DEMONSTRATED_DEPTH_MM, HOVER_MM, Ledger,
                            RUNS_DIR, file_sha256, load_map, run_campaign,
                            scale_from_stiffness, serpentine)
from campaign_random import Lattice, sample_points                       # noqa: E402

RUN_PREFIX = "ladder_"

# The ladder this campaign was built for: 1.0 mm to 5.0 mm in 0.4 mm steps.
# ⚠ 1.0 to 5.0 INCLUSIVE at 0.4 mm is ELEVEN rungs, not ten -- so --points 100 is
# 1100 pokes, not 1000. Stated here because the arithmetic is easy to do wrong in
# either direction and the run is ~3 h long either way.
DEPTH_MIN_MM = 1.0
DEPTH_MAX_MM = 5.0
DEPTH_STEP_MM = 0.4

# Rough, measured from campaign_planb's own timing note (94 pokes in ~14 min).
SECONDS_PER_POKE = 9.0


def ladder(lo, hi, step):
    """Inclusive ladder, with the last rung snapped to `hi` if it lands close.

    Building this with a float accumulator is how a ladder quietly acquires a
    12th rung at 5.000000000000001 mm, or loses the 5.0 mm rung entirely.
    """
    n = int(round((hi - lo) / step))
    if n < 1:
        sys.exit("[ERROR] --depth-max-mm must exceed --depth-min-mm by at least one step")
    rungs = [round(lo + i * step, 6) for i in range(n + 1)]
    if abs(rungs[-1] - hi) > 1e-9:
        # The range is not an exact multiple of the step. Say so rather than
        # silently stopping short of the depth that was asked for.
        print("[WARN] %.3f to %.3f in %.3f mm steps does not land on %.3f exactly; "
              "the ladder ends at %.3f mm." % (lo, hi, step, hi, rungs[-1]))
    return rungs


# Extends campaign_random.PLAN_FIELDS: the analysis needs BOTH the depth-ladder
# columns (level_idx, depth_cmd_mm) and the random-geometry ones (col_f, row_f,
# node_dist_mm), because the two questions it asks are "does the error depend on
# depth" and "does it depend on where in the field the poke landed".
PLAN_FIELDS = list(cr.PLAN_FIELDS)


def build_plan(pts, depths, seed, b=DEFAULT_EXPONENT):
    """Randomised complete block: every sampled location x every depth rung, once.

    Structure is campaign_planb.build_plan's, re-implemented here for one reason:
    that function emits campaign_planb's field set, which DROPS col_f/row_f/
    frac_u/frac_v/node_dist_mm. Those are the columns that say where in the marker
    field a poke landed, and this campaign's second question ("is the estimate
    worse near the edges?") cannot be asked without them. Reusing the function and
    then patching the rows afterwards would leave two places that decide what a
    plan row is.

    PASS ORDERING. Within a pass the locations are walked serpentine (travel time:
    1100 pokes at the servo's 0.03 m/s is hours of the run), while the DEPTH each
    location gets on that pass is drawn from its own permutation of the rungs. So
    position is orderly within a pass and depth is not, which is exactly the
    correlation structure that matters: depth must not track time.
    """
    rng = random.Random(seed)
    n_pass = len(depths)
    perm = {p["point_index"]: rng.sample(range(n_pass), n_pass) for p in pts}

    plan = []
    for p_idx in range(n_pass):
        for loc in serpentine(pts, reverse_first=(p_idx % 2 == 1)):
            li = perm[loc["point_index"]][p_idx]
            d_cmd = float(depths[li])
            plan.append({
                "seq": len(plan),
                "block": "ladder",
                "pass": p_idx,
                "repeat": 0,
                "point_index": loc["point_index"],
                # Cell indices, for the progress line and the serpentine walk only.
                # The real location is (x, y) / (col_f, row_f); nothing downstream
                # may treat these as a map node.
                "row": loc["row"], "col": loc["col"],
                "x": loc["x"], "y": loc["y"],
                "surface_z": loc["surface_z"],
                "stiffness_n_per_mm": loc["k"],
                "level_idx": li,
                "target_depth_mm": d_cmd,
                # Informational only. The measured HEX21 force is the label; this
                # is what the interpolated local stiffness predicts.
                "target_force_n": scale_from_stiffness(loc["k"], b) * d_cmd ** b,
                "depth_cmd_mm": d_cmd,
                "col_f": loc["col_f"], "row_f": loc["row_f"],
                "frac_u": loc["frac_u"], "frac_v": loc["frac_v"],
                "node_dist_mm": loc["node_dist_mm"],
            })
    return plan


def to_locations(pts):
    """sample_points' dicts -> the shape build_plan/serpentine/run_campaign expect."""
    locs = []
    for i, p in enumerate(pts):
        loc = dict(p)
        loc["point_index"] = i
        loc["row"], loc["col"] = p["cell_row"], p["cell_col"]
        locs.append(loc)
    return locs


def report(plan, locs, depths, stats, args, run_dir, ledger, remaining):
    n = len(plan)
    print("\n" + "=" * 78)
    print("  campaign_ladder -- %d locations x %d depth rungs = %d pokes"
          % (len(locs), len(depths), n))
    print("=" * 78)
    print("  run dir              : %s" % run_dir)
    print("  depth ladder (%2d)    : %s mm"
          % (len(depths), " ".join("%.1f" % d for d in depths)))
    print("  deepest rung         : %.2f mm%s"
          % (max(depths),
             "" if max(depths) <= DEMONSTRATED_DEPTH_MM
             else "   ⚠ beyond the %.1f mm demonstrated limit" % DEMONSTRATED_DEPTH_MM))
    print("  seed                 : %d" % args.seed)
    sep = cr.pair_min_sep_mm(locs)
    if sep is not None:
        print("  closest two points   : %.2f mm apart" % sep)
    print("  sampler acceptance   : %.1f%% (%d tries)"
          % (100.0 * stats["acceptance_rate"], stats["tries"]))
    forces = [p["target_force_n"] for p in plan]
    print("  predicted force      : %.2f - %.2f N (informational; HEX21 is the label)"
          % (min(forces), max(forces)))
    already = n - len(remaining)
    if already:
        print("  already in ledger    : %d  ->  %d remaining" % (already, len(remaining)))
    mins = len(remaining) * SECONDS_PER_POKE / 60.0
    print("  estimated duration   : ~%.0f min (~%.1f h) at %.0f s/poke"
          % (mins, mins / 60.0, SECONDS_PER_POKE))
    print("=" * 78 + "\n")


def print_contact_checklist():
    """The one thing that silently ruins this particular campaign."""
    print("""
+----------------------------------------------------------------------------+
|  BEFORE THIS RUN -- the contact log is the POINT of this campaign           |
+----------------------------------------------------------------------------+
  On the WORKSTATION, in E_BTS_GUI (launched from the repo root):

    1. Circle Tracking pane OPEN.  The contact estimate is computed from its
       event windows; with the pane never opened there is no marker baseline
       and <run>_contact.csv will be empty.

    2. Press REBUILD BASELINE with NOTHING TOUCHING THE PAD, and wait for the
       console line

           Baseline collected N stable observed circles ... Circle map built:

       A baseline captured under load defines the deformed shape as
       "undeformed". Every contact coordinate in the run is then wrong, by a
       constant nothing downstream can detect or undo.

    3. Console shows the calibration actually in force, e.g.

           [contact] pixel->mm from calibration/pixel_to_mm.json;
                     pad centre at robot (654.98, -0.82) mm

       "no elastomer origin" means coordinates will be ROBOT-BASE, not
       pad-centred -- still analysable, but not what was asked for.

    4. Force/Torque + Sequence Recording panes open, as for any recorded run.
       The HEX21 must be on the WORKSTATION (not tactile) for this campaign.

  This script cannot check any of the above from tactile. It is on you.
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_argument_group("mode")
    mode.add_argument("--dry-run", action="store_true",
                      help="plan, report and preview only. No ROS, no motion.")

    des = ap.add_argument_group("design")
    des.add_argument("--points", "-n", type=int, default=100,
                     help="number of random locations (default 100)")
    des.add_argument("--seed", type=int, default=7)
    des.add_argument("--map", default=str(pb.MAP_CSV))
    des.add_argument("--depth-min-mm", type=float, default=DEPTH_MIN_MM)
    des.add_argument("--depth-max-mm", type=float, default=DEPTH_MAX_MM)
    des.add_argument("--depth-step-mm", type=float, default=DEPTH_STEP_MM)
    des.add_argument("--min-node-dist-mm", type=float, default=0.0)
    # ⚠ NOT 0.0, unlike campaign_random. Uniform-random sampling clumps: at
    # --points 100 the closest pair came out 0.27 mm apart, which for THIS campaign
    # is two problems. (1) Coverage: the run exists to map localisation error across
    # the pad, and a near-duplicate spends 22 pokes (2 locations x 11 rungs)
    # measuring one spot while leaving a gap elsewhere. (2) The material: 22 pokes
    # to 5 mm in one place over ~3 h invites local creep, which would show up as a
    # position-dependent error that is really a history-dependent one.
    # 1.5 mm is chosen against the sampler's own jamming limit: random sequential
    # packing saturates near 0.696*A/d^2, giving d ~ 2.4 mm as the hard limit for
    # 100 points here (15.9% acceptance, near-jammed). 1.5 mm accepts 44% of draws
    # and still guarantees every location is its own.
    des.add_argument("--min-sep-mm", type=float, default=1.5,
                     help="minimum spacing between sampled locations (default 1.5)")
    des.add_argument("--use-poor-points", action="store_true")
    des.add_argument("--ack-deep", action="store_true")

    res = ap.add_argument_group("resume")
    res.add_argument("--resume", action="store_true")
    res.add_argument("--run-dir", default=None)
    res.add_argument("--fresh", action="store_true")

    mot = ap.add_argument_group("motion")
    mot.add_argument("--dwell-s", type=float, default=pb.DWELL_S)
    mot.add_argument("--tare-s", type=float, default=pb.TARE_S)
    mot.add_argument("--log-hz", type=float, default=200.0)
    mot.add_argument("--no-level", action="store_true")
    mot.add_argument("--no-recovery", action="store_true")
    mot.add_argument("--no-depth-correction", action="store_true")
    args = ap.parse_args()

    depths = ladder(args.depth_min_mm, args.depth_max_mm, args.depth_step_mm)
    if max(depths) > DEMONSTRATED_DEPTH_MM and not args.ack_deep:
        sys.exit("[ERROR] deepest rung %.2f mm is past the %.1f mm demonstrated limit.\n"
                 "        Deeper is extrapolation and the straps are the real "
                 "constraint.\n        Pass --ack-deep only after the strap test at "
                 "full depth." % (max(depths), DEMONSTRATED_DEPTH_MM))

    map_path = Path(args.map)
    all_locs, _ = load_map(map_path, use_poor=True)
    lat = Lattice(all_locs)
    if args.use_poor_points:
        print("[WARN] --use-poor-points: cells with an unmeasured corner are allowed;\n"
              "       those pokes EXTRAPOLATE the surface datum.")

    # ONE rng for the draw, a SEPARATE deterministic one for the depth permutation
    # (seeded off the same seed), so --resume rebuilds the plan bit-for-bit.
    rng = random.Random(args.seed)
    pts, stats = sample_points(lat, args.points, rng,
                               min_node_mm=args.min_node_dist_mm,
                               min_sep_mm=args.min_sep_mm)
    locs = to_locations(pts)
    plan = build_plan(locs, depths, args.seed)

    deepest = max(p["depth_cmd_mm"] for p in plan)
    if deepest > max(depths) + 1e-9:
        sys.exit("[ERROR] internal: planned depth %.4f exceeds the ladder top %.4f."
                 % (deepest, max(depths)))

    params = {
        "sampler": "continuous-uniform in lattice parameter space (campaign_random)",
        "design": "randomised complete block: every location x every depth rung",
        "map_sha256": file_sha256(map_path),
        "n_locations": args.points,
        "n_depth_levels": len(depths),
        "n_pokes": len(plan),
        "seed": args.seed,
        "depths_mm": depths,
        "depth_targeted": True,
        "min_node_dist_mm": args.min_node_dist_mm,
        "min_sep_mm": args.min_sep_mm,
        "use_poor_points": args.use_poor_points,
        "surface_reference": "bilinear interpolation of surface_offset_map",
        "exponent_b": DEFAULT_EXPONENT,
        "dwell_s": args.dwell_s,
        "tare_s": args.tare_s,
        "hover_mm": HOVER_MM,
        "approach_mm": APPROACH_MM,
        "depth_correction": not args.no_depth_correction,
        "lattice_cols": lat.n_c,
        "lattice_rows": lat.n_r,
        "cells_usable": lat.n_cells_ok,
        "purpose": "contact-location accuracy vs depth (SOFTWARE_README 11b)",
    }
    fp = pb.plan_fingerprint(plan, params)
    meta_only = {"map_path": str(map_path.resolve()),
                 "indenter_diameter_mm": cr.INDENTER_MM,
                 "sampler_stats": stats,
                 "script": Path(__file__).name}

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.resume:
        cands = sorted(d for d in RUNS_DIR.glob(RUN_PREFIX + "*")
                       if (d / "state.jsonl").exists())
        if not cands:
            sys.exit("[ERROR] --resume: no %s* run with a ledger under %s."
                     % (RUN_PREFIX, RUNS_DIR))
        run_dir = cands[-1]
        print("[RESUME] newest run with a ledger: %s" % run_dir)
    else:
        run_dir = RUNS_DIR / (RUN_PREFIX + time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(run_dir / "state.jsonl", fp)
    _, err = ledger.load()
    if err and not args.fresh:
        sys.exit("[ERROR] cannot resume: %s\n\n"
                 "        That ledger came from a campaign built with DIFFERENT "
                 "parameters --\n        a different --seed, --points or ladder gives a "
                 "different plan, and\n        continuing would stitch two experiments "
                 "into one dataset.\n\n        Pass the original parameters, or --fresh "
                 "to archive it (nothing is deleted)." % err)
    if args.fresh:
        arch = ledger.archive()
        if arch:
            print("[FRESH] previous ledger archived -> %s" % arch.name)

    remaining = [p for p in plan if p["seq"] not in ledger.done]
    cr.save_plan(run_dir, plan, params, fp, meta_only)
    report(plan, locs, depths, stats, args, run_dir, ledger, remaining)

    if args.dry_run:
        png = cr.preview_png(plan, lat, run_dir / "plan_preview.png")
        print("[DRY RUN] plan -> %s" % (run_dir / "plan.csv"))
        if png:
            print("[DRY RUN] preview -> %s" % png)
        print("[DRY RUN] no ROS, no motion. Nothing has moved.\n")
        return 0

    if not remaining:
        print("  ✅ nothing to do -- all %d pokes are already in the ledger.\n" % len(plan))
        return 0

    print_contact_checklist()
    ledger.open(dict(params, **meta_only))
    run_campaign(args, plan, plan, run_dir, ledger, remaining)
    return 0


if __name__ == "__main__":
    sys.exit(main())
