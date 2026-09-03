#!/usr/bin/env python3
"""
calib_raster.py -- a REGULAR raster of shallow pokes, for camera calibration.

This is not a force experiment. It exists to produce one thing: a dense set of
correspondences

    (where the indenter WAS, in millimetres)  <->  (where the contact APPEARED,
                                                    in sensor pixels)

from which ml/fit_pixel_mm_warp.py fits the pixel -> millimetre map of the lens,
including its radial (fisheye) distortion. See CALIB_RASTER.md for the runbook and
ml/fit_pixel_mm_warp.py for what happens to the data afterwards.

WHY A RASTER AND NOT THE EXISTING CAMPAIGN DATA
------------------------------------------------
The 40 recorded campaigns already pair a robot XY with a camera recording, and
those pairs ARE usable (they are the validation set fit_pixel_mm_warp.py checks
against). They are a poor FITTING set for three reasons:

  * campaign_random.py samples uniformly at random, so coverage of the field is
    lumpy -- and distortion is a smooth function of RADIUS, so what a distortion
    fit needs is points spread evenly out to the corners, which is exactly where
    random sampling is thinnest.
  * every campaign varies depth on purpose. Depth changes how far the elastomer
    shears, which moves the apparent contact centroid. Fitting geometry across a
    depth ladder confounds lens distortion with material response.
  * nothing in those runs measures the REPEATABILITY of the contact-centroid
    estimator, so a residual could not be attributed to the lens rather than to
    the estimator's own noise.

This raster fixes all three: even coverage, ONE depth, and repeats.

⚠ THE INDENTER MUST BE THE SMALL PIN
------------------------------------
Calibration localises a contact. A 16 mm face does not have a location -- it has
an area, and its divergence lobe is broad and flat, so the peak wanders by whole
lattice cells for reasons that have nothing to do with the lens. Fit the 3 mm pin
(or smaller) and record its diameter with --indenter-mm; the script prints a loud
warning above 6 mm and refuses above 10 mm.

⚠ ONE DEPTH, DELIBERATELY
-------------------------
--depths-mm defaults to a single 2.0 mm. 2 mm is deep enough that the divergence
peak is unambiguous (the two-cylinder runs of 2026-09-02 resolved 8 mm-separated
contacts at this depth) and shallow enough to stay far from the strap limit.

Passing TWO depths is a free bias test rather than a mistake: the contact centroid
should not move when the indenter goes deeper, because the indenter axis has not
moved. If it does move, the estimator is depth-biased and fit_pixel_mm_warp.py
will say so per-depth instead of averaging the bias away. It costs one extra pass.

DESIGN
------
  * RASTER IN LATTICE COORDINATES, not in millimetres. The pad is reached through
    the same bilinear interpolation over the measured surface map that
    campaign_random.py uses, so each target carries an interpolated surface_z and
    the depth means the same thing everywhere. A raster in raw robot XY would
    ignore the pad's tilt and press to different depths at different corners.
  * INSET FROM THE BORDER. ml/contact_detect.py masks one lattice cell at every
    border (a central difference cannot be taken there), so a contact under the
    outermost cells has no divergence and cannot be localised. --inset-cells
    keeps the raster inside the region where the estimator is defined.
  * SERPENTINE, ALTERNATING PER PASS. Same reason as campaign_planb: traverse
    efficiency without correlating sweep direction with row parity. Repeats are
    passes, so any creep or thermal drift spreads across the whole field instead
    of aliasing onto one band of it.
  * RESUME. Identical Ledger to campaign_planb, fingerprinted the same way.

Everything a run produces lands in one directory, the same layout the rest of the
post-processing chain already expects:

    ~/E-BTS/recordings/calib_<stamp>/
        plan.csv        the enumerated pokes, in execution order
        plan.json       parameters + fingerprint
        state.jsonl     append-only progress (the resume ledger)
        franka_segNN.csv  one franka_states log per segment

RUN IT FROM THE WORKSTATION (never directly on tactile), so the camera and the
HEX21 are recording the same pokes:

    python3 master_campaign.py calib --remote-script calib_raster.py \\
        --remote-args --indenter-mm 3.0

    --remote-args must come LAST. Add --dry-run to see the plan and the time/size
    estimate without moving anything.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_planb as pb                                          # noqa: E402
from campaign_planb import (Ledger, file_sha256, load_map,           # noqa: E402
                            plan_fingerprint, run_campaign, serpentine,
                            DEMONSTRATED_DEPTH_MM, DWELL_S,
                            HARD_DEPTH_CEILING_MM, MAP_CSV, RUNS_DIR, TARE_S)

RUN_PREFIX = "calib_"

# Wall-clock per poke, measured: the 594-poke pilot took ~1.3 h end to end.
SECONDS_PER_POKE = 7.9
# .raw size per poke at the campaign's event rate, from the same pilot (19 GB/594).
GB_PER_POKE = 0.032

DEFAULT_DEPTH_MM = 2.0
DEFAULT_COLS = 11
DEFAULT_ROWS = 9
DEFAULT_REPEATS = 3
DEFAULT_INSET_CELLS = 0.5

# The contact-centroid estimator degrades with contact size long before it fails
# outright, so warn early and refuse only where the result would be meaningless.
INDENTER_WARN_MM = 6.0
INDENTER_REFUSE_MM = 10.0

# Calibration never needs depth. Anything past this is a typo, not an intention.
CALIB_MAX_DEPTH_MM = 4.0


# =============================================================================
#  the mapped lattice
# =============================================================================

class Lattice(object):
    """The mapped grid, addressable by CONTINUOUS (col_f, row_f).

    A deliberate COPY of campaign_random.Lattice's interpolation, per the repo's
    copy-don't-modify rule (HANDOFF section 12): this script must keep working if
    the random-campaign sampler is changed for reasons of its own, and the two
    have no shared requirement beyond "bilinear over the same map".

    Holds ALL map rows, including the not-quality_ok ones, because the lattice
    geometry needs to be complete; `cell_ok` is what refuses a bad cell. Dropping
    bad nodes instead would close the gap and interpolate ACROSS a hole, which is
    extrapolation wearing interpolation's clothes.
    """

    def __init__(self, all_locs):
        self.cols = sorted(set(l["col"] for l in all_locs))
        self.rows = sorted(set(l["row"] for l in all_locs))
        self.node = {(l["col"], l["row"]): l for l in all_locs}
        missing = [(c, r) for r in self.rows for c in self.cols
                   if (c, r) not in self.node]
        if missing:
            sys.exit("[ERROR] the map is not a complete rectangular lattice: %d "
                     "(col,row) slots absent, e.g. %s.\n"
                     "        Bilinear interpolation needs a full grid."
                     % (len(missing), missing[:5]))
        self.n_c, self.n_r = len(self.cols), len(self.rows)
        if self.n_c < 2 or self.n_r < 2:
            sys.exit("[ERROR] lattice is %dx%d -- need at least 2x2 to interpolate."
                     % (self.n_c, self.n_r))
        self._node_xy = np.array([[l["x"], l["y"]] for l in all_locs], float)

    def corners(self, ci, ri):
        c0, c1 = self.cols[ci], self.cols[ci + 1]
        r0, r1 = self.rows[ri], self.rows[ri + 1]
        return (self.node[(c0, r0)], self.node[(c1, r0)],
                self.node[(c0, r1)], self.node[(c1, r1)])

    def cell_ok(self, ci, ri):
        return all(n["quality_ok"] for n in self.corners(ci, ri))

    def interp(self, cf, rf):
        """Bilinear interpolation at continuous (cf, rf) -- LATTICE indices, not mm."""
        ci = min(int(math.floor(cf)), self.n_c - 2)
        ri = min(int(math.floor(rf)), self.n_r - 2)
        fu, fv = cf - ci, rf - ri
        n00, n10, n01, n11 = self.corners(ci, ri)
        w = ((1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv)

        def blend(key):
            return (w[0] * n00[key] + w[1] * n10[key] +
                    w[2] * n01[key] + w[3] * n11[key])

        return {"cell_col": ci, "cell_row": ri, "col_f": cf, "row_f": rf,
                "cell_ok": self.cell_ok(ci, ri),
                "x": blend("x"), "y": blend("y"),
                "surface_z": blend("surface_z"), "k": blend("k")}

    def span_mm(self):
        xs, ys = self._node_xy[:, 0], self._node_xy[:, 1]
        return (float((xs.max() - xs.min()) * 1e3),
                float((ys.max() - ys.min()) * 1e3))


# =============================================================================
#  the raster
# =============================================================================

def build_raster(lat, n_cols, n_rows, inset, use_poor=False):
    """Evenly spaced targets in continuous lattice coordinates.

    Returns (points, refused). A point in a cell with a failed map corner is
    REFUSED rather than pressed: its surface_z is interpolated from a node whose
    height was never trusted, so the depth there would be unknown -- and an
    unknown depth is exactly the confound this run is designed not to have.
    """
    if n_cols < 2 or n_rows < 2:
        sys.exit("[ERROR] --cols and --rows must both be >= 2.")
    lo_c, hi_c = inset, (lat.n_c - 1) - inset
    lo_r, hi_r = inset, (lat.n_r - 1) - inset
    if hi_c <= lo_c or hi_r <= lo_r:
        sys.exit("[ERROR] --inset-cells %.2f leaves no room inside a %dx%d lattice."
                 % (inset, lat.n_c, lat.n_r))

    pts, refused = [], []
    idx = 0
    for iv, rf in enumerate(np.linspace(lo_r, hi_r, n_rows)):
        for iu, cf in enumerate(np.linspace(lo_c, hi_c, n_cols)):
            p = lat.interp(float(cf), float(rf))
            p.update({"point_index": idx, "row": iv, "col": iu})
            idx += 1
            if p["cell_ok"] or use_poor:
                pts.append(p)
            else:
                refused.append(p)
    if not pts:
        sys.exit("[ERROR] every raster point fell in a cell with an untrusted map "
                 "corner. Re-map the surface, or pass --use-poor-points knowing "
                 "the depths will be unreliable.")
    return pts, refused


def min_separation_mm(pts):
    """Closest pair in the raster -- the resolution the estimator must beat."""
    if len(pts) < 2:
        return float("nan")
    xy = np.array([[p["x"], p["y"]] for p in pts], float) * 1e3
    best = float("inf")
    for i in range(len(xy)):
        d = np.hypot(xy[i + 1:, 0] - xy[i, 0], xy[i + 1:, 1] - xy[i, 1])
        if d.size:
            best = min(best, float(d.min()))
    return best


PLAN_FIELDS = ["seq", "block", "pass", "repeat", "point_index", "row", "col",
               "x", "y", "surface_z", "stiffness_n_per_mm", "level_idx",
               "target_depth_mm", "target_force_n", "depth_cmd_mm",
               "col_f", "row_f", "cell_col", "cell_row"]


def build_plan(pts, depths, repeats):
    """One pass per (repeat, depth), serpentine, direction alternating each pass.

    Order is DETERMINISTIC, not randomised. campaign_planb randomises level order
    because level is its design variable and must be decorrelated from time. Here
    the design variable is POSITION, and position cannot be decorrelated from time
    by shuffling -- the arm still has to travel. Alternating the serpentine is the
    honest control: every pass crosses the field in the opposite sense, so a drift
    that grows with time enters the two passes with opposite spatial gradients and
    shows up as scatter between repeats rather than as a fake distortion.
    """
    plan, seq = [], 0
    for pass_idx, (rep, li) in enumerate(
            [(r, li) for r in range(repeats) for li in range(len(depths))]):
        walk = serpentine(pts, reverse_first=(pass_idx % 2 == 1))
        for p in walk:
            plan.append({
                "seq": seq, "block": "calib", "pass": pass_idx, "repeat": rep,
                "point_index": p["point_index"], "row": p["row"], "col": p["col"],
                "x": p["x"], "y": p["y"], "surface_z": p["surface_z"],
                "stiffness_n_per_mm": p["k"], "level_idx": li,
                "target_depth_mm": depths[li],
                "target_force_n": float("nan"),   # not targeted; measured in post
                "depth_cmd_mm": depths[li],
                "col_f": p["col_f"], "row_f": p["row_f"],
                "cell_col": p["cell_col"], "cell_row": p["cell_row"],
            })
            seq += 1
    return plan


def save_plan(run_dir, plan, params, fp, meta_only=None):
    with open(str(run_dir / "plan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        w.writeheader()
        for p in plan:
            w.writerow({k: p[k] for k in PLAN_FIELDS})
    meta = dict(params)
    meta.update(meta_only or {})
    meta["fingerprint"] = fp
    meta["n_pokes"] = len(plan)
    (run_dir / "plan.json").write_text(json.dumps(meta, indent=2, default=str))


# =============================================================================
#  reporting
# =============================================================================

def report(lat, pts, refused, plan, depths, args, run_dir, ledger, remaining):
    su, sv = lat.span_mm()
    n_pts = len(pts)
    mins = min_separation_mm(pts)
    secs = len(remaining) * SECONDS_PER_POKE

    print("\n" + "=" * 74)
    print("  CALIBRATION RASTER -- geometry only, not a force experiment")
    print("=" * 74)
    print("  map                  : %s" % args.map)
    print("  mapped area          : %.1f x %.1f mm (%d x %d nodes)"
          % (su, sv, lat.n_c, lat.n_r))
    print("  raster               : %d cols x %d rows, inset %.2f cells"
          % (args.cols, args.rows, args.inset_cells))
    print("  targets              : %d usable%s"
          % (n_pts, "" if not refused else
             ", %d REFUSED (untrusted map corner)" % len(refused)))
    print("  closest pair         : %.2f mm" % mins)
    print("  depth(s)             : %s mm" % ", ".join("%.2f" % d for d in depths))
    print("  repeats              : %d  ->  %d passes over the field"
          % (args.repeats, args.repeats * len(depths)))
    print("  indenter             : %.1f mm" % args.indenter_mm)
    print("  pokes                : %d total, %d remaining" % (len(plan), len(remaining)))
    print("  estimated            : %.0f min, ~%.1f GB of .raw"
          % (secs / 60.0, len(remaining) * GB_PER_POKE))
    print("  run dir              : %s" % run_dir)

    if refused:
        cells = sorted(set((p["cell_col"], p["cell_row"]) for p in refused))
        print("\n  refused cells (col,row): %s" % ", ".join(str(c) for c in cells[:12]))
        print("  -> coverage has holes there. The fit will still work; it simply has")
        print("     no data in that part of the field, and extrapolating a distortion")
        print("     polynomial into a hole is how a calibration goes quietly wrong.")

    if args.indenter_mm > INDENTER_WARN_MM:
        print("\n  ⚠ INDENTER %.1f mm IS TOO BIG TO LOCALISE WELL." % args.indenter_mm)
        print("    A broad face makes a broad divergence lobe, and the peak of a")
        print("    plateau is set by noise. Expect the residuals to be dominated by")
        print("    the estimator rather than by the lens. Fit the 3 mm pin.")

    if len(depths) > 1:
        print("\n  Two depths requested: fit_pixel_mm_warp.py will report the fit")
        print("  PER DEPTH. If the contact centroid moves with depth, the estimator")
        print("  is depth-biased and the difference is the size of that bias.")

    print("=" * 74 + "\n")


# =============================================================================
#  main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Regular raster of shallow pokes for camera calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    mode = ap.add_argument_group("mode")
    mode.add_argument("--dry-run", action="store_true",
                      help="build and print the plan, write the sidecar; no motion")
    mode.add_argument("--self-test", action="store_true",
                      help="raster + plan checks; no ROS, no map, no robot")

    res = ap.add_argument_group("resume")
    res.add_argument("--resume", action="store_true",
                     help="continue the run in --run-dir, skipping completed pokes")
    res.add_argument("--run-dir", default=None,
                     help="run directory (default: newest calib_* under "
                          "~/E-BTS/recordings, or a new one)")
    res.add_argument("--fresh", action="store_true",
                     help="archive any existing ledger and start from poke 0")

    des = ap.add_argument_group("design")
    des.add_argument("--map", default=str(MAP_CSV), help="surface_offset_map.csv")
    des.add_argument("--cols", type=int, default=DEFAULT_COLS,
                     help="raster columns (default %d)" % DEFAULT_COLS)
    des.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                     help="raster rows (default %d)" % DEFAULT_ROWS)
    des.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                     help="passes over the whole field (default %d). Repeats are "
                          "what separate estimator noise from lens distortion."
                          % DEFAULT_REPEATS)
    des.add_argument("--depths-mm", type=float, nargs="+", default=[DEFAULT_DEPTH_MM],
                     help="ONE depth by default (%.1f mm). Two depths turns the run "
                          "into a depth-bias test at the cost of one extra pass."
                          % DEFAULT_DEPTH_MM)
    des.add_argument("--inset-cells", type=float, default=DEFAULT_INSET_CELLS,
                     help="keep the raster this many lattice cells clear of the map "
                          "border (default %.1f)" % DEFAULT_INSET_CELLS)
    des.add_argument("--use-poor-points", action="store_true",
                     help="also press into cells with an untrusted map corner")
    des.add_argument("--indenter-mm", type=float, default=3.0,
                     help="indenter diameter at the surface. Recorded in plan.json, "
                          "and checked: this run cannot do its job with a big face.")

    mot = ap.add_argument_group("motion")
    mot.add_argument("--dwell-s", type=float, default=DWELL_S)
    mot.add_argument("--tare-s", type=float, default=TARE_S)
    mot.add_argument("--log-hz", type=float, default=200.0)
    mot.add_argument("--no-level", action="store_true")
    mot.add_argument("--no-recovery", action="store_true")
    mot.add_argument("--no-depth-correction", action="store_true")

    args = ap.parse_args()

    if args.self_test:
        return self_test()

    # ---- safety / sanity gates ---------------------------------------------
    depths = sorted(float(d) for d in args.depths_mm)
    if min(depths) <= 0:
        sys.exit("[ERROR] --depths-mm must all be > 0.")
    if max(depths) > CALIB_MAX_DEPTH_MM:
        sys.exit("[ERROR] --depths-mm %.2f exceeds this script's %.1f mm ceiling.\n"
                 "        Calibration is a GEOMETRY measurement; it has no reason to "
                 "press deep,\n        and %.1f mm is already well inside the %.1f mm "
                 "demonstrated limit.\n        If you want deep presses, that is a "
                 "campaign, not a calibration."
                 % (max(depths), CALIB_MAX_DEPTH_MM, CALIB_MAX_DEPTH_MM,
                    DEMONSTRATED_DEPTH_MM))
    if max(depths) > HARD_DEPTH_CEILING_MM:      # unreachable, kept as a belt
        sys.exit("[ERROR] internal: depth past the shared hard ceiling.")
    if args.indenter_mm >= INDENTER_REFUSE_MM:
        sys.exit("[ERROR] --indenter-mm %.1f cannot localise a contact.\n"
                 "        A %.0f mm face spans several lattice cells, so its "
                 "divergence peak\n        is not a position. Fit the 3 mm pin and "
                 "re-run." % (args.indenter_mm, args.indenter_mm))
    if args.repeats < 1:
        sys.exit("[ERROR] --repeats must be >= 1.")
    if args.repeats < 2:
        print("[WARN] --repeats 1 gives no estimate of the contact-centroid's own\n"
              "       noise, so a residual cannot be attributed to the lens. The fit\n"
              "       will run, but it cannot tell you whether to believe it.")

    # ---- build the raster ---------------------------------------------------
    locs, skipped = load_map(Path(args.map), use_poor=True)   # geometry needs them all
    lat = Lattice(locs)
    pts, refused = build_raster(lat, args.cols, args.rows, args.inset_cells,
                                use_poor=args.use_poor_points)
    plan = build_plan(pts, depths, args.repeats)

    params = {
        "map_sha256": file_sha256(Path(args.map)),
        "purpose": "camera_calibration_raster",
        "cols": args.cols, "rows": args.rows,
        "inset_cells": args.inset_cells,
        "depths_mm": [round(d, 4) for d in depths],
        "repeats": args.repeats,
        "use_poor_points": args.use_poor_points,
        "n_targets": len(pts), "n_refused": len(refused),
        "dwell_s": args.dwell_s, "tare_s": args.tare_s,
    }
    fp = plan_fingerprint(plan, params)
    meta_only = {"map_path": str(Path(args.map).resolve()),
                 "indenter_diameter_mm": args.indenter_mm,
                 "min_separation_mm": round(min_separation_mm(pts), 4)}

    # ---- resolve the run directory -----------------------------------------
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.resume:
        cands = sorted(d for d in RUNS_DIR.glob(RUN_PREFIX + "*")
                       if (d / "state.jsonl").exists())
        if not cands:
            sys.exit("[ERROR] --resume: no calibration run with a ledger under %s.\n"
                     "        A --dry-run preview writes plan.csv but no ledger, so "
                     "it is not\n        an interrupted run." % RUNS_DIR)
        run_dir = cands[-1]
        print("[RESUME] newest calibration run with a ledger: %s" % run_dir)
    else:
        run_dir = RUNS_DIR / (RUN_PREFIX + time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(run_dir / "state.jsonl", fp)
    n_rec, err = ledger.load()
    if err and not args.fresh:
        sys.exit("[ERROR] cannot resume: %s\n\n"
                 "        The ledger in %s was written by a raster built from\n"
                 "        DIFFERENT parameters. Continuing would stitch two different\n"
                 "        geometries into one calibration.\n\n"
                 "        Either pass the original parameters, or --fresh to archive\n"
                 "        that ledger and start over (nothing is deleted)."
                 % (err, ledger.path))
    if args.fresh:
        arch = ledger.archive()
        if arch:
            print("[FRESH] previous ledger archived -> %s" % arch.name)

    remaining = [p for p in plan if p["seq"] not in ledger.done]

    save_plan(run_dir, plan, params, fp, meta_only)
    report(lat, pts, refused, plan, depths, args, run_dir, ledger, remaining)

    if args.dry_run:
        print("[DRY RUN] plan written to %s -- no ROS, no motion."
              % (run_dir / "plan.csv"))
        return 0

    if not remaining:
        print("  ✅ nothing to do -- all %d pokes are already in the ledger." % len(plan))
        return 0

    ledger.open(dict(params, **meta_only))
    # run_campaign is campaign_planb's, unmodified: same phases (travel/tare/dip/
    # dwell/retract), same logger, same ledger, so ml/make_poke_windows.py and the
    # rest of the post-processing chain read a calib_ run exactly as they read a
    # campaign. `pts` is passed as the locations so the preflight reach check
    # tests the RASTER targets, not the map nodes.
    run_campaign(args, plan, pts, run_dir, ledger, remaining)
    return 0


# =============================================================================
#  self-test -- no ROS, no map, no robot
# =============================================================================

def _fake_map(n_c=11, n_r=9, bad=()):
    locs = []
    for r in range(n_r):
        for c in range(n_c):
            locs.append({"point_index": r * n_c + c, "row": r, "col": c,
                         "x": 0.40 + 0.003 * c, "y": -0.015 + 0.003 * r,
                         "surface_z": 0.2690 + 0.00002 * r, "k": 0.5,
                         "fit_r2": 1.0, "quality_ok": (c, r) not in bad})
    return locs


def self_test():
    ok_all = True

    def check(label, cond):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s" % ("ok " if cond else "FAIL", label))

    lat = Lattice(_fake_map())

    # a cell centre is the mean of its four corners -- bilinear, by definition
    mid = lat.interp(2.5, 3.5)
    cs = lat.corners(2, 3)
    check("cell centre == mean of its corners",
          abs(mid["x"] - sum(n["x"] for n in cs) / 4.0) < 1e-12 and
          abs(mid["y"] - sum(n["y"] for n in cs) / 4.0) < 1e-12)

    # interpolated z can never leave the enclosing corners' range
    zs = [l["surface_z"] for l in _fake_map()]
    pts, refused = build_raster(lat, 11, 9, 0.5)
    check("interpolated surface_z stays inside the map's range",
          all(min(zs) - 1e-12 <= p["surface_z"] <= max(zs) + 1e-12 for p in pts))

    # the raster is inset, evenly spaced, and complete
    check("raster is complete (11x9 = 99 targets)", len(pts) == 99 and not refused)
    cfs = sorted(set(round(p["col_f"], 6) for p in pts))
    check("raster inset from the border", cfs[0] >= 0.5 - 1e-9 and
          cfs[-1] <= (lat.n_c - 1) - 0.5 + 1e-9)
    gaps = np.diff(cfs)
    check("raster columns evenly spaced", float(gaps.max() - gaps.min()) < 1e-9)

    # a failed map corner refuses its whole cell, and only that cell
    lat_bad = Lattice(_fake_map(bad=((0, 0),)))
    pts_b, ref_b = build_raster(lat_bad, 11, 9, 0.5)
    check("a bad map corner refuses points, not the run",
          len(ref_b) > 0 and len(pts_b) == 99 - len(ref_b))
    check("refusals are confined to cells touching the bad node",
          all(p["cell_col"] <= 0 and p["cell_row"] <= 0 for p in ref_b))

    # the plan enumerates every target once per pass, and passes alternate
    plan = build_plan(pts, [2.0], 3)
    check("plan = targets x repeats x depths", len(plan) == 99 * 3)
    check("every target appears once per pass",
          all(sorted(p["point_index"] for p in plan if p["pass"] == k) ==
              sorted(p["point_index"] for p in pts) for k in range(3)))
    first = [p["point_index"] for p in plan if p["pass"] == 0]
    second = [p["point_index"] for p in plan if p["pass"] == 1]
    check("consecutive passes traverse in opposite directions",
          first[:11] == list(reversed(second[:11])))
    check("seq is dense and ordered",
          [p["seq"] for p in plan] == list(range(len(plan))))
    check("depth is commanded verbatim at every poke",
          all(abs(p["depth_cmd_mm"] - 2.0) < 1e-12 for p in plan))

    # two depths => two levels per repeat, both present at every target
    plan2 = build_plan(pts, [1.5, 2.5], 2)
    check("two depths give 2 x repeats passes", len(plan2) == 99 * 4)
    check("both depths reach every target",
          all(len({p["depth_cmd_mm"] for p in plan2
                   if p["point_index"] == i}) == 2 for i in range(0, 99, 17)))

    # the fingerprint must move when the geometry does, or resume cannot protect us
    fp_a = plan_fingerprint(plan, {"cols": 11})
    fp_b = plan_fingerprint(build_plan(pts, [2.0], 2), {"cols": 11})
    check("fingerprint changes with the plan", fp_a != fp_b)

    print("\n  %s" % ("✅ all checks passed" if ok_all else "❌ FAILURES above"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
