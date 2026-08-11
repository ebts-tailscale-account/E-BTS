#!/usr/bin/env python3
"""Map the surface height AND local stiffness across the elastomer, on a grid.

WHY THIS EXISTS
---------------
teach_surface.py measures 5 points objectively. That is enough to define a plane, but
the elastomer is not a plane: it is edge-strapped and unbacked, so it sags. Measured on
the 2026-08-08 five-point run the surface spans 1.501 mm across the block, and the
earlier campaign showed a bilinear plane over kissed corners sitting +0.863 mm above
the true contact surface in the interior -- an interpolation error, not a teaching error.
This maps the real shape so no interpolation is needed.

TWO OUTPUTS, BOTH REQUIRED FOR THE NEXT CAMPAIGN
  surface_z_m(x, y)        -- the depth datum. Without it, a commanded 1.5 mm press can
                              land anywhere from no-contact to 2.5 mm deep.
  stiffness_n_per_mm(x, y) -- lets presses target FORCE instead of depth. Because
                              F = k(x,y)*d and k varies several-fold, a fixed depth
                              ladder produces wildly uneven force coverage. That is
                              exactly why the last dataset had only 9% of presses above
                              3 N while the model was expected to work to 5.9 N.

⚠ REUSES teach_surface.probe_one -- it does NOT reimplement the descent. One
implementation of "rise, zero the load cell, descend, extrapolate to zero force" means
the map and the 5-point reference cannot disagree about what "the surface" is.

HOW THE HOVER HEIGHT IS CHOSEN PER POINT
Bilinearly interpolating the 5 measured points gives an estimate of the surface at
every grid location, good to a few tenths of a mm. Each descent starts a fixed height
above THAT estimate rather than above a global plane, so the search range stays tight
and the run stays fast even where the sag is large.

⚠ RESUMABLE. A 25-point map is ~20 min of robot time. If it aborts -- reflex, user
stop, a bad point -- re-running skips what already has a good measurement rather than
starting over. --redo re-probes specific points.

Prereqs (on tactile):
  * teach_surface.py --teach-xy and --probe-z already done: surface_points.json must
    contain 5 points with surface_z_m set.
  * HEX21 ON TACTILE (this reads force locally).
  * servo reach launch up, arm at HOME, franka_control NOT running.
  * Flange is levelled to straight-down before every descent (same as teach_surface).

Usage:
    python3 map_offset.py --dry-run              # grid, trajectory, timing. No motion.
    python3 map_offset.py                         # default 5 x 5 over the inset area
    python3 map_offset.py --nu 6 --nv 7
    python3 map_offset.py --margin-mm 4 4 4 8     # per-edge clearance, u_lo u_hi v_lo v_hi
    python3 map_offset.py --redo 7,12             # re-probe two points
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC_POINTS = HERE / "surface_points.json"
OUT_STEM = "surface_offset_map"
OUT_JSON = HERE / (OUT_STEM + ".json")
OUT_CSV = HERE / (OUT_STEM + ".csv")
OUT_PNG = HERE / (OUT_STEM + ".png")


def set_out_stem(stem):
    """Point the outputs at a different basename.

    ⚠ Needed so a diagnostic run -- e.g. the fine line that checks whether the v
    ripple is really the marker lattice aliased by the sampling pitch -- cannot
    overwrite a 99-point map that took 75 minutes to measure.
    """
    global OUT_STEM, OUT_JSON, OUT_CSV, OUT_PNG
    OUT_STEM = stem
    OUT_JSON = HERE / (stem + ".json")
    OUT_CSV = HERE / (stem + ".csv")
    OUT_PNG = HERE / (stem + ".png")

NU, NV = 5, 5             # grid counts across u (the ~30 mm axis) and v (the ~36 mm axis)
MARGIN_MM = [4.0, 4.0, 4.0, 4.0]   # u_lo, u_hi, v_lo, v_hi clearance from the taught quad
HOVER_MM = 2.0            # rise this far above the ESTIMATED surface before zeroing
SEARCH_BELOW_MM = 1.5     # ...and search this far below it. The estimate is good to a few
                          # tenths of a mm, so this can be tighter than teach_surface's 2.0
APPROACH_MM = 12.0        # travel height between points
MAX_ATTEMPTS = 3          # retry a low-confidence point IMMEDIATELY, while the arm is
                          # still there -- a later --redo pass pays the traverse again
                          # and measures under different conditions.
RETRY_STEP_SCALE = 1.5    # ⚠ each retry ENLARGES the fine step and goes slightly deeper.
RETRY_FORCE_SCALE = 1.25  # Low confidence means too few DISTINCT z levels, and that is
                          # caused by the impedance controller holding then slipping
                          # (HANDOFF §4.4). Repeating the same parameters would just
                          # reproduce the same clumping. A bigger step clears the ~0.5 mm
                          # tracking floor, and a wider force band spans more depth --
                          # both directly increase the number of distinct levels, and a
                          # wider z spread also conditions the line fit better.


def load_reference():
    if not SRC_POINTS.exists():
        sys.exit("[ERROR] %s not found. Run teach_surface.py --teach-xy then --probe-z "
                 "first." % SRC_POINTS.name)
    d = json.loads(SRC_POINTS.read_text())
    pts = d["points"]
    need = ["bottom-left", "top-left", "top-right", "bottom-right"]
    missing = [k for k in need if k not in pts or pts[k].get("surface_z_m") is None]
    if missing:
        sys.exit("[ERROR] these corners have no measured surface_z_m: %s\n"
                 "        Run teach_surface.py --probe-z first." % ", ".join(missing))
    bad = [k for k in pts if pts[k].get("surface_quality_ok") is False]
    if bad:
        print("⚠ WARNING: these reference points are flagged LOW CONFIDENCE: %s"
              % ", ".join(bad))
        print("  The hover estimate leans on them. Re-probe them before trusting this map.")
    return d, pts


def build_grid(pts, nu, nv, margins, order="raster", line_u=None):
    """Grid over the four measured corners, with per-edge margins.

    ⚠ ORDER MATTERS AND IT IS NOT COSMETIC. With serpentine, alternate rows are swept
    in opposite directions, and on 2026-08-08 that produced a 0.26 mm offset between
    them (r = +0.75 / +0.81 across two runs) -- a quarter of the total apparent surface
    variation, and 6x the 0.042 mm repeatability. "raster" sweeps every row the same
    way so any residual direction bias is CONSTANT and cancels out of relative depths.
    "serpentine" is kept only so the two can be compared directly.

    Corner positions use the MEASURED surface_z_m, not the kissed z, so the estimated
    surface a descent starts above is built from force-referenced data throughout.
    """
    def P(k):
        return np.array([pts[k]["xy_m"][0], pts[k]["xy_m"][1], pts[k]["surface_z_m"]])
    BL, TL, TR, BR = P("bottom-left"), P("top-left"), P("top-right"), P("bottom-right")
    u_len = np.linalg.norm(BR - BL) * 1e3
    v_len = np.linalg.norm(TL - BL) * 1e3
    mu_lo, mu_hi, mv_lo, mv_hi = margins
    if mu_lo + mu_hi >= u_len or mv_lo + mv_hi >= v_len:
        sys.exit("[ERROR] margins %s leave nothing inside the %.1f x %.1f mm quad."
                 % (margins, u_len, v_len))
    if line_u is not None:
        # A single column at a fixed fraction along u. np.linspace(a, b, 1) returns
        # [a], which would silently put the column on the u=0 EDGE rather than where
        # you asked -- so set it explicitly.
        us = np.array([float(line_u)])
        nu = 1
    else:
        us = np.linspace(mu_lo / u_len, 1.0 - mu_hi / u_len, nu)
    vs = np.linspace(mv_lo / v_len, 1.0 - mv_hi / v_len, nv)

    def bilinear(u, v):
        return ((1 - u) * (1 - v) * BL + u * (1 - v) * BR +
                (1 - u) * v * TL + u * v * TR)

    grid, i = [], 0
    for iv, v in enumerate(vs):
        cols = (range(nu) if order == "raster"
                else (range(nu) if iv % 2 == 0 else range(nu - 1, -1, -1)))
        for iu in cols:
            p = bilinear(us[iu], v)
            grid.append({"point_index": i, "row": iv, "col": iu,
                         "x_m": float(p[0]), "y_m": float(p[1]),
                         "surface_z_estimate_m": float(p[2]),
                         "surface_z_m": None, "stiffness_n_per_mm": None,
                         "fit_r2": None, "fit_distinct_z": None, "quality_ok": None})
            i += 1
    info = {"n_u": nu, "n_v": nv, "n_points": len(grid), "order": order,
            "taught_u_len_mm": float(u_len), "taught_v_len_mm": float(v_len),
            "margins_mm": {"u_lo": mu_lo, "u_hi": mu_hi, "v_lo": mv_lo, "v_hi": mv_hi},
            "span_u_mm": float(u_len - mu_lo - mu_hi),
            "span_v_mm": float(v_len - mv_lo - mv_hi),
            "pitch_u_mm": float((u_len - mu_lo - mu_hi) / max(nu - 1, 1)),
            "pitch_v_mm": float((v_len - mv_lo - mv_hi) / max(nv - 1, 1)),
            "estimate_range_mm": float((max(g["surface_z_estimate_m"] for g in grid) -
                                        min(g["surface_z_estimate_m"] for g in grid)) * 1e3)}
    return grid, info


# Parameters that change WHAT IS MEASURED, not just how fast. A stored point taken with
# different values of these is not comparable and must not be merged.
METHOD_KEYS = ("order", "settle_before_zero_s", "fine_step_mm", "coarse_step_mm",
               "fit_force_max_n", "hover_mm", "approach", "levelled")


def load_existing(grid, meta):
    """Merge a previous run so this is resumable -- matched on point_index, position AND
    measurement method.

    ⚠ THE METHOD CHECK IS NOT OPTIONAL. Switching serpentine -> raster moved the grid,
    and one point still landed within the position tolerance of an old one, so a
    measurement taken with the OLD diagonal approach and no settle dwell would have been
    silently merged into a run using the new one. Mixing methods inside a single map is
    exactly the error that yields a plausible file and a wrong answer.
    """
    if not OUT_JSON.exists():
        return 0
    try:
        prev = json.loads(OUT_JSON.read_text())
        old = {int(g["point_index"]): g for g in prev["points"]}
    except Exception as e:
        print("[warn] could not read %s (%s) -- starting fresh." % (OUT_JSON.name, e))
        return 0
    pm = prev.get("probe", {})
    differs = [k for k in METHOD_KEYS if pm.get(k) != meta.get(k)]
    if differs:
        print("[note] %s was measured with a DIFFERENT method -- reusing NONE of it."
              % OUT_JSON.name)
        for k in differs:
            print("         %-22s was %-34s now %s" % (k, pm.get(k), meta.get(k)))
        return 0
    n = 0
    for g in grid:
        o = old.get(g["point_index"])
        if not o or o.get("surface_z_m") is None:
            continue
        # ⚠ only reuse if the point is at the same place. A changed grid or re-taught
        # corners moves every location, and silently reusing an old Z there would be
        # attributing a measurement to coordinates it was never taken at.
        if (abs(o["x_m"] - g["x_m"]) > 2e-4) or (abs(o["y_m"] - g["y_m"]) > 2e-4):
            continue
        for k in ("surface_z_m", "stiffness_n_per_mm", "fit_r2", "fit_distinct_z",
                  "quality_ok", "probe_steps"):
            if k in o:
                g[k] = o[k]
        n += 1
    return n


def archive_previous():
    """Copy any existing map aside BEFORE the first write of a new run.

    ⚠ Learned expensively on 2026-08-08: a `--redo` invocation that omitted --nv 9
    rebuilt a 5x5 grid and its first save() destroyed a completed 45-point map. save()
    runs after every point, so by the time anything looks wrong the original is gone.
    One cheap copy makes that mistake recoverable instead of terminal.
    """
    import shutil
    if not OUT_JSON.exists():
        return None
    try:
        prev = json.loads(OUT_JSON.read_text())
        g = prev.get("grid", {})
        tag = "%dx%d" % (g.get("n_u", 0), g.get("n_v", 0))
        n = sum(1 for q in prev.get("points", []) if q.get("surface_z_m") is not None)
    except Exception:
        tag, n = "unknown", 0
    stem = "surface_offset_map.prev_%s_%dpts" % (tag, n)
    for src, ext in ((OUT_JSON, "json"), (OUT_CSV, "csv"), (OUT_PNG, "png")):
        if src.exists():
            shutil.copy2(src, HERE / ("%s.%s" % (stem, ext)))
    print("  archived the previous map -> %s.{json,csv,png}" % stem)
    return stem


def save(grid, info, meta):
    OUT_JSON.write_text(json.dumps(
        {"schema_version": 1, "generated_at": datetime.now().isoformat(timespec="seconds"),
         "frame": "panda_link0", "grid": info, "probe": meta,
         "source_points": SRC_POINTS.name, "points": grid}, indent=2))
    fields = ["point_index", "row", "col", "x_m", "y_m", "surface_z_estimate_m",
              "surface_z_m", "stiffness_n_per_mm", "fit_r2", "fit_distinct_z",
              "quality_ok"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in grid:
            w.writerow({k: g.get(k) for k in fields})


def save_png(grid, info):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (no plot: %s)" % e)
        return
    nu, nv = info["n_u"], info["n_v"]
    Z = np.full((nv, nu), np.nan)
    K = np.full((nv, nu), np.nan)
    E = np.full((nv, nu), np.nan)
    for g in grid:
        if g["surface_z_m"] is not None:
            Z[g["row"], g["col"]] = g["surface_z_m"] * 1e3
            E[g["row"], g["col"]] = (g["surface_z_m"] - g["surface_z_estimate_m"]) * 1e3
        if g["stiffness_n_per_mm"] is not None:
            K[g["row"], g["col"]] = g["stiffness_n_per_mm"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for a, M, t, unit in ((ax[0], Z - np.nanmin(Z), "surface height", "mm above lowest"),
                          (ax[1], K, "local stiffness", "N/mm"),
                          (ax[2], E, "measured − 4-corner estimate", "mm")):
        im = a.imshow(M, origin="lower", aspect="auto", cmap="viridis")
        for i in range(nv):
            for j in range(nu):
                if np.isfinite(M[i, j]):
                    a.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7,
                           color="white")
        a.set_title("%s (%s)" % (t, unit), fontsize=9, loc="left")
        a.set_xlabel("col (u)")
        fig.colorbar(im, ax=a, fraction=.046)
    ax[0].set_ylabel("row (v)")
    fig.suptitle("surface offset + stiffness map, %d x %d over %.1f x %.1f mm"
                 % (nu, nv, info["span_u_mm"], info["span_v_mm"]), x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print("  plot -> %s" % OUT_PNG.name)


def _score(distinct, r2, ok):
    """Rank attempts: passing beats failing, then more distinct z levels, then R2."""
    return (1 if ok else 0, distinct or 0, r2 or 0.0)


def probe_with_retry(tsf, servo, ft, g, args, approach_z, quat):
    """Probe one point, retrying immediately on low confidence with escalated steps.

    Returns (best_result_dict, attempts_log). Keeps the BEST attempt, not the last --
    a retry can come out worse, and silently overwriting a good measurement with a
    worse one would defeat the point.
    """
    import argparse as _ap
    z_est = g["surface_z_estimate_m"]
    best, log = None, []
    for k in range(1, args.max_attempts + 1):
        a = _ap.Namespace(**vars(args))
        if k > 1:
            a.fine_step_mm = args.fine_step_mm * (RETRY_STEP_SCALE ** (k - 1))
            a.fit_force_max_n = min(args.fit_force_max_n * (RETRY_FORCE_SCALE ** (k - 1)),
                                    1.20)
            print("    retry %d/%d with fine step %.2f mm, fit to %.2f N"
                  % (k, args.max_attempts, a.fine_step_mm, a.fit_force_max_n), flush=True)
        out = tsf.probe_one(servo, ft, g["x_m"], g["y_m"],
                            z_est + a.hover_mm / 1000.0, z_est, quat, a,
                            "idx%d" % g["point_index"], approach_z=approach_z)
        sz, stiff, r2, rows = out[0], out[1], out[2], out[3]
        distinct = out[4] if len(out) > 4 else None
        ok = bool(sz is not None and distinct is not None
                  and distinct >= tsf.MIN_DISTINCT_Z and r2 >= tsf.MIN_FIT_R2)
        res = {"surface_z_m": None if sz is None else round(sz, 6),
               "stiffness_n_per_mm": None if stiff is None else round(stiff, 4),
               "fit_r2": None if r2 is None else round(r2, 4),
               "fit_distinct_z": distinct, "quality_ok": ok,
               "probe_steps": rows, "attempt": k,
               "fine_step_mm": a.fine_step_mm, "fit_force_max_n": a.fit_force_max_n}
        log.append({kk: res[kk] for kk in ("attempt", "surface_z_m", "fit_r2",
                                           "fit_distinct_z", "quality_ok",
                                           "fine_step_mm", "fit_force_max_n")})
        if best is None or _score(distinct, r2, ok) > _score(best["fit_distinct_z"],
                                                            best["fit_r2"],
                                                            best["quality_ok"]):
            best = res
        if ok:
            break
    if best and not best["quality_ok"]:
        print("    ⚠ still low confidence after %d attempts -- keeping the best "
              "(%s distinct, R2 %s)" % (len(log), best["fit_distinct_z"], best["fit_r2"]),
              flush=True)
    elif len(log) > 1:
        print("    recovered on attempt %d" % best["attempt"], flush=True)
    return best, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=int, default=NU, help="points across u (default %d)" % NU)
    ap.add_argument("--nv", type=int, default=NV, help="points across v (default %d)" % NV)
    ap.add_argument("--margin-mm", type=float, nargs=4, default=MARGIN_MM,
                    metavar=("U_LO", "U_HI", "V_LO", "V_HI"),
                    help="per-edge clearance from the taught quad (default 4 4 4 4). "
                         "The strapped edge needs more than the others.")
    ap.add_argument("--hover-mm", type=float, default=HOVER_MM)
    ap.add_argument("--search-below-mm", type=float, default=SEARCH_BELOW_MM)
    ap.add_argument("--coarse-step-mm", type=float, default=None)
    ap.add_argument("--fine-step-mm", type=float, default=None)
    ap.add_argument("--fit-force-max-n", type=float, default=None)
    ap.add_argument("--settle-s", type=float, default=None)
    ap.add_argument("--line-u", type=float, default=None, metavar="FRAC",
                    help="probe a SINGLE column at this fraction along u (0=BL edge, "
                         "0.5=middle, 1=BR edge) instead of a full grid. --nv sets how "
                         "many points down v. Use with --out-stem so it does not "
                         "overwrite the real map.")
    ap.add_argument("--out-stem", default=None,
                    help="basename for the outputs (default surface_offset_map)")
    ap.add_argument("--order", choices=["raster", "serpentine"], default="raster",
                    help="raster (default) sweeps every row the same direction so a "
                         "residual direction bias stays constant; serpentine alternates "
                         "and was measured to introduce a 0.26 mm offset between rows.")
    ap.add_argument("--settle-before-zero-s", type=float, default=None,
                    help="dwell at hover before zeroing the F/T (default: inherit "
                         "teach_surface's %.1f s)" % 1.5)
    ap.add_argument("--redo", default="", help="comma-separated point_index to re-probe")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                    help="retry a low-confidence point immediately, up to this many "
                         "times, escalating the fine step each go (default %d). 1 "
                         "disables retrying." % MAX_ATTEMPTS)
    ap.add_argument("--fresh", action="store_true", help="ignore any previous map")
    ap.add_argument("--keep-taught-orientation", action="store_true")
    ap.add_argument("--no-recovery", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import teach_surface as tsf

    # Inherit the descent parameters from teach_surface unless overridden, so the map
    # and the 5-point reference are measured the same way by construction.
    for name, const in (("coarse_step_mm", tsf.COARSE_STEP_MM),
                        ("fine_step_mm", tsf.FINE_STEP_MM),
                        ("fit_force_max_n", tsf.FIT_FORCE_MAX_N),
                        ("settle_s", tsf.SETTLE_S),
                        ("settle_before_zero_s", tsf.SETTLE_BEFORE_ZERO_S)):
        if getattr(args, name) is None:
            setattr(args, name, const)

    if args.out_stem:
        set_out_stem(args.out_stem)

    ref, pts = load_reference()

    # ⚠ ADOPT THE STORED GRID when continuing an existing map. --redo previously
    # rebuilt whatever the CLI defaults implied, so `--redo 1,5,15` without repeating
    # `--nv 9` silently switched to a 5x5 and discarded a finished 45-point map. The
    # grid a map was measured on is part of that map, not a per-invocation flag.
    if args.line_u is None and not args.fresh and OUT_JSON.exists():
        try:
            pg = json.loads(OUT_JSON.read_text()).get("grid", {})
        except Exception:
            pg = {}
        stored = (pg.get("n_u"), pg.get("n_v"), pg.get("margins_mm"), pg.get("order"))
        m = pg.get("margins_mm") or {}
        stored_margins = [m.get(k) for k in ("u_lo", "u_hi", "v_lo", "v_hi")]
        asked = (args.nu, args.nv, list(args.margin_mm), args.order)
        gave_shape = any(f in " ".join(sys.argv) for f in ("--nu", "--nv", "--margin-mm",
                                                          "--order"))
        if stored[0] and (stored[0], stored[1]) != (args.nu, args.nv) \
                or (stored_margins[0] is not None
                    and [float(v) for v in stored_margins] != [float(v) for v in args.margin_mm]):
            if gave_shape:
                sys.exit(
                    "[ABORT] %s holds a %sx%s map with margins %s, but you asked for "
                    "%dx%d with %s.\n"
                    "        Continuing would DESTROY the stored map. Either drop the "
                    "shape flags to\n        continue it, or pass --fresh to start a new "
                    "one (the old is archived either way)."
                    % (OUT_JSON.name, stored[0], stored[1], stored_margins,
                       args.nu, args.nv, list(args.margin_mm)))
            print("  continuing the stored %sx%s map (margins %s) -- grid taken from the "
                  "file, not the defaults" % (stored[0], stored[1], stored_margins))
            args.nu, args.nv = int(stored[0]), int(stored[1])
            args.margin_mm = [float(v) for v in stored_margins]
            if stored[3]:
                args.order = stored[3]

    grid, info = build_grid(pts, args.nu, args.nv, args.margin_mm, args.order,
                            line_u=args.line_u)

    meta = {"coarse_step_mm": args.coarse_step_mm, "fine_step_mm": args.fine_step_mm,
            "hover_mm": args.hover_mm, "search_below_mm": args.search_below_mm,
            "fit_force_max_n": args.fit_force_max_n, "baseline_s": tsf.BASELINE_S,
            "levelled": not args.keep_taught_orientation,
            "order": args.order, "settle_before_zero_s": args.settle_before_zero_s,
            "approach": "traverse at height, then vertical descent to hover",
            "method": "teach_surface.probe_one -- rise, zero, coarse find, back off, "
                      "fine crawl, extrapolate force to zero"}
    reused = 0 if args.fresh else load_existing(grid, meta)
    redo = {int(v) for v in args.redo.split(",") if v.strip()}
    if redo:
        unknown = sorted(i for i in redo if i >= len(grid))
        if unknown:
            sys.exit("[ABORT] --redo %s: this grid only has indices 0..%d. The stored "
                     "map may have a different shape." % (unknown, len(grid) - 1))
        if not reused:
            sys.exit("[ABORT] --redo was given but NO stored points were reusable, so "
                     "this would\n        re-measure the entire grid rather than the %d "
                     "points you asked for.\n        Check the grid shape matches the "
                     "stored map, or use --fresh deliberately." % len(redo))
        for g in grid:
            if g["point_index"] in redo:
                g["surface_z_m"] = None
    todo = [g for g in grid if g["surface_z_m"] is None]

    per_pt = ((args.hover_mm + args.search_below_mm) / args.coarse_step_mm
              + (tsf.BACKOFF_MM + 1.6) / args.fine_step_mm) * (args.settle_s + tsf.SAMPLE_S) \
        + tsf.BASELINE_S + args.settle_before_zero_s + 9.0
    print("=" * 74)
    print(" SURFACE OFFSET + STIFFNESS MAP")
    print("=" * 74)
    print("  reference      : %s (%d points, probed %s)"
          % (SRC_POINTS.name, len(pts), ref.get("probed_at")))
    print("  taught quad    : %.1f x %.1f mm" % (info["taught_u_len_mm"],
                                                 info["taught_v_len_mm"]))
    print("  mapped area    : %.1f x %.1f mm   margins u %.1f/%.1f  v %.1f/%.1f mm"
          % (info["span_u_mm"], info["span_v_mm"], *args.margin_mm))
    if args.line_u is not None:
        print("  MODE           : single column at u = %.2f  (diagnostic line)"
              % args.line_u)
    print("  grid           : %d x %d = %d points, pitch %.1f x %.1f mm, %s"
          % (info["n_u"], info["n_v"], info["n_points"], info["pitch_u_mm"],
             info["pitch_v_mm"], info["order"].upper()))
    print("  4-corner estimate spans %.3f mm -- each descent starts %.1f mm above ITS"
          % (info["estimate_range_mm"], args.hover_mm))
    print("                    own estimate, and searches %.1f mm below it."
          % args.search_below_mm)
    print("  descent        : coarse %.2f / fine %.2f mm, zero %.1f s, fit to %.2f N"
          % (args.coarse_step_mm, args.fine_step_mm, tsf.BASELINE_S, args.fit_force_max_n))
    print("  approach       : traverse at height, then STRAIGHT DOWN, settle %.1f s, "
          "then zero" % args.settle_before_zero_s)
    print("  flange         : %s"
          % ("LEVELLED to straight-down before each descent"
             if not args.keep_taught_orientation else "⚠ taught tilt, NOT levelled"))
    if reused:
        print("  resuming       : %d point(s) already measured, %d to do"
              % (reused, len(todo)))
    if redo:
        print("  re-probing     : %s" % sorted(redo))
    retry_pad = 1.0 + 0.25 * (args.max_attempts > 1)     # ~25% of points needed a retry
    print("  retries        : up to %d attempts/point, escalating the fine step %.2f -> "
          "%.2f mm" % (args.max_attempts, args.fine_step_mm,
                       args.fine_step_mm * RETRY_STEP_SCALE ** (args.max_attempts - 1)))
    print("  estimated time : %.0f s/point x %d = %.0f min (%.0f min allowing for retries)"
          % (per_pt, len(todo), per_pt * len(todo) / 60,
             per_pt * len(todo) * retry_pad / 60))
    print("=" * 74)
    print("  ⚠ HEX21 must be ON TACTILE. Servo reach launch up. Hands off the arm.")
    print("  ⚠ Keep a hand near the user-stop -- the servo does not stop on contact.")
    print("=" * 74)
    if args.dry_run:
        for g in grid[:info["n_u"] * 2]:
            print("   idx %2d  r%-2d c%-2d  [%.5f, %.5f]  est surface %.6f"
                  % (g["point_index"], g["row"], g["col"], g["x_m"], g["y_m"],
                     g["surface_z_estimate_m"]))
        if len(grid) > info["n_u"] * 2:
            print("   ... %d more" % (len(grid) - info["n_u"] * 2))
        print("\n[DRY RUN] no ROS, no motion.")
        return
    if not todo:
        sys.exit("[done] every point already measured. --redo or --fresh to repeat.")

    import rospy
    rospy.init_node("map_offset", disable_signals=True)
    import map_surface as m
    from franka_grid_logger import clear_reflex, warn_if_loaded
    from servo_client import CartesianServo
    from franka_surface_map import WittensteinFT
    from home_and_level import flat_down_quat, tilt_deg
    if not args.no_recovery:
        clear_reflex()
    warn_if_loaded()
    servo = CartesianServo()

    pre = [{"x": g["x_m"], "y": g["y_m"], "z_plane": g["surface_z_estimate_m"]}
           for g in grid]
    # ⚠ preflight_reach requires a "center" key -- see teach_surface.build_reach_points.
    # This dict is numbered, so without the helper it would raise KeyError: 'center'.
    named = {"p%d" % g["point_index"]: [g["x_m"], g["y_m"], g["surface_z_estimate_m"]]
             for g in grid}
    ctr = ([pts["center"]["xy_m"][0], pts["center"]["xy_m"][1],
            pts["center"].get("surface_z_m") or pts["center"]["kissed_z_m"]]
           if "center" in pts else None)
    reach = tsf.build_reach_points(named, ctr)
    saved = m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM
    m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = args.search_below_mm, APPROACH_MM
    try:
        m.preflight_reach(servo, reach, pre, args.hover_mm / 1000.0)
    finally:
        m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = saved

    # One traverse height for the whole run, clear of the highest estimated surface, so
    # a lateral move can never dip toward a high spot.
    approach_z = max(g["surface_z_estimate_m"] for g in grid) + APPROACH_MM / 1000.0
    print("  traverse height: %.5f m (%.1f mm above the highest estimate)"
          % (approach_z, APPROACH_MM))
    archive_previous()          # before the first save() can overwrite anything
    ok = bad = 0
    t0 = time.time()
    with WittensteinFT(port=m.SERIAL_PORT) as ft:
        try:
            _, q0 = servo.current_pose()
            for n, g in enumerate(todo):
                x, y = g["x_m"], g["y_m"]
                z_est = g["surface_z_estimate_m"]
                quat = ([float(v) for v in flat_down_quat(q0)]
                        if not args.keep_taught_orientation else list(q0))
                el = time.time() - t0
                print("\n[%2d/%2d] idx %d  r%d c%d  est %.6f   elapsed %.0f min, ETA %.0f min"
                      % (n + 1, len(todo), g["point_index"], g["row"], g["col"], z_est,
                         el / 60, (el / max(n, 1) * (len(todo) - n)) / 60 if n else 0),
                      flush=True)
                if n == 0:
                    print("    flange tilt %.2f deg -> commanding %.2f deg"
                          % (tilt_deg(q0), tilt_deg(quat)), flush=True)
                best, log = probe_with_retry(tsf, servo, ft, g, args, approach_z, quat)
                g["attempts"] = log
                g["n_attempts"] = len(log)
                if best is None or best["surface_z_m"] is None:
                    print("    -> FAILED after %d attempt(s)" % len(log))
                    g["probe_steps"] = best["probe_steps"] if best else None
                    bad += 1
                else:
                    for kk in ("surface_z_m", "stiffness_n_per_mm", "fit_r2",
                               "fit_distinct_z", "quality_ok", "probe_steps"):
                        g[kk] = best[kk]
                    ok, bad = (ok + 1, bad) if g["quality_ok"] else (ok, bad + 1)
                    print("    -> surface %.6f  (%+.3f mm vs the estimate)   k %.3f N/mm  "
                          "%s  [attempt %d/%d]"
                          % (best["surface_z_m"], (best["surface_z_m"] - z_est) * 1e3,
                             best["stiffness_n_per_mm"],
                             "ok" if g["quality_ok"] else "LOW CONFIDENCE",
                             best["attempt"], len(log)))
                save(grid, info, meta)          # after EVERY point, so a crash loses one
        except KeyboardInterrupt:
            print("\n[interrupted] progress is saved; re-run to continue.")
        finally:
            save(grid, info, meta)

    done = [g for g in grid if g["surface_z_m"] is not None]
    print("\n" + "=" * 74)
    print(" %d/%d points measured  (%d good, %d low-confidence/failed)  in %.1f min"
          % (len(done), len(grid), ok, bad, (time.time() - t0) / 60))
    if done:
        z = np.array([g["surface_z_m"] for g in done])
        e = np.array([g["surface_z_m"] - g["surface_z_estimate_m"] for g in done])
        k = np.array([g["stiffness_n_per_mm"] for g in done
                      if g["stiffness_n_per_mm"] is not None])
        print("  surface height range      : %.3f mm across the mapped area"
              % ((z.max() - z.min()) * 1e3))
        print("  vs the 4-corner estimate  : mean %+.3f mm, worst %+.3f mm"
              % (e.mean() * 1e3, e[np.argmax(np.abs(e))] * 1e3))
        print("     -> that is the error a plane would have made. It is why this map exists.")
        if len(k):
            print("  stiffness                 : %.3f .. %.3f N/mm  (%.1fx spread)"
                  % (k.min(), k.max(), k.max() / max(k.min(), 1e-9)))
            print("     -> use it to target FORCE, not depth, in the next campaign.")
        na = [g.get("n_attempts", 1) for g in done]
        rec = [g for g in done if g.get("n_attempts", 1) > 1 and g["quality_ok"]]
        print("  attempts          : %d of %d points needed a retry, %d recovered"
              % (sum(1 for v in na if v > 1), len(done), len(rec)))
        lowc = [g["point_index"] for g in grid if g["quality_ok"] is False]
        if lowc:
            print("  ⚠ low-confidence points   : %s" % lowc)
            print("     re-probe with:  python3 map_offset.py --redo %s"
                  % ",".join(str(i) for i in lowc))
        save_png(grid, info)
    print("  wrote %s / %s" % (OUT_JSON.name, OUT_CSV.name))
    print("=" * 74)


if __name__ == "__main__":
    main()
