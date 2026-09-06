#!/usr/bin/env python3
"""
contact_error.py -- how good is the camera's contact location, and does depth matter?

    python3 ml/contact_error.py output/<run>
    python3 ml/contact_error.py output/<run> --per-poke --figure
    python3 ml/contact_error.py output/<run> --phase dwell

WHAT IT COMPARES
----------------
Two independent answers to the same question -- where is the indenter touching --
reduced over the same wall-clock window:

    camera   contact.csv   the divergence peak of the marker displacement field,
                           through calibration/pixel_to_mm.json  (E_BTS_GUI writes
                           one row per event window, live)
    robot    franka.csv    the end-effector XY the Franka reports

Both are ROBOT BASE millimetres -- the pixel->mm calibration was fitted from
franka/calib_raster.py against robot XY -- so the error is a subtraction with no
frame conversion in between that could be silently wrong by a sign or a rotation.

THE HEADLINE IS THE PER-DEPTH TABLE
-----------------------------------
franka/campaign_ladder.py pokes every location at every rung of a depth ladder, so
each rung has N independent locations behind it. That is what turns "the error was
0.3 mm" into "0.31 +- 0.04 mm at 1.0 mm and 0.18 +- 0.02 mm at 5.0 mm" -- a
statement about the estimator rather than about one poke. Two failure modes are
expected at opposite ends and the table is what separates them:

    shallow   the divergence lobe is weak and the peak is noise-dominated
    deep      the elastomer shears far enough that the marker field's centre may
              no longer sit under the indenter axis -- a BIAS, not scatter

which is why bias and scatter are always reported apart. A constant offset is a
wrong origin or a stale calibration and is correctable after the fact; scatter
about it is the estimator and is not.

⚠ DEPTH IS TAKEN FROM THE LOG, NOT FROM plan.csv.
`surface_z - ee_z` is the depth the arm ACHIEVED. The commanded rung is in the log
too (`surface_z - target_z`) and pokes are grouped by it, but the achieved column
is reported beside it: the joint-impedance controller sags under load (HANDOFF
4.4), so command and achievement differ, and it is the achieved depth that produced
the image the camera saw.

⚠ MEDIAN OVER THE DWELL, NOT THE PEAK OR THE MEAN.
The dwell is the only phase where the indenter is stationary AND fully engaged, so
it is the only window in which "the contact is at X" is well posed. Within it the
camera estimate still steps by fractions of a lattice cell as the divergence
weighting shifts, and a mean would be dragged by the boundary windows where the
indenter is still settling. The within-dwell spread is reported as `cam_sd` -- if
it is comparable to the error, the error is not what limits you.

⚠ THE TARE CHECK IS NOT OPTIONAL, READ IT.
Every poke is preceded by a ~1 s hold OUT OF CONTACT (phase `tare`). During it the
estimator must find NOTHING. Any tare window reporting a contact means the marker
baseline no longer describes the unloaded pad -- it drifted, or it was captured
under load in the first place. Over a ~3 h run that is the single most likely way
for this measurement to be quietly wrong, and it is free to check, so it is checked
by default and printed before the results. A rising false-contact rate through the
run is drift; a high rate from poke 1 is a bad baseline and the run is void.

WHAT THIS DOES NOT MEASURE
--------------------------
The Franka's own pose is not ground truth to arbitrary precision (~0.5 mm command
tracking under joint impedance, HANDOFF 4.4), and the tool is assumed axial:
O_T_EE's XY is taken as the contact XY, exactly as calib_raster assumed when the
calibration was fitted. Both are inside the number this prints and neither is
separable from it here.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


def load_csv(path):
    """CSV -> dict of numpy arrays; float where possible, object otherwise."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("%s is empty" % path)
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        try:
            out[key] = np.array([float(v) if v not in ("", "nan", "None") else np.nan
                                 for v in values])
        except ValueError:
            out[key] = np.array(values, dtype=object)
    return out


def load_many(paths):
    """Concatenate several segment CSVs that share a header, in order."""
    parts = [load_csv(p) for p in paths]
    keys = [k for k in parts[0] if all(k in p for p in parts)]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def find(run_dir, *names):
    for name in names:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return None


def phase_blocks(t, phase, wanted):
    """Contiguous runs of rows with phase == wanted, as index arrays.

    Contiguity, not point_index: in a ladder campaign a location is visited once
    per rung, so point_index repeats 11 times and grouping by it would merge 11
    different pokes -- at 11 different depths -- into one.
    """
    mask = np.array([str(p) == wanted for p in phase])
    if not mask.any():
        return [], mask
    edges = np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1
    blocks = [b for b in np.split(np.arange(len(mask)), edges) if len(b) and mask[b[0]]]
    return blocks, mask


def summarise(values, label, unit="mm"):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "%s: (none)" % label
    return ("%s: median %.3f  mean %.3f  p90 %.3f  max %.3f %s"
            % (label, np.median(v), v.mean(), np.percentile(v, 90), v.max(), unit))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="an output/<run> directory (post-processed)")
    ap.add_argument("--phase", default="dwell", help="phase to measure over (default: dwell)")
    ap.add_argument("--tare-phase", default="tare", help="the out-of-contact phase")
    ap.add_argument("--per-poke", action="store_true", help="print one row per poke")
    ap.add_argument("--figure", action="store_true", help="write contact_error.png")
    ap.add_argument("--min-windows", type=int, default=3,
                    help="skip a poke with fewer valid camera windows in its dwell")
    ap.add_argument("--depth-tol-mm", type=float, default=0.15,
                    help="commanded depths within this are treated as the same rung")
    ap.add_argument("--min-divergence", type=float, default=None,
                    help="re-decide which windows count as a contact, at this divergence "
                         "threshold in px/cell, instead of using the valid/ambiguous columns "
                         "as recorded. Needs the peak_found and second_divergence columns that "
                         "E_BTS_contact_replay writes -- a live contact.csv only records the "
                         "peaks its own threshold already accepted, so it cannot be re-thresholded "
                         "and this will refuse rather than quietly report a subset. Sweeping this "
                         "is how kContactMinimumDivergence stops being a guess; the robot cannot "
                         "answer it, because you cannot re-poke at a different threshold.")
    ap.add_argument("--contact", default="contact.csv",
                    help="which camera log inside run_dir to read (default: contact.csv, the "
                         "one E_BTS_GUI wrote live). E_BTS_contact_replay writes a second one "
                         "from the same .raw, so the live and replayed estimators can be "
                         "compared on identical pokes by pointing this at it.")
    args = ap.parse_args()

    contact_path = find(args.run_dir, args.contact)
    franka_path = find(args.run_dir, "franka.csv")
    if not franka_path:
        # A campaign run that has not been through postprocess.py still has the
        # logger's own segments. They are the same columns; postprocess only
        # concatenates and renames them, and this analysis needs no force channel,
        # so there is no reason to require that step first.
        import glob as _glob
        segs = sorted(_glob.glob(os.path.join(args.run_dir, "franka_seg*.csv")))
        if segs:
            franka_path = segs
            print("using %d franka segment(s): %s"
                  % (len(segs), ", ".join(os.path.basename(p) for p in segs)))
    meta_path = find(args.run_dir, "metadata.json")
    if not contact_path:
        sys.exit(("no %s in %%s.\n" % args.contact) +
                 "        E_BTS_GUI writes it as <run>_contact.csv beside the .raw and\n"
                 "        postprocess.py carries it in. A run recorded before this feature\n"
                 "        existed, with no pixel->mm calibration loaded, or with the Circle\n"
                 "        Tracking pane never opened, will not have one." % args.run_dir)
    if not franka_path:
        sys.exit("no franka.csv in %s" % args.run_dir)

    offset = 0.0
    if meta_path:
        offset = json.load(open(meta_path)).get("tactile_minus_workstation_offset_s") or 0.0
    print("run: %s   |  tactile->workstation offset %.1f ms" % (args.run_dir, -offset * 1e3))

    cam = load_csv(contact_path)
    fr = load_csv(franka_path) if isinstance(franka_path, str) else load_many(franka_path)
    t_fr = fr["unix_time_s"] - offset            # the shift postprocess.py applies
    t_cam = cam["unix_time_s"]
    if "x_robot_mm" not in cam:
        sys.exit("contact.csv has no x_robot_mm column -- written by an older GUI build")

    if args.min_divergence is not None:
        missing = [c for c in ("peak_found", "divergence_px_per_cell",
                               "second_divergence_px_per_cell") if c not in cam]
        if missing:
            sys.exit("--min-divergence needs %s, which %s does not have.\n"
                     "        Only E_BTS_contact_replay writes them. A live contact.csv records\n"
                     "        a peak's strength ONLY for the windows its own threshold already\n"
                     "        accepted, so re-thresholding it could lower the count but never\n"
                     "        raise it -- the rejected peaks are simply not in the file. That\n"
                     "        would look like a sweep and be a truncation, so it is refused."
                     % (", ".join(missing), os.path.basename(contact_path)))
        # NaN (no peak that window) compares false, which is the intent.
        peak = cam["peak_found"] > 0.5
        cam["valid"] = np.where(peak & (cam["divergence_px_per_cell"] >= args.min_divergence),
                                1.0, 0.0)
        cam["ambiguous"] = np.where(
            (cam["valid"] > 0.5) & (cam["second_divergence_px_per_cell"] >= args.min_divergence),
            1.0, 0.0)
        print("re-thresholded at %.2f px/cell: %d of %d windows have a peak at all, %d clear it"
              % (args.min_divergence, int(peak.sum()), len(peak), int((cam["valid"] > 0.5).sum())))

    valid = cam["valid"] > 0.5
    single = valid & (cam["ambiguous"] < 0.5)
    usable = single & np.isfinite(cam["x_robot_mm"])
    print("camera windows: %d total, %d with a contact, %d single-contact with millimetres"
          % (len(t_cam), int(valid.sum()), int(usable.sum())))
    if not usable.any():
        sys.exit("no usable camera estimates in this run")

    phase = fr.get("phase")
    if phase is None:
        sys.exit("franka.csv has no phase column")

    # ---- the tare check, BEFORE any results -------------------------------
    tare_blocks, _ = phase_blocks(t_fr, phase, args.tare_phase)
    if tare_blocks:
        seen, false_hits, first_half_rate, halves = 0, 0, None, [[0, 0], [0, 0]]
        for i, block in enumerate(tare_blocks):
            w = (t_cam >= t_fr[block[0]]) & (t_cam <= t_fr[block[-1]])
            n, hits = int(w.sum()), int((w & valid).sum())
            seen += n
            false_hits += hits
            half = 0 if i < len(tare_blocks) / 2 else 1
            halves[half][0] += n
            halves[half][1] += hits
        rate = 100.0 * false_hits / seen if seen else float("nan")
        r0 = 100.0 * halves[0][1] / halves[0][0] if halves[0][0] else float("nan")
        r1 = 100.0 * halves[1][1] / halves[1][0] if halves[1][0] else float("nan")
        print("\nBASELINE CHECK (phase '%s', %d holds, %d camera windows)"
              % (args.tare_phase, len(tare_blocks), seen))
        print("  contacts found while OUT OF CONTACT: %d (%.2f%%)   first half %.2f%% -> second half %.2f%%"
              % (false_hits, rate, r0, r1))
        if rate > 5.0:
            print("  ⚠ ABOVE 5%. The marker baseline does not describe the unloaded pad.\n"
                  "    If it was already high at the start, the baseline was captured under\n"
                  "    load and every coordinate below is offset by an amount nothing here\n"
                  "    can recover -- rebuild the baseline unloaded and re-run the campaign.")
        elif r1 > max(3.0 * max(r0, 0.2), 5.0):
            print("  ⚠ RISING through the run: the baseline is drifting, so later pokes are\n"
                  "    measured against a staler reference than earlier ones. Treat any\n"
                  "    trend against time (not depth) as suspect.")
        else:
            print("  ✅ the baseline still describes the unloaded pad.")
    else:
        print("\n[warn] no '%s' phase in franka.csv -- the baseline could not be checked."
              % args.tare_phase)

    # ---- per-poke reduction ------------------------------------------------
    blocks, _ = phase_blocks(t_fr, phase, args.phase)
    if not blocks:
        sys.exit("no rows with phase='%s' in franka.csv (have: %s)"
                 % (args.phase, ", ".join(sorted({str(p) for p in phase}))))

    rows, skipped = [], 0
    for block in blocks:
        t0, t1 = t_fr[block[0]], t_fr[block[-1]]
        w = usable & (t_cam >= t0) & (t_cam <= t1)
        n = int(w.sum())
        if n < args.min_windows:
            skipped += 1
            continue
        # Depth from the log: commanded is surface - target, achieved is surface -
        # where the arm actually ended up. Both in mm, both positive downward.
        sz = float(np.nanmedian(fr["surface_z"][block]))
        d_cmd = (sz - float(np.nanmedian(fr["target_z"][block]))) * 1000.0
        d_ach = (sz - float(np.nanmedian(fr["ee_z"][block]))) * 1000.0
        rows.append({
            "point": int(fr["point_index"][block[0]]) if "point_index" in fr else -1,
            "t_mid": float((t0 + t1) / 2.0),
            "n_windows": n,
            "depth_cmd_mm": d_cmd,
            "depth_ach_mm": d_ach,
            "rob_x": float(np.median(fr["ee_x"][block])) * 1000.0,
            "rob_y": float(np.median(fr["ee_y"][block])) * 1000.0,
            "cam_x": float(np.median(cam["x_robot_mm"][w])),
            "cam_y": float(np.median(cam["y_robot_mm"][w])),
            "cam_sd": float(np.hypot(np.std(cam["x_robot_mm"][w]), np.std(cam["y_robot_mm"][w]))),
        })
    if not rows:
        sys.exit("no poke had >= %d camera windows inside its '%s' phase.\n"
                 "        Was the Circle Tracking baseline built, UNLOADED, before the run?"
                 % (args.min_windows, args.phase))
    for r in rows:
        r["dx"] = r["rob_x"] - r["cam_x"]
        r["dy"] = r["rob_y"] - r["cam_y"]
        r["err"] = float(np.hypot(r["dx"], r["dy"]))

    print("\npokes with a usable estimate: %d of %d (%d skipped for < %d windows)"
          % (len(rows), len(blocks), skipped, args.min_windows))
    if skipped > 0.2 * len(blocks):
        # Not fatal, but it is a selection effect: the pokes that fail to produce an
        # estimate are unlikely to be a random subset (shallow ones are the usual
        # casualty), so the surviving error is optimistic by an unknown amount.
        print("  ⚠ %.0f%% of pokes produced no estimate. Those are probably not a random\n"
              "    subset -- shallow pokes fail first -- so the errors below are optimistic."
              % (100.0 * skipped / len(blocks)))

    # ---- group into rungs --------------------------------------------------
    # ⚠ NOT by the commanded depth in the log. dip_to_depth() closes the loop on
    # measured ee_z, so `target_z` during the dwell holds the SAG-CORRECTED command,
    # which overshoots the nominal rung by up to ~1.5 mm and differs per location.
    # Grouping on it turned an 11-rung ladder into 35 spurious "rungs".
    # The ladder itself is in plan.json; pokes are assigned to the nearest rung by
    # what the arm ACHIEVED, which is also the depth that produced the image.
    plan_json = find(args.run_dir, "plan.json")
    rungs = None
    if plan_json:
        rungs = json.load(open(plan_json)).get("depths_mm")
    if rungs:
        rungs = [float(g) for g in rungs]
        print("depth ladder from plan.json: %s mm" % " ".join("%.1f" % g for g in rungs))
    else:
        # No plan: fall back to clustering the achieved depths.
        rungs = []
        for c in np.sort(np.array([r["depth_ach_mm"] for r in rows])):
            if not rungs or abs(c - rungs[-1]) > args.depth_tol_mm:
                rungs.append(float(c))
        print("no plan.json; %d rungs inferred from achieved depth" % len(rungs))
    for r in rows:
        r["rung_mm"] = min(rungs, key=lambda g: abs(g - r["depth_ach_mm"]))

    dx = np.array([r["dx"] for r in rows])
    dy = np.array([r["dy"] for r in rows])
    err = np.array([r["err"] for r in rows])

    if args.per_poke:
        print("\n point  depth cmd/ach   robot x    robot y  |  camera x   camera y  |"
              "     dx      dy    |err|  cam sd  n")
        for r in rows:
            print("  %4d   %4.1f / %4.2f  %8.2f  %8.2f  |  %8.2f  %8.2f  | %+6.2f  %+6.2f"
                  "  %6.2f  %5.2f %3d"
                  % (r["point"], r["depth_cmd_mm"], r["depth_ach_mm"], r["rob_x"], r["rob_y"],
                     r["cam_x"], r["cam_y"], r["dx"], r["dy"], r["err"], r["cam_sd"],
                     r["n_windows"]))

    print("\n" + "=" * 84)
    print("  LOCALISATION ERROR BY DEPTH   (bias = mean offset, correctable;"
          " scatter = estimator)")
    print("=" * 84)
    print("  rung  achieved      n | bias dx   bias dy  |bias| | scatter |"
          " median |err|   p90   cam sd")
    for g in rungs:
        sel = [r for r in rows if r["rung_mm"] == g]
        if not sel:
            continue
        gdx = np.array([r["dx"] for r in sel])
        gdy = np.array([r["dy"] for r in sel])
        gerr = np.array([r["err"] for r in sel])
        ach = np.array([r["depth_ach_mm"] for r in sel])
        scatter = float(np.hypot(gdx.std(ddof=1), gdy.std(ddof=1))) if len(sel) > 1 else float("nan")
        print("  %4.1f   %5.2f    %4d | %+7.3f  %+7.3f  %6.3f | %7.3f | %10.3f  %6.3f  %6.3f"
              % (g, ach.mean(), len(sel), gdx.mean(), gdy.mean(),
                 float(np.hypot(gdx.mean(), gdy.mean())), scatter,
                 float(np.median(gerr)), float(np.percentile(gerr, 90)),
                 float(np.median([r["cam_sd"] for r in sel]))))
    print("=" * 84)

    print("\nALL POKES (%d)" % len(rows))
    print("  bias   (mean)    : dx %+.3f  dy %+.3f mm   -> |bias| %.3f mm"
          % (dx.mean(), dy.mean(), np.hypot(dx.mean(), dy.mean())))
    print("  scatter about it : sd %.3f mm in x, %.3f mm in y"
          % (dx.std(ddof=1) if len(dx) > 1 else 0.0, dy.std(ddof=1) if len(dy) > 1 else 0.0))
    print("  " + summarise(err, "error  |Δ|      "))
    print("  camera spread    : median %.3f mm within a dwell"
          % np.median([r["cam_sd"] for r in rows]))
    # Bias removed: what the estimator could do if the origin were re-taught
    # perfectly. Quoting only this would be cheating; quoting only the raw error
    # hides that most of it may be one correctable constant.
    res = np.hypot(dx - dx.mean(), dy - dy.mean())
    print("  " + summarise(res, "after removing bias"))

    out_csv = os.path.join(args.run_dir, "contact_error.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s" % out_csv)

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(13, 10))

        by_rung = [[r["err"] for r in rows if r["rung_mm"] == g] for g in rungs]
        ax[0][0].boxplot(by_rung, labels=["%.1f" % g for g in rungs], showfliers=False)
        ax[0][0].set_xlabel("commanded depth (mm)")
        ax[0][0].set_ylabel("|Δ| (mm)")
        ax[0][0].set_title("localisation error vs depth")
        ax[0][0].grid(alpha=0.3)

        for g in rungs:
            sel = [r for r in rows if r["rung_mm"] == g]
            ax[0][1].scatter([np.mean([r["dx"] for r in sel])],
                             [np.mean([r["dy"] for r in sel])], s=40, label="%.1f mm" % g)
        ax[0][1].axhline(0, color="k", lw=0.5)
        ax[0][1].axvline(0, color="k", lw=0.5)
        ax[0][1].set_xlabel("bias dx (mm)")
        ax[0][1].set_ylabel("bias dy (mm)")
        # A bias that MOVES with depth is the shear hypothesis; one that sits still
        # is just a wrong origin.
        ax[0][1].set_title("bias per depth — does it move with depth?")
        ax[0][1].legend(fontsize=7, ncol=2)
        ax[0][1].set_aspect("equal", adjustable="datalim")
        ax[0][1].grid(alpha=0.3)

        ax[1][0].quiver([r["cam_x"] for r in rows], [r["cam_y"] for r in rows], dx, dy,
                        angles="xy", scale_units="xy", scale=1, width=0.003)
        ax[1][0].set_aspect("equal")
        ax[1][0].set_xlabel("x, robot base (mm)")
        ax[1][0].set_ylabel("y, robot base (mm)")
        # True scale: all pointing one way is a bad origin, fanning outward is a
        # scale error, scattered is the estimator.
        ax[1][0].set_title("error field (robot − camera, true scale)")
        ax[1][0].grid(alpha=0.3)

        t0 = min(r["t_mid"] for r in rows)
        ax[1][1].scatter([(r["t_mid"] - t0) / 60.0 for r in rows], err, s=8, alpha=0.6)
        ax[1][1].set_xlabel("minutes into the run")
        ax[1][1].set_ylabel("|Δ| (mm)")
        # Drift shows here and nowhere else: the depth design is randomised in time
        # precisely so a trend on this axis cannot be mistaken for a depth effect.
        ax[1][1].set_title("error vs time — baseline drift check")
        ax[1][1].grid(alpha=0.3)

        fig.tight_layout()
        out_png = os.path.join(args.run_dir, "contact_error.png")
        fig.savefig(out_png, dpi=140)
        print("wrote %s" % out_png)


if __name__ == "__main__":
    main()
