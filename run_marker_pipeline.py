#!/usr/bin/env python3
"""Run the marker-displacement pipeline end to end (HANDOFF §13.7).

This is an ORCHESTRATOR, not a rewrite. Each stage stays a standalone script that can
still be run and debugged on its own; this file just launches them in order, checks that
each one actually produced its artifact, and stops at the first failure instead of letting
a broken stage cascade into the next.

Stages:
  1  postprocess_indent.py   per-run  -> output/<run>_indent/  (F/T + robot slices,
                                         repeats_summary.csv, report_indent.pdf)
  2  event_video.py          per-repeat -> .../video/<stem>.mkv  (LOSSLESS, gray == event
                                         count) + <stem>.json sidecar (phase, tared Fz)
  3  marker_overlay.py       per-repeat -> <stem>_circles.mp4 + <stem>_detections.json
                                         (gapless roster `tracks`, stable indices)
  4  marker_arrows.py        per-repeat -> <stem>_arrows.mp4, peak/final displacement
                                         figures, <stem>_displacement.csv

Stage 1 is per-RUN; stages 2-4 are per-REPEAT, so --all runs stage 1 once and loops 2-4.

Force vs displacement is deliberately NOT a stage here -- see HANDOFF §13.5. It is blocked
on knowing the indentation xy from CAD, not on tooling.

Usage:
  python3 run_marker_pipeline.py                       # newest mid*/ run, repeat 1
  python3 run_marker_pipeline.py mid5_20260805_235328 --repeat 1
  python3 run_marker_pipeline.py --all                 # every repeat in the run
  python3 run_marker_pipeline.py --fps 25              # non-overlapping frames
  python3 run_marker_pipeline.py --roi 306 91 71 37    # crop (stages 2-4 follow it)
  python3 run_marker_pipeline.py --skip postprocess    # skip stages by name
  python3 run_marker_pipeline.py --dry-run             # print the commands only
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(REPO, "recordings")
OUTPUT = os.path.join(REPO, "output")

STAGE_NAMES = ("postprocess", "video", "overlay", "arrows")


def resolve_run(arg):
    if arg:
        p = arg if os.path.isdir(arg) else os.path.join(RECORDINGS, arg)
        if not os.path.isdir(p):
            sys.exit("[ABORT] no such run folder: %s" % p)
        return p
    c = [d for d in glob.glob(os.path.join(RECORDINGS, "*")) if os.path.isdir(d)
         and os.path.exists(os.path.join(d, "camera.raw"))
         and os.path.basename(d).startswith("mid")]
    if not c:
        sys.exit("[ABORT] no mid*/ run folder with a camera.raw under %s.\n"
                 "        Record one first:  python3 master_midpoint.py <name>" % RECORDINGS)
    return max(c, key=os.path.getmtime)


def n_repeats(run_dir):
    """How many indentations the run contains, from the summary if present."""
    summ = os.path.join(OUTPUT, os.path.basename(run_dir) + "_indent",
                        "repeats_summary.csv")
    if os.path.exists(summ):
        with open(summ) as f:
            return sum(1 for _ in csv.DictReader(f))
    # fall back to the recorded metadata
    meta = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta):
        import json
        return int(json.load(open(meta)).get("repeats", 1) or 1)
    return 1


def stem_for(run_dir, repeat, accum_us, fps, roi):
    """Mirror event_video.py's naming so later stages can find its output."""
    name = "repeat%02d_accum%dus_%gfps" % (repeat, accum_us, fps)
    if roi:
        name += "_roi%d-%d-%dx%d" % tuple(roi)
    return os.path.join(OUTPUT, os.path.basename(run_dir) + "_indent", "video", name)


def run(cmd, label, expect, dry):
    print("\n" + "=" * 78)
    print(" %s" % label)
    print("=" * 78)
    print("$ " + " ".join(cmd))
    if dry:
        return True
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    dt = time.time() - t0
    if rc != 0:
        print("\n[ABORT] %s failed (rc=%d) after %.1fs. Pipeline stopped -- later stages"
              " would build on missing output." % (label, rc, dt))
        return False
    missing = [p for p in expect if not os.path.exists(p)]
    if missing:
        print("\n[ABORT] %s exited 0 but did not produce:" % label)
        for p in missing:
            print("          %s" % p)
        return False
    print("  [ok] %s in %.1fs" % (label, dt))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None,
                    help="run folder name or path (default: newest mid*/)")
    ap.add_argument("--repeat", type=int, default=1, help="which indentation (1-based)")
    ap.add_argument("--all", action="store_true", help="loop stages 2-4 over every repeat")
    ap.add_argument("--accum-us", type=int, default=40000,
                    help="event accumulation per frame, us (default 40000 = one 25 Hz "
                         "illumination cycle)")
    ap.add_argument("--fps", type=float, default=60.0,
                    help="frame rate; 25 = non-overlapping 40 ms windows (default 60)")
    ap.add_argument("--roi", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"))
    ap.add_argument("--radius-px", type=float, default=0.0,
                    help="fixed marker radius in px (0 = measure from the baseline mean)")
    ap.add_argument("--view-scale", type=int, default=2)
    ap.add_argument("--net-gain", type=float, default=6.0)
    ap.add_argument("--step-gain", type=float, default=30.0)
    ap.add_argument("--fig-gain", type=float, default=8.0)
    ap.add_argument("--skip", nargs="*", default=[], choices=STAGE_NAMES,
                    help="stage names to skip: %s" % " ".join(STAGE_NAMES))
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()

    run_dir = resolve_run(args.run)
    run_id = os.path.basename(run_dir.rstrip("/"))
    out_dir = os.path.join(OUTPUT, run_id + "_indent")
    py = sys.executable or "python3"

    reps = list(range(1, n_repeats(run_dir) + 1)) if args.all else [args.repeat]

    print("=" * 78)
    print(" E-BTS marker pipeline")
    print("=" * 78)
    print("  run          : %s" % run_dir)
    print("  repeats      : %s" % ", ".join(str(r) for r in reps))
    print("  accumulation : %d us (%.0f ms)" % (args.accum_us, args.accum_us / 1e3))
    print("  fps          : %.4g  (%s)"
          % (args.fps, "non-overlapping" if abs(args.fps - 1e6 / args.accum_us) < 1e-6
             else "sliding window, %.1f%% overlap"
                  % (100 * max(0.0, args.accum_us - 1e6 / args.fps) / args.accum_us)))
    if args.roi:
        print("  ROI          : x=%d y=%d w=%d h=%d" % tuple(args.roi))
    if args.skip:
        print("  skipping     : %s" % " ".join(args.skip))
    print("  NOTE force-vs-displacement is intentionally not a stage (HANDOFF §13.5)")

    produced = []

    # ---- stage 1: per-run post-processing
    if "postprocess" not in args.skip:
        ok = run([py, "postprocess_indent.py", run_id],
                 "stage 1/4  postprocess_indent.py  (per-run)",
                 [os.path.join(out_dir, "repeats_summary.csv")], args.dry_run)
        if not ok:
            sys.exit(1)
        produced += [os.path.join(out_dir, "repeats_summary.csv"),
                     os.path.join(out_dir, "report_indent.pdf")]
        if args.all:                       # summary may now exist -> recount
            reps = list(range(1, n_repeats(run_dir) + 1))
            print("\n  repeats found in the summary: %s"
                  % ", ".join(str(r) for r in reps))

    # ---- stages 2-4: per repeat
    for rep in reps:
        stem = stem_for(run_dir, rep, args.accum_us, args.fps, args.roi)
        mkv = stem + ".mkv"
        det = stem + "_detections.json"

        if "video" not in args.skip:
            cmd = [py, "event_video.py", run_id, "--repeat", str(rep),
                   "--accum-us", str(args.accum_us), "--fps", "%g" % args.fps]
            if args.roi:
                cmd += ["--roi"] + [str(v) for v in args.roi]
            if not run(cmd, "stage 2/4  event_video.py  (repeat %d)" % rep,
                       [mkv, stem + ".json"], args.dry_run):
                sys.exit(1)
            produced += [mkv, stem + ".json", stem + "_annotated.mp4"]

        if "overlay" not in args.skip:
            cmd = [py, "marker_overlay.py", "--mkv", mkv,
                   "--view-scale", str(args.view_scale)]
            if args.radius_px > 0:
                cmd += ["--radius-px", "%g" % args.radius_px]
            if not run(cmd, "stage 3/4  marker_overlay.py  (repeat %d)" % rep,
                       [stem + "_circles.mp4", det], args.dry_run):
                sys.exit(1)
            produced += [stem + "_circles.mp4", det]

        if "arrows" not in args.skip:
            cmd = [py, "marker_arrows.py", "--detections", det,
                   "--net-gain", "%g" % args.net_gain,
                   "--step-gain", "%g" % args.step_gain,
                   "--fig-gain", "%g" % args.fig_gain,
                   "--view-scale", str(args.view_scale)]
            if not run(cmd, "stage 4/4  marker_arrows.py  (repeat %d)" % rep,
                       [stem + "_arrows.mp4", stem + "_displacement.csv"], args.dry_run):
                sys.exit(1)
            produced += [stem + "_arrows.mp4", stem + "_displacement.csv",
                         stem + "_peak_displacement.png",
                         stem + "_final_displacement.png"]

    print("\n" + "=" * 78)
    print(" PIPELINE COMPLETE" if not args.dry_run else " DRY RUN COMPLETE")
    print("=" * 78)
    if args.dry_run:
        return
    for p in produced:
        if os.path.exists(p):
            print("  %8.2f MB  %s" % (os.path.getsize(p) / 1e6, os.path.relpath(p, REPO)))
        else:
            print("  %8s     %s  [not produced]" % ("-", os.path.relpath(p, REPO)))
    print("\nNext step (force vs displacement) is deferred until the indentation xy is")
    print("known from CAD -- see HANDOFF §13.5.")


if __name__ == "__main__":
    main()
