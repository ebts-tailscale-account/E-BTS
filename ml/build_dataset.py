#!/usr/bin/env python3
"""Campaign run -> ML dataset: 40 ms frames + force/depth labels, in one HDF5.

Replaces postprocess.py's event slicing for campaign-scale runs. See HANDOFF §15.2e.

WHY NOT postprocess.py
----------------------
It buffers every poke's events in RAM and writes only at the end -- ~48 GB for a
648-indent run, which OOM'd the workstation. It also `shutil.move`s the sources out
of recordings/. This script holds ONE indent at a time, writes incrementally, and
only ever READS the recording.

WHY FRAMES, NOT EVENTS (measured, pilot_20260807_134855)
--------------------------------------------------------
A 40 ms window holds ~92,000 events over 288,000 px:
    events x,y,t (u2,u2,u4) = 8 B  ->  722 KB
    dense 640x480 uint8 count frame ->  281 KB     2.6x smaller, uncompressed
and gzip takes the frame ~9x further (15% pixel occupancy). Lossless here because
counts peak at 27 (uint8 exact -- NEVER rescale, §13.1) and every event is one
polarity (bias_diff_off=0 kills OFF in hardware: 923,529 ON / 0 OFF).

WHY THE WHOLE RAMP, NOT JUST tare+peak
--------------------------------------
Tested on the two extremes of the pilot's 4 mm level:
    stiffest (5.945 N): max marker displacement  9.24 px = 0.31 x pitch -> 2 frames OK
    softest  (1.381 N): max marker displacement 21.67 px = 0.74 x pitch -> 2 frames FAIL
                        (5 of 211 markers beyond the 12 px match radius)
⚠ Force does NOT predict displacement -- the SOFTEST indent moves markers furthest.
So a tare->peak nearest-neighbour match can bind to the wrong marker, which is the
§13.3 roster failure. Keeping the dip frames lets the tracker propagate stepwise
(~0.3 px per frame), which is what makes the roster gapless.

Output layout (one file):
    /frames        uint8  [N, H, W]  gzip, chunk = 1 frame
    /frame_indent  int32  [N]        which indent each frame belongs to
    /frame_phase   int8   [N]        0=tare 1=dip 2=dwell
    /frame_t_us    int64  [N]        frame start, camera re-based scale
    /indents/...   per-indent labels (depth, location, achieved depth, forces)

Usage:
    python3 build_dataset.py recordings/pilot_20260807_134855            # all
    python3 build_dataset.py <run> --limit 5                             # smoke test
    python3 build_dataset.py <run> --indents 598,633                     # specific
"""

import os as _os, sys as _sys
# ml/ lives one level below the repo root. Resolve every path against the ROOT,
# never the working directory, so these run correctly from anywhere -- and put
# the root on sys.path so modules that stayed there (marker_overlay) import.
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUTPUT = _os.path.join(REPO, "output")
RECORDINGS = _os.path.join(REPO, "recordings")
_sys.path.insert(0, REPO)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))


import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evt2_frames import Evt2Reader                                   # noqa: E402

ACCUM_US = 40000        # one 25 Hz illumination cycle (§12.5) -- frames are comparable
PHASES = {"tare": 0, "dip": 1, "dwell": 2}


def load_run(run):
    import pandas as pd
    need = ["franka.csv", "ft.csv", "camera.raw", "metadata.json"]
    missing = [n for n in need if not os.path.exists(os.path.join(run, n))]
    if missing:
        sys.exit("[ERROR] %s is missing: %s" % (run, ", ".join(missing)))
    meta = json.load(open(os.path.join(run, "metadata.json")))
    # campaign_plan.csv is OPTIONAL. It only carries the commanded depth, which is a
    # label-side variable the model never sees -- so a midpoint-style recording with
    # no plan can still be evaluated. Depth is filled with NaN when absent.
    pf = os.path.join(run, "campaign_plan.csv")
    if os.path.exists(pf):
        plan = {int(r["point_index"]): r for r in csv.DictReader(open(pf))}
    else:
        print("[note] no campaign_plan.csv -- depth_mm will be NaN (not a model input).")
        idxs = sorted({int(v) for v in
                       __import__("pandas").read_csv(os.path.join(run, "franka.csv"),
                                                     usecols=["point_index"])
                       ["point_index"].unique() if int(v) >= 0})
        plan = {i: {"point_index": i, "depth_mm": "nan", "row": 0, "col": 0,
                    "x": "nan", "y": "nan"} for i in idxs}
    fk = pd.read_csv(os.path.join(run, "franka.csv"),
                     usecols=["unix_time_s", "phase", "point_index",
                              "ee_z", "surface_z"])
    fk = fk[fk.phase.isin(PHASES)]
    ft = pd.read_csv(os.path.join(run, "ft.csv"),
                     usecols=["unix_time_s", "Fx_N", "Fy_N", "Fz_N"]).to_numpy()
    return meta, plan, fk, ft


def indent_windows(g):
    """{phase: (t_start_unix, t_end_unix)} for one indent, or None if incomplete."""
    w = {}
    for p in PHASES:
        s = g[g.phase == p]
        if len(s) < 3:
            return None
        w[p] = (float(s.unix_time_s.min()), float(s.unix_time_s.max()))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--accum-us", type=int, default=ACCUM_US)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N indents (smoke test)")
    ap.add_argument("--indents", default="",
                    help="comma-separated point_index values to extract")
    ap.add_argument("--no-dip", action="store_true",
                    help="tare+dwell only. FASTER and ~2x smaller, but the tracker "
                         "then has to jump tare->dwell, which was MEASURED to fail on "
                         "soft locations (21.7 px > the 12 px match radius). Only use "
                         "if you have re-verified it for your data.")
    ap.add_argument("--compress", type=int, default=4, help="gzip level 0-9")
    args = ap.parse_args()

    import h5py
    run = args.run.rstrip("/")
    out = args.out or os.path.join(OUTPUT, os.path.basename(run) + "_frames.h5")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    meta, plan, fk, ft = load_run(run)
    o0 = meta["tactile_minus_workstation_offset_s"]
    o1 = meta.get("offset_after_s", o0)
    tf, fx, fy, fz = ft[:, 0], ft[:, 1], ft[:, 2], ft[:, 3]
    cam_t0 = tf[0]                       # camera device t=0 == F/T start (§9)
    T0, T1 = fk.unix_time_s.min(), fk.unix_time_s.max()

    def to_ws(t):
        """Tactile clock -> workstation clock, interpolating the measured drift."""
        return t - (o0 + (o1 - o0) * (t - T0) / max(T1 - T0, 1e-9))

    want = sorted(plan)
    if args.indents:
        keep = {int(v) for v in args.indents.split(",") if v.strip()}
        want = [i for i in want if i in keep]
    if args.limit:
        want = want[: args.limit]

    phases = ["tare", "dwell"] if args.no_dip else ["tare", "dip", "dwell"]
    print("run        : %s" % run)
    print("out        : %s" % out)
    print("indents    : %d   phases: %s   accum %d us" % (len(want), "+".join(phases),
                                                          args.accum_us))
    if args.no_dip:
        print("  [WARN] --no-dip: tracker must jump tare->dwell. Measured to FAIL on "
              "soft\n         locations (21.7 px vs a 12 px match radius). See §15.2f.")

    reader = Evt2Reader(os.path.join(run, "camera.raw"))
    if reader.torn:
        sys.exit("[ERROR] camera.raw ends mid-word -- the writer died. Refusing.")
    print("camera.raw : %d words, device base %.3f s\n" % (reader.n_words,
                                                           reader.base_us() / 1e6))

    h5 = h5py.File(out, "w")
    H, W = 480, 640
    fr = h5.create_dataset("frames", shape=(0, H, W), maxshape=(None, H, W),
                           dtype="u1", chunks=(1, H, W),
                           compression="gzip", compression_opts=args.compress)
    d_idx = h5.create_dataset("frame_indent", (0,), maxshape=(None,), dtype="i4")
    d_ph = h5.create_dataset("frame_phase", (0,), maxshape=(None,), dtype="i1")
    d_t = h5.create_dataset("frame_t_us", (0,), maxshape=(None,), dtype="i8")

    labels, n_written, t_start = [], 0, time.time()
    grouped = dict(list(fk.groupby("point_index")))
    for k, pi in enumerate(want):
        g = grouped.get(pi)
        if g is None:
            print("  [skip] indent %d has no logged samples" % pi)
            continue
        w = indent_windows(g)
        if w is None:
            print("  [skip] indent %d is missing a phase window" % pi)
            continue

        s_us = (to_ws(w[phases[0]][0]) - cam_t0) * 1e6
        e_us = (to_ws(w[phases[-1]][1]) - cam_t0) * 1e6
        F, ts = reader.frames(s_us, e_us, accum_us=args.accum_us)

        # Label each frame by phase, then DROP the final truncated window: the last
        # 40 ms bin is cut short by the phase boundary, so it holds a fraction of the
        # events and its marker detections collapse (17 vs ~211 on the pilot).
        ends = {p: (to_ws(w[p][1]) - cam_t0) * 1e6 for p in phases}
        ph = np.full(len(ts), PHASES[phases[-1]], np.int8)
        for p in reversed(phases[:-1]):
            ph[ts <= ends[p]] = PHASES[p]
        if len(F) > 1:
            F, ts, ph = F[:-1], ts[:-1], ph[:-1]

        # forces on this indent's OWN tare window (§4.10 -- never a global baseline)
        def win(p):
            a = np.searchsorted(tf, to_ws(w[p][0]))
            b = np.searchsorted(tf, to_ws(w[p][1]))
            return a, b
        ta, tb = win("tare")
        da, db = win("dwell")
        # ⚠ The F/T stream can drop out. Measured on hard_20260807_223744: a 1.08 s
        # gap 5.5 s into the run swallowed indent 0's entire tare window. Rather than
        # lose the indent, fall back to the nearest samples within FALLBACK_S. The
        # sensor drifts ~0.6 mN/s, so a baseline 3 s away costs under 2 mN -- far
        # below the 0.116 N noise floor, and vastly better than no label at all.
        FALLBACK_S = 3.0
        tare_fallback = 0
        if tb <= ta:
            # ⚠ The fallback MUST stay out of contact. A window centred on the tare
            # midpoint reaches forward into this indent's own dip and dwell, which
            # drags the "baseline" toward the loaded value and under-reports the
            # force. (Seen for real: indent 0 of hard_20260807_223744 came out at
            # 1.050 N for a 4.0 mm press, below the pilot's 1.381 N minimum at that
            # depth.) So take only samples strictly BEFORE the dip begins.
            dip_start = to_ws(w["dip"][0]) if "dip" in w else to_ws(w["dwell"][0])
            t_ref = to_ws(w["tare"][0])
            near = (tf < dip_start - 0.05) & (tf >= t_ref - FALLBACK_S)
            if near.sum() >= 100:
                idx = np.where(near)[0]
                ta, tb = int(idx[0]), int(idx[-1]) + 1
                tare_fallback = 1
                print("    [warn] indent %d: no F/T in its tare window (stream dropout). "
                      "Using %d out-of-contact samples from the %.0f s before its dip."
                      % (pi, tb - ta, FALLBACK_S), flush=True)
            else:
                print("    [warn] indent %d: no F/T near its tare window -- label will be "
                      "NaN and downstream will skip it." % pi, flush=True)
        base = fz[ta:tb].mean() if tb > ta else np.nan
        base_sd = fz[ta:tb].std() if tb > ta else np.nan
        dwell_fz = abs(fz[da:db].mean() - base) if db > da else np.nan
        peak_fz = abs(fz[da:db] - base).max() if db > da else np.nan
        dw = g[g.phase == "dwell"]
        achieved = float((dw.surface_z - dw.ee_z).mean() * 1000)

        n = len(F)
        fr.resize(n_written + n, axis=0)
        fr[n_written:n_written + n] = F
        for ds, val in ((d_idx, np.full(n, pi, np.int32)), (d_ph, ph),
                        (d_t, ts.astype(np.int64))):
            ds.resize(n_written + n, axis=0)
            ds[n_written:n_written + n] = val

        p = plan[pi]
        labels.append((pi, float(p["depth_mm"]), int(p["row"]), int(p["col"]),
                       float(p["x"]), float(p["y"]), achieved,
                       float(base), float(base_sd), float(dwell_fz), float(peak_fz),
                       n_written, n, tare_fallback))
        n_written += n
        if (k + 1) % 10 == 0 or k == 0 or k == len(want) - 1:
            el = time.time() - t_start
            print("  [%4d/%4d] indent %3d  %.1f mm  %2d frames  |Fz| %.3f N   "
                  "%.0f frames total, %.1f min elapsed, ETA %.0f min"
                  % (k + 1, len(want), pi, float(p["depth_mm"]), n, dwell_fz,
                     n_written, el / 60, el / (k + 1) * (len(want) - k - 1) / 60),
                  flush=True)

    L = np.array(labels, dtype=[("point_index", "i4"), ("depth_mm", "f4"),
                                ("row", "i4"), ("col", "i4"), ("x", "f8"), ("y", "f8"),
                                ("achieved_mm", "f4"), ("tare_Fz_N", "f4"),
                                ("tare_sd_N", "f4"), ("dwell_Fz_N", "f4"),
                                ("peak_Fz_N", "f4"), ("frame_offset", "i8"),
                                ("n_frames", "i4"), ("tare_fallback", "i4")])
    h5.create_dataset("indents", data=L)
    h5.attrs["run"] = os.path.basename(run)
    h5.attrs["accum_us"] = args.accum_us
    h5.attrs["phases"] = ",".join(phases)
    h5.attrs["phase_codes"] = "tare=0,dip=1,dwell=2"
    h5.attrs["geometry"] = "%dx%d" % (W, H)
    h5.attrs["camera_roi"] = open(os.path.join(run, "camera.roi")).read() \
        if os.path.exists(os.path.join(run, "camera.roi")) else ""
    h5.attrs["note"] = ("frames are RAW EVENT COUNTS per 40 ms (uint8; observed max 27, so "
                        "counts fit exactly) "
                        "-- never rescale, §13.1. Single polarity (OFF killed in "
                        "hardware). Force labels are tared on each indent's OWN tare "
                        "window. Clock drift interpolated between the two measured "
                        "offsets.")
    h5.close()
    reader.close()
    sz = os.path.getsize(out)
    print("\nwrote %d frames for %d indents -> %s" % (n_written, len(labels), out))
    print("  %.2f GB on disk, %.1fx compression, %.1f min"
          % (sz / 1e9, n_written * H * W / max(sz, 1), (time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
