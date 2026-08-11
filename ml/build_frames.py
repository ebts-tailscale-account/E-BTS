#!/usr/bin/env python3
"""
build_frames.py -- ONE frame per poke, paired with that poke's median force.

WHY ONE FRAME AND NOT FIFTY
---------------------------
A 2 s dwell at 40 ms accumulation yields 50 frames, so the campaign would give
50,400. That number is an illusion. Measured on this very run:

    within-dwell force sd   0.1022 N
    tare-window force sd    0.1022 N      <- identical

All within-dwell variation IS sensor noise: the force is flat, the arm is holding
still, and the 50 frames are 50 replicates of ONE (image, force) pair. They average
label noise; they add no degrees of freedom. EFFECTIVE N = 1008 either way.

Keeping all 50 would cost 50x the storage and, worse, invite a split that puts
frames from the same poke on both sides of a train/test boundary -- which inflates
every score it touches. One frame per poke makes that mistake impossible.

WHAT IS EXTRACTED
-----------------
Per poke, two frames at the TEMPORAL MIDPOINT of their phase windows:

    tare  frame : midpoint of the tare phase  (arm still, OUT of contact)
    dwell frame : midpoint of the dwell phase (arm holding the indent)

The midpoint is the right sample because it is furthest from both transitions --
the settling right after the dip and the release at retract.

The model input is the DIFFERENCE, dwell - tare, which cancels the static marker
lattice and leaves only displacement. Both frames are stored so the difference can
be recomputed, or the raw dwell used instead, without re-decoding 68 GB.

The label is `force_n` from pokes.csv: median(dwell Fz) - median(tare Fz), i.e. the
median over the whole 2 s window, not the force at the frame's instant. That is the
correct pairing -- the force is constant across the dwell, so its median is the
lowest-variance estimate of the quantity the single frame depicts.

TWO CLOCKS
----------
The franka log is on the TACTILE clock; ft.csv and camera.raw are on the WORKSTATION
clock. Phase windows are therefore converted tactile -> workstation with the
measured offset, interpolating the drift across the run. Camera device time 0 is the
first F/T sample (HANDOFF section 9) -- both are opened by the GUI on one clock.

    python3 build_frames.py recordings/planb_3mm_20260810_145750
    python3 build_frames.py <run> --limit 5 --png   # eyeball it first
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evt2_frames import Evt2Reader

ACCUM_US = 40000          # 40 ms == exactly one 25 Hz strobed-illumination cycle
W, H = 640, 480


def phase_windows(franka_csv):
    """{seq: {phase: (t_start, t_end)}} on the TACTILE clock, from the phase tags.

    The logger tags every sample, so this is exact -- no thresholding on depth or
    force, which is what silently lost 30% of the pilot (HANDOFF section 12.6).
    """
    out = {}
    cur_seq = cur_phase = None
    t0 = t1 = None
    with open(franka_csv) as f:
        for r in csv.DictReader(f):
            ph, pid = r["phase"], r["point_index"]
            t = float(r["unix_time_s"])
            if pid in ("", "None"):
                key = cur_seq
            else:
                key = int(pid)
            if ph != cur_phase or key != cur_seq:
                if cur_phase is not None and cur_seq is not None:
                    out.setdefault(cur_seq, {})[cur_phase] = (t0, t1)
                cur_phase, cur_seq, t0 = ph, key, t
            t1 = t
    if cur_phase is not None and cur_seq is not None:
        out.setdefault(cur_seq, {})[cur_phase] = (t0, t1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory")
    ap.add_argument("--out", default=None, help="output .h5 (default <run>/frames.h5)")
    ap.add_argument("--accum-us", type=int, default=ACCUM_US,
                    help="accumulation window in us (default 40000 = one strobe "
                         "cycle). Force is static during the dwell, so a LONGER "
                         "window is free SNR if you want it.")
    ap.add_argument("--limit", type=int, default=None, help="first N pokes only")
    ap.add_argument("--png", action="store_true",
                    help="also dump a contact sheet for eyeballing")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "frames.h5"
    meta = json.loads((run / "metadata.json").read_text())

    pokes = list(csv.DictReader(open(run / "pokes.csv")))
    if args.limit:
        pokes = pokes[:args.limit]

    # --- clocks --------------------------------------------------------------
    ft_t0 = None
    with open(run / "ft.csv") as f:
        f.readline()
        ft_t0 = float(f.readline().split(",")[0])
    cam_t0 = ft_t0                    # camera device 0 == first F/T sample

    o0 = float(meta["tactile_minus_workstation_offset_s"])
    o1 = float(meta.get("offset_after_s", o0))
    fk = run / "franka_seg00.csv"
    with open(fk) as f:
        f.readline()
        T0 = float(f.readline().split(",")[0])
    T1 = T0
    with open(fk, "rb") as f:                      # last line, cheaply
        f.seek(max(0, os.path.getsize(fk) - 4096))
        T1 = float(f.read().decode(errors="ignore").strip().split("\n")[-1].split(",")[0])

    def to_ws(t):
        """Tactile clock -> workstation clock, interpolating the measured drift."""
        a = 0.0 if T1 == T0 else (t - T0) / (T1 - T0)
        return t - (o0 + (o1 - o0) * min(max(a, 0.0), 1.0))

    print("clock: tactile-workstation %.4f -> %.4f s (drift %.4f)" % (o0, o1, o1 - o0))
    print("camera device t=0 == %.6f (first F/T sample)" % cam_t0)

    wins = phase_windows(fk)
    print("phase windows for %d pokes" % len(wins))

    # --- decode --------------------------------------------------------------
    n = len(pokes)
    h5 = h5py.File(out, "w")
    d_tare = h5.create_dataset("tare_frame", (n, H, W), dtype="u1",
                               compression="gzip", compression_opts=1,
                               chunks=(1, H, W))
    d_dwell = h5.create_dataset("dwell_frame", (n, H, W), dtype="u1",
                                compression="gzip", compression_opts=1,
                                chunks=(1, H, W))
    cols = ["seq", "point_index", "row", "col", "level_idx", "block", "pass",
            "target_force_n", "stiffness_map", "depth_cmd_mm",
            "depth_achieved_mm", "force_n", "force_sd_n", "tare_n", "t_rel_s"]
    store = {c: [] for c in cols}
    ev_tare, ev_dwell, tmid_us, skipped = [], [], [], []

    t_start = time.time()
    with Evt2Reader(str(run / "camera.raw")) as r:
        for i, p in enumerate(pokes):
            seq = int(p["seq"])
            w = wins.get(seq, {})
            if "tare" not in w or "dwell" not in w:
                skipped.append((seq, "missing phase window")); continue

            def midframe(phase):
                a, b = w[phase]
                mid = 0.5 * (to_ws(a) + to_ws(b))
                us = (mid - cam_t0) * 1e6
                s = int(us - args.accum_us / 2)
                fr, _ = r.frames(s, s + args.accum_us, accum_us=args.accum_us)
                return fr[0], us

            try:
                ft_, _ = midframe("tare")
                fd_, us = midframe("dwell")
            except Exception as e:
                skipped.append((seq, str(e))); continue

            d_tare[i] = ft_
            d_dwell[i] = fd_
            ev_tare.append(int(ft_.sum())); ev_dwell.append(int(fd_.sum()))
            tmid_us.append(us)
            for c in cols:
                store[c].append(p[c])

            if (i + 1) % 50 == 0 or i + 1 == n:
                el = time.time() - t_start
                print("  %4d/%d   %.1f s elapsed, ETA %.0f s   (tare %d ev, dwell %d ev)"
                      % (i + 1, n, el, el / (i + 1) * (n - i - 1),
                         ev_tare[-1], ev_dwell[-1]), flush=True)

    keep = len(ev_tare)
    if keep < n:
        d_tare.resize((keep, H, W)); d_dwell.resize((keep, H, W))

    for c in cols:
        v = store[c]
        try:
            arr = np.array([float(x) for x in v], dtype="f4")
            if c in ("seq", "point_index", "row", "col", "level_idx", "pass"):
                arr = arr.astype("i4")
        except ValueError:
            arr = np.array([str(x).encode() for x in v])
        h5.create_dataset(c, data=arr)
    h5.create_dataset("events_tare", data=np.array(ev_tare, "i8"))
    h5.create_dataset("events_dwell", data=np.array(ev_dwell, "i8"))
    h5.create_dataset("frame_t_us", data=np.array(tmid_us, "f8"))
    h5.attrs["accum_us"] = args.accum_us
    h5.attrs["run"] = run.name
    h5.attrs["cam_t0_unix"] = cam_t0
    h5.attrs["note"] = ("one frame per poke at the temporal midpoint of each phase "
                        "window; label force_n is the median over the whole dwell")
    h5.close()

    et, ed = np.array(ev_tare), np.array(ev_dwell)
    print("\nwrote %s  (%d pokes, %.0f MB)" % (out, keep, out.stat().st_size / 1e6))
    print("  events/frame  tare  median %6d   (%d - %d)" % (np.median(et), et.min(), et.max()))
    print("  events/frame  dwell median %6d   (%d - %d)" % (np.median(ed), ed.min(), ed.max()))
    print("  dwell/tare event ratio: median %.2f" % (np.median(ed) / max(1, np.median(et))))
    if skipped:
        print("  SKIPPED %d: %s" % (len(skipped), skipped[:5]))
    print("  %.1f s total" % (time.time() - t_start))

    if args.png:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with h5py.File(out, "r") as f:
            k = min(4, keep)
            fig, ax = plt.subplots(3, k, figsize=(3.1 * k, 8.4))
            ax = np.atleast_2d(ax).reshape(3, k)
            for j in range(k):
                t_, d_ = f["tare_frame"][j], f["dwell_frame"][j]
                diff = d_.astype("i2") - t_.astype("i2")
                for rown, (img, cm, ttl) in enumerate((
                        (t_, "gray", "tare"), (d_, "gray", "dwell"),
                        (diff, "coolwarm", "dwell - tare"))):
                    v = np.abs(diff).max() if rown == 2 else None
                    ax[rown, j].imshow(img, cmap=cm,
                                       vmin=-v if v else None, vmax=v if v else None)
                    ax[rown, j].set_title("%s  seq %d  %.2f N"
                                          % (ttl, int(f["seq"][j]), f["force_n"][j]),
                                          fontsize=8)
                    ax[rown, j].axis("off")
            fig.tight_layout()
            fig.savefig(run / "frames_check.png", dpi=110)
            print("  contact sheet -> %s" % (run / "frames_check.png"))


if __name__ == "__main__":
    main()
