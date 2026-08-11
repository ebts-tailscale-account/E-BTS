#!/usr/bin/env python3
"""Post-process a REPEATED-INDENT run (master_midpoint.py) -- separate from postprocess.py.

postprocess.py is for the grid sweep: it finds pokes by thresholding ee_z against
surface_z with --press-margin. That is fragile here -- the achieved indent (~1.9 mm)
sits barely above the 1.5 mm default, which is what silently dropped 28 of 97 pokes
in the run1 sweep.

This script does not threshold anything. The logger already tags every sample with
`phase` (home/travel/tare/dip/dwell/retract/park) and `point_index` (the repeat), so
each indentation is segmented EXACTLY:

  contact window  = phase == "dwell"  for that point_index
  its own F/T zero = phase == "tare"  for that point_index  (held out of contact
                     immediately before the dip, so drift cancels per repeat)

Other differences from postprocess.py:
  * DRIFT-AWARE clock alignment. metadata.json carries the offset measured BEFORE and
    AFTER the run; this interpolates linearly across the run instead of using one
    constant (4.8 ms of drift on the mid5 run).
  * NON-DESTRUCTIVE. It only READS recordings/<run>/ and writes output/<run>_indent/.
    postprocess.py shutil.move()s the source files out of recordings/; this copies.

Outputs (output/<run>_indent/):
  repeats_summary.csv            one row per indentation
  repeats/repeatNN_ft.csv        F/T slice, tared on that repeat's own window
  repeats/repeatNN_events.csv    event slice (x,y,polarity,t_us) over the window
  repeats/repeatNN_franka.csv    franka_states slice, workstation-clock aligned
  event_rate.csv                 whole-run event rate
  report_indent.pdf              overview + the 5 repeats overlaid + per-repeat images
  metadata.json                  copied, plus what this script derived

Usage:
  python3 postprocess_indent.py [run]        # default: newest mid*/ run folder
  python3 postprocess_indent.py --no-events  # skip the .raw pass (fast)
  python3 postprocess_indent.py --no-report  # data only, no PDF
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(REPO, "recordings")
OUTPUT = os.path.join(REPO, "output")

EVENT_BIN_S   = 0.02    # 20 ms bins for the event-rate trace
ACCUM_US      = 40000   # accumulation for the per-repeat event image (one 25 Hz cycle)
WIN_PRE_S     = 0.40    # event/plot window: this much before the dwell starts
WIN_POST_S    = 0.40    # ... and after it ends
CONTACT_K     = 5.0     # contact onset = |Fz - tare| > max(K*tare_sigma, CONTACT_MIN_N)
CONTACT_MIN_N = 0.05
SENSOR_W, SENSOR_H = 640, 480


# ----------------------------------------------------------------- inputs

def resolve_run(arg):
    if arg:
        p = arg if os.path.isdir(arg) else os.path.join(RECORDINGS, arg)
        if not os.path.isdir(p):
            sys.exit("no such run folder: %s" % p)
        return p
    cands = [d for d in glob.glob(os.path.join(RECORDINGS, "*")) if os.path.isdir(d)
             and os.path.exists(os.path.join(d, "franka.csv"))
             and os.path.basename(d).startswith("mid")]
    if not cands:
        sys.exit("no mid*/ run folder with a franka.csv found under %s" % RECORDINGS)
    return max(cands, key=os.path.getmtime)


def load(run_dir):
    def need(n):
        p = os.path.join(run_dir, n)
        if not os.path.exists(p):
            sys.exit("missing %s" % p)
        return p

    with open(need("metadata.json")) as f:
        meta = json.load(f)
    ft = np.genfromtxt(need("ft.csv"), delimiter=",", names=True)
    fr = np.genfromtxt(need("franka.csv"), delimiter=",", names=True,
                       dtype=None, encoding="utf-8")
    raw = os.path.join(run_dir, "camera.raw")
    return meta, ft, fr, (raw if os.path.exists(raw) else None)


def franka_time_ws(fr, meta):
    """franka unix_time_s (TACTILE clock) -> workstation clock, drift-aware.

    metadata carries the offset measured before and after the run. Interpolating
    across the run beats one constant: the mid5 run drifted 4.8 ms.
    """
    t = fr["unix_time_s"].astype(float)
    o0 = float(meta.get("tactile_minus_workstation_offset_s", 0.0))
    o1 = meta.get("offset_after_s")
    if o1 is None:
        return t - o0, o0, o0
    o1 = float(o1)
    # linear in wall time across the franka log
    frac = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    return t - (o0 + frac * (o1 - o0)), o0, o1


# ----------------------------------------------------------------- segment

def segment(fr, t_ws):
    """Exact per-repeat windows from the logged phase / point_index tags."""
    phase = fr["phase"].astype(str)
    idx = fr["point_index"].astype(int)
    reps = sorted(set(int(i) for i in idx if i >= 0))
    out = []
    for r in reps:
        d = (idx == r) & (phase == "dwell")
        a = (idx == r) & (phase == "tare")
        if not d.any():
            print("  [WARN] repeat %d has no dwell samples -- skipped" % r)
            continue
        rec = {
            "repeat": r,
            "dwell_t0": float(t_ws[d].min()), "dwell_t1": float(t_ws[d].max()),
            "tare_t0": float(t_ws[a].min()) if a.any() else None,
            "tare_t1": float(t_ws[a].max()) if a.any() else None,
            "surface_z": float(np.nanmedian(fr["surface_z"][d])),
            "ee_x": float(np.nanmedian(fr["ee_x"][d])),
            "ee_y": float(np.nanmedian(fr["ee_y"][d])),
            "ee_z_min": float(np.nanmin(fr["ee_z"][d])),
            "dwell_mask": d,
        }
        rec["indent_mm"] = (rec["surface_z"] - float(np.nanmedian(fr["ee_z"][d]))) * 1e3
        rec["indent_mm_max"] = (rec["surface_z"] - rec["ee_z_min"]) * 1e3
        out.append(rec)
    return out


def tare_and_contact(ft, rep):
    """Per-repeat F/T zero from its own tare window + contact-onset time."""
    t = ft["unix_time_s"]
    cols = ["Fx_N", "Fy_N", "Fz_N", "Mx_mNm", "My_mNm", "Mz_mNm"]
    base, sig = {}, {}
    if rep["tare_t0"] is not None:
        m = (t >= rep["tare_t0"]) & (t <= rep["tare_t1"])
        if m.sum() >= 20:
            for c in cols:
                base[c] = float(ft[c][m].mean())
                sig[c] = float(ft[c][m].std())
            rep["tare_n"] = int(m.sum())
    if not base:
        print("  [WARN] repeat %d: tare window too sparse, using the run's first 1 s"
              % rep["repeat"])
        m = t < t[0] + 1.0
        for c in cols:
            base[c] = float(ft[c][m].mean())
            sig[c] = float(ft[c][m].std())
        rep["tare_n"] = int(m.sum())
    rep["tare"] = base
    rep["tare_sigma_Fz"] = sig["Fz_N"]

    # contact onset: first sample in [tare_end, dwell_end] whose tared |Fz| clears the
    # noise floor of THIS repeat's tare window
    thr = max(CONTACT_K * sig["Fz_N"], CONTACT_MIN_N)
    lo = rep["tare_t1"] if rep["tare_t1"] is not None else rep["dwell_t0"]
    m = (t >= lo) & (t <= rep["dwell_t1"])
    fz = ft["Fz_N"][m] - base["Fz_N"]
    tt = t[m]
    hit = np.where(np.abs(fz) > thr)[0]
    rep["contact_t"] = float(tt[hit[0]]) if hit.size else float(rep["dwell_t0"])
    rep["contact_thr_N"] = float(thr)
    return rep


# ----------------------------------------------------------------- events

def decode(raw, windows_us, want_rate=True):
    """One streaming pass: per-window events + a whole-run rate trace."""
    try:
        from metavision_core.event_io import EventsIterator
    except ImportError:
        print("  metavision_core not importable -- skipping events.")
        return None, None
    stop = max(w[1] for w in windows_us)
    bufs = [[] for _ in windows_us]
    rate_t, rate_n = [], []
    print("  decoding %s up to t=%.2f s ..." % (os.path.basename(raw), stop / 1e6))
    for evs in EventsIterator(input_path=raw, delta_t=int(EVENT_BIN_S * 1e6)):
        if want_rate:
            rate_t.append(len(rate_t) * EVENT_BIN_S + EVENT_BIN_S / 2)
            rate_n.append(int(evs.size))
        if evs.size == 0:
            continue
        if evs["t"][0] > stop:
            if not want_rate:
                break
            continue
        for i, (a, b) in enumerate(windows_us):
            if evs["t"][-1] < a or evs["t"][0] >= b:
                continue
            m = (evs["t"] >= a) & (evs["t"] < b)
            if m.any():
                bufs[i].append(np.stack([evs["x"][m].astype(np.int32),
                                         evs["y"][m].astype(np.int32),
                                         evs["p"][m].astype(np.int8),
                                         evs["t"][m].astype(np.int64)]))
    out = []
    for chunks in bufs:
        if not chunks:
            out.append(None)
            continue
        arr = np.concatenate(chunks, axis=1)
        o = np.argsort(arr[3], kind="stable")
        out.append(dict(x=arr[0][o].astype(np.int32), y=arr[1][o].astype(np.int32),
                        p=arr[2][o].astype(np.int8), t=arr[3][o]))
    return out, (np.array(rate_t), np.array(rate_n))


def image_of(ev, t0_us, t1_us):
    img = np.zeros((SENSOR_H, SENSOR_W), np.int64)
    if ev is None:
        return img
    m = (ev["t"] >= t0_us) & (ev["t"] < t1_us)
    if not m.any():
        return img
    flat = ev["y"][m].astype(np.int64) * SENSOR_W + ev["x"][m]
    return np.bincount(flat, minlength=SENSOR_H * SENSOR_W).reshape(SENSOR_H, SENSOR_W)


# ----------------------------------------------------------------- report

def make_report(path, run_id, meta, ft, fr, t_ws, reps, evs, rate, cam_epoch):
    with PdfPages(path) as pdf:
        # ---- page 1: whole run
        fig, ax = plt.subplots(4, 1, figsize=(11.7, 8.3), sharex=True,
                               gridspec_kw=dict(hspace=0.12))
        fig.suptitle("%s -- %d repeated indentations at one taught point"
                     % (run_id, len(reps)), fontsize=13, fontweight="bold")
        t_ft = ft["unix_time_s"] - cam_epoch
        g0 = reps[0]["tare"]
        ax[0].plot(t_ft, ft["Fz_N"] - g0["Fz_N"], lw=0.5, color="#c44e52", label="Fz")
        ax[0].set_ylabel("Fz (N)\n(repeat-1 tare)")
        ax[0].legend(fontsize=7, loc="lower left")
        res = np.sqrt((ft["Fx_N"] - g0["Fx_N"]) ** 2 + (ft["Fy_N"] - g0["Fy_N"]) ** 2
                      + (ft["Fz_N"] - g0["Fz_N"]) ** 2)
        ax[1].plot(t_ft, res, lw=0.5, color="#8172b2")
        ax[1].set_ylabel("|F| (N)")
        ax[2].plot(t_ws - cam_epoch, fr["ee_z"] * 1e3, lw=0.7, color="#333333", label="ee_z")
        ax[2].plot(t_ws - cam_epoch, fr["surface_z"] * 1e3, lw=0.7, ls=":",
                   color="#c44e52", label="surface_z")
        ax[2].set_ylabel("height (mm)")
        ax[2].invert_yaxis()
        ax[2].legend(fontsize=7, loc="lower left")
        if rate is not None:
            ax[3].plot(rate[0], rate[1] / 1e3, lw=0.5, color="#dd8452")
            ax[3].set_ylabel("events / %.0f ms\n(x1e3)" % (EVENT_BIN_S * 1e3))
        ax[3].set_xlabel("time since camera t=0 (s)")
        for a in ax:
            for r in reps:
                a.axvspan(r["dwell_t0"] - cam_epoch, r["dwell_t1"] - cam_epoch,
                          color="#4c72b0", alpha=0.15, zorder=0)
            a.grid(alpha=0.25)
        for r in reps:
            ax[0].annotate("#%d" % (r["repeat"] + 1),
                           ((r["dwell_t0"] + r["dwell_t1"]) / 2 - cam_epoch,
                            ax[0].get_ylim()[1]), fontsize=7, ha="center",
                           color="#1f3d6b", xytext=(0, -9), textcoords="offset points")
        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.10, right=0.97)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: the repeats overlaid on contact onset (the repeatability view)
        fig, ax = plt.subplots(1, 2, figsize=(11.7, 5.2))
        fig.suptitle("Repeatability: all %d indentations aligned on contact onset"
                     % len(reps), fontsize=13, fontweight="bold")
        # The raw 1 kHz traces overlap so tightly that whichever is drawn last hides the
        # rest, so show them faint and put a 20 ms moving average on top -- the smoothed
        # curves are what you can actually tell apart.
        cmap = plt.get_cmap("viridis")
        SMOOTH_S = 0.020
        for k, r in enumerate(reps):
            m = ((ft["unix_time_s"] >= r["contact_t"] - 0.5)
                 & (ft["unix_time_s"] <= r["contact_t"] + 3.0))
            tt = ft["unix_time_s"][m] - r["contact_t"]
            fz = ft["Fz_N"][m] - r["tare"]["Fz_N"]
            col = cmap(k / max(len(reps) - 1, 1))
            ax[0].plot(tt, fz, lw=0.4, color=col, alpha=0.18)
            if tt.size > 4:
                # Mean interval, NOT median(diff): the Wittenstein stamps samples in
                # USB bursts (~11-14 us apart inside a packet, then a gap), so the
                # median diff is ~1e-5 s while the true rate is ~1 kHz. Using the
                # median here inflated the window to ~2000 samples.
                dt = (tt[-1] - tt[0]) / max(tt.size - 1, 1)
                w = max(1, int(round(SMOOTH_S / max(dt, 1e-9))))
                kern = np.ones(w) / w
                sm = np.convolve(fz, kern, mode="valid")
                ax[0].plot(tt[w - 1:], sm, lw=1.1, color=col,
                           label="#%d" % (r["repeat"] + 1))
        ax[0].axvline(0, color="#333333", lw=0.7, ls="--")
        ax[0].set_xlabel("time from contact onset (s)")
        ax[0].set_ylabel("Fz, per-repeat tare (N)")
        ax[0].legend(fontsize=7, ncol=2, title="%.0f ms moving avg" % (SMOOTH_S * 1e3),
                     title_fontsize=6)
        ax[0].grid(alpha=0.25)
        ax[0].set_title("Fz, each on its own zero (faint = raw 1 kHz)", fontsize=9)

        w = 0.35
        n = np.arange(len(reps))
        pk = [r["peak_Fz"] for r in reps]
        ind = [r["indent_mm"] for r in reps]
        ax[1].bar(n - w / 2, np.abs(pk), w, color="#c44e52", label="|peak Fz| (N)")
        ax[1].bar(n + w / 2, ind, w, color="#4c72b0", label="indent (mm)")
        ax[1].set_xticks(n)
        ax[1].set_xticklabels(["#%d" % (r["repeat"] + 1) for r in reps])
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.25, axis="y")
        ax[1].set_title("spread: |Fz| %.3f+-%.3f N,  indent %.3f+-%.3f mm"
                        % (np.mean(np.abs(pk)), np.std(np.abs(pk)),
                           np.mean(ind), np.std(ind)), fontsize=9)
        fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.22)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: one accumulated event image per repeat
        if evs is not None and any(e is not None for e in evs):
            fig, axes = plt.subplots(1, len(reps), figsize=(2.4 * len(reps), 3.0))
            axes = np.atleast_1d(axes)
            fig.suptitle("Events accumulated over each dwell (ON polarity, absolute "
                         "sensor px)", fontsize=12, fontweight="bold")
            imgs = [image_of(e, int((r["dwell_t0"] - cam_epoch) * 1e6),
                             int((r["dwell_t1"] - cam_epoch) * 1e6))
                    for e, r in zip(evs, reps)]
            allnz = np.concatenate([i[i > 0].ravel() for i in imgs if (i > 0).any()]) \
                if any((i > 0).any() for i in imgs) else np.array([1])
            vmax = max(1.0, float(np.percentile(allnz, 99.5)))
            for a, img, r in zip(axes, imgs, reps):
                a.imshow(img, cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
                a.set_xticks([]); a.set_yticks([])
                a.set_title("#%d  %d ev\n%.3f N" % (r["repeat"] + 1, img.sum(),
                                                    r["peak_Fz"]), fontsize=8)
            fig.subplots_adjust(top=0.80, bottom=0.03, left=0.02, right=0.98, wspace=0.08)
            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "E-BTS repeated-indent report (%s)" % run_id


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None,
                    help="run folder name or path (default: newest mid*/)")
    ap.add_argument("--no-events", action="store_true", help="skip the camera.raw pass")
    ap.add_argument("--no-report", action="store_true", help="skip the PDF")
    args = ap.parse_args()

    run_dir = resolve_run(args.run)
    run_id = os.path.basename(run_dir.rstrip("/"))
    print("Run: %s" % run_dir)
    meta, ft, fr, raw = load(run_dir)

    out_dir = os.path.join(OUTPUT, run_id + "_indent")
    reps_dir = os.path.join(out_dir, "repeats")
    os.makedirs(reps_dir, exist_ok=True)

    # camera device t=0 == first F/T sample (both fire from begin_recording)
    cam_epoch = float(ft["unix_time_s"][0])
    t_ws, o0, o1 = franka_time_ws(fr, meta)
    print("Clock: offset before %+.1f ms, after %+.1f ms -> drift-corrected linearly"
          % (o0 * 1e3, o1 * 1e3))

    reps = segment(fr, t_ws)
    if not reps:
        sys.exit("no dwell phases found -- is this a master_midpoint.py run?")
    print("Segmented %d indentations from the logged phase tags (no thresholding)."
          % len(reps))

    for r in reps:
        tare_and_contact(ft, r)

    # ---- events, one pass
    evs = rate = None
    if raw and not args.no_events:
        wins = [(int((r["dwell_t0"] - WIN_PRE_S - cam_epoch) * 1e6),
                 int((r["dwell_t1"] + WIN_POST_S - cam_epoch) * 1e6)) for r in reps]
        evs, rate = decode(raw, wins)
    elif not raw:
        print("  no camera.raw in the run folder -- skipping events.")

    # ---- per-repeat slices + stats
    t_ft = ft["unix_time_s"]
    ftcols = ["Fx_N", "Fy_N", "Fz_N", "Mx_mNm", "My_mNm", "Mz_mNm"]
    for k, r in enumerate(reps):
        w0, w1 = r["dwell_t0"] - WIN_PRE_S, r["dwell_t1"] + WIN_POST_S
        m = (t_ft >= w0) & (t_ft <= w1)
        d = (t_ft >= r["dwell_t0"]) & (t_ft <= r["dwell_t1"])
        fz_d = ft["Fz_N"][d] - r["tare"]["Fz_N"]
        r["peak_Fz"] = float(fz_d[np.argmax(np.abs(fz_d))]) if fz_d.size else float("nan")
        r["mean_Fz"] = float(fz_d.mean()) if fz_d.size else float("nan")
        r["std_Fz"] = float(fz_d.std()) if fz_d.size else float("nan")
        resd = np.sqrt((ft["Fx_N"][d] - r["tare"]["Fx_N"]) ** 2
                       + (ft["Fy_N"][d] - r["tare"]["Fy_N"]) ** 2 + fz_d ** 2)
        r["peak_Fres"] = float(resd.max()) if resd.size else float("nan")
        r["n_ft"] = int(d.sum())

        with open(os.path.join(reps_dir, "repeat%02d_ft.csv" % (r["repeat"] + 1)), "w",
                  newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t_rel_contact_s", "unix_time_s"] + ftcols + ["in_dwell"])
            for i in np.where(m)[0]:
                wr.writerow(["%.6f" % (t_ft[i] - r["contact_t"]), "%.6f" % t_ft[i]]
                            + ["%.4f" % (ft[c][i] - r["tare"][c]) for c in ftcols]
                            + [int(r["dwell_t0"] <= t_ft[i] <= r["dwell_t1"])])

        fm = (t_ws >= w0) & (t_ws <= w1)
        names = list(fr.dtype.names)
        with open(os.path.join(reps_dir, "repeat%02d_franka.csv" % (r["repeat"] + 1)), "w",
                  newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t_ws_aligned_s"] + names)
            for i in np.where(fm)[0]:
                wr.writerow(["%.6f" % t_ws[i]] + [fr[n][i] for n in names])

        ev = evs[k] if evs else None
        if ev is not None:
            r["n_events"] = int(ev["t"].size)
            dt = (r["dwell_t1"] - r["dwell_t0"])
            dm = ((ev["t"] >= int((r["dwell_t0"] - cam_epoch) * 1e6))
                  & (ev["t"] < int((r["dwell_t1"] - cam_epoch) * 1e6)))
            r["n_events_dwell"] = int(dm.sum())
            r["event_rate_dwell_kHz"] = r["n_events_dwell"] / dt / 1e3 if dt > 0 else 0.0
            with open(os.path.join(reps_dir, "repeat%02d_events.csv" % (r["repeat"] + 1)),
                      "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["x", "y", "polarity", "t_us", "t_rel_contact_s"])
                c0 = (r["contact_t"] - cam_epoch) * 1e6
                for x, y, p, t in zip(ev["x"], ev["y"], ev["p"], ev["t"]):
                    wr.writerow([x, y, int(p), int(t), "%.6f" % ((t - c0) / 1e6)])
        else:
            r["n_events"] = r["n_events_dwell"] = 0
            r["event_rate_dwell_kHz"] = float("nan")

    # ---- summary
    fields = ["repeat", "contact_t_ws", "dwell_t0", "dwell_t1", "dwell_s",
              "ee_x", "ee_y", "surface_z", "indent_mm", "indent_mm_max",
              "peak_Fz_N", "mean_Fz_N", "std_Fz_N", "peak_Fres_N",
              "tare_Fz_N", "tare_sigma_Fz_N", "tare_n", "contact_thr_N",
              "n_ft", "n_events_window", "n_events_dwell", "event_rate_dwell_kHz"]
    spath = os.path.join(out_dir, "repeats_summary.csv")
    with open(spath, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(fields)
        for r in reps:
            wr.writerow([r["repeat"] + 1, "%.6f" % r["contact_t"], "%.6f" % r["dwell_t0"],
                         "%.6f" % r["dwell_t1"], "%.3f" % (r["dwell_t1"] - r["dwell_t0"]),
                         "%.5f" % r["ee_x"], "%.5f" % r["ee_y"], "%.6f" % r["surface_z"],
                         "%.3f" % r["indent_mm"], "%.3f" % r["indent_mm_max"],
                         "%.4f" % r["peak_Fz"], "%.4f" % r["mean_Fz"], "%.4f" % r["std_Fz"],
                         "%.4f" % r["peak_Fres"], "%.4f" % r["tare"]["Fz_N"],
                         "%.4f" % r["tare_sigma_Fz"], r.get("tare_n", 0),
                         "%.4f" % r["contact_thr_N"], r["n_ft"], r["n_events"],
                         r["n_events_dwell"], "%.2f" % r["event_rate_dwell_kHz"]])

    if rate is not None:
        with open(os.path.join(out_dir, "event_rate.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t_center_s", "n_events"])
            for t, n in zip(*rate):
                wr.writerow(["%.4f" % t, int(n)])

    # copy provenance (COPY -- never move the source data)
    for n in ("metadata.json", "camera.bias", "camera.roi"):
        s = os.path.join(run_dir, n)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out_dir, n))
    derived = dict(meta)
    derived["postprocess_indent"] = {
        "segmentation": "exact, from logged phase/point_index (no ee_z thresholding)",
        "clock": {"offset_before_s": o0, "offset_after_s": o1,
                  "correction": "linear interpolation across the franka log"},
        "tare": "per repeat, from its own phase=='tare' window",
        "n_repeats": len(reps),
        "accum_us_for_images": ACCUM_US,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(derived, f, indent=2)

    # ---- console table
    print("\n%-4s %8s %9s %10s %9s %9s %10s %8s"
          % ("rep", "indent", "peak Fz", "mean Fz", "std Fz", "|F| peak", "tare Fz", "events"))
    print("%-4s %8s %9s %10s %9s %9s %10s %8s"
          % ("", "mm", "N", "N", "N", "N", "N", "in dwell"))
    for r in reps:
        print("%-4d %8.3f %9.4f %10.4f %9.4f %9.4f %10.4f %8d"
              % (r["repeat"] + 1, r["indent_mm"], r["peak_Fz"], r["mean_Fz"],
                 r["std_Fz"], r["peak_Fres"], r["tare"]["Fz_N"], r["n_events_dwell"]))
    ind = np.array([r["indent_mm"] for r in reps])
    pk = np.array([abs(r["peak_Fz"]) for r in reps])
    print("-" * 78)
    print("indent   mean %.3f mm   sd %.3f mm   spread %.3f mm"
          % (ind.mean(), ind.std(), ind.max() - ind.min()))
    print("|peak Fz| mean %.4f N   sd %.4f N   spread %.4f N  (CV %.1f%%)"
          % (pk.mean(), pk.std(), pk.max() - pk.min(), 100 * pk.std() / max(pk.mean(), 1e-9)))

    if not args.no_report:
        rp = os.path.join(out_dir, "report_indent.pdf")
        make_report(rp, run_id, meta, ft, fr, t_ws, reps, evs, rate, cam_epoch)
        print("\nreport -> %s" % rp)
    print("output -> %s" % out_dir)


if __name__ == "__main__":
    main()
