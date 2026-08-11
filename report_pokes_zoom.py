#!/usr/bin/env python3
"""Zoomed report for the first N pokes: force/displacement + correlating events + accumulated images.

Standalone and additive -- reads an existing output/<run>/ folder and writes ONE new PDF
next to it. Touches no other file and re-runs nothing in the pipeline.

Unlike the pre-sliced pokes/pokeNN_events.csv (which cover only the "pressed" window, so
they start already in contact), this re-decodes camera.raw over a window with a pre-contact
MARGIN_PRE_S baseline, so the event burst is visible against quiet. Only the first ~20 s of
the .raw is streamed -- pokes 1-2 are ~7.6-16.3 s in -- then decoding stops.

Usage:
  python3 report_pokes_zoom.py [run_folder] [--pokes 2] [--accum-us 40000] [--out FILE]

Time base (per output/<run>/metadata.json):
  camera device t=0  ~=  first ft.csv sample (workstation clock)
  franka.csv is on the TACTILE clock -> subtract tactile_minus_workstation_offset_s
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

REPO = os.path.dirname(os.path.abspath(__file__))

ACCUM_US_DEFAULT = 40000   # user-specified accumulation time (microseconds = 40 ms/frame)
MARGIN_PRE_S     = 0.80    # decode this much BEFORE the poke window (quiet baseline)
MARGIN_POST_S    = 0.40    # ... and this much after
TARE_WINDOW_S    = 0.60    # per-poke F/T zero: mean over this window ...
TARE_GUARD_S     = 0.20    # ... ending this far before the poke window starts
SENSOR_W, SENSOR_H = 640, 480   # .raw geometry is always full-frame; ROI only makes it sparse

FRAME_CMAP = "inferno"
FRAME_PCTL = 99.5          # robust vmax so a few hot pixels don't wash the markers out


# ---------------------------------------------------------------- inputs

def load_run(run_dir):
    """Read metadata, F/T, franka, poke summary, ROI. Returns a dict."""
    def need(name):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            sys.exit("missing required file: %s" % p)
        return p

    with open(need("metadata.json")) as f:
        meta = json.load(f)

    ft = np.genfromtxt(need("ft.csv"), delimiter=",", names=True)
    fr = np.genfromtxt(need("franka.csv"), delimiter=",", names=True,
                       dtype=None, encoding="utf-8")
    summ = np.genfromtxt(os.path.join(run_dir, "pokes", "pokes_summary.csv"),
                         delimiter=",", names=True)

    # camera device t=0 == first F/T sample (both fire from begin_recording)
    cam_epoch = float(ft["unix_time_s"][0])
    # franka is on the tactile clock
    offset = float(meta.get("tactile_minus_workstation_offset_s", 0.0))

    roi = None
    roi_path = os.path.join(run_dir, "camera.roi")
    if os.path.exists(roi_path):
        with open(roi_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower() == "disabled":
                    break
                parts = line.split()
                if len(parts) == 4:
                    roi = tuple(int(v) for v in parts)
                break

    biases = {}
    bias_path = os.path.join(run_dir, "camera.bias")
    if os.path.exists(bias_path):
        with open(bias_path) as f:
            for line in f:
                if "%" in line:
                    val, name = line.split("%", 1)
                    biases[name.strip()] = val.strip()

    return dict(run_dir=run_dir, meta=meta, ft=ft, fr=fr, summ=summ,
                cam_epoch=cam_epoch, offset=offset, roi=roi, biases=biases)


def decode_windows(raw_path, windows_us, accum_us):
    """Stream camera.raw once and keep only events inside windows_us.

    windows_us: list of (t0_us, t1_us) in camera device time.
    Returns list of dicts {x, y, t} (int arrays), one per window.
    """
    try:
        from metavision_core.event_io import EventsIterator
    except ImportError:
        sys.exit("metavision_core not importable -- cannot decode camera.raw")

    stop_us = max(w[1] for w in windows_us)
    bufs = [[] for _ in windows_us]
    print("Decoding %s up to t=%.2f s ..." % (os.path.basename(raw_path), stop_us / 1e6))

    for evs in EventsIterator(input_path=raw_path, delta_t=100000):
        if evs.size == 0:
            continue
        if evs["t"][0] > stop_us:
            break
        for i, (a, b) in enumerate(windows_us):
            if evs["t"][-1] < a or evs["t"][0] > b:
                continue
            m = (evs["t"] >= a) & (evs["t"] < b)   # half-open: b belongs to the next frame
            if m.any():
                bufs[i].append(np.stack([evs["x"][m].astype(np.int32),
                                         evs["y"][m].astype(np.int32),
                                         evs["t"][m].astype(np.int64)]))

    out = []
    for i, chunks in enumerate(bufs):
        if not chunks:
            out.append(dict(x=np.empty(0, np.int32), y=np.empty(0, np.int32),
                            t=np.empty(0, np.int64)))
            continue
        arr = np.concatenate(chunks, axis=1)
        order = np.argsort(arr[2], kind="stable")
        out.append(dict(x=arr[0][order].astype(np.int32),
                        y=arr[1][order].astype(np.int32),
                        t=arr[2][order]))
        print("  window %d: %d events" % (i + 1, arr.shape[1]))
    return out


def build_frames(ev, anchor_us, accum_us, n_frames):
    """Accumulate events into n_frames images of accum_us each, anchored at anchor_us.

    Frame boundaries are anchored at the poke-window start so one edge lands exactly on
    contact. Returns (frames uint16 [n,H,W], per_frame_counts int64 [n]).
    """
    frames = np.zeros((n_frames, SENSOR_H, SENSOR_W), np.uint16)
    counts = np.zeros(n_frames, np.int64)
    if ev["t"].size == 0:
        return frames, counts

    fidx = (ev["t"] - anchor_us) // accum_us
    flat = ev["y"].astype(np.int64) * SENSOR_W + ev["x"].astype(np.int64)
    # events are time-sorted, so each frame is a contiguous slice
    edges = np.searchsorted(fidx, np.arange(n_frames + 1))
    for k in range(n_frames):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        bc = np.bincount(flat[a:b], minlength=SENSOR_H * SENSOR_W)
        frames[k] = np.minimum(bc, 65535).astype(np.uint16).reshape(SENSOR_H, SENSOR_W)
        counts[k] = b - a
    return frames, counts


# ---------------------------------------------------------------- drawing

def draw_frame(ax, img, roi, title=None, vmax=None, cmap=FRAME_CMAP, show_roi=True):
    if vmax is None:
        nz = img[img > 0]
        vmax = max(1.0, np.percentile(nz, FRAME_PCTL)) if nz.size else 1.0
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest",
              origin="upper", extent=[0, SENSOR_W, SENSOR_H, 0])
    if roi and show_roi:
        x, y, w, h = roi
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#39d0ff",
                               lw=0.7, ls="--", alpha=0.8))
    ax.set_xlim(0, SENSOR_W)
    ax.set_ylim(SENSOR_H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=6.5, pad=1.5)
    return vmax


def cover_page(pdf, R, pokes, accum_us):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("E-BTS - zoomed report: first %d poke(s)" % len(pokes),
                 fontsize=15, fontweight="bold", y=0.97)

    roi_s = ("%d %d %d %d" % R["roi"]) if R["roi"] else "disabled / absent"
    bias_s = ", ".join("%s=%s" % (k, v) for k, v in sorted(R["biases"].items())) or "n/a"
    lines = [
        "run_id: %s" % R["meta"].get("run_id", "?"),
        "source: %s" % R["run_dir"],
        "",
        "accumulation time: %d us  (%.0f ms per frame)  <- as specified" % (accum_us, accum_us / 1e3),
        "decode window: poke window  -%.2f s / +%.2f s  (re-decoded from camera.raw for a"
        " pre-contact baseline)" % (MARGIN_PRE_S, MARGIN_POST_S),
        "camera ROI: %s   (events outside are never read out; .raw coords stay ABSOLUTE)" % roi_s,
        "camera biases: %s" % bias_s,
        "polarity: bias_diff_off=0 kills OFF events in hardware -> this run is ON-polarity ONLY",
        "F/T zero: per-poke tare, mean over the %.1f s ending %.1f s before the poke window"
        % (TARE_WINDOW_S, TARE_GUARD_S),
        "clocks: camera t=0 == first ft.csv sample; franka.csv shifted by -%.4f s onto the"
        " workstation clock" % R["offset"],
    ]
    fig.text(0.06, 0.90, "\n".join(lines), fontsize=8.5, va="top", family="monospace")

    # poke summary table
    cols = ["poke", "col", "row", "dur_s", "indent_mm", "peak_Fz_ft_N",
            "peak_F_res_N", "n_events"]
    cell = []
    for p in pokes:
        cell.append(["%d" % p["poke"], "%d" % p["col"], "%d" % p["row"],
                     "%.3f" % p["dur_s"], "%.2f" % p["indent_mm"],
                     "%.3f" % p["peak_Fz"], "%.3f" % p["peak_res"],
                     "%d" % p["n_events_summary"]])
    ax = fig.add_axes([0.06, 0.42, 0.88, 0.14])
    ax.axis("off")
    tb = ax.table(cellText=cell, colLabels=cols, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.35)
    ax.set_title("pokes_summary.csv rows for the pokes in this report", fontsize=9,
                 loc="left", pad=6)

    # whole-run event rate with these pokes highlighted
    er_path = os.path.join(R["run_dir"], "event_rate.csv")
    ax2 = fig.add_axes([0.06, 0.08, 0.88, 0.26])
    if os.path.exists(er_path):
        er = np.genfromtxt(er_path, delimiter=",", names=True)
        ax2.plot(er["t_center_s"], er["n_pos"] / 1e3, lw=0.4, color="#c44e52")
        for p in pokes:
            ax2.axvspan(p["t0_us"] / 1e6, p["t1_us"] / 1e6, color="#4c72b0", alpha=0.35)
            ax2.annotate("poke %d" % p["poke"],
                         (p["t0_us"] / 1e6, ax2.get_ylim()[1]),
                         fontsize=7, color="#1f3d6b",
                         xytext=(0, -10), textcoords="offset points")
        ax2.set_xlabel("camera device time (s)")
        ax2.set_ylabel("ON events / 50 ms bin  (x1e3)")
        ax2.set_title("whole-run event rate - shaded = the pokes zoomed in this report",
                      fontsize=9, loc="left")
        ax2.grid(alpha=0.25)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "event_rate.csv not found", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def timeline_page(pdf, R, p, counts, accum_us, n_frames):
    """Zoomed force / displacement / event-rate for one poke, on one shared time axis."""
    ft, fr = R["ft"], R["fr"]
    t_ft = (ft["unix_time_s"] - R["cam_epoch"])
    t_fr = (fr["unix_time_s"] - R["offset"] - R["cam_epoch"])
    w0, w1 = p["win_us"][0] / 1e6, p["win_us"][1] / 1e6
    pk0, pk1 = p["t0_us"] / 1e6, p["t1_us"] / 1e6

    mft = (t_ft >= w0) & (t_ft <= w1)
    mfr = (t_fr >= w0) & (t_fr <= w1)

    # per-poke tare on the quiet pre-contact window
    tb = (t_ft >= pk0 - TARE_GUARD_S - TARE_WINDOW_S) & (t_ft <= pk0 - TARE_GUARD_S)
    def z(c):
        base = ft[c][tb].mean() if tb.any() else 0.0
        return ft[c][mft] - base

    fig, axes = plt.subplots(4, 1, figsize=(11.7, 8.3), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.15, 1, 1, 1], hspace=0.12))
    fig.suptitle("Poke %d  (col %d, row %d)  -  zoom: force / displacement / events"
                 % (p["poke"], p["col"], p["row"]), fontsize=13, fontweight="bold")

    Fx, Fy, Fz = z("Fx_N"), z("Fy_N"), z("Fz_N")
    res = np.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)

    ax = axes[0]
    ax.plot(t_ft[mft], Fz, lw=0.9, color="#c44e52", label="Fz")
    ax.plot(t_ft[mft], Fx, lw=0.6, color="#4c72b0", alpha=0.8, label="Fx")
    ax.plot(t_ft[mft], Fy, lw=0.6, color="#55a868", alpha=0.8, label="Fy")
    ax.plot(t_ft[mft], res, lw=0.9, color="#8172b2", ls="--", label="|F|")
    ax.set_ylabel("force (N)\nWittenstein, tared")
    ax.legend(fontsize=7, ncol=4, loc="lower left")

    ax = axes[1]
    ax.plot(t_ft[mft], ft["Mx_mNm"][mft] - (ft["Mx_mNm"][tb].mean() if tb.any() else 0),
            lw=0.6, color="#4c72b0", label="Mx")
    ax.plot(t_ft[mft], ft["My_mNm"][mft] - (ft["My_mNm"][tb].mean() if tb.any() else 0),
            lw=0.6, color="#55a868", label="My")
    ax.plot(t_ft[mft], ft["Mz_mNm"][mft] - (ft["Mz_mNm"][tb].mean() if tb.any() else 0),
            lw=0.6, color="#c44e52", label="Mz")
    ax.set_ylabel("torque (mNm)")
    ax.legend(fontsize=7, ncol=3, loc="lower left")

    ax = axes[2]
    if mfr.any():
        ax.plot(t_fr[mfr], fr["ee_z"][mfr] * 1e3, lw=0.9, color="#333333", label="ee_z")
        sz = fr["surface_z"][mfr]
        if np.isfinite(sz).any():
            ax.plot(t_fr[mfr], sz * 1e3, lw=0.8, ls=":", color="#c44e52",
                    label="mapped surface_z")
        ax.set_ylabel("height (mm)")
        ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.invert_yaxis()

    ax = axes[3]
    fedges = (p["win_us"][0] + np.arange(n_frames + 1) * accum_us) / 1e6
    fcent = 0.5 * (fedges[:-1] + fedges[1:])
    ax.bar(fcent, counts / 1e3, width=accum_us / 1e6 * 0.95, color="#dd8452",
           edgecolor="none")
    ax.set_ylabel("ON events\nper %.0f ms frame (x1e3)" % (accum_us / 1e3))
    ax.set_xlabel("camera device time (s)   -   frame bins = the %.0f ms accumulation windows"
                  % (accum_us / 1e3))

    for ax in axes:
        ax.axvspan(pk0, pk1, color="#4c72b0", alpha=0.12, zorder=0)
        ax.axvline(pk0, color="#4c72b0", lw=0.8, alpha=0.8)
        ax.axvline(pk1, color="#4c72b0", lw=0.8, alpha=0.8)
        ax.grid(alpha=0.25)
    axes[3].annotate("pressed window (postprocess)", (pk0, 0),
                     fontsize=7, color="#1f3d6b", xytext=(3, 4),
                     textcoords="offset points")
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.09, right=0.97)
    pdf.savefig(fig)
    plt.close(fig)


def correlation_page(pdf, R, p, ev, frames, counts, accum_us):
    """Why the event RATE is flat but the image still encodes contact.

    Left column proves the frame grid is phase-locked to the strobed illumination
    (the Arduino complementary 25 Hz driver -> 40 ms period, two bursts per cycle).
    Right column is the displacement view: a pre-contact reference image, the
    in-contact image, and their difference -- which IS the contact signal.
    """
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("Poke %d  -  event/contact correlation: illumination phase lock + marker"
                 " displacement" % p["poke"], fontsize=13, fontweight="bold")

    n = p["n_frames"]
    k_on = int(np.round((p["t0_us"] - p["win_us"][0]) / accum_us))
    k_off = int(np.round((p["t1_us"] - p["win_us"][0]) / accum_us))
    nref = max(1, int(round(0.40 / (accum_us / 1e6))))       # ~0.4 s of frames each side

    # 1 ms event-rate series and its autocorrelation, used by two panels below
    rate_ms = ac = None
    r_at_period = float("nan")
    if ev["t"].size:
        rate_ms = np.bincount(((ev["t"] - p["win_us"][0]) // 1000).astype(int)).astype(float)
        x = rate_ms - rate_ms.mean()
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac = ac / ac[0]
        k = int(accum_us / 1000)
        if ac.size > k:
            r_at_period = float(ac[k])

    # ---- 1. folded illumination phase profile
    ax = fig.add_subplot(3, 3, 1)
    if ev["t"].size:
        rel = ev["t"] - p["win_us"][0]
        ph = (rel % accum_us) // 1000
        prof = np.bincount(ph.astype(int), minlength=accum_us // 1000)
        ax.bar(np.arange(prof.size), prof / 1e3, width=1.0, color="#dd8452")
        ax.set_title("events folded on the %.0f ms frame period"
                     % (accum_us / 1e3), fontsize=8)
        ax.set_xlabel("phase within frame (ms)", fontsize=7)
        ax.set_ylabel("events (x1e3)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)

    # ---- 2. autocorrelation -> the period
    ax = fig.add_subplot(3, 3, 4)
    if ac is not None:
        lag = np.arange(min(220, ac.size))
        ax.plot(lag, ac[:lag.size], lw=0.8, color="#4c72b0")
        ax.axvline(accum_us / 1000.0, color="#c44e52", ls="--", lw=0.9)
        ax.set_title("autocorrelation of the 1 ms event rate\nr(%.0f ms) = %+.3f"
                     % (accum_us / 1e3, r_at_period), fontsize=8)
        ax.set_xlabel("lag (ms)", fontsize=7)
        ax.set_ylabel("r", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)

    # ---- 3. per-frame event count vs |Fz|
    ax = fig.add_subplot(3, 3, 7)
    fz = p["fz_at_frame"]
    inw = np.zeros(n, bool)
    inw[k_on:min(k_off, n)] = True
    ax.scatter(np.abs(fz[~inw]), counts[~inw] / 1e3, s=6, alpha=0.6,
               color="#8c8c8c", label="out of contact")
    ax.scatter(np.abs(fz[inw]), counts[inw] / 1e3, s=6, alpha=0.7,
               color="#4c72b0", label="in contact")
    ok = np.isfinite(fz) & (counts > 0)
    if ok.sum() > 2:
        r = np.corrcoef(np.abs(fz[ok]), counts[ok])[0, 1]
        ax.set_title("event count vs |Fz| per frame\nPearson r = %+.3f" % r, fontsize=8)
    ax.set_xlabel("|Fz| (N)", fontsize=7)
    ax.set_ylabel("events / frame (x1e3)", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25)

    # ---- 4-6. reference / contact / difference images
    ref = frames[max(0, k_on - nref):k_on].mean(axis=0) if k_on > 0 else \
        np.zeros((SENSOR_H, SENSOR_W))
    mid = (k_on + min(k_off, n)) // 2
    con = frames[max(k_on, mid - nref // 2):min(n, mid + nref // 2)].mean(axis=0)

    ax = fig.add_subplot(3, 3, 2)
    vm = draw_frame(ax, ref, R["roi"],
                    title="REFERENCE: mean of %d pre-contact frames\n(%.2f s before contact)"
                          % (nref, nref * accum_us / 1e6))
    ax = fig.add_subplot(3, 3, 3)
    draw_frame(ax, con, R["roi"], vmax=vm,
               title="IN CONTACT: mean of %d mid-dwell frames\n(same colour scale)" % nref)

    diff = con - ref
    ax = fig.add_subplot(3, 3, 5)
    lim = np.percentile(np.abs(diff), 99.8) or 1.0
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                   origin="upper", extent=[0, SENSOR_W, SENSOR_H, 0])
    if R["roi"]:
        x, y, w, h = R["roi"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#333333", lw=0.7, ls="--"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("DIFFERENCE  (contact - reference)\nred = more events, blue = fewer",
                 fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=5)

    # ---- 7. marginals of the DIFFERENCE -> which rows/cols the markers moved in.
    # Plotting ref and contact on top of each other is useless (they overlap to within
    # ~1%), so show contact-minus-reference, which is the displacement itself.
    drow = con.sum(axis=1) - ref.sum(axis=1)
    dcol = con.sum(axis=0) - ref.sum(axis=0)

    ax = fig.add_subplot(3, 3, 6)
    rows = np.arange(SENSOR_H)
    ax.fill_betweenx(rows, 0, drow, where=drow >= 0, color="#c44e52", lw=0)
    ax.fill_betweenx(rows, 0, drow, where=drow < 0, color="#4c72b0", lw=0)
    ax.axvline(0, color="#333333", lw=0.6)
    if R["roi"]:
        ax.axhspan(R["roi"][1], R["roi"][1] + R["roi"][3], color="#39d0ff", alpha=0.06)
    ax.invert_yaxis()
    ax.set_title("row profile of the DIFFERENCE\n(contact - reference)", fontsize=8)
    ax.set_xlabel("delta events", fontsize=7)
    ax.set_ylabel("sensor row (px)", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.25)

    ax = fig.add_subplot(3, 3, 8)
    cols = np.arange(SENSOR_W)
    ax.fill_between(cols, 0, dcol, where=dcol >= 0, color="#c44e52", lw=0)
    ax.fill_between(cols, 0, dcol, where=dcol < 0, color="#4c72b0", lw=0)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_title("column profile of the DIFFERENCE\n(contact - reference)", fontsize=8)
    ax.set_xlabel("sensor col (px)", fontsize=7)
    ax.set_ylabel("delta events", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.25)

    ax = fig.add_subplot(3, 3, 9)
    ax.axis("off")
    ax.text(0.0, 1.0,
            "READ THIS PANEL FIRST\n"
            "\n"
            "The event RATE carries almost no contact\n"
            "signal: it sits at ~%.0fk events/frame whether\n"
            "the tool is pressed or hovering.\n"
            "\n"
            "Reason: the illumination is STROBED. The\n"
            "1 ms rate autocorrelates at r=%.3f on a\n"
            "%.0f ms lag, and the folded profile shows TWO\n"
            "bursts per cycle -- the Arduino complementary\n"
            "two-phase 25 Hz driver (1/25 Hz = %.0f ms).\n"
            "\n"
            "So the requested %.0f ms accumulation is exactly\n"
            "ONE full illumination cycle: every frame\n"
            "integrates both phases, which is why the\n"
            "frames are directly comparable.\n"
            "\n"
            "The contact signal is therefore SPATIAL, not\n"
            "temporal -- it lives in WHERE the marker\n"
            "events land. That is what the difference\n"
            "image and the row/column profiles show."
            % (np.median(counts[counts > 0]) / 1e3 if (counts > 0).any() else 0.0,
               r_at_period, accum_us / 1e3, accum_us / 1e3, accum_us / 1e3),
            fontsize=7, va="top", family="monospace")

    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.06, right=0.97,
                        wspace=0.30, hspace=0.42)
    pdf.savefig(fig)
    plt.close(fig)


def key_frames_page(pdf, R, p, frames, counts, accum_us, keys, integrated):
    """Six labelled 40 ms frames at the interesting moments + the integrated poke image."""
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("Poke %d  -  accumulated event images (%.0f ms per frame)"
                 % (p["poke"], accum_us / 1e3), fontsize=13, fontweight="bold")

    vmax = None
    nz = frames[frames > 0]
    if nz.size:
        vmax = max(1.0, np.percentile(nz, FRAME_PCTL))

    for i, (k, label) in enumerate(keys):
        ax = fig.add_subplot(2, 4, i + 1)
        t_rel = (p["win_us"][0] + k * accum_us - p["t0_us"]) / 1e6
        draw_frame(ax, frames[k], R["roi"], vmax=vmax,
                   title="%s\nt=%+.3f s from contact | %d ev | Fz=%.3f N"
                         % (label, t_rel, counts[k], p["fz_at_frame"][k]))

    ax = fig.add_subplot(2, 4, 7)
    draw_frame(ax, integrated, R["roi"],
               title="INTEGRATED over pressed window\n%d events, %.3f s"
                     % (integrated.sum(), (p["t1_us"] - p["t0_us"]) / 1e6))

    ax = fig.add_subplot(2, 4, 8)
    ax.axis("off")
    ax.text(0.0, 0.95,
            "ON-polarity event counts per pixel.\n"
            "Colour = events in that %.0f ms window.\n\n"
            "Dashed cyan box = hardware ROI\n(%s).\n\n"
            "Coordinates are ABSOLUTE sensor px --\nthe .raw is sparse, not cropped.\n\n"
            "peak |Fz| = %.3f N\nindent (vs global plane) = %.2f mm"
            % (accum_us / 1e3,
               ("%d %d %d %d" % R["roi"]) if R["roi"] else "disabled",
               p["peak_Fz"], p["indent_mm"]),
            fontsize=7.5, va="top", family="monospace")
    fig.subplots_adjust(top=0.86, bottom=0.05, left=0.03, right=0.98,
                        wspace=0.10, hspace=0.28)
    pdf.savefig(fig)
    plt.close(fig)


def contact_sheet_page(pdf, R, p, frames, counts, accum_us):
    """Every 40 ms frame across the window, in time order."""
    n = frames.shape[0]
    ncol = 10
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.75 * ncol, 1.62 * nrow))
    axes = np.atleast_2d(axes).ravel()
    fig.suptitle("Poke %d  -  contact sheet: all %d frames at %.0f ms accumulation "
                 "(pressed window shaded blue)"
                 % (p["poke"], n, accum_us / 1e3), fontsize=13, fontweight="bold")

    nz = frames[frames > 0]
    vmax = max(1.0, np.percentile(nz, FRAME_PCTL)) if nz.size else 1.0

    for k in range(n):
        ax = axes[k]
        t_rel = (p["win_us"][0] + k * accum_us - p["t0_us"]) / 1e6
        pressed = 0.0 <= t_rel < (p["t1_us"] - p["t0_us"]) / 1e6
        draw_frame(ax, frames[k], R["roi"], vmax=vmax, show_roi=False,
                   title="%+.2fs | %.0fk | %.2fN" % (t_rel, counts[k] / 1e3,
                                                     p["fz_at_frame"][k]))
        if pressed:
            for s in ax.spines.values():
                s.set_color("#4c72b0")
                s.set_linewidth(1.6)
        else:
            for s in ax.spines.values():
                s.set_alpha(0.25)
    for k in range(n, axes.size):
        axes[k].axis("off")
    fig.subplots_adjust(top=1 - 0.5 / nrow, bottom=0.01, left=0.01, right=0.99,
                        wspace=0.06, hspace=0.34)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="run1_20260805_180345",
                    help="output/<run> folder name or path")
    ap.add_argument("--pokes", type=int, default=2, help="how many leading pokes")
    ap.add_argument("--accum-us", type=int, default=ACCUM_US_DEFAULT,
                    help="accumulation time per frame, MICROseconds (default %d)"
                         % ACCUM_US_DEFAULT)
    ap.add_argument("--out", default=None, help="output PDF path")
    args = ap.parse_args()

    run_dir = args.run if os.path.isdir(args.run) else os.path.join(REPO, "output", args.run)
    if not os.path.isdir(run_dir):
        sys.exit("no such run folder: %s" % run_dir)
    R = load_run(run_dir)
    accum_us = args.accum_us

    raw_path = os.path.join(run_dir, "camera.raw")
    if not os.path.exists(raw_path):
        sys.exit("camera.raw not found in %s" % run_dir)

    # ---- build the poke descriptors (first N rows of pokes_summary.csv)
    summ = np.atleast_1d(R["summ"])
    order = np.argsort(summ["poke"])[:args.pokes]
    pokes = []
    for i in order:
        t0 = int(summ["cam_t0_us"][i])
        t1 = int(summ["cam_t1_us"][i])
        w0 = max(0, t0 - int(MARGIN_PRE_S * 1e6))
        w1 = t1 + int(MARGIN_POST_S * 1e6)
        # Anchor frame boundaries so one edge lands exactly on contact, then snap the
        # decode window to a whole number of frames -- otherwise the final frame is
        # truncated by the window and its event count reads as a spurious dropout.
        n_pre = int(np.ceil((t0 - w0) / accum_us))
        w0 = t0 - n_pre * accum_us
        n_frames = n_pre + int(np.ceil((w1 - t0) / accum_us))
        w1 = w0 + n_frames * accum_us
        pokes.append(dict(poke=int(summ["poke"][i]), col=int(summ["col"][i]),
                          row=int(summ["row"][i]), dur_s=float(summ["dur_s"][i]),
                          indent_mm=float(summ["indent_mm"][i]),
                          peak_Fz=float(summ["peak_Fz_ft_N"][i]),
                          peak_res=float(summ["peak_F_res_N"][i]),
                          n_events_summary=int(summ["n_events"][i]),
                          t0_us=t0, t1_us=t1,
                          win_us=(w0, w1), n_frames=n_frames))

    print("Pokes: %s" % ", ".join("#%d (col %d,row %d) %d frames"
                                  % (p["poke"], p["col"], p["row"], p["n_frames"])
                                  for p in pokes))
    print("Accumulation time: %d us (%.0f ms/frame)" % (accum_us, accum_us / 1e3))

    # ---- one streaming pass over the .raw for both windows
    evs = decode_windows(raw_path, [p["win_us"] for p in pokes], accum_us)

    # ---- per-frame Fz (mean over each accumulation window) for the labels
    ft = R["ft"]
    t_ft = ft["unix_time_s"] - R["cam_epoch"]
    for p in pokes:
        tb = (t_ft >= p["t0_us"] / 1e6 - TARE_GUARD_S - TARE_WINDOW_S) & \
             (t_ft <= p["t0_us"] / 1e6 - TARE_GUARD_S)
        base = ft["Fz_N"][tb].mean() if tb.any() else 0.0
        fz = np.zeros(p["n_frames"])
        for k in range(p["n_frames"]):
            a = (p["win_us"][0] + k * accum_us) / 1e6
            b = a + accum_us / 1e6
            m = (t_ft >= a) & (t_ft < b)
            fz[k] = (ft["Fz_N"][m].mean() - base) if m.any() else np.nan
        p["fz_at_frame"] = fz

    out = args.out or os.path.join(run_dir, "report_pokes_1_%d_zoom.pdf" % args.pokes)
    with PdfPages(out) as pdf:
        cover_page(pdf, R, pokes, accum_us)
        for p, ev in zip(pokes, evs):
            frames, counts = build_frames(ev, p["win_us"][0], accum_us, p["n_frames"])

            # integrated image over the pressed window only
            m = (ev["t"] >= p["t0_us"]) & (ev["t"] <= p["t1_us"])
            integrated = np.zeros((SENSOR_H, SENSOR_W), np.int64)
            if m.any():
                bc = np.bincount(ev["y"][m].astype(np.int64) * SENSOR_W + ev["x"][m],
                                 minlength=SENSOR_H * SENSOR_W)
                integrated = bc.reshape(SENSOR_H, SENSOR_W)

            # key frames: pre-contact, contact onset, peak |Fz|, mid-dwell, retract, post
            n = p["n_frames"]
            k_on = int(np.round((p["t0_us"] - p["win_us"][0]) / accum_us))
            k_off = int(np.round((p["t1_us"] - p["win_us"][0]) / accum_us))
            fz = p["fz_at_frame"]
            inwin = np.arange(n)
            sel = inwin[(inwin >= k_on) & (inwin < min(k_off, n))]
            k_peak = int(sel[np.nanargmin(fz[sel])]) if sel.size else k_on
            keys = [(max(0, k_on - int(round(0.4 / (accum_us / 1e6)))), "pre-contact (quiet)"),
                    (min(n - 1, k_on), "contact onset"),
                    (min(n - 1, k_peak), "peak |Fz|"),
                    (min(n - 1, (k_on + k_off) // 2), "mid-dwell"),
                    (min(n - 1, k_off), "retract onset"),
                    (n - 1, "post-retract")]
            print("Poke %d: frames=%d  peak frame=%d  events=%d"
                  % (p["poke"], n, k_peak, counts.sum()))

            timeline_page(pdf, R, p, counts, accum_us, n)
            key_frames_page(pdf, R, p, frames, counts, accum_us, keys, integrated)
            correlation_page(pdf, R, p, ev, frames, counts, accum_us)
            contact_sheet_page(pdf, R, p, frames, counts, accum_us)
            del frames

        d = pdf.infodict()
        d["Title"] = "E-BTS zoomed poke report (%s)" % R["meta"].get("run_id", "")
        d["Subject"] = "First %d pokes, %d us accumulation" % (args.pokes, accum_us)

    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()
