#!/usr/bin/env python3
"""Render a batch's aligned data to a single PDF report.

Reads an organized batch folder (output/<batch>/) and writes
output/<batch>/report.pdf:

  Page 1 -- Wittenstein AND Franka force + torque overlaid on one shared time
            axis (both on the common UNIX clock; the Franka stream is shifted by
            the recorded workstation<->tactile offset). Each of the 12 series
            gets its own colour, line style and stroke width. Indentation
            windows are shaded.
  Page 2 -- per-poke Wittenstein Fz force curves (small multiples).

Usage:  python3 visualize.py <batch>            # e.g. output/sweep or just "sweep"
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

REPO = os.path.dirname(os.path.abspath(__file__))

# (label, csv_column, colour, linewidth, linestyle) -- every series distinct.
FORCE_SERIES = [
    ("Wit Fx",  "ft", "Fx_N",      "#1f77b4", 1.3, "-"),
    ("Wit Fy",  "ft", "Fy_N",      "#ff7f0e", 1.3, "-"),
    ("Wit Fz",  "ft", "Fz_N",      "#2ca02c", 1.9, "-"),
    ("Fr Fx",   "fr", "Fx_ext_N",  "#d62728", 1.1, "--"),
    ("Fr Fy",   "fr", "Fy_ext_N",  "#9467bd", 1.1, ":"),
    ("Fr Fz",   "fr", "Fz_ext_N",  "#8c564b", 1.6, "-."),
]
# Wittenstein torque is mNm; Franka torque is Nm -> scale to mNm so they share an axis.
TORQUE_SERIES = [
    ("Wit Mx",  "ft", "Mx_mNm",    "#1f77b4", 1.3, "-",  1.0),
    ("Wit My",  "ft", "My_mNm",    "#ff7f0e", 1.3, "-",  1.0),
    ("Wit Mz",  "ft", "Mz_mNm",    "#2ca02c", 1.9, "-",  1.0),
    ("Fr Tx",   "fr", "Tx_ext_Nm", "#d62728", 1.1, "--", 1000.0),
    ("Fr Ty",   "fr", "Ty_ext_Nm", "#9467bd", 1.1, ":",  1000.0),
    ("Fr Tz",   "fr", "Tz_ext_Nm", "#8c564b", 1.6, "-.", 1000.0),
]

BASELINE_SECONDS = 5.0   # no-load lead-in used as the zero offset
EVENT_BIN_S      = 0.05  # 50 ms bins for the event-polarity histogram


def resolve_batch(arg):
    if os.path.isdir(arg):
        return arg
    cand = os.path.join(REPO, "output", arg)
    if os.path.isdir(cand):
        return cand
    sys.exit("Batch folder not found: %s" % arg)


def baseline_zero(t, values, before):
    pre = t < before
    base = values[pre].mean() if pre.any() else 0.0
    return values - base


def latest_batch():
    out = os.path.join(REPO, "output")
    if not os.path.isdir(out):
        return None
    dirs = [os.path.join(out, d) for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def event_rate(batch_dir, bin_s=EVENT_BIN_S):
    """Positive/negative polarity event counts per time bin over the whole run.

    Streams camera.raw once and caches the result to event_rate.csv (the raw
    pass is the slow part; cached after). t is device time = seconds since
    recording start, which matches the force panels' x-axis.
    """
    cache = os.path.join(batch_dir, "event_rate.csv")
    if os.path.exists(cache):
        return np.atleast_1d(np.genfromtxt(cache, delimiter=",", names=True))
    raw = os.path.join(batch_dir, "camera.raw")
    if not os.path.exists(raw):
        return None
    try:
        from metavision_core.event_io import EventsIterator
    except ImportError:
        return None

    print("Binning event polarity from camera.raw (one streaming pass; cached to event_rate.csv)...")
    bin_us = int(bin_s * 1e6)
    pos, neg = {}, {}
    for evs in EventsIterator(input_path=raw, delta_t=100000):
        if evs.size == 0:
            continue
        b = (evs["t"] // bin_us).astype(np.int64)
        p = evs["p"]
        bp, cp = np.unique(b[p == 1], return_counts=True)
        bn, cn = np.unique(b[p == 0], return_counts=True)
        for i, c in zip(bp, cp):
            pos[int(i)] = pos.get(int(i), 0) + int(c)
        for i, c in zip(bn, cn):
            neg[int(i)] = neg.get(int(i), 0) + int(c)
    if not pos and not neg:
        return None
    maxb = max(list(pos) + list(neg))
    with open(cache, "w") as f:
        f.write("t_center_s,n_pos,n_neg\n")
        for i in range(maxb + 1):
            f.write("%.4f,%d,%d\n" % ((i + 0.5) * bin_s, pos.get(i, 0), neg.get(i, 0)))
    return np.atleast_1d(np.genfromtxt(cache, delimiter=",", names=True))


def overview_figure(name, offset, data, windows, step, events=None, xlim=None):
    """Force (top) + torque (mid) + event polarity (bottom), shared time axis. xlim zooms."""
    fig, (ax_f, ax_t, ax_e) = plt.subplots(3, 1, figsize=(11.7, 9.6), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 3, 2]})
    tag = "" if xlim is None else "  [zoom %.0f-%.0f s]" % xlim
    fig.suptitle("E-BTS run '%s'%s  --  Wittenstein + Franka F/T + event polarity, common UNIX clock "
                 "(Franka offset %.1f ms, baseline-zeroed)" % (name, tag, -offset * 1e3), fontsize=11)

    def shade(ax):
        for x0, x1, k in windows:
            ax.axvspan(x0, x1, color="0.85", alpha=0.35, lw=0)

    def draw(ax, series):
        for spec in series:
            label, src, col, colour, lw, ls = spec[:6]
            scale = spec[6] if len(spec) > 6 else 1.0
            arr, x = data[src]
            if col not in arr.dtype.names:
                continue
            y = baseline_zero(x, arr[col].astype(float), BASELINE_SECONDS) * scale
            dec = step if src == "ft" else 1
            ax.plot(x[::dec], y[::dec], color=colour, lw=lw, ls=ls, label=label,
                    rasterized=(src == "ft"))
        shade(ax)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=6, fontsize=8, loc="upper right")

    draw(ax_f, FORCE_SERIES)
    ax_f.set_ylabel("Force (N)")
    for x0, x1, k in windows:  # poke numbers along the top of the force axis
        ax_f.annotate(str(k), (0.5 * (x0 + x1), 1.0), xycoords=("data", "axes fraction"),
                      ha="center", va="bottom", fontsize=7, color="0.4")

    draw(ax_t, TORQUE_SERIES)
    ax_t.set_ylabel("Torque (mNm)")

    # Event polarity: ON (positive) events up, OFF (negative) events down.
    if events is not None and len(events) > 0:
        te, npos, nneg = events["t_center_s"], events["n_pos"], events["n_neg"]
        ax_e.fill_between(te, 0, npos, step="mid", color="#2ca02c", alpha=0.6, lw=0, label="ON (+) up")
        ax_e.fill_between(te, 0, -nneg, step="mid", color="#9467bd", alpha=0.6, lw=0, label="OFF (-) down")
        ax_e.axhline(0, color="0.5", lw=0.6)
        ax_e.legend(ncol=2, fontsize=8, loc="upper right")
    else:
        ax_e.text(0.5, 0.5, "no event data (camera.raw missing?)", ha="center", va="center",
                  transform=ax_e.transAxes, color="0.5")
    shade(ax_e)
    ax_e.grid(True, alpha=0.25)
    ax_e.set_ylabel("events / %d ms bin\n(ON up, OFF down)" % int(EVENT_BIN_S * 1000))
    ax_e.set_xlabel("time (s since recording start)  --  shaded = indentation windows")

    if xlim is not None:
        ax_f.set_xlim(xlim)  # shared x -> all three panels zoom together
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def perpoke_figure(name, batch_dir):
    poke_files = sorted(f for f in os.listdir(os.path.join(batch_dir, "pokes")) if f.endswith("_ft.csv"))
    if not poke_files:
        return None
    n = len(poke_files)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.7, 8.3), squeeze=False)
    fig.suptitle("Per-indentation Wittenstein Fz (N, zeroed)  --  '%s'" % name, fontsize=11)
    for i, pf in enumerate(poke_files):
        ax = axes[i // ncols][i % ncols]
        d = np.genfromtxt(os.path.join(batch_dir, "pokes", pf), delimiter=",", names=True)
        ax.plot(d["t_rel_s"], d["Fz_N"], color="#2ca02c", lw=1.0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(pf.replace("_ft.csv", ""), fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", nargs="?", default=None, help="batch under output/ (default: most recent)")
    ap.add_argument("--zoom", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="also write a separate zoomed PDF spanning pokes FIRST..LAST")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="seconds of context before/after the zoom range (default 5)")
    args = ap.parse_args()

    batch_dir = resolve_batch(args.batch) if args.batch else latest_batch()
    if not batch_dir:
        sys.exit("usage: python3 visualize.py <batch> [--zoom FIRST LAST] [--margin S]")
    if not args.batch:
        print("No batch given; using most recent: %s" % os.path.basename(batch_dir))
    name = os.path.basename(batch_dir.rstrip("/"))

    ft = np.genfromtxt(os.path.join(batch_dir, "ft.csv"), delimiter=",", names=True)
    fr = np.genfromtxt(os.path.join(batch_dir, "franka.csv"), delimiter=",", names=True)
    meta = json.load(open(os.path.join(batch_dir, "metadata.json")))
    offset = meta.get("tactile_minus_workstation_offset_s") or 0.0

    # Common clock: everything relative to the F/T recording start.
    t0 = ft["unix_time_s"][0]
    x_ft = ft["unix_time_s"] - t0
    x_fr = (fr["unix_time_s"] - offset) - t0  # Franka onto the workstation clock
    data = {"ft": (ft, x_ft), "fr": (fr, x_fr)}

    windows = []
    summ_path = os.path.join(batch_dir, "pokes", "pokes_summary.csv")
    if os.path.exists(summ_path):
        s = np.atleast_1d(np.genfromtxt(summ_path, delimiter=",", names=True))
        for row in s:
            windows.append((float(row["t_start_ws"]) - t0, float(row["t_end_ws"]) - t0, int(row["poke"])))
    step = max(1, len(x_ft) // 12000)  # decimate 1 kHz F/T for a light vector PDF
    events = event_rate(batch_dir)     # +/- polarity histogram (streams/caches camera.raw)

    # ---- main report (overview + per-poke) ----
    pdf_path = os.path.join(batch_dir, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        fig = overview_figure(name, offset, data, windows, step, events=events)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        pp = perpoke_figure(name, batch_dir)
        if pp is not None:
            pdf.savefig(pp, dpi=150)
            plt.close(pp)
    print("Wrote %s" % pdf_path)

    # ---- optional zoomed PDF (same styling, restricted time window) ----
    if args.zoom:
        p1, p2 = args.zoom
        wmap = {k: (x0, x1) for x0, x1, k in windows}
        if p1 not in wmap or p2 not in wmap:
            sys.exit("zoom pokes %d..%d not found (available: %s)" % (p1, p2, sorted(wmap)))
        lo = min(wmap[p1][0], wmap[p2][0]) - args.margin
        hi = max(wmap[p1][1], wmap[p2][1]) + args.margin
        zpath = os.path.join(batch_dir, "report_zoom_poke%d-%d.pdf" % (p1, p2))
        with PdfPages(zpath) as pdf:
            fig = overview_figure(name, offset, data, windows, step, events=events, xlim=(lo, hi))
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
        print("Wrote %s" % zpath)


if __name__ == "__main__":
    main()
