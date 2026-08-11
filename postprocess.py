#!/usr/bin/env python3
"""E-BTS post-processing: turn one synchronized run into per-indentation data.

Takes the four files master.py produced for a run and:
  1. Aligns clocks   -- shifts the Franka (tactile-clock) rows onto the
                        workstation clock using the offset in the metadata.
  2. Zeros baselines -- subtracts the no-load lead-in mean from the F/T and the
                        Franka external-wrench channels.
  3. Segments        -- finds the 16 indentations from the Franka ee_z dips
                        (each ~2 s dwell at ~2 mm below the surface).
  4. Emits per poke  -- a summary row (contact xy, peak/mean force, timing) plus
                        the sliced F/T force curve, and -- unless --no-events --
                        the event-camera slice pulled from the .raw for that
                        indentation window (one streaming pass).

All output lands in recordings/<run>_pokes/ .

Usage:
  python3 postprocess.py <run_name> [--recordings DIR] [--no-events] [--press-margin M]
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

PRESS_MARGIN_M   = 0.0015  # ee_z within this of its minimum counts as "pressed"
MIN_DWELL_S      = 1.0     # a real indentation dwells >= this (rejects transit dips)
HOVER_M          = 0.005   # the master's hover offset above the surface
BASELINE_SECONDS = 5.0     # mean of the first N s (no-load lead-in) = fallback zero
# Per-poke tare (see tare_baseline): the sweep holds out of contact at hover for ~1 s
# before each dip, so we zero each indentation on its OWN window -> drift cancels.
TARE_WINDOW_S    = 0.6     # length of the no-load window used as that poke's zero
TARE_GUARD_S     = 0.20    # end the window this far before contact (avoid touchdown)
TARE_MIN_SAMPLES = 50      # need at least this many F/T samples, else fall back


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def load_phase_columns(path):
    """(phase[str], point_index[int]) arrays aligned with load_csv's rows.

    np.genfromtxt(names=True) turns the text 'phase' column into NaN, so read it
    separately. The sweep TAGS every sample with its phase, which is far more
    reliable than inferring contact from a depth threshold: the servo's impedance
    controller yields under load and only reaches ~1.3-1.9 mm of a commanded 2 mm,
    so a fixed press-margin silently drops indentations (it found 69 of 99 once).
    """
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
    if "phase" not in header or "point_index" not in header:
        return None, None
    ip, ii = header.index("phase"), header.index("point_index")
    phases, idxs = [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(ip, ii):
                continue
            phases.append(parts[ip])
            try:
                idxs.append(int(float(parts[ii])))
            except ValueError:
                idxs.append(-1)
    return np.array(phases), np.array(idxs, dtype=int)


def contiguous_runs(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, breaks + 1)]


def snake_order(n_rows=4, n_cols=4):
    order = []
    for c in range(n_cols):
        rows = list(range(n_rows))
        if c % 2 == 1:
            rows = rows[::-1]
        for r in rows:
            order.append((c, r))
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--recordings", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings"))
    ap.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    ap.add_argument("--no-events", action="store_true", help="skip the .raw event slicing")
    ap.add_argument("--events-npy", action="store_true",
                    help="write per-poke events as compact .npy instead of CSV (smaller, faster to load)")
    ap.add_argument("--press-margin", type=float, default=PRESS_MARGIN_M)
    args = ap.parse_args()

    rec = args.recordings
    run = args.run
    batch = os.path.join(args.output, run)  # the organized per-batch folder
    os.makedirs(os.path.join(batch, "pokes"), exist_ok=True)

    # master.py now gives every run its OWN folder, recordings/<run>_<stamp>/, with
    # canonical filenames inside. Accept: a run-folder name, a path to one, or (for
    # older runs) the flat recordings/<run>_*.<ext> layout.
    run_dir = run if os.path.isdir(run) else os.path.join(rec, run)
    if os.path.isdir(run_dir):
        run = os.path.basename(os.path.normpath(run_dir))
        batch = os.path.join(args.output, run)
        os.makedirs(os.path.join(batch, "pokes"), exist_ok=True)
        print("Reading run folder: %s" % run_dir)
    else:
        run_dir = None

    # Inputs come from the run folder / the flat raw dump (recordings/) OR, if this
    # batch was already organized, from output/<run>/ itself.
    def resolve(rec_glob, batch_name):
        if run_dir:                                   # per-run folder, canonical names
            pf = os.path.join(run_dir, batch_name)
            if os.path.exists(pf):
                return pf, True
        p = find_one(os.path.join(rec, rec_glob))      # legacy flat layout
        if p and os.path.exists(p):
            return p, True  # from recordings -> will be moved into the batch
        pb = os.path.join(batch, batch_name)
        return (pb, False) if os.path.exists(pb) else (None, False)

    raw_path, raw_from_rec       = resolve(run + "_*.raw", "camera.raw")
    bias_path, bias_from_rec     = resolve(run + "_*.bias", "camera.bias")  # camera bias sidecar (optional)
    roi_path, roi_from_rec       = resolve(run + "_*.roi", "camera.roi")    # hardware ROI record (optional)
    ft_path, ft_from_rec         = resolve(run + "_*_ft.csv", "ft.csv")
    franka_path, franka_from_rec = resolve(run + "_franka.csv", "franka.csv")
    meta_path, meta_from_rec     = resolve(run + "_metadata.json", "metadata.json")
    for label, p in [("F/T", ft_path), ("Franka", franka_path), ("metadata", meta_path)]:
        if not p:
            sys.exit("Missing %s file for run '%s' (looked in %s and %s)" % (label, run, rec, batch))

    meta = json.load(open(meta_path))
    offset = meta.get("tactile_minus_workstation_offset_s") or 0.0
    print("Run '%s'  |  tactile->workstation offset %.1f ms" % (run, -offset * 1e3))

    ft = load_csv(ft_path)
    fr = load_csv(franka_path)

    # --- 1. clock align: Franka rows onto the workstation clock ---
    t_fr = fr["unix_time_s"] - offset
    t_ft = ft["unix_time_s"]
    cam_t0 = t_ft[0]  # camera device t=0 ~= F/T recording start (both fire from begin_recording)

    # --- 3. segment the indentations ---
    # PREFERRED: use the phase tag the sweep logs -- one window per 'dwell' point.
    # Exact and threshold-free. Fall back to a depth threshold for older logs.
    eez = fr["ee_z"]
    phase_arr, pidx_arr = load_phase_columns(franka_path)
    windows, window_pids = [], []
    if phase_arr is not None and phase_arr.size == eez.size and (phase_arr == "dwell").any():
        dwell = phase_arr == "dwell"
        for pid in sorted(set(pidx_arr[dwell].tolist())):
            sel = dwell & (pidx_arr == pid)
            if not sel.any():
                continue
            ts = t_fr[sel]
            if ts.max() - ts.min() < MIN_DWELL_S * 0.5:
                continue
            windows.append((ts.min(), ts.max()))
            window_pids.append(pid)
        pressed = dwell
        print("Detected %d indentation windows (from the logged 'dwell' phase)."
              % len(windows))
    else:
        if "surface_z" in (fr.dtype.names or ()):
            pressed = eez < (fr["surface_z"] - args.press_margin)
        else:
            pressed = eez < (eez.min() + args.press_margin)
        for a, b in contiguous_runs(pressed):
            if t_fr[b] - t_fr[a] >= MIN_DWELL_S:
                windows.append((t_fr[a], t_fr[b]))
                window_pids.append(None)
        print("Detected %d indentation windows (depth threshold, margin %.1f mm)."
              % (len(windows), args.press_margin * 1e3))
    if not windows:
        sys.exit("No indentations found -- check --press-margin.")

    # Fallback surface estimate (median ee_z while NOT pressed). New logs carry the
    # MAPPED surface height per sample, so each poke uses its own real reference.
    surface_z = np.median(eez[~pressed]) - HOVER_M
    has_surface_z = "surface_z" in (fr.dtype.names or ())

    # --- 2. zero baselines ---
    # GLOBAL fallback: the no-load lead-in at the very start of the recording.
    base_end = t_ft[0] + BASELINE_SECONDS
    ft_pre = t_ft < base_end
    fr_pre = t_fr < base_end
    ft_cols = ["Fx_N", "Fy_N", "Fz_N", "Mx_mNm", "My_mNm", "Mz_mNm"]
    ft_base = {c: (ft[c][ft_pre].mean() if ft_pre.any() else 0.0) for c in ft_cols}
    fr_wrench = ["Fx_ext_N", "Fy_ext_N", "Fz_ext_N", "Tx_ext_Nm", "Ty_ext_Nm", "Tz_ext_Nm"]
    fr_base = {c: (fr[c][fr_pre].mean() if fr_pre.any() else 0.0) for c in fr_wrench}

    def tare_baseline(t_contact):
        """PER-POKE tare: mean F/T over the no-load hover window just before contact.

        The sweep holds still out of contact for TARE_S at hover height immediately
        before each dip, so this window is a fresh zero for THIS indentation. Using it
        instead of one global baseline cancels the Wittenstein's Fz drift over a long
        run. Falls back to the global baseline if the window is too sparse.
        """
        lo = t_contact - TARE_GUARD_S - TARE_WINDOW_S
        hi = t_contact - TARE_GUARD_S
        w = (t_ft >= lo) & (t_ft <= hi)
        if w.sum() < TARE_MIN_SAMPLES:
            return ft_base, False
        return {c: ft[c][w].mean() for c in ft_cols}, True

    outdir = os.path.join(batch, "pokes")
    # Prefer the real per-sample col/row logged by the servo sweep (works for any
    # grid size); fall back to the assumed snake order only for old 16-point CSVs.
    fr_names = fr.dtype.names or ()
    has_grid_cols = "col" in fr_names and "row" in fr_names
    order = snake_order() if len(windows) == 16 else [(None, None)] * len(windows)

    # --- per-poke summary + F/T force-curve slices ---
    summary = []
    n_tared = 0
    for k, (t0, t1) in enumerate(windows, 1):
        m_fr = (t_fr >= t0) & (t_fr <= t1)

        if has_grid_cols:
            cvals, rvals = fr["col"][m_fr], fr["row"][m_fr]
            col = int(np.median(cvals)) if cvals.size else None
            row = int(np.median(rvals)) if rvals.size else None
            col = None if col is None or col < 0 else col   # -1 = unset sentinel
            row = None if row is None or row < 0 else row
        else:
            col, row = order[k - 1]

        m_ft = (t_ft >= t0) & (t_ft <= t1)
        ee_x, ee_y = fr["ee_x"][m_fr].mean(), fr["ee_y"][m_fr].mean()
        ee_z_min = eez[m_fr].min()

        # PER-POKE TARE: zero this indentation on its own pre-contact hover window
        poke_base, tared = tare_baseline(t0)
        n_tared += int(tared)

        fz_ft = ft["Fz_N"][m_ft] - poke_base["Fz_N"]
        peak_fz_ft = fz_ft[np.argmax(np.abs(fz_ft))] if fz_ft.size else float("nan")
        mean_fz_ft = fz_ft.mean() if fz_ft.size else float("nan")
        res_ft = np.sqrt((ft["Fx_N"][m_ft] - poke_base["Fx_N"]) ** 2 +
                         (ft["Fy_N"][m_ft] - poke_base["Fy_N"]) ** 2 + fz_ft ** 2)
        peak_res_ft = res_ft.max() if res_ft.size else float("nan")
        fz_ext = fr["Fz_ext_N"][m_fr] - fr_base["Fz_ext_N"]
        peak_fz_ext = fz_ext[np.argmax(np.abs(fz_ext))] if fz_ext.size else float("nan")

        # true achieved depth: prefer this point's MAPPED surface height
        if has_surface_z:
            sz = np.nanmedian(fr["surface_z"][m_fr])
            surf_ref = sz if np.isfinite(sz) else surface_z
        else:
            surf_ref = surface_z

        summary.append({
            "poke": k, "point_id": window_pids[k - 1], "col": col, "row": row,
            "t_start_ws": "%.6f" % t0, "t_end_ws": "%.6f" % t1, "dur_s": "%.3f" % (t1 - t0),
            "ee_x": "%.5f" % ee_x, "ee_y": "%.5f" % ee_y,
            "indent_mm": "%.2f" % ((surf_ref - ee_z_min) * 1e3),
            "surface_z": "%.5f" % surf_ref,
            "peak_Fz_ft_N": "%.3f" % peak_fz_ft, "mean_Fz_ft_N": "%.3f" % mean_fz_ft,
            "peak_F_res_N": "%.3f" % peak_res_ft,
            "peak_Fz_ext_N": "%.3f" % peak_fz_ext,
            "cam_t0_us": int((t0 - cam_t0) * 1e6), "cam_t1_us": int((t1 - cam_t0) * 1e6),
            "n_events": 0,
        })

        # tared F/T force curve for this indentation (zeroed on ITS pre-contact window)
        with open(os.path.join(outdir, "poke%02d_ft.csv" % k), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_rel_s", "unix_time_s"] + ft_cols)
            for i in np.where(m_ft)[0]:
                w.writerow(["%.6f" % (t_ft[i] - t0), "%.6f" % t_ft[i]] +
                           ["%.4f" % (ft[c][i] - poke_base[c]) for c in ft_cols])

    print("Per-poke tare: %d/%d pokes zeroed on their own pre-contact window%s."
          % (n_tared, len(windows),
             "" if n_tared == len(windows) else " (rest used the global baseline)"))

    # --- 4. event slices from the .raw (one streaming pass) ---
    if not args.no_events and raw_path:
        try:
            from metavision_core.event_io import EventsIterator
        except ImportError:
            print("metavision_core not importable -- skipping event slicing.")
            EventsIterator = None
        if EventsIterator:
            print("Slicing events from %s (one streaming pass over the .raw)..." % os.path.basename(raw_path))
            dev = [(s["cam_t0_us"], s["cam_t1_us"]) for s in summary]
            last_end = max(e for _, e in dev)
            buckets = [[] for _ in windows]  # lists of event sub-arrays per poke
            for evs in EventsIterator(input_path=raw_path, delta_t=100000):
                if evs.size == 0:
                    continue
                if evs["t"][0] > last_end:
                    break  # past every window; stop early
                for k, (d0, d1) in enumerate(dev):
                    sel = (evs["t"] >= d0) & (evs["t"] < d1)
                    if sel.any():
                        buckets[k].append(evs[sel].copy())  # vectorized -- no per-event loop

            for k in range(len(windows)):
                ev = np.concatenate(buckets[k]) if buckets[k] else None
                n = 0 if ev is None else ev.size
                summary[k]["n_events"] = int(n)
                base = os.path.join(outdir, "poke%02d_events" % (k + 1))
                if args.events_npy:
                    np.save(base + ".npy", ev if ev is not None else np.empty(0))
                else:
                    with open(base + ".csv", "w", newline="") as f:
                        f.write("x,y,polarity,t_us\n")
                        if n:
                            np.savetxt(f, np.column_stack([ev["x"], ev["y"], ev["p"], ev["t"]]),
                                       fmt="%d", delimiter=",")
            print("  event slicing done: %d events total across %d pokes." %
                  (sum(s["n_events"] for s in summary), len(windows)))
    elif not raw_path:
        print("No .raw found -- skipping event slicing (numeric streams still processed).")

    # --- write the summary ---
    summ_path = os.path.join(outdir, "pokes_summary.csv")
    with open(summ_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # --- organize the raw-dump inputs into the batch folder (clean names) ---
    for src, from_rec, dst in [(raw_path, raw_from_rec, "camera.raw"),
                               (bias_path, bias_from_rec, "camera.bias"),
                               (roi_path, roi_from_rec, "camera.roi"),
                               (ft_path, ft_from_rec, "ft.csv"),
                               (franka_path, franka_from_rec, "franka.csv"),
                               (meta_path, meta_from_rec, "metadata.json")]:
        if src and from_rec and os.path.exists(src):
            shutil.move(src, os.path.join(batch, dst))

    print("\nPer-poke summary (%s):" % summ_path)
    print("  poke (col,row)   ee_x    ee_y   indent  peakFz  peak|F|  peakFz_ext  n_events")
    for s in summary:
        print("   %2d  (%s,%s)  %7s %7s  %5smm  %6s  %6s  %8s N  %8s" % (
            s["poke"], s["col"], s["row"], s["ee_x"], s["ee_y"], s["indent_mm"],
            s["peak_Fz_ft_N"], s["peak_F_res_N"], s["peak_Fz_ext_N"], s["n_events"]))
    print("\nDone. Batch organized in %s/  (then: python3 visualize.py %s)" % (batch, run))


if __name__ == "__main__":
    main()
