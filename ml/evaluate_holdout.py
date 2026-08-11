#!/usr/bin/env python3
"""Run the saved force model on a FRESH recording and compare it to the load cell.

THIS IS THE ONLY FULLY HONEST TEST WE CAN RUN.
Cross-validation is careful, but it still scores a model on data collected in the
same session, on the same silicone, in the same sitting. A new recording shares none
of that: nothing about it could have leaked into training. Whatever number comes out
of here is the one to believe.

WHAT IT DOES
  1. reads a run folder recorded the normal way (master_campaign.py / master_midpoint.py)
  2. rebuilds the tare-mean and dwell-mean frames per indent, exactly as training did
  3. runs the CNN ensemble, the marker linear model, and their average
  4. compares all three against the Wittenstein, per indent, and writes a CSV + plot

⚠ WHAT TO EXPECT, so a bad number is not a surprise:
  * Interior of the elastomer (rows 0-8 of the pilot grid): cross-validated error was
    ~0.33 N. Anything near that is the model behaving.
  * Near the strapped edge: cross-validated error was ~0.84 N and EVERY model failed
    there, robot-only included. A bad number there is the PREDICTED weakness, not a bug.
  * Outside 1.5-4.0 mm depth or 0.24-5.95 N the model has no experience at all.
  * One indent proves nothing -- the seed-to-seed spread alone is +-0.09 R^2. Judge on
    10-15 presses, not one.

Usage:
    python3 evaluate_holdout.py recordings/<new_run>
    python3 evaluate_holdout.py recordings/<new_run> --model output/force_model.pt
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_run_frames(run, downsample, nt=10, nd=10):
    """Rebuild (tare_mean, dwell_mean) per indent from a run folder's HDF5 frames.

    Requires build_dataset.py to have been run on the recording first, which is what
    produces the frame file. We deliberately reuse that rather than re-decoding the
    .raw here, so training and evaluation share one extraction path.
    """
    import h5py
    import cv2
    h5 = os.path.join(OUTPUT, os.path.basename(run.rstrip("/")) + "_frames.h5")
    if not os.path.exists(h5):
        sys.exit("[ERROR] %s not found.\n"
                 "        Build it first:  python3 build_dataset.py %s" % (h5, run))
    f = h5py.File(h5, "r")
    L = f["indents"][:]
    FR, PH = f["frames"], f["frame_phase"][:]
    T, D = [], []
    for r in L:
        o, cnt = int(r["frame_offset"]), int(r["n_frames"])
        p = PH[o:o + cnt]
        ti = np.where(p == 0)[0]
        di = np.where(p == 2)[0]
        ti = ti[np.linspace(0, len(ti) - 1, min(nt, len(ti))).astype(int)]
        di = di[np.linspace(len(di) * 0.25, len(di) - 1, min(nd, len(di))).astype(int)]
        tm = np.stack([FR[o + int(i)] for i in ti]).astype(np.float32).mean(0)
        dm = np.stack([FR[o + int(i)] for i in di]).astype(np.float32).mean(0)
        T.append(cv2.resize(tm, (320, 240), interpolation=cv2.INTER_AREA))
        D.append(cv2.resize(dm, (320, 240), interpolation=cv2.INTER_AREA))
    return np.array(T, np.float32), np.array(D, np.float32), L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default=_os.path.join(OUTPUT, "force_model.pt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from train_final import build_net, make_input, MAG, SHP

    # weights_only=False is deliberate: we save python objects (the marker model
    # coefficients and metadata) alongside the tensors. The file is ours.
    try:
        ck = torch.load(args.model, map_location="cpu", weights_only=False)
    except TypeError:                      # torch < 2.0 has no weights_only kwarg
        ck = torch.load(args.model, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("model    : %s" % args.model)
    print("  trained on: %s" % ck.get("trained_on"))
    print("  members   : %d CNN + %s marker model"
          % (len(ck["states"]), "1" if ck.get("linear_marker") else "no"))
    ref = ck.get("cv_scores_for_reference", {})
    if ref:
        print("  cross-validated reference: CNN %.3f N, blend %.3f N, depth-only %.3f N"
              % (ref.get("cnn_ensemble_rmse_N", np.nan), ref.get("blend_rmse_N", np.nan),
                 ref.get("depth_only_rmse_N", np.nan)))

    tare, dwell, L = load_run_frames(args.run, ck["downsample"])
    X = make_input(tare, dwell, ck["downsample"])
    y = L["dwell_Fz_N"].astype(float)

    # ⚠ A single NaN label used to propagate into EVERY metric and print a table of
    # nan. NaNs happen for real: the Wittenstein dropped out for 1.08 s during
    # hard_20260807_223744 and indent 0's whole tare window fell inside the gap, so
    # it had no baseline. Drop those rows loudly and score the rest.
    bad = ~np.isfinite(y)
    if bad.any():
        print("\n[WARN] %d of %d indents have no usable force label and are EXCLUDED:"
              % (bad.sum(), len(y)))
        for i in np.where(bad)[0]:
            print("         point_index %d (row %d, col %d) -- check ft.csv for a dropout "
                  "over its tare window" % (L["point_index"][i], L["row"][i], L["col"][i]))
    keep = ~bad
    print("\nrun      : %s   %d indents scored, measured force %.3f..%.3f N"
          % (args.run, int(keep.sum()), y[keep].min(), y[keep].max()))

    # --- CNN ensemble ---
    preds = []
    with torch.no_grad():
        for st in ck["states"]:
            net = build_net(ck["in_channels"], ck["dropout"], nn)
            net.load_state_dict(st)
            net.to(dev).eval()
            p = net(torch.tensor(X, device=dev)).cpu().numpy()
            preds.append(p * ck["target_sd"] + ck["target_mean"])
    cnn = np.mean(preds, 0)
    spread = np.std(preds, 0)          # disagreement between members = a confidence proxy

    # --- marker linear model, if features exist for this run ---
    mk = None
    fcsv = os.path.join(OUTPUT, os.path.basename(args.run.rstrip("/")) + "_features.csv")
    if ck.get("linear_marker") and os.path.exists(fcsv):
        import pandas as pd
        Fdf = pd.read_csv(fcsv).sort_values("point_index").reset_index(drop=True)
        if len(Fdf) == len(y):
            A = np.column_stack([np.ones(len(Fdf)),
                                 Fdf[ck["linear_marker"]["columns"]].values.astype(float)])
            mk = A @ np.array(ck["linear_marker"]["coef"])
        else:
            print("[warn] %s has %d rows, run has %d -- skipping the marker model"
                  % (fcsv, len(Fdf), len(y)))
    elif ck.get("linear_marker"):
        print("[note] no %s -- marker model skipped. Generate it with extract_features.py"
              % os.path.basename(fcsv))

    def sc(p):
        e = (p - y)[keep]
        return (float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean()),
                float(1 - (e ** 2).sum() / max(((y[keep] - y[keep].mean()) ** 2).sum(), 1e-9)),
                float(e.mean()))

    print("\n  %-28s %8s %8s %8s %8s" % ("model", "RMSE", "MAE", "R2", "bias"))
    print("  " + "-" * 62)
    rows = [("CNN ensemble", cnn)]
    if mk is not None:
        rows += [("marker linear", mk), ("blend (CNN + marker)/2", (cnn + mk) / 2)]
    for nm, p in rows:
        r, m, q, b = sc(p)
        print("  %-28s %7.4f %8.4f %8.3f %+8.4f" % (nm, r, m, q, b))
    print("  " + "-" * 62)
    print("  %-28s %7.4f  (predicting the mean would give this)"
          % ("measured force sd", y[keep].std()))

    print("\n  per-indent detail")
    print("  %4s %6s %6s %9s %9s %9s %8s" %
          ("idx", "row", "col", "measured", "CNN", "blend", "±ens"))
    best = (cnn + mk) / 2 if mk is not None else cnn
    for i, r in enumerate(L):
        print("  %4d %6d %6d %9.3f %9.3f %9.3f %8.3f"
              % (r["point_index"], r["row"], r["col"], y[i], cnn[i], best[i], spread[i]))

    out = args.out or os.path.join(
        OUTPUT, os.path.basename(args.run.rstrip("/")) + "_holdout_eval.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["point_index", "row", "col", "depth_mm", "achieved_mm",
                    "measured_Fz_N", "pred_cnn_N", "pred_marker_N", "pred_blend_N",
                    "ensemble_spread_N"])
        for i, r in enumerate(L):
            w.writerow([r["point_index"], r["row"], r["col"], r["depth_mm"],
                        r["achieved_mm"], "%.4f" % y[i], "%.4f" % cnn[i],
                        "" if mk is None else "%.4f" % mk[i],
                        "%.4f" % best[i], "%.4f" % spread[i]])
    print("\nwrote %s" % out)

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, a = plt.subplots(figsize=(5.4, 5.2))
            lim = [0, max(y[keep].max(), best[keep].max()) * 1.08]
            a.plot(lim, lim, color="#c2521f", lw=1.5, zorder=1, label="perfect")
            a.errorbar(y[keep], best[keep], yerr=spread[keep], fmt="o", ms=7, color="#2a78d6",
                       ecolor="#9dc0ec", elinewidth=1.2, capsize=0, zorder=3,
                       label="prediction ±ensemble spread")
            a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
            a.set_xlabel("measured |Fz| (N)  ·  Wittenstein")
            a.set_ylabel("predicted |Fz| (N)  ·  camera")
            a.set_title("%s\nRMSE %.3f N on %d fresh indents"
                        % (os.path.basename(args.run.rstrip("/")), sc(best)[0], int(keep.sum())),
                        fontsize=10, loc="left")
            a.grid(color="#e6e4dd", lw=.6); a.set_axisbelow(True)
            a.spines[["top", "right"]].set_visible(False)
            a.legend(fontsize=8, frameon=False, loc="upper left")
            png = out.replace(".csv", ".png")
            fig.tight_layout(); fig.savefig(png, dpi=160)
            print("wrote %s" % png)
        except Exception as e:
            print("[note] no plot (%s)" % e)


if __name__ == "__main__":
    main()
