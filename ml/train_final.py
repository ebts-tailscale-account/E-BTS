#!/usr/bin/env python3
"""Train the deployable force model on ALL 648 indents and save it to disk.

DIFFERENT FROM train_cnn.py, AND THAT DIFFERENCE MATTERS
--------------------------------------------------------
`train_cnn.py` answers "how well does this approach work?" It trains 4 separate
models, each on 3/4 of the data, scores each on the quarter it never saw, and then
THROWS ALL FOUR AWAY. Its output is a number, not a model.

This script answers "give me the thing I can actually use." It trains on all 648 and
saves the result. It deliberately reports NO accuracy score, because any score it
could compute would be on data it trained on, and would be a lie. The honest numbers
come from train_cnn.py (cross-validated) or from a fresh recording (evaluate_holdout.py).

WHAT GETS SAVED  ->  output/force_model.pt
  * five CNNs, one per random seed. Per-seed R^2 measured 0.52-0.77 on 648 samples,
    so a single network is not a stable estimate -- we average five. This was worth
    +0.06 R^2 in the cross-validated test.
  * the target normalisation (mean, sd) used at training time.
  * the linear marker-feature model (coefficients + column names), because the
    CNN+marker average scored better than either alone (0.509 N vs 0.579 / 0.669).
  * the preprocessing constants, so inference cannot silently disagree with training.

⚠ depth is NEVER an input. The model must work from the camera alone.

Usage:
    python3 train_final.py                 # ~3 min, writes output/force_model.pt
    python3 train_final.py --seeds 3       # fewer members, faster
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
import time

import numpy as np

CACHE = _os.path.join(OUTPUT, "pilot_cnn_cache.npz")
FEATS = _os.path.join(OUTPUT, "pilot_20260807_134855_features.csv")
OUT = _os.path.join(OUTPUT, "force_model.pt")

MAG = ["disp_mean_mm", "disp_median_mm", "disp_rms_mm", "disp_max_mm",
       "disp_top10_mm", "disp_sum_mm"]
SHP = ["cx_mm", "cy_mm", "r50_mm", "r25_mm", "aniso", "axis_major_mm",
       "axis_minor_mm", "mean_radial_mm", "mean_tangential_mm", "radial_frac",
       "frac_within_5mm"]


def build_net(cin, p, nn):
    def blk(a, b, ks=3, st=2):
        return nn.Sequential(nn.Conv2d(a, b, ks, st, ks // 2, bias=False),
                             nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(blk(cin, 8, 5), blk(8, 16), blk(16, 32), blk(32, 32))
            self.pool = nn.AdaptiveAvgPool2d((3, 4))   # keeps coarse LOCATION, see train_cnn.py
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(32 * 12, 1))

        def forward(self, x):
            return self.head(self.pool(self.f(x))).squeeze(1)
    return Net()


def make_input(tare, dwell, downsample=2):
    """The exact preprocessing the model was trained on. Inference MUST reuse this."""
    k = downsample
    if k > 1:
        tare, dwell = tare[:, ::k, ::k], dwell[:, ::k, ::k]
    diff = (dwell - tare) / 4.0
    n, Hh, Ww = diff.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    coord = np.stack([xx / Ww * 2 - 1, yy / Hh * 2 - 1])[None].repeat(n, 0)
    return np.concatenate([diff[:, None], coord], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=3e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pandas as pd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE)
    y = d["force"].astype(np.float32)
    X = make_input(d["tare"].astype(np.float32), d["dwell"].astype(np.float32),
                   args.downsample)
    n, C, Hh, Ww = X.shape
    print("training on ALL %d indents  ·  input %d×%d×%d  ·  device %s"
          % (n, C, Hh, Ww, dev))
    print("force %.3f..%.3f N (sd %.3f)\n" % (y.min(), y.max(), y.std()))

    mu, sd = float(y.mean()), float(y.std())
    Xt = torch.tensor(X, device=dev)
    yt = torch.tensor((y - mu) / sd, device=dev)

    states, t0 = [], time.time()
    for s in range(args.seeds):
        torch.manual_seed(s)
        net = build_net(C, args.dropout, nn).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        # No early stopping here: there is no honest validation split left once we
        # train on everything. We use the epoch count the cross-validated run settled
        # on instead -- borrowing that decision is legitimate, peeking at a score is not.
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(n, device=dev)
            for b in range(0, n, args.batch):
                sel = perm[b:b + args.batch]
                opt.zero_grad()
                F.smooth_l1_loss(net(Xt[sel]), yt[sel]).backward()
                opt.step()
            sched.step()
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        print("  seed %d trained (%.0f s elapsed)" % (s, time.time() - t0), flush=True)

    # --- the linear marker-feature model, fitted on the same 648 ---
    lin = None
    try:
        Fdf = pd.read_csv(FEATS).sort_values("point_index").reset_index(drop=True)
        cols = MAG + SHP
        A = np.column_stack([np.ones(len(Fdf)), Fdf[cols].values.astype(float)])
        w, *_ = np.linalg.lstsq(A, Fdf.dwell_Fz_N.values.astype(float), rcond=None)
        lin = {"columns": cols, "coef": w.tolist()}
        print("\nmarker linear model fitted on %d rows, %d features" % (len(Fdf), len(cols)))
    except Exception as e:
        print("\n[warn] marker features unavailable (%s) -- saving the CNN only." % e)

    torch.save({"states": states, "target_mean": mu, "target_sd": sd,
                "in_channels": C, "height": Hh, "width": Ww,
                "downsample": args.downsample, "dropout": args.dropout,
                "linear_marker": lin,
                "trained_on": "pilot_20260807_134855, all 648 indents",
                "preprocess": "diff=(dwell-tare)/4.0 then 2 coord channels; "
                              "tare/dwell are frame MEANS at 320x240 then ::downsample",
                "cv_scores_for_reference": {
                    "cnn_ensemble_rmse_N": 0.579, "cnn_ensemble_r2": 0.657,
                    "marker_linear_rmse_N": 0.669, "marker_linear_r2": 0.542,
                    "blend_rmse_N": 0.509, "blend_r2": 0.734,
                    "depth_only_rmse_N": 0.692, "depth_only_r2": 0.510,
                    "note": "from train_cnn.py, strict row-band CV. NOT from this script."}},
               args.out)
    print("\nsaved %s  (%d CNN members + %s marker model)"
          % (args.out, len(states), "a" if lin else "NO"))
    print("⚠ this script reports no accuracy on purpose — it trained on everything.")
    print("  For an honest number: train_cnn.py (cross-validated), or")
    print("  evaluate_holdout.py on a NEW recording.")


if __name__ == "__main__":
    main()
