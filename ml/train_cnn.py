#!/usr/bin/env python3
"""Small CNN: 40 ms event frames -> normal force. Grouped CV by grid location.

WHY A CNN AT ALL (HANDOFF §15.2f)
---------------------------------
Hand-built marker features that only summarise the SIZE of the deformation were
measured to fail: mean displacement alone gives R^2 = -0.012, worse than predicting
the average. The reason is physical -- F = k(x,y) * deformation, and k varies 9.4x
across this elastomer, so a soft spot deforms a lot for little force while a stiff
spot barely moves under a lot. Magnitude cannot separate those. The information that
would separate them is in the SHAPE and LOCATION of the field, which a convnet can
read directly off the frame.

THREE DESIGN DECISIONS THAT MATTER
----------------------------------
1. INPUT IS THE DIFFERENCE IMAGE (dwell - tare), not the raw frame. The static marker
   lattice is a nuisance variable; differencing removes it and leaves only what moved.
   Each indent has its own tare, held out of contact 1 s before its own dip, so this
   is exactly the reference the sensor would have at inference.

2. ⚠ NO GLOBAL AVERAGE POOLING, AND COORDINATE CHANNELS ARE APPENDED. GAP makes a
   convnet translation-invariant -- normally a feature, here a defect: force depends
   on WHERE you pressed, because stiffness is a function of position. So we pool to a
   coarse 3x4 grid rather than 1x1, and feed two coordinate channels, so the network
   can represent "soft middle vs stiff edge" at all.

3. ⚠ SPLITS ARE GROUPED BY GRID ROW-BAND, NEVER RANDOM. The deformation field has a
   measured half-width of ~8 mm and is still at 10% of peak 18 mm away, while the grid
   pitch is 3 mm. Neighbouring presses are therefore heavily overlapping measurements;
   a random split would train and test on effectively the same data and report a
   score that means nothing. We hold out contiguous bands of 3 rows.

⚠ depth_mm and achieved_mm are NEVER model inputs. Depth alone already explains
R^2 ~ 0.43 through the robot; including it would hide whether the camera works.
They appear only as the reference baseline.

Usage:
    python3 train_cnn.py                 # difference input, 4-fold grouped CV
    python3 train_cnn.py --channels both # tare + dwell as separate channels
    python3 train_cnn.py --epochs 400
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


def grouped_folds(row, n_folds=4):
    """Contiguous row-bands. Returns a list of boolean test masks."""
    rows = np.unique(row)
    bands = np.array_split(rows, n_folds)
    return [np.isin(row, b) for b in bands]


def linear_loo_baseline(feat, y, folds):
    """Same folds, same metric -- so the CNN is compared like for like."""
    pred = np.zeros_like(y)
    X = np.column_stack([np.ones(len(y))] + feat)
    for te in folds:
        tr = ~te
        w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ w
    return pred


def metrics(y, p):
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    r2 = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    mae = float(np.abs(y - p).mean())
    return rmse, r2, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=["diff", "both"], default="diff")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=3e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--downsample", type=int, default=2, help="from the cached 320x240")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(CACHE)
    tare = d["tare"].astype(np.float32)
    dwell = d["dwell"].astype(np.float32)
    y = d["force"].astype(np.float32)
    row, col, depth = d["row"], d["col"], d["depth_mm"].astype(np.float32)

    k = args.downsample
    if k > 1:
        tare = tare[:, ::k, ::k]
        dwell = dwell[:, ::k, ::k]
    n, Hh, Ww = tare.shape

    # Difference is the signal; scale so it lands near unit variance.
    diff = (dwell - tare) / 4.0
    if args.channels == "diff":
        X = diff[:, None]
    else:
        X = np.stack([diff, tare / 8.0], 1)

    # Coordinate channels -- see design note 2.
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    coord = np.stack([xx / Ww * 2 - 1, yy / Hh * 2 - 1])[None].repeat(n, 0)
    X = np.concatenate([X, coord], 1)
    C = X.shape[1]

    folds = grouped_folds(row, args.folds)
    print("input   : %s  (%d channels, %dx%d)" % (args.channels, C, Hh, Ww))
    print("samples : %d   force %.3f..%.3f N (sd %.3f)" % (n, y.min(), y.max(), y.std()))
    print("folds   : %d grouped row-bands, test sizes %s"
          % (args.folds, [int(t.sum()) for t in folds]))
    print("device  : %s\n" % dev)

    class Net(nn.Module):
        def __init__(self, cin, p):
            super().__init__()
            def blk(a, b, ks=3, st=2):
                return nn.Sequential(nn.Conv2d(a, b, ks, st, ks // 2, bias=False),
                                     nn.BatchNorm2d(b), nn.ReLU(inplace=True))
            self.f = nn.Sequential(blk(cin, 8, 5), blk(8, 16), blk(16, 32), blk(32, 32))
            self.pool = nn.AdaptiveAvgPool2d((3, 4))     # keeps coarse LOCATION
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(32 * 12, 1))

        def forward(self, x):
            return self.head(self.pool(self.f(x))).squeeze(1)

    pred = np.zeros(n, np.float32)
    t0 = time.time()
    for fi, te in enumerate(folds):
        tr = ~te
        mu, sd = y[tr].mean(), y[tr].std()
        Xtr = torch.tensor(X[tr], device=dev)
        ytr = torch.tensor((y[tr] - mu) / sd, device=dev)
        Xte = torch.tensor(X[te], device=dev)

        # inner split for early stopping, also grouped so it stays honest
        rtr = row[tr]
        inner = np.isin(rtr, np.unique(rtr)[-2:])
        net = Net(C, args.dropout).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        best, best_state, bad = 1e9, None, 0
        idx = np.where(~inner)[0]
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(len(idx), device=dev)
            for b in range(0, len(idx), args.batch):
                sel = torch.tensor(idx, device=dev)[perm[b:b + args.batch]]
                opt.zero_grad()
                loss = F.smooth_l1_loss(net(Xtr[sel]), ytr[sel])
                loss.backward()
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                v = F.mse_loss(net(Xtr[inner]), ytr[inner]).item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k2: v2.detach().clone() for k2, v2 in net.state_dict().items()}
            else:
                bad += 1
                if bad > 60:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pred[te] = (net(Xte).cpu().numpy() * sd + mu)
        r, q, _ = metrics(y[te], pred[te])
        print("  fold %d  rows %-9s n=%3d   RMSE %.4f N  R2 %+.3f   (%d epochs)"
              % (fi + 1, str(sorted(set(row[te]))[:3])[:9], te.sum(), r, q, ep + 1))

    print("\n%s" % ("=" * 66))
    base_mean = np.zeros(n, np.float32)
    for te in folds:
        base_mean[te] = y[~te].mean()
    rows = [("predict the training mean", base_mean),
            ("depth only (ROBOT, no camera)", linear_loo_baseline([depth], y, folds)),
            ("CNN on frames (CAMERA only)", pred)]
    print("  %-32s %8s %8s %8s" % ("model", "RMSE", "R2", "MAE"))
    for nm, p in rows:
        r, q, m = metrics(y, p)
        print("  %-32s %7.4f %8.3f %8.4f" % (nm, r, q, m))
    print("=" * 66)
    print("wall time %.1f min" % ((time.time() - t0) / 60))
    np.savez(_os.path.join(OUTPUT, "cnn_predictions.npz"),
             y=y, pred=pred, row=row, col=col, depth=depth,
             baseline_depth=linear_loo_baseline([depth], y, folds))


if __name__ == "__main__":
    main()
