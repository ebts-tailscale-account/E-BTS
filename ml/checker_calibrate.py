#!/usr/bin/env python3
"""
checker_calibrate.py -- pixel -> millimetre from a printed checkerboard.

    python3 ml/checker_calibrate.py recordings/checker3mm_<stamp>/camera.raw --square-mm 3.0

WHY NOT cv2.fisheye, WHICH EVERY TUTORIAL REACHES FOR
------------------------------------------------------
Two reasons, and both are about this rig rather than about the tutorials.

  * cv2.fisheye implements Kannala-Brandt, which models a >90 deg barrel lens.
    This macro lens is about 65 deg and its distortion is PINCUSHION -- the dome
    lattice and the checkerboard independently agree that the periphery is
    MAGNIFIED, not compressed. Fitting KB to that is fitting the wrong sign of the
    wrong model. Brown-Conrady (cv2.calibrateCamera) is the correct intrinsic model
    if intrinsics are what you want.
  * Nothing here wants intrinsics. The markers live on ONE rigid plane at ONE fixed
    distance, so the whole 3D problem collapses to a 2D -> 2D warp. Intrinsics
    would additionally need the board imaged at MANY tilts; a board fixed flat in
    front of the sensor is a single pose, and a single planar pose cannot separate
    focal length from distortion from pose. It is not that it is hard -- it is
    degenerate, and the optimiser will still hand you numbers.

WHY SQUARE CENTROIDS AND NOT findChessboardCorners
---------------------------------------------------
Corner detection wants smooth intensity gradients. An event frame has none: it is
a sparse count image, and the operator's own capture came through as pure binary.
cv2.findChessboardCorners was tried on that image at five grid sizes and failed
every one. Connected-component centroids of the black squares are far more robust
on this data -- they integrate over the whole square instead of localising a
gradient -- and they gave a clean lattice fit on the same image.

The cost is that centroids are the centres of squares rather than corners, so the
lattice is STAGGERED: black squares tile only the cells where (row + col) is even.
Within a row they sit 2 squares apart, and consecutive rows are offset by one.
Indexing that as a plain rectangular grid is the single easiest way to get this
wrong, and it produces a lattice fit that looks converged and is nonsense.

INDEX BY POSITION, NEVER BY RANK
---------------------------------
Assigning a column from a square's rank within its row breaks the moment one
square is missing -- and the missing ones are always at the edges, where the
illumination falls off and the detector rejects them. On the operator's capture
the rows came out 4/5/4/5/4/5/5/5 instead of a clean alternation, and rank-based
indexing put the residual at 0.64 of a square pitch. The same points indexed by
POSITION land at 0.077. Same data, same detector; only the bookkeeping differed.

WHAT COMES OUT
--------------
A pixel -> millimetre warp anchored to the PRINTED pitch, so its scale is
traceable to a ruler rather than to a robot. That is the one thing the poke raster
could not supply well: it measured scale to 1.4% and implied a 38.4 mm dome field
across a 36 mm sheet.

⚠ THE SCALE ONLY TRANSFERS IF THE BOARD SAT AT THE MARKER PLANE.
Magnification depends on distance. A board a few millimetres nearer or further
than the silicone gives a perfectly self-consistent calibration of the WRONG
plane. The distortion SHAPE still transfers; the millimetres do not. This script
cannot check that from the image, so it cross-checks the implied dome pitch
against the sensor's own lattice and prints both for you to judge.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml"))

from evt2_frames import Evt2Reader                                   # noqa: E402
from undistort import MODELS, fit_model, normaliser, predict         # noqa: E402

# Only the plane-to-plane models make sense here; lattice_affine is defined against
# the DOME grid and has nothing to say about a checkerboard.
CHECKER_MODELS = ("affine", "poly2", "poly3", "poly4", "tps")


def accumulate(raw_path, start_s, seconds, accum_us):
    """One high-SNR count image: many short frames summed.

    A static board only makes events because the LEDs flicker, so a single 40 ms
    frame is sparse and speckled -- which is why the operator's screenshot was
    pure binary. Summing several seconds of them turns it into a smooth intensity
    image that a centroid can actually be measured on.
    """
    with Evt2Reader(str(raw_path)) as r:
        frames, t0s = r.frames(start_s * 1e6, (start_s + seconds) * 1e6,
                               accum_us=accum_us)
    if not len(frames):
        sys.exit("[ERROR] no events in %.1f-%.1f s of %s"
                 % (start_s, start_s + seconds, raw_path))
    acc = frames.astype(np.float64).sum(0)
    return acc, len(frames)


def find_squares(acc, min_area_frac=0.15, max_area_frac=4.0, close_px=5):
    """Centroids of the BLACK squares: event-sparse islands in an event-dense field.

    White paper reflects the flicker and fires events; the printed black squares do
    not. So the squares are the LOW-count regions, and Otsu on the smoothed count
    image separates them without a hand-set threshold.
    """
    img = acc / max(acc.max(), 1e-9)
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    sm = cv2.GaussianBlur(u8, (0, 0), 2.0)
    _t, white = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)

    inv = 255 - white
    n, lab, st, cen = cv2.connectedComponentsWithStats(inv, 8)
    H, W = acc.shape
    blobs = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        if w < 3 or h < 3:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue                       # clipped by the frame edge
        blobs.append((cen[i][0], cen[i][1], a, w, h))
    if len(blobs) < 12:
        sys.exit("[ERROR] only %d square candidates -- check the capture actually "
                 "shows the board." % len(blobs))
    b = np.array(blobs)
    med = np.median(b[:, 2])
    keep = (b[:, 2] > min_area_frac * med) & (b[:, 2] < max_area_frac * med)
    asp = b[:, 3] / np.maximum(b[:, 4], 1)
    keep &= (asp > 0.5) & (asp < 2.0)
    return b[keep][:, :2], white


def cluster(v, tol):
    order = np.argsort(v)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if v[b] - v[a] > tol:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    return groups


def index_lattice(pts):
    """(row, col) for each black square, on the staggered checkerboard sublattice.

    Rows come from clustering y, which is reliable. Columns are then solved by
    POSITION through an iterated affine fit -- see the module docstring for why
    rank-based indexing silently fails here. Both stagger phases are tried and the
    lower-residual one wins.
    """
    n = len(pts)
    span = pts[:, 1].max() - pts[:, 1].min()
    rows = sorted(cluster(pts[:, 1], max(8.0, 0.35 * span / 8.0)),
                  key=lambda g: pts[g, 1].mean())
    r_idx = np.zeros(n, int)
    for i, g in enumerate(rows):
        r_idx[g] = i

    gaps = []
    for g in rows:
        if len(g) > 1:
            gaps += list(np.diff(np.sort(pts[g, 0])))
    if not gaps:
        sys.exit("[ERROR] could not estimate the square pitch (no row has 2 squares).")
    gaps = np.array(gaps)
    pitch = float(np.median(gaps[gaps < np.percentile(gaps, 80) * 1.6])) / 2.0

    def fit(c_idx):
        A = np.column_stack([np.ones(n), c_idx, r_idx])
        sol, *_ = np.linalg.lstsq(A, pts, rcond=None)
        return sol, pts - A @ sol

    best = None
    for phase in (0, 1):
        x0 = pts[:, 0].min()
        c_idx = np.zeros(n, int)
        for i in range(n):
            raw = (pts[i, 0] - x0) / pitch
            want = (r_idx[i] + phase) % 2
            c = int(round(raw))
            if (c % 2) != want:
                c = c + 1 if raw > c else c - 1
            c_idx[i] = c
        for _ in range(10):
            sol, _dev = fit(c_idx)
            M = np.array([[sol[1, 0], sol[2, 0]], [sol[1, 1], sol[2, 1]]])
            cr = np.linalg.solve(M, (pts - sol[0]).T)
            new = c_idx.copy()
            for i in range(n):
                want = (r_idx[i] + phase) % 2
                c = int(round(cr[0, i]))
                if (c % 2) != want:
                    c = c + 1 if cr[0, i] > c else c - 1
                new[i] = c
            if (new == c_idx).all():
                break
            c_idx = new
        sol, dev = fit(c_idx)
        rms = float(np.sqrt((dev ** 2).sum(1).mean()))
        if best is None or rms < best[0]:
            best = (rms, phase, c_idx.copy(), r_idx.copy(), pitch, sol, dev)
    return best


def radial_report(pts, dev):
    def resid(c):
        d = pts - np.asarray(c)
        r = np.linalg.norm(d, axis=1)
        u = d / np.maximum(r, 1e-9)[:, None]
        proj = (dev * u).sum(1)
        M = np.column_stack([r ** 2, r ** 4])
        kk, *_ = np.linalg.lstsq(M, proj, rcond=None)
        return proj - M @ kk, kk, proj, r

    c0 = pts.mean(0)
    for scale in (200., 60., 20., 6., 2., 0.7):
        grid = [(c0[0] + dx * scale, c0[1] + dy * scale)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        c0 = np.array(sorted((float(np.sqrt((resid(c)[0] ** 2).mean())), c)
                             for c in grid)[0][1])
    res, kk, proj, r = resid(c0)
    mag = np.linalg.norm(dev, axis=1)
    out = {"centre_px": [float(c0[0]), float(c0[1])],
           "k1": float(kk[0]), "k2": float(kk[1]),
           "radial_r2": float(1 - np.var(res) / max(np.var(proj), 1e-12)),
           "radial_fraction": float(np.mean(np.abs(proj)) / max(np.mean(mag), 1e-12)),
           "sign": "pincushion" if np.mean(proj[r > np.percentile(r, 70)]) > 0
                   else "barrel"}
    return out


def grouped_cv(uv, xy, model, k=6, seed=0):
    """Plain k-fold. Each square is one independent observation of a RIGID target,
    so there are no repeat-groups to leak the way the poke raster had."""
    rng = np.random.RandomState(seed)
    fold = rng.permutation(len(uv)) % k
    err = np.full(len(uv), np.nan)
    for f in range(k):
        te = fold == f
        tr = ~te
        if te.sum() == 0 or tr.sum() < 12:
            continue
        fit = fit_model(uv[tr], xy[tr], model, normaliser(uv[tr]))
        err[te] = np.linalg.norm(predict(fit, uv[te]) - xy[te], axis=1)
    e = err[np.isfinite(err)]
    return {"model": model, "rms_mm": float(np.sqrt((e ** 2).mean())),
            "median_mm": float(np.median(e)), "p95_mm": float(np.percentile(e, 95)),
            "max_mm": float(e.max())}


def main():
    ap = argparse.ArgumentParser(
        description="pixel -> mm from a printed checkerboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help=".raw file, or a run directory containing camera.raw")
    ap.add_argument("--square-mm", type=float, required=True,
                    help="printed side of ONE square, in mm (the scale anchor)")
    ap.add_argument("--start-s", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--accum-us", type=float, default=40000.0)
    ap.add_argument("--model", default="poly3", choices=CHECKER_MODELS)
    ap.add_argument("--out", default=str(REPO / "calibration" / "pixel_to_mm_checker.json"))
    ap.add_argument("--debug-png", default=None)
    args = ap.parse_args()

    src = Path(args.source)
    raw = src / "camera.raw" if src.is_dir() else src
    if not raw.exists():
        sys.exit("[ERROR] %s not found." % raw)

    print("\n" + "=" * 72)
    print("  CHECKERBOARD CALIBRATION")
    print("=" * 72)
    print("  raw          : %s" % raw)
    print("  square       : %.3f mm" % args.square_mm)

    acc, nfr = accumulate(raw, args.start_s, args.seconds, args.accum_us)
    print("  accumulated  : %d frames over %.1f s (peak count %d)"
          % (nfr, args.seconds, int(acc.max())))

    pts, white = find_squares(acc)
    print("  squares      : %d detected" % len(pts))

    rms_px, phase, c_idx, r_idx, pitch_px, sol, dev = index_lattice(pts)
    nrow, ncol = len(set(r_idx)), len(set(c_idx))
    print("  lattice      : %d rows x %d column slots, stagger phase %d"
          % (nrow, ncol, phase))
    print("  pitch        : %.2f px per square" % pitch_px)
    print("  affine fit   : %.3f px RMS (%.4f of a square)" % (rms_px, rms_px / pitch_px))
    if rms_px / pitch_px > 0.25:
        print("  ⚠ that residual is a large fraction of a square: the indexing is")
        print("    probably wrong rather than the lens being bad. Check --debug-png.")

    d = np.linalg.norm(dev, axis=1)
    print("  deviation from the best affine lattice:")
    print("    RMS %.2f px   p95 %.2f px   max %.2f px"
          % (np.sqrt((d ** 2).mean()), np.percentile(d, 95), d.max()))
    rad = radial_report(pts, dev)
    print("    %.0f%% radial about (%.1f, %.1f), explains %.0f%%  ->  %s"
          % (100 * rad["radial_fraction"], rad["centre_px"][0], rad["centre_px"][1],
             100 * rad["radial_r2"], rad["sign"].upper()))

    # ---- the fit: pixel -> millimetre, anchored on the printed pitch ----------
    xy = np.column_stack([c_idx * args.square_mm, r_idx * args.square_mm]).astype(float)
    uv = pts.astype(float)
    cvs = [grouped_cv(uv, xy, m) for m in CHECKER_MODELS]
    print("\n  CROSS-VALIDATED RESIDUALS (6-fold over squares)")
    print("    %-8s %9s %9s %9s" % ("model", "RMS", "median", "p95"))
    for c in cvs:
        print("    %-8s %8.4f  %8.4f  %8.4f"
              % (c["model"], c["rms_mm"], c["median_mm"], c["p95_mm"]))
    aff = [c for c in cvs if c["model"] == "affine"][0]["rms_mm"]
    bst = min(c["rms_mm"] for c in cvs)
    print("    affine (do nothing) %.4f mm -> best %.4f mm  = %.0f%% removed"
          % (aff, bst, 100 * (1 - bst / aff)))

    fit = fit_model(uv, xy, args.model, normaliser(uv))
    M = np.array(fit_model(uv, xy, "affine", normaliser(uv))["coeffs"])[1:].T \
        / normaliser(uv)["scale"]
    S = np.linalg.svd(M, compute_uv=False)
    ppm = (1.0 / S[0], 1.0 / S[1])
    print("\n  MEASURED SCALE (traceable to the printed pitch)")
    print("    %.3f and %.3f px/mm  (anisotropy %.4f)" % (ppm[0], ppm[1], S[0] / S[1]))
    print("    poke raster gave 13.153 and 14.033 px/mm; config implies 17.78 / 16.00")

    dome_pitch_mm = 30.9 / (0.5 * (ppm[0] + ppm[1]))
    print("\n  CROSS-CHECK -- was the board at the marker plane?")
    print("    the sensor's dome lattice is 30.9 px; at this scale that is %.3f mm."
          % dome_pitch_mm)
    print("    The poke raster implied 2.40 mm. A big disagreement means the board")
    print("    sat at a different distance than the silicone, and the MILLIMETRES")
    print("    do not transfer even though the distortion shape does.")

    if args.debug_png:
        vis = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
        for (x, y), r_, c_ in zip(pts, r_idx, c_idx):
            cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, "%d,%d" % (r_, c_), (int(x) - 14, int(y) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.imwrite(args.debug_png, vis)
        print("\n  debug image  -> %s" % args.debug_png)

    out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_raw": str(raw), "target": "printed checkerboard",
        "square_mm": args.square_mm,
        "image_size": [int(acc.shape[1]), int(acc.shape[0])],
        "frame": "checkerboard plane, millimetres",
        "pixel_to_mm": fit,
        "mm_to_pixel": fit_model(xy, uv, args.model, normaliser(xy)),
        "cross_validated_residuals_mm": cvs,
        "px_per_mm": [float(ppm[0]), float(ppm[1])],
        "anisotropy": float(S[0] / S[1]),
        "lattice_deviation_px": {"rms_px": float(np.sqrt((d ** 2).mean())),
                                 "p95_px": float(np.percentile(d, 95)),
                                 "max_px": float(d.max()),
                                 "n_squares": int(len(pts))},
        "radial_fit": rad,
        "square_pitch_px": float(pitch_px),
        "implied_dome_pitch_mm": float(dome_pitch_mm),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("  calibration  -> %s" % outp)
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
