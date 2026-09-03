#!/usr/bin/env python3
"""
export_undistorted_video.py -- side-by-side mp4 of raw vs rectified event frames.

    python3 ml/export_undistorted_video.py three_cyl_24mm_20260902_204649
    python3 ml/export_undistorted_video.py <run> --seq 4 --seconds 6 --overlay none

⚠ THIS IS A PICTURE, NOT A MEASUREMENT PATH
--------------------------------------------
ml/undistort.py says, repeatedly, that the correction belongs on detected marker
CENTRES and not on images: resampling a 640x480 event-count frame interpolates
intensity that no event produced, and it blurs domes that are only ~19 px across.
That warning still stands. This script exists because a warp you cannot see is a
warp nobody trusts, and the honest way to show one is to render it.

Nothing downstream should ever read these frames back as data.

WHY SIDE BY SIDE, AND WHY THE GRID
-----------------------------------
The distortion is 16 px peak on a 640 px field -- about 2.5%. Played on its own,
a rectified event video looks exactly like the original, and "looks the same"
would be a fair conclusion from a bad visualisation rather than a fact about the
lens. So each panel gets a STRAIGHT reference grid fitted in its own space:

  left  (raw)       the best affine fit of the dome lattice, in sensor pixels.
                    This is the null model -- rotation, shear and a different
                    scale per axis are all allowed, so any bowing that survives
                    is the lens.
  right (rectified) the same construction in rectified millimetres.

The domes are drawn as circles on both. On the left they drift off the lines
toward the corners; on the right they should sit on them. That difference IS the
calibration, and it is visible at a glance where a raw A/B is not.

RECTIFIED SPACE
---------------
The output image is linear in millimetres, at --px-per-mm (default: the measured
mean, so resolution is roughly preserved rather than silently resampled up). Its
axes are the ROBOT frame's, because that is the frame the raster measured, so the
image may sit at a slight angle to the sensor -- about 1 degree on this rig, which
the calibration reports as not significantly different from zero.

Outside the dome field the calibration is undefined -- pixel_to_mm returns NaN
there by design -- so those pixels are filled flat grey rather than extrapolated.
The grey border is the honest edge of what was measured.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml"))

from evt2_frames import Evt2Reader                                   # noqa: E402
from undistort import load as load_cal                               # noqa: E402

REC = REPO / "recordings"
GRID = REPO / "ml" / "grid_out"

# frames_png convention (ml/export_frames.py, ml/blue_circle_grid.cpp): counts peak
# around 26, so x255/8 lands a typical dome near white without clipping the tail.
COUNT_TO_U8 = 255.0 / 8.0

INVALID_GREY = 38          # outside the calibrated field
GRID_COLOUR = (70, 170, 255)
DOME_COLOUR = (255, 140, 40)


def resolve_run(name):
    p = Path(name)
    if not p.is_dir():
        p = REC / name
    if not p.is_dir():
        sys.exit("[ERROR] no such run: %s" % name)
    return p


def node_array(raw):
    return np.array([[(np.nan, np.nan) if p is None or p[0] is None else p
                      for p in row] for row in raw], float)


def affine_grid_lines(pos, step=1):
    """Straight lines through the best affine fit of index -> position.

    Returns (lines, ok_mask). `pos` is (rows, cols, 2) with NaN holes; the affine
    fit uses whatever is measured and then predicts every node, so a missing dome
    does not break the line it sits on.
    """
    rows, cols = pos.shape[:2]
    rr, cc = np.mgrid[0:rows, 0:cols]
    ok = np.isfinite(pos[..., 0])
    A = np.column_stack([np.ones(ok.sum()), cc[ok].ravel(), rr[ok].ravel()])
    sol, *_ = np.linalg.lstsq(A, pos[ok], rcond=None)
    full = np.column_stack([np.ones(rows * cols), cc.ravel(), rr.ravel()]) @ sol
    return full.reshape(rows, cols, 2), ok


def draw_overlay(img, pos, pred, ok):
    """Straight affine grid + a circle on every measured dome."""
    rows, cols = pred.shape[:2]
    for r in range(rows):
        a, b = pred[r, 0], pred[r, -1]
        cv2.line(img, (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))), GRID_COLOUR, 1, cv2.LINE_AA)
    for c in range(cols):
        a, b = pred[0, c], pred[-1, c]
        cv2.line(img, (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))), GRID_COLOUR, 1, cv2.LINE_AA)
    for r in range(rows):
        for c in range(cols):
            if ok[r, c]:
                p = pos[r, c]
                cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 5,
                           DOME_COLOUR, 1, cv2.LINE_AA)
    return img


def build_rectify_map(cal, px_per_mm, pad_mm):
    """Output-pixel -> source-pixel maps for cv2.remap, plus the valid-region mask.

    The mask is the polygon of the OUTERMOST MEASURED DOMES, not a per-pixel
    round-trip test. Both give the same region; the polygon costs milliseconds
    where inverting the node table 300k times costs minutes, and the boundary of
    the calibrated field is exactly the boundary of the domes anyway.
    """
    node = node_array(cal.d["node_px"])
    rows, cols = node.shape[:2]
    ok = np.isfinite(node[..., 0])

    nm = np.full((rows, cols, 2), np.nan)
    idx = np.argwhere(ok)
    xs, ys = cal.pixel_to_mm(node[ok][:, 0], node[ok][:, 1])
    nm[ok] = np.column_stack([xs, ys])
    good = np.isfinite(nm[..., 0])
    if good.sum() < 10:
        sys.exit("[ERROR] the calibration maps too few domes to millimetres.")

    x0, x1 = np.nanmin(nm[..., 0]) - pad_mm, np.nanmax(nm[..., 0]) + pad_mm
    y0, y1 = np.nanmin(nm[..., 1]) - pad_mm, np.nanmax(nm[..., 1]) + pad_mm
    W = int(round((x1 - x0) * px_per_mm))
    H = int(round((y1 - y0) * px_per_mm))

    def mm_to_out(x, y):
        return ((np.asarray(x) - x0) * px_per_mm, (np.asarray(y) - y0) * px_per_mm)

    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    X = x0 + jj / px_per_mm
    Y = y0 + ii / px_per_mm
    su, sv = cal.mm_to_pixel(X.ravel(), Y.ravel())
    map_x = np.asarray(su, np.float32).reshape(H, W)
    map_y = np.asarray(sv, np.float32).reshape(H, W)

    # Valid region = the CONVEX HULL of the measured domes. An edge-walk of the node
    # grid looks more precise and is worse: wherever a dome is missing the walk steps
    # inward, and the polygon self-intersects into visible wedges cut out of the
    # image. The dome field is convex, so the hull is both correct and robust to the
    # ~10 domes that are never detected.
    ox, oy = mm_to_out(nm[good][:, 0], nm[good][:, 1])
    pts = np.column_stack([ox, oy]).astype(np.float32)
    hull = cv2.convexHull(pts).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # dome positions in the rectified image, for the overlay
    rect_node = np.full((rows, cols, 2), np.nan)
    gx, gy = mm_to_out(nm[good][:, 0], nm[good][:, 1])
    rect_node[good] = np.column_stack([gx, gy])

    return map_x, map_y, mask, rect_node, (W, H), (x1 - x0, y1 - y0)


def measure_transfer(run, grid_dir, cal, ppm):
    """Does the calibration straighten domes it was NOT fitted on?

    Detects this run's own domes from its tare frames, then measures how far they
    bow off a straight grid before and after the warp. Because the calibration was
    fitted on a different run, neither number is circular -- and a calibration that
    only works on its own recording is not a calibration.

    Returns (raw_bow_px, rectified_bow_px_equivalent). Falls back to the source
    run's numbers, clearly labelled, when this run has no frames.h5.
    """
    import fit_pixel_mm_warp as F
    if not (run / "frames.h5").exists():
        print("  ⚠ %s has no frames.h5, so the transfer check cannot run; the bow\n"
              "    shown is the CALIBRATION run's, which its own model straightens\n"
              "    by construction and therefore proves nothing here." % run.name)
        dv = cal.d.get("lattice_deviation_px", {})
        return float(dv.get("max_px", float("nan"))), float("nan")

    lat = F.load_lattice(grid_dir)
    own, seen, st = F.measure_nodes(run, lat, cal.d.get("detector_radius_px", 9.6),
                                    60, progress=False)
    ok = np.isfinite(own[..., 0])
    _sol, _pred, _dev, ds = F.affine_from_index(own)
    raw_bow = float(ds["max_px"])

    mm = np.full(own.shape, np.nan)
    xs, ys = cal.pixel_to_mm(own[ok][:, 0], own[ok][:, 1])
    mm[ok] = np.column_stack([xs, ys])
    good = np.isfinite(mm[..., 0])
    if good.sum() < 10:
        return raw_bow, float("nan")
    pred_mm, _ = affine_grid_lines(mm)
    d = np.linalg.norm(mm - pred_mm, axis=-1)
    rect_bow_mm = float(np.nanmax(np.where(good, d, np.nan)))

    print("  transfer check: %d of %d domes in THIS run, calibration from %s"
          % (int(ok.sum()), ok.size, cal.d.get("source_run")))
    print("  dome bow off a straight grid (independent domes, not the fit's own):")
    print("    raw        %.2f px" % raw_bow)
    print("    rectified  %.2f px  (%.3f mm)" % (rect_bow_mm * ppm, rect_bow_mm))
    print("    -> %.0f%% of the bow removed"
          % (100 * (1 - (rect_bow_mm * ppm) / max(raw_bow, 1e-9))))
    return raw_bow, rect_bow_mm * ppm


def pick_window(run, grid_dir, seq, seconds):
    """[start, end] in re-based microseconds, centred on a poke if we know one."""
    pw = grid_dir / "poke_windows.csv"
    if pw.exists():
        rows = list(csv.DictReader(open(pw)))
        if rows:
            i = min(max(seq, 0), len(rows) - 1)
            tare = float(rows[i]["tare_mid_us"])
            dwell = float(rows[i]["dwell_mid_us"])
            mid = 0.5 * (tare + dwell)
            half = seconds * 1e6 / 2.0
            return mid - half, mid + half, i, float(rows[i].get("force_mean_n", "nan"))
    return 0.0, seconds * 1e6, None, float("nan")


def main():
    ap = argparse.ArgumentParser(
        description="Side-by-side raw vs rectified event video.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--grid-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seq", type=int, default=0, help="poke index to centre on")
    ap.add_argument("--seconds", type=float, default=5.0, help="event-time span")
    ap.add_argument("--accum-us", type=float, default=20000.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="rectified scale (default: the calibration's measured mean)")
    ap.add_argument("--pad-mm", type=float, default=1.0)
    ap.add_argument("--overlay", choices=("grid", "none"), default="grid")
    ap.add_argument("--no-labels", action="store_true",
                    help="drop the panel captions and clock too. With --overlay none "
                         "this gives a bare A/B, which is the honest way to look at "
                         "the imagery itself -- but note that 16 px on a 640 px field "
                         "is genuinely hard to see without a reference.")
    args = ap.parse_args()

    run = resolve_run(args.run)
    grid_dir = Path(args.grid_dir) if args.grid_dir else GRID / run.name
    raw = run / "camera.raw"
    if not raw.exists():
        sys.exit("[ERROR] %s not found." % raw)

    cal = load_cal(args.calibration)
    g = cal.d.get("affine_geometry", {})
    ppm = args.px_per_mm or (0.5 * (g.get("px_per_mm_major", 13.5) +
                                    g.get("px_per_mm_minor", 13.5)))

    print("\n" + "=" * 72)
    print("  UNDISTORTED VIDEO")
    print("=" * 72)
    print("  run          : %s" % run.name)
    print("  calibration  : %s  (model %s, from %s)"
          % (cal.path, cal.d["pixel_to_mm"]["model"], cal.d.get("source_run")))
    print("  rectified at : %.2f px/mm" % ppm)

    map_x, map_y, mask, rect_node, (W, H), (spanx, spany) = \
        build_rectify_map(cal, ppm, args.pad_mm)
    print("  output size  : %d x %d px  (%.1f x %.1f mm of calibrated field)"
          % (W, H, spanx, spany))

    src_node = node_array(cal.d["node_px"])
    src_pred, src_ok = affine_grid_lines(src_node)
    rect_pred, rect_ok = affine_grid_lines(rect_node)

    # ⚠ DO NOT SCORE THE CALIBRATION ON ITS OWN DOMES. Under lattice_affine the
    # rectified node grid is an affine function of lattice index BY CONSTRUCTION --
    # the model is built from these very nodes -- so its bow is exactly 0.00 px and
    # "100% of the distortion removed" would be a tautology, not a result.
    # The real question is whether a calibration transfers, so the check below uses
    # THIS run's own domes, detected independently, against a calibration fitted on
    # a different run and a different day.
    bow_src, bow_rect = measure_transfer(run, grid_dir, cal, ppm)

    s_us, e_us, seq_i, force = pick_window(run, grid_dir, args.seq, args.seconds)
    print("  window       : %.3f -> %.3f s%s"
          % (s_us / 1e6, e_us / 1e6,
             "" if seq_i is None else "  (poke %d, %.2f N)" % (seq_i, force)))

    with Evt2Reader(str(raw)) as r:
        frames, t0s = r.frames(max(0.0, s_us), e_us, accum_us=args.accum_us)
    print("  frames       : %d at %.0f us" % (len(frames), args.accum_us))

    out_path = Path(args.out) if args.out else \
        REPO / "output" / "runs" / run.name / "undistorted_sidebyside.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # panels are made the same height so the pair reads as one image
    ph = max(480, H)
    lw = int(round(640 * ph / 480.0))
    rw = int(round(W * ph / float(H)))
    gap = 12
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (lw + gap + rw, ph + (0 if args.no_labels else 34)))
    if not vw.isOpened():
        sys.exit("[ERROR] could not open the video writer for %s" % out_path)

    # Display gain from the data, not from a constant. The frames_png convention
    # (x255/8) is calibrated for 40 ms accumulation; at 20 ms the counts are halved
    # and the markers come out nearly black, which makes the correction impossible
    # to see -- the whole point of rendering it. Put the 99.5th percentile of the
    # occupied pixels near white instead, and say what was chosen.
    nz = np.concatenate([f[f > 0].ravel() for f in frames[:40] if (f > 0).any()])
    gain = (230.0 / max(np.percentile(nz, 99.5), 1.0)) if nz.size else COUNT_TO_U8
    print("  display gain : x%.1f (99.5th pct of occupied pixels -> 230)" % gain)

    grey = np.full((H, W), INVALID_GREY, np.uint8)
    for k, f in enumerate(frames):
        u8 = np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        rect = cv2.remap(u8, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rect = np.where(mask > 0, rect, grey)

        L = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        R = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
        if args.overlay == "grid":
            L = draw_overlay(L, src_node, src_pred, src_ok)
            R = draw_overlay(R, rect_node, rect_pred, rect_ok)

        L = cv2.resize(L, (lw, ph), interpolation=cv2.INTER_NEAREST)
        R = cv2.resize(R, (rw, ph), interpolation=cv2.INTER_NEAREST)
        top = 0 if args.no_labels else 34
        canvas = np.zeros((ph + top, lw + gap + rw, 3), np.uint8)
        canvas[top:, :lw] = L
        canvas[top:, lw + gap:] = R
        if not args.no_labels:
            cv2.putText(canvas,
                        "RAW  (domes bow off the straight grid: %.1f px)" % bow_src,
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, "RECTIFIED  (%.1f px)" % bow_rect,
                        (lw + gap + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (140, 255, 160), 1, cv2.LINE_AA)
            cv2.putText(canvas, "t=%.2fs" % ((t0s[k] - t0s[0]) / 1e6),
                        (lw + gap + rw - 92, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()

    print("  video        -> %s" % out_path)
    print("  ⚠ display only -- never read these frames back as measurements.")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
