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


def build_rectify_map(cal, px_per_mm, pad_mm, ref_node, frame_hw=(480, 640),
                      masked=True):
    """Output-pixel -> source-pixel maps for cv2.remap, plus the valid-region mask.

    Works for ANY calibration. An earlier version read the dome grid out of the
    calibration file, which only the dome fit carries, so pointing it at the
    checkerboard fit was a bare KeyError. Two changes make it general:

      * the output extent comes from the FRAME BORDER pushed through pixel_to_mm
        rather than from the calibration's stored features, so the whole sensor
        stays in view instead of being cropped to whatever the target covered;
      * the mask comes from `ref_node`, the features measured in THIS recording,
        supplied by the caller. That is the more honest boundary anyway: it marks
        where this frame has data, not where some other run's target happened to sit.
    """
    Hf, Wf = frame_hw

    # Sample the whole border, not the four corners: the warp is nonlinear, so an
    # edge midpoint can fall outside the box the corners imply.
    bx = np.concatenate([np.linspace(0, Wf - 1, 80), np.full(80, Wf - 1),
                         np.linspace(Wf - 1, 0, 80), np.zeros(80)])
    by = np.concatenate([np.zeros(80), np.linspace(0, Hf - 1, 80),
                         np.full(80, Hf - 1), np.linspace(Hf - 1, 0, 80)])
    mx, my = cal.pixel_to_mm(bx, by)
    if not np.isfinite(mx).any():
        sys.exit("[ERROR] this calibration maps none of the frame to millimetres.")
    x0, x1 = np.nanmin(mx) - pad_mm, np.nanmax(mx) + pad_mm
    y0, y1 = np.nanmin(my) - pad_mm, np.nanmax(my) + pad_mm
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

    # This recording's own features, carried into the rectified frame -- used for
    # both the overlay and the mask.
    rect_node = None
    mask = np.full((H, W), 255, np.uint8)
    if ref_node is not None:
        ok = np.isfinite(ref_node[..., 0])
        nm = np.full(ref_node.shape, np.nan)
        xs, ys = cal.pixel_to_mm(ref_node[ok][:, 0], ref_node[ok][:, 1])
        nm[ok] = np.column_stack([xs, ys])
        good = np.isfinite(nm[..., 0])
        if good.sum() >= 10:
            ox, oy = mm_to_out(nm[good][:, 0], nm[good][:, 1])
            rect_node = np.full(ref_node.shape, np.nan)
            rect_node[good] = np.column_stack([ox, oy])
            if masked:
                # Convex hull, not an edge-walk of the grid: wherever a feature is
                # missing the walk steps inward and the polygon self-intersects
                # into visible wedges cut out of the image.
                hull = cv2.convexHull(
                    np.column_stack([ox, oy]).astype(np.float32)).astype(np.int32)
                mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(mask, [hull], 255)

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
        return float(dv.get("max_px", float("nan"))), float("nan"), None

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
        return raw_bow, float("nan"), own
    pred_mm, _ = affine_grid_lines(mm)
    d = np.linalg.norm(mm - pred_mm, axis=-1)
    rect_bow_mm = float(np.nanmax(np.where(good, d, np.nan)))

    print("  transfer check: %d of %d domes in THIS run, calibration from %s"
          % (int(ok.sum()), ok.size,
             cal.d.get("source_run") or Path(cal.d.get("source_raw", "?")).name))
    print("  dome bow off a straight grid (independent domes, not the fit's own):")
    print("    raw        %.2f px" % raw_bow)
    print("    rectified  %.2f px  (%.3f mm)" % (rect_bow_mm * ppm, rect_bow_mm))
    print("    -> %.0f%% of the bow removed"
          % (100 * (1 - (rect_bow_mm * ppm) / max(raw_bow, 1e-9))))
    return raw_bow, rect_bow_mm * ppm, own


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
    ap.add_argument("--no-mask", action="store_true",
                    help="keep the whole warped frame instead of clipping to "
                         "the features measured in this recording")
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
    if args.px_per_mm:
        ppm = args.px_per_mm
    elif "px_per_mm" in cal.d:                       # checkerboard fit
        ppm = float(np.mean(cal.d["px_per_mm"]))
    else:                                            # dome fit
        ppm = 0.5 * (g.get("px_per_mm_major", 13.5) + g.get("px_per_mm_minor", 13.5))

    print("\n" + "=" * 72)
    print("  UNDISTORTED VIDEO")
    print("=" * 72)
    print("  run          : %s" % run.name)
    # the dome fit records source_run, the checkerboard fit source_raw
    src_label = cal.d.get("source_run") or Path(cal.d.get("source_raw", "?")).name
    print("  calibration  : %s  (model %s, from %s)"
          % (cal.path, cal.d["pixel_to_mm"]["model"], src_label))
    print("  rectified at : %.2f px/mm" % ppm)

    # Measure THIS recording's own domes first: they serve as the transfer check,
    # the overlay reference and the mask, and none of those should be read out of
    # the calibration file -- the checkerboard fit contains no dome grid at all,
    # and scoring a fit on its own features is circular (see measure_transfer).
    bow_src, bow_rect, own_node = measure_transfer(run, grid_dir, cal, ppm)

    map_x, map_y, mask, rect_node, (W, H), (spanx, spany) = \
        build_rectify_map(cal, ppm, args.pad_mm, own_node, frame_hw=(480, 640),
                          masked=not args.no_mask)
    print("  output size  : %d x %d px  (%.1f x %.1f mm of the sensor field)"
          % (W, H, spanx, spany))

    src_node = own_node
    src_pred, src_ok = (affine_grid_lines(src_node) if src_node is not None
                        else (None, None))
    rect_pred, rect_ok = (affine_grid_lines(rect_node) if rect_node is not None
                          else (None, None))

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
        if args.overlay == "grid" and src_node is not None:
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
