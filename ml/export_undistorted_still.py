#!/usr/bin/env python3
"""
export_undistorted_still.py -- one PNG: raw event frame beside the rectified one.

    python3 ml/export_undistorted_still.py recordings/checker3mm_<stamp>.raw \\
        --calibration calibration/pixel_to_mm_checker.json

Takes whichever calibration you point it at, so it works for the checkerboard fit
(pixel_to_mm_checker.json) and for the dome fit (pixel_to_mm.json) alike.

⚠ Display only. Resampling an event-count frame interpolates intensity that no
event produced; real measurements go through cal.pixel_to_mm on detected feature
centres, never through a remapped image. See ml/undistort.py.

The rectified panel is linear in millimetres, so anything straight on the physical
plane comes out straight here. On a checkerboard that is the whole demonstration
and needs no overlay: the rows bow on the left and run straight on the right.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml"))

from undistort import load as load_cal                               # noqa: E402
from checker_calibrate import accumulate, find_squares               # noqa: E402

INVALID_GREY = 34


def main():
    ap = argparse.ArgumentParser(description="Raw vs rectified, as one PNG.")
    ap.add_argument("source", help=".raw file, or a run dir containing camera.raw")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start-s", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--accum-us", type=float, default=40000.0)
    ap.add_argument("--px-per-mm", type=float, default=None)
    ap.add_argument("--pad-mm", type=float, default=0.5)
    ap.add_argument("--no-mask", action="store_true",
                    help="render the whole warped plane instead of clipping to the "
                         "region the calibration was actually fitted on")
    args = ap.parse_args()

    src = Path(args.source)
    raw = src / "camera.raw" if src.is_dir() else src
    if not raw.exists():
        sys.exit("[ERROR] %s not found." % raw)

    cal = load_cal(args.calibration)
    d = cal.d
    ppm = args.px_per_mm or (np.mean(d["px_per_mm"]) if "px_per_mm" in d
                             else 0.5 * (d["affine_geometry"]["px_per_mm_major"] +
                                         d["affine_geometry"]["px_per_mm_minor"]))
    print("  calibration : %s (model %s)" % (cal.path, d["pixel_to_mm"]["model"]))
    print("  rectify at  : %.3f px/mm" % ppm)

    acc, nfr = accumulate(raw, args.start_s, args.seconds, args.accum_us)
    H, W = acc.shape
    print("  accumulated : %d frames, %dx%d" % (nfr, W, H))

    # mm extent of the frame, sampled around the border rather than at 4 corners --
    # the warp is nonlinear, so an edge midpoint can lie outside the corner bbox
    bx = np.concatenate([np.linspace(0, W - 1, 60), np.full(60, W - 1),
                         np.linspace(W - 1, 0, 60), np.zeros(60)])
    by = np.concatenate([np.zeros(60), np.linspace(0, H - 1, 60),
                         np.full(60, H - 1), np.linspace(H - 1, 0, 60)])
    mx, my = cal.pixel_to_mm(bx, by)
    x0, x1 = np.nanmin(mx) - args.pad_mm, np.nanmax(mx) + args.pad_mm
    y0, y1 = np.nanmin(my) - args.pad_mm, np.nanmax(my) + args.pad_mm
    OW = int(round((x1 - x0) * ppm))
    OH = int(round((y1 - y0) * ppm))
    print("  rectified   : %dx%d px (%.1f x %.1f mm)" % (OW, OH, x1 - x0, y1 - y0))

    jj, ii = np.meshgrid(np.arange(OW), np.arange(OH))
    su, sv = cal.mm_to_pixel((x0 + jj / ppm).ravel(), (y0 + ii / ppm).ravel())
    map_x = np.asarray(su, np.float32).reshape(OH, OW)
    map_y = np.asarray(sv, np.float32).reshape(OH, OW)

    gain = 230.0 / max(np.percentile(acc[acc > 0], 99.5), 1.0)
    u8 = np.clip(acc * gain, 0, 255).astype(np.uint8)
    rect = cv2.remap(u8, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if not args.no_mask:
        # Clip to where the calibration was actually fitted. A polynomial is happy to
        # evaluate anywhere; that does not make it a measurement, and an unmarked
        # extrapolated border is the kind of thing that quietly becomes data later.
        pts, _white = find_squares(acc)
        pmx, pmy = cal.pixel_to_mm(pts[:, 0], pts[:, 1])
        ox = (np.asarray(pmx) - x0) * ppm
        oy = (np.asarray(pmy) - y0) * ppm
        hull = cv2.convexHull(np.column_stack([ox, oy]).astype(np.float32)).astype(np.int32)
        mask = np.zeros((OH, OW), np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        rect = np.where(mask > 0, rect, INVALID_GREY).astype(np.uint8)
        print("  masked to the hull of %d detected squares" % len(pts))

    ph = max(H, OH)
    lw = int(round(W * ph / float(H)))
    rw = int(round(OW * ph / float(OH)))
    gap = 10
    canvas = np.zeros((ph, lw + gap + rw), np.uint8)
    canvas[:, :lw] = cv2.resize(u8, (lw, ph), interpolation=cv2.INTER_NEAREST)
    canvas[:, lw + gap:] = cv2.resize(rect, (rw, ph), interpolation=cv2.INTER_NEAREST)

    out = Path(args.out) if args.out else \
        REPO / "output" / (raw.stem + "_undistorted.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print("  -> %s   (left raw, right rectified)" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
