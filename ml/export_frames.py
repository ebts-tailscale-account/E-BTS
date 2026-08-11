#!/usr/bin/env python3
"""
export_frames.py -- frames.h5 -> a CSV you can open + one PNG per frame you can look at.

The HDF5 stays the source of truth; this is the human-readable mirror of it.

    <run>/frames.csv          one row per poke, every scalar + 3 image paths
    <run>/frames_png/         3 x 1008 PNGs
    <run>/frames_png/README   the scaling, written down so nobody guesses

⚠ FIXED DISPLAY SCALING, NOT PER-IMAGE AUTO-CONTRAST
----------------------------------------------------
Event counts are tiny and sparse: nonzero pixels sit at 2-4 (p99 = 6, max 20) over a
~15% occupied frame. Rendered raw, every image is nearly black.

The obvious fix -- normalise each image to its own min/max -- would be WRONG here, and
badly so: it would rescale a 0.2 N poke and a 5.7 N poke to the same apparent
brightness, destroying by eye exactly the comparison these images exist to support.

So the scale is FIXED and global:

    tare / dwell   value * 255 / 8, clipped        (8 = p99.9 of nonzero counts)
    difference     +-6 counts -> blue . white . red, clipped

Two images are therefore directly comparable: a brighter frame really did have more
events, and a redder difference really did move more markers. Values above the clip
are rare (<0.1%) and saturate rather than rescale the rest.

Exact counts are in frames.h5. These PNGs are for looking.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import h5py
from PIL import Image

FRAME_CLIP = 8.0      # counts -> full white
DIFF_CLIP = 6.0       # counts -> full red / full blue


def gray(a):
    """uint8 count frame -> viewable grayscale, fixed scale."""
    return np.clip(a.astype(np.float32) * (255.0 / FRAME_CLIP), 0, 255).astype(np.uint8)


def diverging(d):
    """signed difference -> blue(neg) . white(0) . red(pos), fixed scale."""
    x = np.clip(d.astype(np.float32) / DIFF_CLIP, -1.0, 1.0)
    pos, neg = np.clip(x, 0, 1), np.clip(-x, 0, 1)
    # white background so zero reads as blank paper, not as mid-grey noise
    r = 255 - neg * 195
    g = 255 - (pos + neg) * 205
    b = 255 - pos * 195
    return np.stack([r, g, b], -1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--h5", default=None)
    ap.add_argument("--subdir", default="frames_png")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-images", action="store_true", help="CSV only")
    args = ap.parse_args()

    run = Path(args.run)
    h5p = Path(args.h5) if args.h5 else run / "frames.h5"
    imgdir = run / args.subdir
    imgdir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(h5p, "r")
    n = int(f["force_n"].shape[0])
    if args.limit:
        n = min(n, args.limit)

    # every 1-D dataset becomes a CSV column, in a sensible order
    scalars = [k for k in f.keys() if f[k].ndim == 1]
    order = ["seq", "point_index", "row", "col", "block", "pass", "level_idx",
             "target_force_n", "force_n", "force_sd_n", "tare_n",
             "depth_cmd_mm", "depth_achieved_mm", "stiffness_map",
             "events_tare", "events_dwell", "frame_t_us", "t_rel_s"]
    cols = [c for c in order if c in scalars] + sorted(set(scalars) - set(order))
    data = {c: f[c][:n] for c in cols}

    rows = []
    for i in range(n):
        seq = int(data["seq"][i])
        stem = "poke_%04d" % seq
        rec = {}
        for c in cols:
            v = data[c][i]
            if isinstance(v, (bytes, np.bytes_)):
                v = v.decode()
            elif isinstance(v, (np.floating, float)):
                v = round(float(v), 6)
            elif isinstance(v, (np.integer, int)):
                v = int(v)
            rec[c] = v
        rec["tare_image"] = "%s/%s_tare.png" % (args.subdir, stem)
        rec["dwell_image"] = "%s/%s_dwell.png" % (args.subdir, stem)
        rec["diff_image"] = "%s/%s_diff.png" % (args.subdir, stem)
        rows.append(rec)

        if not args.no_images:
            t = f["tare_frame"][i]
            d = f["dwell_frame"][i]
            Image.fromarray(gray(t), "L").save(imgdir / ("%s_tare.png" % stem),
                                               optimize=True)
            Image.fromarray(gray(d), "L").save(imgdir / ("%s_dwell.png" % stem),
                                               optimize=True)
            Image.fromarray(diverging(d.astype(np.int16) - t.astype(np.int16)),
                            "RGB").save(imgdir / ("%s_diff.png" % stem), optimize=True)
            if (i + 1) % 100 == 0 or i + 1 == n:
                print("  %4d/%d images" % (i + 1, n), flush=True)

    out = run / "frames.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    f.close()

    (imgdir / "README.txt").write_text(
        "PNGs exported from frames.h5 by ml/export_frames.py\n"
        "\n"
        "  <stem>_tare.png   grayscale, arm out of contact (undeformed reference)\n"
        "  <stem>_dwell.png  grayscale, arm holding the indent\n"
        "  <stem>_diff.png   dwell - tare.  RED = more events, BLUE = fewer, WHITE = 0\n"
        "\n"
        "FIXED display scaling -- images ARE comparable to each other:\n"
        "  grayscale : count * 255/%g, clipped   (%g = p99.9 of nonzero counts)\n"
        "  difference: +-%g counts full scale, clipped\n"
        "\n"
        "Per-image auto-contrast was deliberately NOT used: it would make a 0.2 N\n"
        "poke and a 5.7 N poke look identical.\n"
        "\n"
        "Exact integer counts live in frames.h5. These are for looking at.\n"
        "Row -> image mapping is in ../frames.csv (tare_image/dwell_image/diff_image).\n"
        % (FRAME_CLIP, FRAME_CLIP, DIFF_CLIP))

    print("\nwrote %s  (%d rows x %d cols)" % (out, len(rows), len(rows[0])))
    if not args.no_images:
        pngs = list(imgdir.glob("*.png"))
        mb = sum(p.stat().st_size for p in pngs) / 1e6
        print("wrote %s  (%d PNGs, %.0f MB)" % (imgdir, len(pngs), mb))


if __name__ == "__main__":
    main()
