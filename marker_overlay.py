#!/usr/bin/env python3
"""Detect visible markers and render a BLUE-CIRCLE-ONLY overlay, FIXED circle size.

Step 2 of the displacement-vs-force pipeline (HANDOFF §13.3). Reads the LOSSLESS,
identity-mapped .mkv from event_video.py (gray value == event count) plus its .json
sidecar (per-frame phase / tared Fz), finds the marker discs, and writes a video of ONLY
blue circles on black -- no event pixels -- so detection can be judged on its own and the
marker pair of interest can be picked.

FIXED CIRCLE SIZE. The markers are physically 0.75 mm in RADIUS, so their image size
cannot vary: every circle is drawn at ONE radius. Letting the radius follow each blob's
thresholded area (the first version did) made identical markers appear to jiggle in size,
which is threshold noise, not signal.

The radius is measured, not guessed. Averaging the `tare` frames gives a high-SNR baseline
image on which blob geometry is stable -- 170 blobs, median radius 9.6-9.8 px, bbox
19x19 px, at thresholds from 1.25 to 1.50, i.e. threshold-insensitive. At 0.75 mm radius
that is ~12.8 px/mm, making the measured 31 px column pitch ~2.4 mm. --radius-px overrides.

SIZE GATE. A blob too small to BE a marker is discarded rather than circled, so markers
that are deliberately not visible stay uncircled. Gates are fractions of the expected
marker area pi*R^2:
  area < MIN_AREA_FRAC * pi R^2         -> too small to be a marker  (dropped)
  area > MAX_AREA_FRAC * pi R^2         -> merged neighbours / smear  (dropped)
  bbox aspect outside 1/ASPECT..ASPECT  -> not circular              (dropped)
The aspect gate also suppresses the fast-motion artefact where a smeared marker fragments:
without it, `dip`/`retract` frames spiked from ~161 to 191 detections.

Detection uses LOCAL-CONTRAST NORMALISATION (blur / large-sigma blur). Marker brightness
varies ~9x across the frame, so a single global threshold found only 84-89 of ~165 markers
and missed y 80-160 and 240-280 entirely.

Centres are INTENSITY-WEIGHTED centroids (weights = event counts) -> sub-pixel.

Usage:
  python3 marker_overlay.py --probe        # measured radius + detection stats, then exit
  python3 marker_overlay.py                # the blue-circle-only video
  python3 marker_overlay.py --radius-px 10 --filled
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import cv2

REPO = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(REPO, "output")

MARKER_RADIUS_MM = 0.75    # physical marker radius -- fixed by the sensor design

BLUR_SIGMA    = 3.0        # px; smooths sparse event counts into solid discs
BG_SIGMA      = 40.0       # px; illumination envelope -- must exceed the ~31 px pitch
THRESH_REL    = 1.25       # keep pixels this many x their LOCAL mean
MIN_AREA_FRAC = 0.35       # of pi*R^2 -- below this a blob is not a marker
MAX_AREA_FRAC = 2.20       # of pi*R^2 -- above this it is merged neighbours / smear
ASPECT_MAX    = 1.70       # bbox w/h must lie within 1/ASPECT_MAX .. ASPECT_MAX
VIEW_TARGET   = 900
BLUE_RGB      = (40, 90, 255)


def find_mkv(arg):
    if arg:
        if not os.path.exists(arg):
            sys.exit("no such file: %s" % arg)
        return arg
    c = glob.glob(os.path.join(OUTPUT, "*_indent", "video", "*.mkv"))
    c = [p for p in c if "_roi" not in os.path.basename(p)] or c
    if not c:
        sys.exit("no event-video .mkv under %s/*_indent/video/" % OUTPUT)
    return max(c, key=os.path.getmtime)


def read_gray_video(path):
    """Decode gray8 video -> (n, H, W) uint8. Values ARE event counts (identity map)."""
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                        capture_output=True, text=True)
    w, h = [int(v) for v in pr.stdout.strip().split(",")[:2]]
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path,
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True)
    a = np.frombuffer(out.stdout, np.uint8)
    n = a.size // (w * h)
    return a[:n * w * h].reshape(n, h, w), w, h


def _mask(frame, blur_sigma, thresh_rel, bg_sigma):
    f = frame.astype(np.float32)
    b = cv2.GaussianBlur(f, (0, 0), blur_sigma)
    if (b > 0).sum() < 50:
        return f, None
    bg = cv2.GaussianBlur(f, (0, 0), bg_sigma)
    return f, ((b / (bg + 0.05)) > thresh_rel).astype(np.uint8)


def measure_radius(avg_img, blur_sigma=BLUR_SIGMA, thresh_rel=THRESH_REL,
                   bg_sigma=BG_SIGMA, min_area=60):
    """Median blob radius (px) on the high-SNR averaged baseline image."""
    _f, m = _mask(avg_img, blur_sigma, thresh_rel, bg_sigma)
    if m is None:
        return None, 0
    nl, _lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    r = [np.sqrt(st[i, cv2.CC_STAT_AREA] / np.pi) for i in range(1, nl)
         if st[i, cv2.CC_STAT_AREA] >= min_area]
    if not r:
        return None, 0
    return float(np.median(r)), len(r)


def detect(frame, radius_px, blur_sigma=BLUR_SIGMA, thresh_rel=THRESH_REL,
           bg_sigma=BG_SIGMA, min_frac=MIN_AREA_FRAC, max_frac=MAX_AREA_FRAC,
           aspect_max=ASPECT_MAX, stats_out=None):
    """-> list of (cx, cy). Markers are fixed size, so no per-blob radius is returned."""
    f, m = _mask(frame, blur_sigma, thresh_rel, bg_sigma)
    if m is None:
        return []
    expect = np.pi * radius_px ** 2
    amin, amax = min_frac * expect, max_frac * expect
    nl, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    n_small = n_big = n_shape = 0
    for i in range(1, nl):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < amin:
            n_small += 1
            continue
        if area > amax:
            n_big += 1
            continue
        bw = int(st[i, cv2.CC_STAT_WIDTH])
        bh = int(st[i, cv2.CC_STAT_HEIGHT])
        asp = bw / max(bh, 1)
        if asp > aspect_max or asp < 1.0 / aspect_max:
            n_shape += 1
            continue
        msk = lab == i
        wgt = f[msk]
        tot = float(wgt.sum())
        ys, xs = np.nonzero(msk)
        if tot > 0:
            out.append((float((xs * wgt).sum() / tot), float((ys * wgt).sum() / tot)))
        else:
            out.append((float(cent[i][0]), float(cent[i][1])))
    if stats_out is not None:
        for key, val in (("small", n_small), ("big", n_big), ("shape", n_shape)):
            stats_out[key] = stats_out.get(key, 0) + val
    out.sort(key=lambda t: (round(t[1] / 20.0), t[0]))
    return out


def mean_positions(dets, idxs, match_r=14.0):
    """Mean marker positions over frames, clustered by proximity.

    Per-frame detections are unordered, so seed from the frame with the most detections and
    accumulate every other frame's onto the nearest seed within match_r. A seed hit in fewer
    than half the frames is dropped as spurious.
    """
    if not idxs:
        return []
    seed_i = max(idxs, key=lambda k: len(dets[k]))
    if not dets[seed_i]:
        return []
    S = np.array(dets[seed_i], float)
    acc = [[S[j].copy()] for j in range(S.shape[0])]
    for k in idxs:
        if k == seed_i or not dets[k]:
            continue
        for p in dets[k]:
            q = np.array(p, float)
            d = np.linalg.norm(S - q, axis=1)
            j = int(np.argmin(d))
            if d[j] <= match_r:
                acc[j].append(q)
    out = []
    for pts in acc:
        if len(pts) < 0.5 * len(idxs):
            continue
        P = np.vstack(pts)
        out.append({"x": float(P[:, 0].mean()), "y": float(P[:, 1].mean()),
                    "sx": float(P[:, 0].std()), "sy": float(P[:, 1].std()),
                    "n": int(P.shape[0])})
    out.sort(key=lambda d: (round(d["y"] / 20.0), d["x"]))
    return out


def track_markers(dets, base_pos, match_r=12.0, max_missing=0):
    """Persistent roster with STABLE INDICES, present in every kept frame.

    Why this exists: a marker that flickers out for even one frame breaks the arrow stage,
    because nearest-blue-circle matching can then bind two different arrows to the SAME
    circle. So identity must come from an index, not from a per-frame proximity search.

    Roster is seeded from the baseline mean positions, then propagated SEQUENTIALLY: each
    marker is matched to the nearest detection to *where it was in the previous frame*, not
    to its baseline position. That matters once displacement accumulates -- anchoring on the
    baseline would lose markers near peak indentation, while frame-to-frame motion at 60 fps
    is well under a pixel.

    Any marker missing in more than max_missing frames is deleted outright (default 0, i.e.
    one dropout is fatal). Survivors therefore appear in EVERY frame.

    Returns (tracks[n_frames, n_kept, 2], kept_idx, n_roster, missing_per_roster).
    """
    n = len(dets)
    m = len(base_pos)
    if n == 0 or m == 0:
        return np.zeros((n, 0, 2)), [], m, []
    tracks = np.full((n, m, 2), np.nan)
    missing = np.zeros(m, int)
    prev = np.array([[p["x"], p["y"]] for p in base_pos], float)
    for k in range(n):
        D = np.array(dets[k], float) if dets[k] else np.zeros((0, 2))
        for j in range(m):
            if D.shape[0] == 0:
                missing[j] += 1
                tracks[k, j] = prev[j]
                continue
            d = np.linalg.norm(D - prev[j], axis=1)
            i = int(np.argmin(d))
            if d[i] <= match_r:
                tracks[k, j] = D[i]
                prev[j] = D[i]          # propagate: next frame searches from here
            else:
                missing[j] += 1
                tracks[k, j] = prev[j]  # hold last known so the search can recover
    kept = [j for j in range(m) if missing[j] <= max_missing]
    return tracks[:, kept, :], kept, m, missing


def draw_circles(hw, pts, radius_px, filled=False, scale=1, out_hw=None):
    """Blue circles of ONE fixed radius on BLACK. No event pixels are drawn."""
    h, w = hw
    oh, ow = out_hw if out_hw else (h * scale, w * scale)
    img = np.zeros((oh, ow, 3), np.uint8)
    rr = max(2, int(round(radius_px * scale)))
    for (cx, cy) in pts:
        c = (int(round(cx * scale)), int(round(cy * scale)))
        cv2.circle(img, c, rr, BLUE_RGB, -1 if filled else max(1, scale // 2 + 1),
                   lineType=cv2.LINE_AA)
        if not filled:
            cv2.circle(img, c, max(1, scale // 3), BLUE_RGB, -1, lineType=cv2.LINE_AA)
    return img


def encode_rgb(gen, path, fps, w, h):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
           "-r", "%.6f" % fps, "-i", "-", "-c:v", "libx264", "-crf", "0",
           "-preset", "medium", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in gen:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed for %s" % path)
    print("  wrote %s (%.2f MB)" % (path, os.path.getsize(path) / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mkv", default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--filled", action="store_true")
    ap.add_argument("--radius-px", type=float, default=0.0,
                    help="fixed circle radius in px (0 = measure from the baseline mean)")
    ap.add_argument("--blur", type=float, default=BLUR_SIGMA)
    ap.add_argument("--thresh-rel", type=float, default=THRESH_REL)
    ap.add_argument("--bg-sigma", type=float, default=BG_SIGMA)
    ap.add_argument("--min-frac", type=float, default=MIN_AREA_FRAC,
                    help="drop blobs below this fraction of pi*R^2 (default %.2f)"
                         % MIN_AREA_FRAC)
    ap.add_argument("--max-frac", type=float, default=MAX_AREA_FRAC)
    ap.add_argument("--aspect-max", type=float, default=ASPECT_MAX)
    ap.add_argument("--match-r", type=float, default=12.0,
                    help="frame-to-frame match radius in px; must stay well under half "
                         "the ~31 px marker pitch (default 12)")
    ap.add_argument("--max-missing", type=int, default=0,
                    help="delete a marker that goes missing in more than this many "
                         "frames (default 0 = one dropout is fatal)")
    ap.add_argument("--view-scale", type=int, default=0)
    args = ap.parse_args()

    mkv = find_mkv(args.mkv)
    side = os.path.splitext(mkv)[0] + ".json"
    if not os.path.exists(side):
        sys.exit("missing sidecar %s (written by event_video.py)" % side)
    meta = json.load(open(side))
    fps = float(meta["fps"])
    phases = meta["frame_phase"]
    fz = np.array(meta["frame_Fz_N_tared"], float)

    print("Video : %s" % mkv)
    frames, w, h = read_gray_video(mkv)
    n = min(frames.shape[0], len(phases))
    print("  %d frames of %dx%d, gray max %d (== event count)"
          % (frames.shape[0], w, h, frames.max()))

    tare_idx = [k for k in range(n) if phases[k] == "tare"]
    dwell_idx = [k for k in range(n) if phases[k] == "dwell"]

    if args.radius_px > 0:
        R = args.radius_px
        print("  fixed radius: %.2f px (given)" % R)
    else:
        src = tare_idx or list(range(min(n, 60)))
        R, nb = measure_radius(frames[src].mean(axis=0), args.blur, args.thresh_rel,
                               args.bg_sigma)
        if not R:
            sys.exit("could not measure a marker radius")
        print("  fixed radius: %.2f px  (median of %d blobs on the mean of %d %s frames)"
              % (R, nb, len(src), "tare" if tare_idx else "leading"))
    print("  implied scale: %.2f px/mm at %.2f mm marker radius"
          % (R / MARKER_RADIUS_MM, MARKER_RADIUS_MM))
    expect = np.pi * R ** 2
    print("  size gate: area %.0f..%.0f px^2 (expected marker %.0f), aspect %.2f..%.2f"
          % (args.min_frac * expect, args.max_frac * expect, expect,
             1 / args.aspect_max, args.aspect_max))

    kw = dict(blur_sigma=args.blur, thresh_rel=args.thresh_rel, bg_sigma=args.bg_sigma,
              min_frac=args.min_frac, max_frac=args.max_frac, aspect_max=args.aspect_max)

    if args.probe:
        print("\nprobe:")
        for k in [0, n // 8, n // 4, n // 2, int(n * 0.62), int(n * 0.8), n - 1]:
            st = {}
            d = detect(frames[k], R, stats_out=st, **kw)
            print("  frame %3d %-8s Fz=%+.3f -> %3d markers   dropped: %d small,"
                  " %d oversized, %d non-circular"
                  % (k, phases[k], fz[k], len(d), st.get("small", 0),
                     st.get("big", 0), st.get("shape", 0)))
        return

    print("\nDetecting in %d frames ..." % n)
    st = {}
    dets = [detect(frames[k], R, stats_out=st, **kw) for k in range(n)]
    counts = np.array([len(d) for d in dets])
    print("markers/frame: min %d  median %d  max %d  std %.2f"
          % (counts.min(), int(np.median(counts)), counts.max(), counts.std()))
    print("dropped over the clip: %d too small, %d oversized, %d non-circular"
          % (st.get("small", 0), st.get("big", 0), st.get("shape", 0)))
    if counts.std() > 2.0:
        print("  [WARN] detection count still varies by more than 2 across frames.")

    peak_idx = []
    if dwell_idx:
        pk = dwell_idx[int(np.argmax(np.abs(fz[dwell_idx])))]
        peak_idx = [k for k in dwell_idx if abs(k - pk) <= max(3, len(dwell_idx) // 6)]
    base_pos = mean_positions(dets, tare_idx)
    peak_pos = mean_positions(dets, peak_idx)
    print("baseline (tare, %d frames) : %d markers" % (len(tare_idx), len(base_pos)))
    print("peak     (dwell, %d frames): %d markers" % (len(peak_idx), len(peak_pos)))
    if base_pos:
        print("  baseline per-frame centroid jitter: sx %.2f px, sy %.2f px"
              % (np.mean([p["sx"] for p in base_pos]),
                 np.mean([p["sy"] for p in base_pos])))

    # ---- persistent roster: same markers, same indices, in EVERY frame
    tracks, kept, n_roster, missing = track_markers(dets, base_pos, args.match_r,
                                                    args.max_missing)
    print("\nroster: %d markers seeded from the baseline" % n_roster)
    print("  kept %d present in all %d frames (max_missing=%d, match_r=%.1f px)"
          % (len(kept), n, args.max_missing, args.match_r))
    print("  deleted %d that dropped out at least once" % (n_roster - len(kept)))
    if n_roster:
        md = np.array(missing)
        print("  dropout distribution: %d perfect, %d missed 1-5, %d missed >5 frames"
              % (int((md == 0).sum()), int(((md >= 1) & (md <= 5)).sum()),
                 int((md > 5).sum())))
    if not kept:
        sys.exit("[ABORT] no marker survived the all-frames requirement -- raise "
                 "--max-missing or --match-r.")
    disp = tracks - tracks[0]
    print("  max |displacement| over the clip: %.2f px (%.3f mm)"
          % (np.nanmax(np.linalg.norm(disp, axis=2)),
             np.nanmax(np.linalg.norm(disp, axis=2)) / (R / MARKER_RADIUS_MM)))

    out_dir = os.path.dirname(mkv)
    stem = os.path.splitext(os.path.basename(mkv))[0]
    scale = args.view_scale or max(1, int(round(VIEW_TARGET / max(w, h))))
    vw, vh = w * scale, h * scale
    vw += vw % 2
    vh += vh % 2

    def gen():
        for k in range(n):
            yield draw_circles((h, w), [tuple(p) for p in tracks[k]], R,
                               filled=args.filled, scale=scale, out_hw=(vh, vw))

    print("\nRendering blue-circle-only video (%dx%d, %dx, fixed r=%.2f px, %d markers "
          "in every frame) ..." % (vw, vh, scale, R, len(kept)))
    encode_rgb(gen(), os.path.join(out_dir, stem + "_circles.mp4"), fps, vw, vh)

    with open(os.path.join(out_dir, stem + "_detections.json"), "w") as f:
        json.dump({
            "source_mkv": os.path.basename(mkv), "n_frames": n, "fps": fps,
            "geometry": [w, h], "view_scale": scale,
            "marker_radius_px": R, "marker_radius_mm": MARKER_RADIUS_MM,
            "scale_px_per_mm": R / MARKER_RADIUS_MM,
            "params": {"blur_sigma": args.blur, "thresh_rel": args.thresh_rel,
                       "bg_sigma": args.bg_sigma, "min_area_frac": args.min_frac,
                       "max_area_frac": args.max_frac, "aspect_max": args.aspect_max,
                       "normalisation": "local contrast: blur(blur_sigma)/blur(bg_sigma)"},
            "centre_definition": "intensity-weighted centroid, weights = event counts",
            "radius_note": ("markers are physically fixed at %.2f mm radius, so ONE radius "
                            "is used for every circle; per-blob radii were threshold noise"
                            % MARKER_RADIUS_MM),
            "dropped_totals": {"too_small": st.get("small", 0),
                               "oversized": st.get("big", 0),
                               "non_circular": st.get("shape", 0)},
            "counts_per_frame": [int(c) for c in counts],
            "raw_frames": [[[round(p[0], 4), round(p[1], 4)] for p in dets[k]]
                           for k in range(n)],
            "roster": {
                "n_seeded": n_roster, "n_kept": len(kept), "kept_indices": kept,
                "match_r_px": args.match_r, "max_missing": args.max_missing,
                "missing_per_seeded": [int(v) for v in missing],
                "note": ("`tracks` is [frame][marker][x,y] with STABLE marker indices and "
                         "NO gaps -- every kept marker exists in every frame. Use the index, "
                         "never a nearest-circle search, to keep arrow identity."),
            },
            "tracks": [[[round(float(tracks[k, j, 0]), 4),
                         round(float(tracks[k, j, 1]), 4)]
                        for j in range(tracks.shape[1])] for k in range(n)],
            "baseline_tare": base_pos, "peak_dwell": peak_pos,
            "tare_frames": tare_idx, "peak_frames": peak_idx,
            "frame_phase": phases[:n],
            "frame_Fz_N_tared": [float(v) for v in fz[:n]],
        }, f, indent=1)
    print("  wrote %s_detections.json" % os.path.join(out_dir, stem))


if __name__ == "__main__":
    main()
