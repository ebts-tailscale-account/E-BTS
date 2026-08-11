#!/usr/bin/env python3
"""Render one indentation's event stream to LOSSLESS video (and an annotated preview).

FRAME RATE vs ACCUMULATION TIME -- these are independent knobs:
  * accumulation (--accum-us) is how much event history each frame integrates.
  * frame rate (--fps) is how often a frame is emitted.
Non-overlapping 40 ms frames give only 25 fps. Sliding the 40 ms window at a 16.67 ms
stride gives 60 fps instead, and -- because the illumination is strobed at 25 Hz by the
Arduino two-phase driver (40 ms period) -- a 40 ms window always contains EXACTLY ONE
full illumination cycle whatever its phase, so every frame carries the same brightness
content. Consecutive 60 fps frames then share 23.3 ms of events (58% overlap), so the
video is smooth but the effective temporal resolution of anything measured from it is
still 40 ms. That is a property of the accumulation, not a defect of the encoding.

GRAY VALUE == EVENT COUNT, with no scaling at all. Counts in a 40 ms window are small
(median 1, global max 25 here), so they fit in 8 bits directly. That makes the analysis
video an exactly invertible event-count image: nothing to un-normalise, and no
saturation of the marker cores, which is where sub-pixel centroid precision lives.
(Scaling to a 99.5th-percentile vmax instead clipped 0.8% of nonzero pixels -- precisely
the marker centres.) Per-frame autoscaling would be worse still: a pixel's intensity
would depend on the rest of its frame.

Outputs (default next to the run's _indent output dir):
  <name>.mkv           FFV1 gray8, LOSSLESS, gray == event count -- USE THIS FOR ANALYSIS
                       (verified: ffmpeg-decoded frames are byte-identical to the source
                       counts)
  <name>_preview.mp4   x264 -crf 0, contrast-stretched so it is actually visible
  <name>_annotated.mp4 stretched + time / phase / Fz burned in -- FOR VIEWING ONLY, the
                       text overwrites pixels, never measure from this one
  <name>.json          per-frame window centre, phase and tared Fz -- the join key for
                       the displacement-vs-force plots

Usage:
  python3 event_video.py                          # newest mid*/ run, repeat 1, 60 fps
  python3 event_video.py --repeat 1 --fps 60
  python3 event_video.py --roi X Y W H            # crop (comes later)
  python3 event_video.py --no-annotated
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(REPO, "recordings")
OUTPUT = os.path.join(REPO, "output")

ACCUM_US_DEFAULT = 40000     # one full 25 Hz illumination cycle
FPS_DEFAULT      = 60.0
PAD_PRE_S        = 0.08      # a little quiet before the tare phase starts
PAD_POST_S       = 0.10      # ... and after retract ends
VMAX_PCTL        = 99.5      # global normalisation percentile over nonzero pixels
SENSOR_W, SENSOR_H = 640, 480


def resolve_run(arg):
    if arg:
        p = arg if os.path.isdir(arg) else os.path.join(RECORDINGS, arg)
        if not os.path.isdir(p):
            sys.exit("no such run folder: %s" % p)
        return p
    c = [d for d in glob.glob(os.path.join(RECORDINGS, "*")) if os.path.isdir(d)
         and os.path.exists(os.path.join(d, "camera.raw"))
         and os.path.basename(d).startswith("mid")]
    if not c:
        sys.exit("no mid*/ run folder with a camera.raw under %s" % RECORDINGS)
    return max(c, key=os.path.getmtime)


def phase_windows(run_dir, repeat0):
    """Camera-relative phase windows + contact onset for one repeat (0-based)."""
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    ft = np.genfromtxt(os.path.join(run_dir, "ft.csv"), delimiter=",", names=True)
    fr = np.genfromtxt(os.path.join(run_dir, "franka.csv"), delimiter=",", names=True,
                       dtype=None, encoding="utf-8")
    cam = float(ft["unix_time_s"][0])
    o0 = float(meta.get("tactile_minus_workstation_offset_s", 0.0))
    o1 = float(meta.get("offset_after_s", o0))
    t = fr["unix_time_s"].astype(float)
    frac = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    t_ws = t - (o0 + frac * (o1 - o0))

    ph = fr["phase"].astype(str)
    idx = fr["point_index"].astype(int)
    win = {}
    for p in ("travel", "tare", "dip", "dwell", "retract"):
        m = (idx == repeat0) & (ph == p)
        if m.any():
            win[p] = (float(t_ws[m].min() - cam), float(t_ws[m].max() - cam))
    if "dwell" not in win:
        sys.exit("repeat %d has no dwell phase in this run." % (repeat0 + 1))

    # contact onset from the F/T, tared on this repeat's own tare window
    if "tare" in win:
        tm = ((ft["unix_time_s"] - cam >= win["tare"][0])
              & (ft["unix_time_s"] - cam <= win["tare"][1]))
        base, sig = float(ft["Fz_N"][tm].mean()), float(ft["Fz_N"][tm].std())
    else:
        base, sig = float(ft["Fz_N"][:1000].mean()), float(ft["Fz_N"][:1000].std())
    thr = max(5.0 * sig, 0.05)
    lo = win.get("tare", win["dwell"])[1]
    sm = ((ft["unix_time_s"] - cam >= lo) & (ft["unix_time_s"] - cam <= win["dwell"][1]))
    fz = ft["Fz_N"][sm] - base
    tt = ft["unix_time_s"][sm] - cam
    hit = np.where(np.abs(fz) > thr)[0]
    contact = float(tt[hit[0]]) if hit.size else win["dwell"][0]
    return win, contact, base, ft, cam


def decode(raw, t0_us, t1_us):
    try:
        from metavision_core.event_io import EventsIterator
    except ImportError:
        sys.exit("metavision_core not importable -- cannot decode camera.raw")
    xs, ys, ts = [], [], []
    print("Decoding %s over %.3f..%.3f s ..."
          % (os.path.basename(raw), t0_us / 1e6, t1_us / 1e6))
    for evs in EventsIterator(input_path=raw, delta_t=100000):
        if evs.size == 0:
            continue
        if evs["t"][0] >= t1_us:
            break
        if evs["t"][-1] < t0_us:
            continue
        m = (evs["t"] >= t0_us) & (evs["t"] < t1_us)
        if m.any():
            xs.append(evs["x"][m].astype(np.int32))
            ys.append(evs["y"][m].astype(np.int32))
            ts.append(evs["t"][m].astype(np.int64))
    if not xs:
        sys.exit("no events in that window")
    x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts)
    o = np.argsort(t, kind="stable")
    print("  %d events" % t.size)
    return x[o], y[o], t[o]


def build_frames(x, y, t, t0_us, n_frames, accum_us, stride_us, roi):
    """(n_frames, H, W) uint16 event counts. Sliding window, fixed geometry."""
    if roi:
        rx, ry, rw, rh = roi
        keep = (x >= rx) & (x < rx + rw) & (y >= ry) & (y < ry + rh)
        x, y, t = x[keep] - rx, y[keep] - ry, t[keep]
        H, W = rh, rw
    else:
        H, W = SENSOR_H, SENSOR_W
    flat = y.astype(np.int64) * W + x.astype(np.int64)
    frames = np.zeros((n_frames, H, W), np.uint16)
    for k in range(n_frames):
        a = t0_us + k * stride_us
        b = a + accum_us
        i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
        if i1 > i0:
            bc = np.bincount(flat[i0:i1], minlength=H * W)
            frames[k] = np.minimum(bc, 65535).astype(np.uint16).reshape(H, W)
    return frames


def to_gray8_raw(frames):
    """IDENTITY mapping: gray value == event count. Exactly invertible.

    Event counts in a 40 ms window are small (median 1, max ~20), so they fit in 8 bits
    with room to spare. Storing them unscaled means the analysis video IS the event-count
    image -- no normalisation to undo, and no saturation of the marker cores, which is
    exactly where sub-pixel centroid precision comes from. (Scaling to a 99.5th-percentile
    vmax clipped 0.8% of the nonzero pixels here, and those were the marker centres.)
    """
    gmax = int(frames.max())
    clipped = int((frames > 255).sum())
    if clipped:
        print("  [WARN] %d pixel-samples exceed 255 events and are clipped" % clipped)
    g = np.minimum(frames, 255).astype(np.uint8)
    print("  analysis video: IDENTITY mapping, gray == event count "
          "(global max %d events/px, no saturation)" % gmax)
    return g, gmax


def to_gray8_scaled(frames, gmax):
    """Contrast-stretched copy for HUMAN VIEWING only.

    Stretching by the global max makes this far too dim: gmax is set by a couple of hot
    pixels, while a typical marker pixel sees only 1-6 events, so markers land at gray
    10-60. Stretch on a robust percentile instead -- it saturates a fraction of a percent
    of pixels, which is fine here precisely because nothing is ever measured from this
    copy (the identity-mapped .mkv is).
    """
    nz = frames[frames > 0]
    s = max(1.0, float(np.percentile(nz, VMAX_PCTL))) if nz.size else 1.0
    print("  viewing copies: contrast stretched at %.1f events/px (%.1fth pctl of "
          "nonzero); global max was %d" % (s, VMAX_PCTL, gmax))
    return np.clip(frames.astype(np.float32) * (255.0 / s), 0, 255).astype(np.uint8)


def upscale_for_view(frames8, scale):
    """Nearest-neighbour integer upscale, then pad to even dims.

    Needed for small ROIs: a 71x37 crop is unwatchable at native size, and x264's
    yuv420p requires EVEN width/height, which 71x37 is not. np.repeat is exact
    nearest-neighbour, so no pixel is invented or blended -- the block structure of the
    original stays visible. Analysis never touches these frames; the .mkv keeps the
    native geometry.
    """
    if scale > 1:
        frames8 = np.repeat(np.repeat(frames8, scale, axis=1), scale, axis=2)
    n, h, w = frames8.shape
    ph, pw = h % 2, w % 2
    if ph or pw:
        out = np.zeros((n, h + ph, w + pw), np.uint8)
        out[:, :h, :w] = frames8
        frames8 = out
    return frames8


def auto_view_scale(w, h, target=560, cap=16):
    """Smallest integer factor that gets the long edge near `target`."""
    if max(w, h) >= target:
        return 1
    return int(min(cap, max(1, round(target / max(w, h)))))


def encode(frames8, path, fps, lossless_mkv=True):
    H, W = frames8.shape[1:]
    if lossless_mkv:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "gray", "-s", "%dx%d" % (W, H),
               "-r", "%.6f" % fps, "-i", "-",
               "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray", path]
    else:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "gray", "-s", "%dx%d" % (W, H),
               "-r", "%.6f" % fps, "-i", "-",
               "-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
               "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(frames8.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed for %s" % path)
    print("  wrote %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))


def encode_annotated(frames8, path, fps, times, phases, fzs, contact):
    """Burn time/phase/Fz in. VIEWING ONLY -- text overwrites pixels."""
    from PIL import Image, ImageDraw
    H, W = frames8.shape[1:]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", "%.6f" % fps, "-i", "-",
           "-c:v", "libx264", "-crf", "0", "-preset", "medium",
           "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(frames8.shape[0]):
        im = Image.fromarray(frames8[k]).convert("RGB")
        d = ImageDraw.Draw(im)
        rel = times[k] - contact
        d.text((6, 4), "t=%+.3f s from contact   %s" % (rel, phases[k]), fill=(0, 255, 255))
        d.text((6, 16), "Fz=%+.3f N" % fzs[k], fill=(255, 200, 80))
        # force bar, 0..1.5 N
        bl = int(np.clip(abs(fzs[k]) / 1.5, 0, 1) * (W - 20))
        d.rectangle([6, H - 12, 6 + bl, H - 6], fill=(255, 80, 80))
        p.stdin.write(np.asarray(im, np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed for %s" % path)
    print("  wrote %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--repeat", type=int, default=1, help="which indentation (1-based)")
    ap.add_argument("--accum-us", type=int, default=ACCUM_US_DEFAULT)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--roi", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"), help="crop to this ROI")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--view-scale", type=int, default=0,
                    help="integer nearest-neighbour upscale for the VIEWING copies "
                         "only (0 = auto; the .mkv always stays native size)")
    ap.add_argument("--no-annotated", action="store_true")
    ap.add_argument("--no-preview-mp4", action="store_true")
    args = ap.parse_args()

    run_dir = resolve_run(args.run)
    run_id = os.path.basename(run_dir.rstrip("/"))
    r0 = args.repeat - 1
    win, contact, fz_base, ft, cam = phase_windows(run_dir, r0)

    t_start = win.get("tare", win["dwell"])[0] - PAD_PRE_S
    t_end = win.get("retract", win["dwell"])[1] + PAD_POST_S
    stride_us = int(round(1e6 / args.fps))
    # frames must all be fully inside the decoded span
    n_frames = int(np.floor(((t_end - t_start) * 1e6 - args.accum_us) / stride_us)) + 1
    if n_frames < 2:
        sys.exit("window too short for that accumulation/fps")

    dur = n_frames * stride_us / 1e6
    print("Run %s, repeat %d" % (run_id, args.repeat))
    print("  clip           : %.3f .. %.3f s (camera clock), %.2f s of video"
          % (t_start, t_end, dur))
    print("  contact onset  : %.3f s" % contact)
    print("  accumulation   : %d us (%.1f ms) = %.2f illumination cycles at 25 Hz"
          % (args.accum_us, args.accum_us / 1e3, args.accum_us / 40000.0))
    print("  fps            : %.2f  (stride %.2f ms, overlap %.1f%%)"
          % (args.fps, stride_us / 1e3,
             100.0 * max(0.0, args.accum_us - stride_us) / args.accum_us))
    print("  frames         : %d" % n_frames)
    if args.roi:
        print("  ROI            : x=%d y=%d w=%d h=%d" % tuple(args.roi))

    t0_us = int(round(t_start * 1e6))
    t1_us = t0_us + (n_frames - 1) * stride_us + args.accum_us
    x, y, t = decode(os.path.join(run_dir, "camera.raw"), t0_us, t1_us)
    frames = build_frames(x, y, t, t0_us, n_frames, args.accum_us, stride_us, args.roi)
    g8, gmax = to_gray8_raw(frames)          # analysis: gray == event count, native size
    g8v = to_gray8_scaled(frames, gmax)      # viewing: contrast stretched
    fh, fw = g8.shape[1:]
    vs = args.view_scale or auto_view_scale(fw, fh)
    g8v = upscale_for_view(g8v, vs)
    if vs > 1 or g8v.shape[1:] != (fh, fw):
        print("  viewing copies upscaled %dx (nearest) to %dx%d; the .mkv stays %dx%d"
              % (vs, g8v.shape[2], g8v.shape[1], fw, fh))

    out_dir = args.out_dir or os.path.join(OUTPUT, run_id + "_indent", "video")
    os.makedirs(out_dir, exist_ok=True)
    name = args.name or ("repeat%02d_accum%dus_%gfps%s"
                         % (args.repeat, args.accum_us, args.fps,
                            "_roi%d-%d-%dx%d" % tuple(args.roi) if args.roi else ""))

    encode(g8, os.path.join(out_dir, name + ".mkv"), args.fps, lossless_mkv=True)
    if not args.no_preview_mp4:
        encode(g8v, os.path.join(out_dir, name + "_preview.mp4"), args.fps,
               lossless_mkv=False)

    # per-frame time / phase / Fz, sampled at each frame's window CENTRE
    centres = t_start + np.arange(n_frames) * stride_us / 1e6 + args.accum_us / 2e6
    tft = ft["unix_time_s"] - cam
    fzs = np.interp(centres, tft, ft["Fz_N"] - fz_base)
    phases = []
    for c in centres:
        lab = "-"
        for p in ("travel", "tare", "dip", "dwell", "retract"):
            if p in win and win[p][0] <= c <= win[p][1]:
                lab = p
                break
        phases.append(lab)
    if not args.no_annotated:
        encode_annotated(g8v, os.path.join(out_dir, name + "_annotated.mp4"), args.fps,
                         centres, phases, fzs, contact)

    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump({
            "run_id": run_id, "repeat": args.repeat,
            "accum_us": args.accum_us, "fps": args.fps, "stride_us": stride_us,
            "overlap_frac": max(0.0, args.accum_us - stride_us) / args.accum_us,
            "n_frames": n_frames, "clip_start_s": t_start, "clip_end_s": t_end,
            "contact_onset_s": contact, "roi": args.roi,
            "frame_geometry": list(g8.shape[1:][::-1]),
            "global_max_events_per_px": gmax,
            "normalisation": ("<name>.mkv is an IDENTITY mapping -- gray value == event "
                              "count, exactly invertible, no saturation. The _preview "
                              "and _annotated mp4s are contrast-stretched for viewing "
                              "only; never measure from those."),
            "frame_window_centres_s": [float(v) for v in centres],
            "frame_phase": phases,
            "frame_Fz_N_tared": [float(v) for v in fzs],
            "fz_tare_baseline_N": fz_base,
            "note": ("frame k integrates events in [clip_start + k*stride, +accum). "
                     "Fz is sampled at each window centre and tared on this repeat's "
                     "own tare phase."),
        }, f, indent=2)
    print("  wrote %s.json (per-frame time/phase/Fz, for the displacement plots)"
          % os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
