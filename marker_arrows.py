#!/usr/bin/env python3
"""Three-frame displacement arrows over the tracked markers (HANDOFF §13.4).

For every frame k, three poses of each marker are drawn and two arrows connect them:

    baseline  ---- NET arrow ---->  current  ---- STEP arrow ---->  next
    tracks[0][m]                    tracks[k][m]                    tracks[k+1][m]

Then k advances: `current` becomes `next`, `next` moves on one frame. The net arrow is the
end-result displacement so far; the step arrow is the incremental motion into the next
frame, which is what keeps a marker's identity anchored across the sequence.

CORRESPONDENCE IS ALREADY SOLVED -- AND IS NOT RE-SOLVED HERE BY PROXIMITY.
marker_overlay.py produced a gapless roster: `tracks[frame][marker]` has stable indices and
no missing entries (141 markers x 313 frames on mid5 repeat 1). So arrow m is always
tracks[0][m] -> tracks[k][m] -> tracks[k+1][m]. Nothing searches for a nearest circle, so
two arrows can never bind to the same circle -- which is the failure this design exists to
prevent.

ARROWS ARE MAGNIFIED AND THE GAIN IS PRINTED ON EVERY FRAME.
Net displacement peaks at ~8.6 px and the frame-to-frame step is often sub-pixel, so at
1-2x view scale an unmagnified step arrow is invisible. Net and step have separate gains;
both are shown in the HUD and in the legend. Never read length off these frames directly.

Outputs (next to the source detections JSON):
  <stem>_arrows.mp4          the video, labelled
  <stem>_final_displacement.png/.pdf   baseline frame + FINAL location of every marker
  <stem>_displacement.csv    per-frame per-marker net displacement (px and mm)

Usage:
  python3 marker_arrows.py                                  # newest detections JSON
  python3 marker_arrows.py --net-gain 6 --step-gain 30
  python3 marker_arrows.py --label-top 8 --view-scale 2
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(REPO, "output")

NET_GAIN    = 6.0     # magnification for baseline->current arrows
STEP_GAIN   = 30.0    # magnification for current->next arrows (usually sub-pixel)
LABEL_TOP   = 6       # annotate this many largest-net-displacement markers
VIEW_TARGET = 1100

C_BASE = (70, 70, 70)      # baseline circle  (grey)
C_CUR  = (40, 90, 255)     # current circle   (blue, matches marker_overlay)
C_NEXT = (0, 190, 255)     # next circle      (cyan)
C_NET  = (60, 230, 90)     # net arrow        (green)
C_STEP = (255, 120, 40)    # step arrow       (orange)
C_TXT  = (235, 235, 235)


def find_detections(arg):
    if arg:
        if not os.path.exists(arg):
            sys.exit("no such file: %s" % arg)
        return arg
    c = glob.glob(os.path.join(OUTPUT, "*_indent", "video", "*_detections.json"))
    if not c:
        sys.exit("no *_detections.json under %s/*_indent/video/ -- run marker_overlay.py"
                 % OUTPUT)
    return max(c, key=os.path.getmtime)


def arrow(img, p0, p1, colour, scale, gain, thickness=1, min_len=3.0):
    """Draw p0 -> p0 + gain*(p1-p0), in view coordinates. Returns the true length (px)."""
    v = np.array(p1, float) - np.array(p0, float)
    true_len = float(np.linalg.norm(v))
    a = (np.array(p0, float) * scale)
    b = a + v * gain * scale
    if np.linalg.norm(b - a) < min_len:          # too short to render an arrowhead
        return true_len
    tipl = min(0.45, 6.0 / max(np.linalg.norm(b - a), 1e-6))
    cv2.arrowedLine(img, (int(round(a[0])), int(round(a[1]))),
                    (int(round(b[0])), int(round(b[1]))), colour, thickness,
                    line_type=cv2.LINE_AA, tipLength=tipl)
    return true_len


def draw_frame(shape_hw, base, cur, nxt, radius_px, scale, out_hw, net_gain, step_gain,
               hud=None, label_idx=(), px_per_mm=1.0):
    h, w = shape_hw
    oh, ow = out_hw
    img = np.zeros((oh, ow, 3), np.uint8)
    rr = max(2, int(round(radius_px * scale)))

    for m in range(base.shape[0]):
        cv2.circle(img, (int(round(base[m, 0] * scale)), int(round(base[m, 1] * scale))),
                   rr, C_BASE, 1, lineType=cv2.LINE_AA)
    for m in range(cur.shape[0]):
        cv2.circle(img, (int(round(cur[m, 0] * scale)), int(round(cur[m, 1] * scale))),
                   rr, C_CUR, max(1, scale // 2), lineType=cv2.LINE_AA)
    if nxt is not None:
        for m in range(nxt.shape[0]):
            cv2.circle(img, (int(round(nxt[m, 0] * scale)),
                             int(round(nxt[m, 1] * scale))),
                       max(2, rr // 2), C_NEXT, 1, lineType=cv2.LINE_AA)

    nets = np.zeros(base.shape[0])
    for m in range(base.shape[0]):
        nets[m] = arrow(img, base[m], cur[m], C_NET, scale, net_gain, 2)
    if nxt is not None:
        for m in range(cur.shape[0]):
            arrow(img, cur[m], nxt[m], C_STEP, scale, step_gain, 1)

    for m in label_idx:
        p = cur[m] * scale
        cv2.putText(img, "%d:%.2fmm" % (m, nets[m] / px_per_mm),
                    (int(p[0]) + rr + 2, int(p[1]) - rr), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, C_TXT, 1, cv2.LINE_AA)

    # legend + HUD
    y = 16
    for txt, col in (("baseline circle", C_BASE), ("current circle", C_CUR),
                     ("next circle", C_NEXT),
                     ("NET arrow  baseline->current  x%.0f" % net_gain, C_NET),
                     ("STEP arrow current->next     x%.0f" % step_gain, C_STEP)):
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        y += 15
    if hud:
        for line in hud:
            cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TXT, 1,
                        cv2.LINE_AA)
            y += 16
    cv2.putText(img, "ARROWS ARE MAGNIFIED - do not read length directly",
                (8, oh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1,
                cv2.LINE_AA)
    return img, nets


def pose_figure(path_png, path_pdf, base, final, radius_px, px_per_mm, geom, meta,
                title, pose_label, pose_frame, note, fig_gain=8.0):
    """Baseline frame + the displacement location of every marker at one pose.

    fig_gain magnifies the quiver ONLY so direction is legible: at true scale the peak
    displacement is ~7 px across a 640 px field and reads as a dot. The gain is stated in
    the panel title. Circle positions are always true -- only arrow LENGTH is scaled.
    """
    w, h = geom
    d = final - base
    mag = np.linalg.norm(d, axis=1)
    mag_mm = mag / px_per_mm

    fig = plt.figure(figsize=(14.0, 6.4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = fig.add_subplot(1, 2, 1)
    ax.set_facecolor("black")
    for m in range(base.shape[0]):
        ax.add_patch(plt.Circle(base[m], radius_px, fill=False, ec="#888888", lw=0.6))
        ax.add_patch(plt.Circle(final[m], radius_px, fill=False, ec="#2a5aff", lw=1.0))
    q = ax.quiver(base[:, 0], base[:, 1], d[:, 0] * fig_gain, d[:, 1] * fig_gain,
                  mag_mm, angles="xy", scale_units="xy", scale=1.0, cmap="viridis",
                  width=0.004, headwidth=4, headlength=5)
    cb = fig.colorbar(q, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("net displacement (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("grey = baseline (frame 0)   blue = %s (frame %d)   "
                 "arrow length x%.0f (circles at true position)"
                 % (pose_label, pose_frame, fig_gain), fontsize=9)
    ax.set_xlabel("sensor x (px)", fontsize=8)
    ax.set_ylabel("sensor y (px)", fontsize=8)
    ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(mag_mm, bins=24, color="#4c72b0")
    ax2.set_xlabel("net displacement (mm)", fontsize=8)
    ax2.set_ylabel("markers", fontsize=8)
    ax2.set_title("distribution over %d markers" % base.shape[0], fontsize=9)
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.axis("off")
    order = np.argsort(-mag_mm)[:8]
    txt = ["markers: %d   baseline = frame 0   %s = frame %d"
           % (base.shape[0], pose_label.lower(), pose_frame),
           "scale: %.2f px/mm  (marker r = %.2f px = %.2f mm)"
           % (px_per_mm, radius_px, radius_px / px_per_mm),
           "",
           "net displacement:  mean %.3f mm   median %.3f mm" % (mag_mm.mean(),
                                                                 np.median(mag_mm)),
           "                   max  %.3f mm   min    %.3f mm" % (mag_mm.max(),
                                                                 mag_mm.min()),
           "",
           "largest movers (index: mm):",
           "  " + "   ".join("%d:%.3f" % (m, mag_mm[m]) for m in order[:4]),
           "  " + "   ".join("%d:%.3f" % (m, mag_mm[m]) for m in order[4:8]),
           "",
           ] + note
    ax3.text(0.0, 1.0, "\n".join(txt), fontsize=8, va="top", family="monospace")

    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.97, wspace=0.22,
                        hspace=0.45)
    fig.savefig(path_png, dpi=170)
    with PdfPages(path_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return mag_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default=None)
    ap.add_argument("--net-gain", type=float, default=NET_GAIN)
    ap.add_argument("--step-gain", type=float, default=STEP_GAIN)
    ap.add_argument("--label-top", type=int, default=LABEL_TOP)
    ap.add_argument("--view-scale", type=int, default=0)
    ap.add_argument("--fig-gain", type=float, default=8.0,
                    help="arrow-length magnification in the STATIC figures (default 8)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    dj = find_detections(args.detections)
    d = json.load(open(dj))
    if "tracks" not in d:
        sys.exit("%s has no 'tracks' -- regenerate with the current marker_overlay.py" % dj)
    T = np.array(d["tracks"], float)              # (frames, markers, 2)
    n, M = T.shape[0], T.shape[1]
    w, h = d["geometry"]
    R = float(d["marker_radius_px"])
    px_per_mm = float(d["scale_px_per_mm"])
    fps = float(d["fps"])
    phases = d.get("frame_phase", ["-"] * n)
    fz = np.array(d.get("frame_Fz_N_tared", [0.0] * n), float)

    print("Detections : %s" % dj)
    print("  tracks %s  (%d markers x %d frames), no gaps: %s"
          % (T.shape, M, n, not bool(np.isnan(T).any())))
    print("  radius %.2f px, scale %.2f px/mm, %.0f fps" % (R, px_per_mm, fps))

    base = T[0]
    net = np.linalg.norm(T - base, axis=2)        # (frames, markers) px
    print("  net displacement: max %.2f px (%.3f mm) at frame %d"
          % (net.max(), net.max() / px_per_mm, int(np.argmax(net.max(axis=1)))))

    out_dir = os.path.dirname(dj)
    stem = os.path.basename(dj).replace("_detections.json", "")

    # ---- per-frame per-marker displacement CSV
    csv_path = os.path.join(out_dir, stem + "_displacement.csv")
    with open(csv_path, "w") as f:
        f.write("frame,phase,Fz_N_tared,marker,x_px,y_px,dx_px,dy_px,net_px,net_mm\n")
        for k in range(n):
            for m in range(M):
                dx, dy = T[k, m] - base[m]
                f.write("%d,%s,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f\n"
                        % (k, phases[k], fz[k], m, T[k, m, 0], T[k, m, 1], dx, dy,
                           net[k, m], net[k, m] / px_per_mm))
    print("  wrote %s" % csv_path)

    # ---- static figure: baseline + FINAL location of every marker
    png = os.path.join(out_dir, stem + "_final_displacement.png")
    pdf = os.path.join(out_dir, stem + "_final_displacement.pdf")
    mag_mm = pose_figure(png, pdf, base, T[-1], R, px_per_mm, (w, h), d,
                         "%s -- baseline (frame 0) vs FINAL (frame %d)"
                         % (d.get("source_mkv", stem), n - 1),
                         "FINAL", n - 1,
                         ["NOTE the final frame is POST-RETRACT, so what remains here is",
                          "elastomer recovery error, not the peak. Peak is in the dwell",
                          "-- see the _peak_displacement figure."],
                         fig_gain=args.fig_gain)
    print("  wrote %s" % png)
    print("  final displacement: mean %.3f mm, median %.3f mm, max %.3f mm"
          % (mag_mm.mean(), np.median(mag_mm), mag_mm.max()))

    # peak-of-dwell figure too -- the physically interesting pose
    peak_frames = d.get("peak_frames") or []
    if peak_frames:
        pk = int(peak_frames[len(peak_frames) // 2])
        png2 = os.path.join(out_dir, stem + "_peak_displacement.png")
        pdf2 = os.path.join(out_dir, stem + "_peak_displacement.pdf")
        mag_pk = pose_figure(png2, pdf2, base, T[pk], R, px_per_mm, (w, h), d,
                             "%s -- baseline (frame 0) vs PEAK DWELL (frame %d, Fz=%.3f N)"
                             % (d.get("source_mkv", stem), pk, fz[pk]),
                             "PEAK DWELL", pk,
                             ["This is the physically meaningful pose: maximum indentation",
                              "at Fz = %.3f N. Compare with the _final figure to separate" % fz[pk],
                              "elastic response from unrecovered offset."],
                             fig_gain=args.fig_gain)
        print("  wrote %s" % png2)
        print("  peak  displacement: mean %.3f mm, median %.3f mm, max %.3f mm"
              % (mag_pk.mean(), np.median(mag_pk), mag_pk.max()))

    if args.no_video:
        return

    # ---- the arrow video
    scale = args.view_scale or max(1, int(round(VIEW_TARGET / max(w, h))))
    vw, vh = w * scale, h * scale
    vw += vw % 2
    vh += vh % 2
    label_idx = list(np.argsort(-net[-1])[:max(0, args.label_top)])
    print("\nRendering arrow video (%dx%d, %dx, net x%.0f, step x%.0f) ..."
          % (vw, vh, scale, args.net_gain, args.step_gain))

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (vw, vh),
           "-r", "%.6f" % fps, "-i", "-", "-c:v", "libx264", "-crf", "0",
           "-preset", "medium", "-pix_fmt", "yuv420p",
           os.path.join(out_dir, stem + "_arrows.mp4")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n):
        nxt = T[k + 1] if k + 1 < n else None
        hud = ["frame %3d/%d   %-8s   Fz = %+.3f N" % (k, n - 1, phases[k], fz[k]),
               "net displacement: mean %.3f mm   max %.3f mm"
               % (net[k].mean() / px_per_mm, net[k].max() / px_per_mm),
               "%d markers, fixed indices (no proximity matching)" % M]
        img, _ = draw_frame((h, w), base, T[k], nxt, R, scale, (vh, vw),
                            args.net_gain, args.step_gain, hud=hud,
                            label_idx=label_idx, px_per_mm=px_per_mm)
        p.stdin.write(np.ascontiguousarray(img).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    ap_path = os.path.join(out_dir, stem + "_arrows.mp4")
    print("  wrote %s (%.2f MB)" % (ap_path, os.path.getsize(ap_path) / 1e6))


if __name__ == "__main__":
    main()
