#!/usr/bin/env python3
"""FAST marker detection / baseline / tracking -- a drop-in-semantics copy of the
relevant half of `marker_overlay.py`, ~50x faster, for campaign-scale extraction.

WHY A NEW FILE. `marker_overlay.py` is validated and in use (§13.3), and the repo rule is
"a variant gets a copy, never a flag bolted onto the validated original". Nothing here
changes what a marker centre *is*; only how fast it is computed.

WHAT IS DIFFERENT (and why it is still the same answer):

1. BACKGROUND AT 1/8 SCALE. The reference does `cv2.GaussianBlur(f, (0,0), 40.0)` at
   640x480 -- measured 221 ms/frame, which alone makes 648 indents a 6 hour job. The
   illumination envelope is smooth by construction (sigma 40 px against a 29.4 px marker
   pitch), so it is band-limited far below the 1/8-scale Nyquist. Estimating it as
   resize(INTER_AREA) -> GaussianBlur(sigma=40/8=5) -> resize(INTER_LINEAR) is
   mathematically near-identical and ~50x cheaper. INTER_AREA is a box-average, i.e. an
   extra tiny low-pass, which is harmless on an already-smooth field.

2. VECTORISED CENTROIDS. The reference does `msk = lab == i` per blob -- 211 full-frame
   boolean passes per frame. Here the intensity-weighted centroid is a `np.bincount` over
   the mask pixels only. Same weights (event counts), same sub-pixel definition.

3. cKDTree instead of Python loops in `mean_positions` / `track_markers`. Both are pure
   nearest-neighbour searches; a KD-tree returns the same neighbour the O(n^2) argmin does.
   Sequential propagation is preserved exactly -- within one frame the detections are fixed,
   so updating every roster marker at once is identical to updating them one at a time.

Semantics kept unchanged on purpose: gapless roster, stable indices, max_missing=0,
match_r 12 px for tracking / 14 px for baseline clustering, intensity-weighted sub-pixel
centroids, the same area/aspect gates, the same (round(y/20), x) ordering.

Validate with:  python3 marker_features.py --validate 598,633,324
"""

import os
import sys
import time

import numpy as np
import cv2
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml/ -> repo root
sys.path.insert(0, REPO)   # so marker_overlay.py (still in the root) imports
OUTPUT = os.path.join(REPO, "output")

# --- identical to marker_overlay.py ---------------------------------------------------
MARKER_RADIUS_MM = 0.75
BLUR_SIGMA    = 3.0
BG_SIGMA      = 40.0
THRESH_REL    = 1.25
MIN_AREA_FRAC = 0.35
MAX_AREA_FRAC = 2.20
ASPECT_MAX    = 1.70
BASE_MATCH_R  = 14.0    # mean_positions()
TRACK_MATCH_R = 12.0    # track_markers()

BG_DOWNSCALE  = 8       # the one new knob


# ---------------------------------------------------------------------------- masking

_GRID = {}


def _grid(h, w):
    key = (h, w)
    g = _GRID.get(key)
    if g is None:
        g = np.meshgrid(np.arange(w, dtype=np.float64),
                        np.arange(h, dtype=np.float64))
        _GRID[key] = g
    return g


def _background(f, bg_sigma=BG_SIGMA, down=BG_DOWNSCALE):
    """Illumination envelope, estimated at 1/`down` scale. See module docstring."""
    h, w = f.shape
    sh, sw = max(2, h // down), max(2, w // down)
    small = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), bg_sigma * sw / float(w))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _mask(frame, blur_sigma=BLUR_SIGMA, thresh_rel=THRESH_REL, bg_sigma=BG_SIGMA,
          down=BG_DOWNSCALE):
    f = frame.astype(np.float32)
    b = cv2.GaussianBlur(f, (0, 0), blur_sigma)
    if (b > 0).sum() < 50:
        return f, None
    bg = _background(f, bg_sigma, down)
    return f, ((b / (bg + 0.05)) > thresh_rel).astype(np.uint8)


def measure_radius(avg_img, blur_sigma=BLUR_SIGMA, thresh_rel=THRESH_REL,
                   bg_sigma=BG_SIGMA, min_area=60, down=BG_DOWNSCALE):
    """Median blob radius (px) on the high-SNR averaged baseline image."""
    _f, m = _mask(avg_img, blur_sigma, thresh_rel, bg_sigma, down)
    if m is None:
        return None, 0
    nl, _lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    a = st[1:, cv2.CC_STAT_AREA]
    a = a[a >= min_area]
    if a.size == 0:
        return None, 0
    return float(np.median(np.sqrt(a / np.pi))), int(a.size)


# -------------------------------------------------------------------------- detection

def detect(frame, radius_px, blur_sigma=BLUR_SIGMA, thresh_rel=THRESH_REL,
           bg_sigma=BG_SIGMA, min_frac=MIN_AREA_FRAC, max_frac=MAX_AREA_FRAC,
           aspect_max=ASPECT_MAX, stats_out=None, down=BG_DOWNSCALE):
    """-> (k,2) float64 array of (cx, cy), intensity-weighted sub-pixel centroids."""
    f, m = _mask(frame, blur_sigma, thresh_rel, bg_sigma, down)
    if m is None:
        return np.zeros((0, 2))
    expect = np.pi * radius_px ** 2
    amin, amax = min_frac * expect, max_frac * expect
    nl, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
    if nl <= 1:
        return np.zeros((0, 2))

    area = st[1:, cv2.CC_STAT_AREA].astype(np.float64)
    bw = st[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    bh = np.maximum(st[1:, cv2.CC_STAT_HEIGHT], 1).astype(np.float64)
    asp = bw / bh
    small = area < amin
    big = (~small) & (area > amax)
    shape = (~small) & (~big) & ((asp > aspect_max) | (asp < 1.0 / aspect_max))
    keep = ~(small | big | shape)
    if stats_out is not None:
        for key, val in (("small", int(small.sum())), ("big", int(big.sum())),
                         ("shape", int(shape.sum()))):
            stats_out[key] = stats_out.get(key, 0) + val
    if not keep.any():
        return np.zeros((0, 2))

    # intensity-weighted centroid per label, over mask pixels only
    ys, xs = np.nonzero(m)
    labs = lab[ys, xs]
    wgt = f[ys, xs].astype(np.float64)
    tot = np.bincount(labs, weights=wgt, minlength=nl)
    sx = np.bincount(labs, weights=wgt * xs, minlength=nl)
    sy = np.bincount(labs, weights=wgt * ys, minlength=nl)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = sx / tot
        cy = sy / tot
    bad = ~np.isfinite(cx) | (tot <= 0)          # reference falls back to the plain centroid
    cx[bad] = cent[:, 0][bad]
    cy[bad] = cent[:, 1][bad]

    idx = np.nonzero(keep)[0] + 1
    out = np.stack([cx[idx], cy[idx]], axis=1)
    order = np.lexsort((out[:, 0], np.round(out[:, 1] / 20.0)))
    return out[order]


# --------------------------------------------------------------------------- baseline

def mean_positions(dets, idxs, match_r=BASE_MATCH_R):
    """Mean marker positions over `idxs` frames, clustered by proximity to a seed frame.

    Same rule as marker_overlay.mean_positions: seed from the frame with the most
    detections, snap every other frame's detections onto the nearest seed within match_r,
    drop a seed hit in fewer than half the frames.
    """
    idxs = list(idxs)
    if not idxs:
        return []
    seed_i = max(idxs, key=lambda k: len(dets[k]))
    S = np.asarray(dets[seed_i], float)
    if S.shape[0] == 0:
        return []
    ns = S.shape[0]
    tree = cKDTree(S)
    s1 = S.copy()                      # the seed itself is the first accumulated point
    s2 = S ** 2
    cnt = np.ones(ns, int)
    for k in idxs:
        if k == seed_i:
            continue
        D = np.asarray(dets[k], float)
        if D.shape[0] == 0:
            continue
        d, j = tree.query(D, k=1)
        ok = d <= match_r
        if not ok.any():
            continue
        j = j[ok]
        P = D[ok]
        np.add.at(s1, j, P)
        np.add.at(s2, j, P ** 2)
        np.add.at(cnt, j, 1)

    good = cnt >= 0.5 * len(idxs)
    mean = s1[good] / cnt[good, None]
    var = np.maximum(s2[good] / cnt[good, None] - mean ** 2, 0.0)
    sd = np.sqrt(var)
    out = [{"x": float(mean[i, 0]), "y": float(mean[i, 1]),
            "sx": float(sd[i, 0]), "sy": float(sd[i, 1]),
            "n": int(cnt[good][i])} for i in range(mean.shape[0])]
    out.sort(key=lambda d: (round(d["y"] / 20.0), d["x"]))
    return out


# --------------------------------------------------------------------------- tracking

def track_markers(dets, base_pos, match_r=TRACK_MATCH_R, max_missing=0):
    """Persistent gapless roster with stable indices. Same semantics as marker_overlay.

    Sequential propagation: each roster marker is matched to the detection nearest to
    where it was in the PREVIOUS frame, and a miss holds the last known position so the
    search can recover. Any marker missing more than `max_missing` times is deleted.

    -> (tracks[n_frames, n_kept, 2], kept_idx, n_roster, missing_per_roster)
    """
    n = len(dets)
    m = len(base_pos)
    if n == 0 or m == 0:
        return np.zeros((n, 0, 2)), [], m, []
    tracks = np.full((n, m, 2), np.nan)
    missing = np.zeros(m, int)
    if isinstance(base_pos, np.ndarray):
        prev = np.asarray(base_pos, float).reshape(m, 2).copy()
    else:
        prev = np.array([[p["x"], p["y"]] for p in base_pos], float)
    for k in range(n):
        D = np.asarray(dets[k], float)
        if D.shape[0] == 0:
            missing += 1
            tracks[k] = prev
            continue
        d, i = cKDTree(D).query(prev, k=1)
        ok = d <= match_r
        prev[ok] = D[i[ok]]            # propagate: next frame searches from here
        missing[~ok] += 1
        tracks[k] = prev               # a miss holds the last known position
    kept = [j for j in range(m) if missing[j] <= max_missing]
    return tracks[:, kept, :], kept, m, missing


# ------------------------------------------------------------------ per-indent driver

def process_indent(frames, phases, radius_px=None, match_r=TRACK_MATCH_R, max_missing=0,
                   down=BG_DOWNSCALE, **kw):
    """frames[n,H,W] uint8 + phases[n] int8 (0 tare, 1 dip, 2 dwell) -> dict.

    Returns baseline / peak positions and displacement in px for the middle dwell frame,
    measured from the TARE-MEAN baseline of the tracked roster.
    """
    n = frames.shape[0]
    phases = np.asarray(phases)
    tare_idx = np.nonzero(phases == 0)[0]
    dwell_idx = np.nonzero(phases == 2)[0]
    if tare_idx.size == 0 or dwell_idx.size == 0:
        return None

    if radius_px is None:
        radius_px, _nb = measure_radius(frames[tare_idx].mean(axis=0), down=down, **kw)
        if not radius_px:
            return None

    dets = [detect(frames[k], radius_px, down=down, **kw) for k in range(n)]
    base_pos = mean_positions(dets, list(tare_idx))
    if not base_pos:
        return None
    tracks, kept, n_seed, missing = track_markers(dets, base_pos, match_r, max_missing)
    if not kept:
        return None

    baseline = tracks[tare_idx].mean(axis=0)          # (n_kept, 2) px
    peak_k = int(dwell_idx[len(dwell_idx) // 2])      # middle dwell frame
    peak = tracks[peak_k]
    return {
        "radius_px": float(radius_px),
        "px_per_mm": float(radius_px / MARKER_RADIUS_MM),
        "n_seeded": int(n_seed),
        "n_kept": int(len(kept)),
        "kept": kept,
        "baseline_px": baseline,
        "peak_px": peak,
        "disp_px": peak - baseline,
        "tracks": tracks,
        "dets_per_frame": np.array([d.shape[0] for d in dets]),
        "peak_frame": peak_k,
        "n_frames": n,
    }


# -------------------------------------------------------------------------- features

def shape_features(baseline_px, disp_px, px_per_mm, near_mm=5.0):
    """Magnitude + SHAPE descriptors of one displacement field, everything in mm.

    Plain magnitude features were measured to carry almost no force information
    (R^2 = -0.012), so the shape terms are the point of this function: where the contact
    is, how fast the field decays away from it, how round it is, and how much of it is
    radial (pushing outward from the contact) vs tangential (shear).
    """
    P = np.asarray(baseline_px, float) / px_per_mm          # mm
    D = np.asarray(disp_px, float) / px_per_mm              # mm
    mag = np.linalg.norm(D, axis=1)
    n = mag.size
    f = {}

    # ---- magnitude
    f["disp_mean_mm"] = float(mag.mean())
    f["disp_median_mm"] = float(np.median(mag))
    f["disp_rms_mm"] = float(np.sqrt((mag ** 2).mean()))
    f["disp_max_mm"] = float(mag.max())
    k = min(10, n)
    f["disp_top10_mm"] = float(np.sort(mag)[-k:].mean())
    f["disp_sum_mm"] = float(mag.sum())

    tot = mag.sum()
    if tot <= 0:
        for key in ("cx_mm", "cy_mm", "r50_mm", "r25_mm", "aniso", "axis_major_mm",
                    "axis_minor_mm", "mean_radial_mm", "mean_tangential_mm",
                    "radial_frac", "frac_within_5mm"):
            f[key] = np.nan
        return f

    # ---- displacement-weighted contact centroid
    w = mag
    c = (w[:, None] * P).sum(axis=0) / tot
    f["cx_mm"] = float(c[0])
    f["cy_mm"] = float(c[1])

    R = P - c
    r = np.linalg.norm(R, axis=1)

    # ---- radial decay: distance at which the profile falls to 50% / 25% of peak.
    # The field is sampled at 2.3 mm marker pitch and is noisy, so the profile is a
    # running mean of the 5 markers nearest in radius, not the raw per-marker values.
    o = np.argsort(r)
    rs, ms = r[o], mag[o]
    win = min(5, n)
    ker = np.ones(win) / win
    prof = np.convolve(ms, ker, mode="same")
    # fix the edges of 'same' convolution (they divide by the full window)
    csum = np.concatenate([[0.0], np.cumsum(ms)])
    for i in range(n):
        a = max(0, i - win // 2)
        b = min(n, a + win)
        a = max(0, b - win)
        prof[i] = (csum[b] - csum[a]) / (b - a)
    pk = prof.max()

    def _cross(frac):
        thr = frac * pk
        below = np.nonzero(prof <= thr)[0]
        if below.size == 0:
            return float(rs[-1])
        i = int(below[0])
        if i == 0:
            return float(rs[0])
        r0, r1 = rs[i - 1], rs[i]
        p0, p1 = prof[i - 1], prof[i]
        if p0 == p1:
            return float(r1)
        return float(r0 + (p0 - thr) * (r1 - r0) / (p0 - p1))

    f["r50_mm"] = _cross(0.50)
    f["r25_mm"] = _cross(0.25)

    # ---- anisotropy of the displacement-weighted second-moment matrix
    M = (w[:, None, None] * (R[:, :, None] * R[:, None, :])).sum(axis=0) / tot
    ev = np.linalg.eigvalsh(M)
    l2, l1 = float(max(ev[0], 0.0)), float(max(ev[1], 0.0))   # l1 >= l2
    f["axis_major_mm"] = float(np.sqrt(l1))
    f["axis_minor_mm"] = float(np.sqrt(l2))
    f["aniso"] = float(np.sqrt(l1 / l2)) if l2 > 1e-12 else np.nan

    # ---- radial (divergence-like) vs tangential decomposition about the centroid
    rr = np.maximum(r, 1e-9)
    u = R / rr[:, None]
    radial = (D * u).sum(axis=1)
    tangential = D[:, 0] * (-u[:, 1]) + D[:, 1] * u[:, 0]
    f["mean_radial_mm"] = float(radial.mean())
    f["mean_tangential_mm"] = float(np.abs(tangential).mean())
    ar, at = np.abs(radial).mean(), np.abs(tangential).mean()
    f["radial_frac"] = float(ar / (ar + at)) if (ar + at) > 0 else np.nan

    # ---- concentration
    f["frac_within_5mm"] = float(mag[r <= near_mm].sum() / tot)
    return f


FEATURE_COLS = [
    "disp_mean_mm", "disp_median_mm", "disp_rms_mm", "disp_max_mm", "disp_top10_mm",
    "disp_sum_mm",
    "cx_mm", "cy_mm", "r50_mm", "r25_mm", "aniso", "axis_major_mm", "axis_minor_mm",
    "mean_radial_mm", "mean_tangential_mm", "radial_frac", "frac_within_5mm",
]
MAG_COLS = FEATURE_COLS[:6]
SHAPE_COLS = FEATURE_COLS[6:]


# -------------------------------------------------------------------------- validation

def _validate(point_indices, h5path):
    import h5py
    import marker_overlay as ref

    h5 = h5py.File(h5path, "r")
    ind = h5["/indents"][:]
    print("validating %s against marker_overlay.py" % os.path.basename(__file__))
    print("%-6s %-26s %-26s %s" % ("pt", "reference", "fast", "agreement (px)"))
    all_mean, all_max = [], []
    for pi in point_indices:
        row = ind[ind["point_index"] == pi]
        if row.size == 0:
            print("  point_index %d not in the dataset" % pi)
            continue
        row = row[0]
        o, nfr = int(row["frame_offset"]), int(row["n_frames"])
        frames = h5["/frames"][o:o + nfr]
        phases = h5["/frame_phase"][o:o + nfr]
        tare_idx = list(np.nonzero(phases == 0)[0])
        dwell_idx = list(np.nonzero(phases == 2)[0])
        peak_k = int(dwell_idx[len(dwell_idx) // 2])

        # ---------------- reference
        t0 = time.time()
        R_ref, _ = ref.measure_radius(frames[tare_idx].mean(axis=0))
        d_ref = [ref.detect(frames[k], R_ref) for k in range(nfr)]
        b_ref = ref.mean_positions(d_ref, tare_idx)
        tr_ref, kept_ref, seed_ref, _ = ref.track_markers(d_ref, b_ref, 12.0, 0)
        t_ref = time.time() - t0
        base_ref = tr_ref[tare_idx].mean(axis=0)
        disp_ref = tr_ref[peak_k] - base_ref

        # ---------------- fast
        t0 = time.time()
        R_f, _ = measure_radius(frames[tare_idx].mean(axis=0))
        d_f = [detect(frames[k], R_f) for k in range(nfr)]
        b_f = mean_positions(d_f, tare_idx)
        tr_f, kept_f, seed_f, _ = track_markers(d_f, b_f, 12.0, 0)
        t_fast = time.time() - t0
        base_f = tr_f[tare_idx].mean(axis=0)
        disp_f = tr_f[peak_k] - base_f

        # ---------------- match the two rosters by baseline position
        if base_f.shape[0] and base_ref.shape[0]:
            dd, jj = cKDTree(base_f).query(base_ref, k=1)
            ok = dd <= 3.0
            dm_ref = np.linalg.norm(disp_ref[ok], axis=1)
            dm_f = np.linalg.norm(disp_f[jj[ok]], axis=1)
            vdiff = np.linalg.norm(disp_ref[ok] - disp_f[jj[ok]], axis=1)
            adiff = np.abs(dm_ref - dm_f)
            n_pair = int(ok.sum())
        else:
            adiff = vdiff = np.array([np.nan])
            n_pair = 0

        all_mean.append(np.nanmean(adiff))
        all_max.append(np.nanmax(adiff))
        print("-" * 96)
        print("point_index %d  depth %.1f mm  dwell Fz %.3f N  (%d frames)"
              % (pi, row["depth_mm"], row["dwell_Fz_N"], nfr))
        print("  radius px         ref %.3f          fast %.3f" % (R_ref, R_f))
        print("  detections/frame  ref %.1f           fast %.1f"
              % (np.mean([len(d) for d in d_ref]), np.mean([d.shape[0] for d in d_f])))
        print("  baseline markers  ref %d             fast %d" % (len(b_ref), len(b_f)))
        print("  seeded / kept     ref %d/%d          fast %d/%d"
              % (seed_ref, len(kept_ref), seed_f, len(kept_f)))
        print("  paired markers    %d" % n_pair)
        print("  |disp| agreement  mean %.4f px  max %.4f px  (= %.5f / %.5f mm)"
              % (np.nanmean(adiff), np.nanmax(adiff),
                 np.nanmean(adiff) / (R_f / MARKER_RADIUS_MM),
                 np.nanmax(adiff) / (R_f / MARKER_RADIUS_MM)))
        print("  vector agreement  mean %.4f px  max %.4f px"
              % (np.nanmean(vdiff), np.nanmax(vdiff)))
        print("  max |disp|        ref %.3f px       fast %.3f px"
              % (np.nanmax(np.linalg.norm(disp_ref, axis=1)),
                 np.nanmax(np.linalg.norm(disp_f, axis=1))))
        print("  time              ref %.2f s        fast %.2f s   -> %.1fx"
              % (t_ref, t_fast, t_ref / max(t_fast, 1e-9)))
    print("=" * 96)
    print("OVERALL mean |disp| disagreement %.4f px, worst-case max %.4f px"
          % (np.mean(all_mean), np.max(all_max)))
    print("threshold is 0.15 px mean -> %s"
          % ("PASS" if np.mean(all_mean) <= 0.15 else "FAIL"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", default="598,633,324",
                    help="comma-separated point_index values to check against "
                         "marker_overlay.py")
    ap.add_argument("--h5", default=os.path.join(OUTPUT, "pilot_20260807_134855_frames.h5"))
    a = ap.parse_args()
    _validate([int(v) for v in a.validate.split(",") if v.strip()], a.h5)
