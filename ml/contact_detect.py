#!/usr/bin/env python3
"""Shared contact detection for the multi-cylinder runs.

WHY THIS MODULE EXISTS
----------------------
`plot_two_cyl.py` and `plot_rot_cyl.py` each carried their OWN copy of `divergence`
and the peak finder. On 2026-09-02 a genuine bug was fixed in one copy and not the
other, so the two scripts silently disagreed about how many contacts a run had --
the vector fields said two, the rotation fit still said three, and both were written
into the same report. One implementation, imported by both.

WHAT IS DETECTED
----------------
Markers are pushed AWAY from a contact, so a contact is a source in the displacement
field and shows up as a positive peak in its divergence.

THREE THINGS THIS GETS RIGHT, EACH LEARNED FROM A WRONG ANSWER
--------------------------------------------------------------
1. Holes are not zeros. Untracked lattice cells are filled with dx = dy = 0 before
   differentiating. Differencing ACROSS such a hole invents a gradient from nothing:
   on the 24 mm run at -30 deg that manufactured a peak of 7.54 at cell (6,3), with
   -1.05 and -2.65 as immediate neighbours, in a region where columns 1-2 had ZERO
   tracked markers. It was reported as a contact with 98.9 % balance. A hole means
   "unknown", so the derivative there is NaN.

2. A peak finder asked for N peaks returns N peaks. It cannot decline. Given a broad
   plateau it returns the N highest points on it, which on the 3-cylinder run were
   mostly single noisy cells at the field edge. Candidates must therefore EARN their
   place: a real contact is separated from its neighbours by a saddle, and is a broad
   lobe rather than one hot cell.

3. Returning fewer contacts than expected is a result, not a failure. If only one
   survives, the separation, balance, scale and resolvability are undefined -- and
   must be reported as such, never as 0.0 px and 100 % balance.
"""
import numpy as np

EDGE = 1                  # lattice cells masked at each border
MIN_RESOLVE = 0.15        # a peak needs this much of a dip against a stronger peak
MIN_COHERENCE = 0.60      # ...and must be a broad lobe, not one hot cell


def divergence(dx, dy, ok=None, edge=EDGE):
    """Central-difference divergence; NaN wherever the stencil touches missing data."""
    d = np.zeros_like(dx, dtype=float)
    d[:, 1:-1] += (dx[:, 2:] - dx[:, :-2]) / 2.0
    d[1:-1, :] += (dy[2:, :] - dy[:-2, :]) / 2.0
    if ok is not None:
        bad = np.zeros(d.shape, bool)
        bad[:, 1:-1] |= ~ok[:, 2:] | ~ok[:, :-2]
        bad[1:-1, :] |= ~ok[2:, :] | ~ok[:-2, :]
        bad |= ~ok
        d[bad] = np.nan
    if edge:
        d[:edge, :] = d[-edge:, :] = d[:, :edge] = d[:, -edge:] = np.nan
    return d


def saddle_between(div, a, b):
    """Minimum divergence strictly between two peaks -- the dip that separates them."""
    t = np.linspace(0.0, 1.0, 120)
    rr = a[1] + t * (b[1] - a[1])
    cc = a[2] + t * (b[2] - a[2])
    pr = np.array([div[int(round(x)), int(round(y))]
                   if 0 <= round(x) < div.shape[0] and 0 <= round(y) < div.shape[1]
                   else np.nan for x, y in zip(rr, cc)])
    inner = (t > 0.05) & (t < 0.95)
    return float(np.nanmin(pr[inner])) if inner.any() else float("nan")


def coherence(div, r, c):
    """3x3 mean over the peak value: how LOBE-like a candidate is.

    Measured on the runs of 2026-09-02:
        two_cyl_24mm  A 0.80  B 0.71    <- genuine contacts
        two_cyl_16mm  A 0.83  B 0.79    <- genuine contacts
        three_cyl_24mm  A 0.42          <- div 7.3 with neighbours 1.1 and 0.6
    MIN_COHERENCE = 0.60 separates them with margin on both sides. Calibrated, not
    guessed; re-measure the same way before changing it.
    """
    r, c = int(round(r)), int(round(c))
    if not (0 <= r < div.shape[0] and 0 <= c < div.shape[1]):
        return float("nan")
    if not div[r, c] > 0:
        return float("nan")
    w = div[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
    return float(np.nanmean(w) / div[r, c])


def prune_unsupported(pk, div, min_resolve=MIN_RESOLVE, min_coherence=MIN_COHERENCE,
                      verbose=True):
    """Keep only candidates a real contact would produce. May return fewer than asked."""
    kept = []
    for cand in sorted(pk, key=lambda t: -t[0]):
        co = coherence(div, cand[1], cand[2])
        if np.isfinite(co) and co < min_coherence:
            if verbose:
                print("  [drop] cell (%.2f, %.2f) div %.1f -- coherence %.2f < %.2f; "
                      "a one-cell spike, not a lobe"
                      % (cand[1], cand[2], cand[0], co, min_coherence))
            continue
        ok_ = True
        for k in kept:
            sad = saddle_between(div, cand, k)
            weaker = min(cand[0], k[0])
            if not np.isfinite(sad) or weaker <= 0 or (1.0 - sad / weaker) < min_resolve:
                if verbose:
                    print("  [drop] cell (%.2f, %.2f) div %.1f -- no saddle against the "
                          "peak at (%.2f, %.2f); a shoulder, not a contact"
                          % (cand[1], cand[2], cand[0], k[1], k[2]))
                ok_ = False
                break
        if ok_:
            kept.append(cand)
    return kept


def find_peaks(div, n=2, min_sep=3.0, prune=True, verbose=True):
    """Up to n well-separated divergence maxima, sub-cell refined, then pruned."""
    v = np.where(np.isnan(div), -1e9, div)
    order = sorted(((v[r, c], r, c) for r in range(v.shape[0]) for c in range(v.shape[1])),
                   reverse=True)
    pk = []
    for val, r, c in order:
        if val <= 0:
            break
        if all((r - pr) ** 2 + (c - pc) ** 2 > min_sep ** 2 for _, pr, pc in pk):
            pk.append((val, r, c))
        if len(pk) == n:
            break
    out = []
    for val, r, c in pk:                       # divergence-weighted 3x3 centroid
        rs = slice(max(0, r - 1), r + 2)
        cs = slice(max(0, c - 1), c + 2)
        w = np.clip(np.nan_to_num(div[rs, cs]), 0, None)
        rr, cc = np.mgrid[rs, cs]
        if w.sum() > 0:
            out.append((float(val), float((rr * w).sum() / w.sum()),
                        float((cc * w).sum() / w.sum())))
        else:
            out.append((float(val), float(r), float(c)))
    return prune_unsupported(out, div, verbose=verbose) if prune else out


def poke_groups(run_dir):
    """seq -> (kind, value) from plan.csv: how the pokes should be GROUPED.

    ⛔ WHY THIS IS SHARED. Both plot_two_cyl and undistort_ab parsed `rot([-+]\\d+)` out
    of the block name to find the orientations. The translation campaign writes
    `..._off-8.0` instead, which matches neither -- so every poke fell into one group and
    all five tool POSITIONS were averaged into a single displacement field. That is
    exactly the smearing bug of HANDOFF 26.22, wearing different clothes: a run at five
    places on the pad is not five repeats of one measurement.

    Returns ({seq: value}, kind) where kind is "rotation_deg" or "offset_mm", or
    ({}, None) for a single-position run.
    """
    import csv as _csv
    import re as _re
    from pathlib import Path as _Path
    pc = _Path(run_dir) / "plan.csv"
    if not pc.exists():
        return {}, None
    rots, offs, poses = {}, {}, {}
    with open(pc) as f:
        for row in _csv.DictReader(f):
            b = row.get("block", "")
            mp = _re.search(r"pose(\d+)", b)
            mo = _re.search(r"off([-+]?\d+(?:\.\d+)?)", b)
            mr = _re.search(r"rot([-+]?\d+(?:\.\d+)?)", b)
            if mp:
                poses[int(row["seq"])] = float(mp.group(1))
            elif mo:
                offs[int(row["seq"])] = float(mo.group(1))
            elif mr:
                rots[int(row["seq"])] = float(mr.group(1))
    # A random-pose run groups by POSE INDEX: yaw and position both change together, so
    # neither alone identifies the group, and averaging two poses is the same smearing
    # mistake as averaging two orientations (HANDOFF 26.22).
    if len(set(poses.values())) > 1:
        return poses, "pose"
    if len(set(offs.values())) > 1:
        return offs, "offset_mm"
    if len(set(rots.values())) > 1:
        return rots, "rotation_deg"
    for d, k in ((poses, "pose"), (offs, "offset_mm"), (rots, "rotation_deg")):
        if d:
            return d, k
    return {}, None
