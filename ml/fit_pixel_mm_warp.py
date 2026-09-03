#!/usr/bin/env python3
"""
fit_pixel_mm_warp.py -- fit the sensor-pixel -> millimetre map, distortion included.

Consumes a calibration raster (franka/calib_raster.py) that has already been through
the standard post-processing chain, and writes calibration/pixel_to_mm.json plus a
diagnostic figure. ml/undistort.py is the consumer; nothing else needs to know how
the fit was done.

    # the full fit, from a calib_ run
    python3 ml/fit_pixel_mm_warp.py calib_20260903_141500

    # just measure the distortion, from ANY run that has frames.h5 + grid_out
    python3 ml/fit_pixel_mm_warp.py two_cyl_8mm_20260902_203821 --nodes-only

WHAT IS ACTUALLY BEING FITTED
-----------------------------
Not camera intrinsics. The markers live on ONE rigid plane at ONE fixed distance,
forever, so the entire 3D calibration problem collapses to a single fixed 2D -> 2D
warp: sensor pixel -> millimetre on the marker plane. That one function absorbs
radial distortion, lens tilt, magnification and any chamber misalignment together.
Fitting Brown-Conrady intrinsics here would be solving a harder problem than we
have, and would need a target we do not own.

TWO INDEPENDENT MEASUREMENTS, AND WHY BOTH ARE HERE
---------------------------------------------------
1. THE MARKER LATTICE MEASURES THE LENS, FOR FREE (--nodes-only).
   The moulded dome grid is a calibration target we already own. Pool the tare
   (undeformed) frames of a run, detect every dome, and compare where each one
   actually lands against the best AFFINE fit of its lattice index. Affine is the
   right null model -- it already allows rotation, shear and different scales on
   the two axes -- so whatever is left over is nonlinear, i.e. the distortion.
   This needs no robot and works on runs that already exist.

   Its limit: it gives the SHAPE of the distortion and nothing about absolute
   scale, because the true dome pitch in millimetres is a construction value we
   have not measured. It also assumes the mould itself is regular.

2. THE ROBOT RASTER MEASURES THE SCALE AND PINS IT TO A FRAME (the default mode).
   Every poke is a known robot XY paired with a contact the camera localises, so
   the fit is anchored to real millimetres. This is what makes the output usable
   as a measurement rather than as a shape.

They are cross-checks on each other: if the raster fit's nonlinear part does not
look like the lattice's deviation field, one of the two is wrong, and the figure
puts them side by side so that is visible rather than averaged away.

WHY THE CONTACT CENTROID IS TAKEN IN BASELINE (TARE) COORDINATES
-----------------------------------------------------------------
ml/blue_circle_grid indexes every marker's displacement by the cell it occupied in
the TARE frame, so the divergence field -- and therefore the contact peak that
ml/contact_detect.py finds -- lives in undeformed material coordinates. The node
pixel positions this script measures are also tare positions, so the two are
consistent by construction. That the material directly under the indenter barely
moves sideways (it is the source of a radial field, so its centre is near
stationary) is why the distinction stays small in practice, but the consistency is
what makes it correct rather than merely small.

WHAT THE RESIDUALS MEAN
-----------------------
Every residual reported here is from GROUPED k-fold cross-validation: all repeats
of one raster point go into the same fold. Splitting repeats across folds would
let the model see a point's own answer while predicting it, and a degree-3
polynomial with 297 samples is more than able to exploit that. The number printed
is therefore what the fit does on positions it has never seen -- which is the only
number that means anything for a calibration.

The affine baseline is fitted and reported alongside. The DIFFERENCE between the
affine residual and the polynomial residual is the entire value of de-fisheyeing:
if it is small, the honest conclusion is that this lens does not need correcting
and the crop is already good enough.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml"))

import contact_detect as cd                                          # noqa: E402
from marker_overlay import detect as detect_markers                  # noqa: E402
# The warp maths lives in undistort.py, the module every CONSUMER calls, so that a
# calibration can never be evaluated by a different polynomial than the one that
# was fitted to produce it. See that file's header.
from undistort import (MODELS, design, fit_model, lattice_forward,   # noqa: E402
                       lattice_inverse, normaliser, predict)

REC = REPO / "recordings"
GRID = REPO / "ml" / "grid_out"
CALIB_OUT = REPO / "calibration" / "pixel_to_mm.json"

# blue_circle_grid's default. The two must agree or the domes detected here are not
# the domes it tracked.
DEFAULT_RADIUS_PX = 9.6

# A dome must be seen in at least this fraction of the pooled tare frames before its
# median position is trusted as a node. An edge dome that appears in 3 frames of 291
# has a median dominated by whichever 3 frames those were.
MIN_NODE_SEEN_FRAC = 0.25

DEFAULT_MODEL = "poly3"     # MODELS is undistort.py's -- one list, not two


# =============================================================================
#  inputs
# =============================================================================

def resolve_run(name):
    """Accept a run name, a path, or nothing (newest calib_ run)."""
    if name:
        p = Path(name)
        if not p.is_dir():
            p = REC / name
        if not p.is_dir():
            sys.exit("[ERROR] no such run: %s" % name)
        return p
    cands = sorted((d for d in REC.glob("calib_*") if (d / "plan.csv").exists()),
                   key=lambda d: d.name)
    if not cands:
        sys.exit("[ERROR] no calib_* run under %s, and no run named on the command "
                 "line.\n        Run franka/calib_raster.py first (see "
                 "franka/CALIB_RASTER.md)." % REC)
    return cands[-1]


def load_lattice(grid_dir):
    """blue_circle_grid's fitted absolute lattice, from the line it prints.

    The C++ only ever printed this; run_two_cyl_pipeline.py captures it into
    lattice.txt. Without it there is no way to know which cell a pixel belongs to.
    """
    p = grid_dir / "lattice.txt"
    if not p.exists():
        sys.exit("[ERROR] %s not found.\n"
                 "        It is written by run_two_cyl_pipeline.py's "
                 "blue_circle_grid stage;\n        run the pipeline over this run "
                 "first." % p)
    t = p.read_text()
    # "lattice: 17 cols x 13 rows detected (want 17x13), pitch 31.0 x 30.0, origin (76.1,42.9)"
    try:
        cols = int(t.split("lattice:")[1].split("cols")[0])
        rows = int(t.split("x")[1].split("rows")[0])
        pitch = t.split("pitch")[1].split(",")[0]
        px, py = (float(v) for v in pitch.split("x"))
        org = t.split("origin")[1].strip().strip("()\n")
        ox, oy = (float(v) for v in org.strip("()").split(","))
    except (IndexError, ValueError) as e:
        sys.exit("[ERROR] could not parse %s (%s):\n        %s" % (p, e, t.strip()))
    return {"cols": cols, "rows": rows, "pitch_x": px, "pitch_y": py,
            "origin_x": ox, "origin_y": oy}


def load_plan(run):
    """seq -> commanded robot XY in millimetres, plus the raster identity."""
    import csv
    out = {}
    with open(run / "plan.csv") as f:
        for r in csv.DictReader(f):
            out[int(r["seq"])] = {
                "x_mm": float(r["x"]) * 1e3, "y_mm": float(r["y"]) * 1e3,
                "point_index": int(r["point_index"]),
                "depth_mm": float(r.get("depth_cmd_mm", "nan") or "nan"),
                "row": int(r["row"]), "col": int(r["col"]),
            }
    if not out:
        sys.exit("[ERROR] %s is empty." % (run / "plan.csv"))
    return out


def load_windows(grid_dir):
    """Row order of the .npy stacks: index i in disp_*.npy is this seq."""
    import csv
    p = grid_dir / "poke_windows.csv"
    if not p.exists():
        sys.exit("[ERROR] %s not found -- run the pipeline over this run first." % p)
    with open(p) as f:
        return [int(r["seq"]) for r in csv.DictReader(f)]


# =============================================================================
#  1. the lattice measures the lens
# =============================================================================

def measure_nodes(run, lat, radius_px, max_frames, progress=True):
    """Median pixel position of every dome, pooled over tare frames.

    Returns (node_px, seen, stats). node_px is (rows, cols, 2) with NaN where a
    dome was never reliably seen; seen is the frame count behind each median.

    The MEDIAN, not the mean: a dome that is occasionally merged with its
    neighbour by the detector contributes a position halfway between the two, and
    a mean would let a handful of those drag the node. The median ignores them as
    long as they are a minority, which they are.
    """
    import h5py
    h5p = run / "frames.h5"
    if not h5p.exists():
        sys.exit("[ERROR] %s not found -- run ml/build_frames.py over this run "
                 "first." % h5p)

    rows, cols = lat["rows"], lat["cols"]
    acc = [[[] for _ in range(cols)] for _ in range(rows)]
    n_det = []

    with h5py.File(h5p, "r") as f:
        tare = f["tare_frame"]
        n = tare.shape[0]
        idx = np.arange(n) if max_frames <= 0 or n <= max_frames else \
            np.linspace(0, n - 1, max_frames).round().astype(int)
        t0 = time.time()
        for k, i in enumerate(idx):
            pts = detect_markers(np.asarray(tare[i]), radius_px)
            n_det.append(len(pts))
            for (u, v) in pts:
                c = int(round((u - lat["origin_x"]) / lat["pitch_x"]))
                r = int(round((v - lat["origin_y"]) / lat["pitch_y"]))
                if 0 <= r < rows and 0 <= c < cols:
                    acc[r][c].append((u, v))
            if progress and (k + 1) % 25 == 0:
                el = time.time() - t0
                print("    tare frames %4d/%d  (%.0f markers/frame, ETA %.0f s)"
                      % (k + 1, len(idx), np.mean(n_det), el / (k + 1) * (len(idx) - k - 1)),
                      flush=True)

    need = max(1, int(MIN_NODE_SEEN_FRAC * len(idx)))
    node_px = np.full((rows, cols, 2), np.nan)
    seen = np.zeros((rows, cols), int)
    for r in range(rows):
        for c in range(cols):
            seen[r, c] = len(acc[r][c])
            if seen[r, c] >= need:
                node_px[r, c] = np.median(np.array(acc[r][c], float), axis=0)

    stats = {"n_frames": int(len(idx)), "markers_per_frame": float(np.mean(n_det)),
             "nodes_total": rows * cols, "nodes_measured": int(np.isfinite(node_px[..., 0]).sum()),
             "min_frames_per_node": need}
    return node_px, seen, stats


def affine_from_index(node_px):
    """Best affine map lattice index -> pixel, and the residual it cannot explain.

    Affine, not similarity: the null model must already contain rotation, shear and
    a DIFFERENT scale per axis, otherwise a perfectly rectilinear but anisotropic
    image would be reported as distortion. What survives an affine fit is nonlinear
    by construction, and nonlinear is what a lens does.
    """
    rows, cols = node_px.shape[:2]
    rr, ccc = np.mgrid[0:rows, 0:cols]
    ok = np.isfinite(node_px[..., 0])
    if ok.sum() < 6:
        sys.exit("[ERROR] only %d nodes measured -- an affine fit needs at least 6."
                 % ok.sum())
    A = np.column_stack([np.ones(ok.sum()), ccc[ok].ravel(), rr[ok].ravel()])
    sol, *_ = np.linalg.lstsq(A, node_px[ok], rcond=None)
    pred = np.full_like(node_px, np.nan)
    pred[ok] = A @ sol
    dev = node_px - pred
    d = np.hypot(dev[..., 0], dev[..., 1])
    return sol, pred, dev, {
        "rms_px": float(np.sqrt(np.nanmean(d ** 2))),
        "max_px": float(np.nanmax(d)),
        "p95_px": float(np.nanpercentile(d[np.isfinite(d)], 95)),
        "n_nodes": int(ok.sum()),
    }


def radial_fit(node_px, dev):
    """Fit dev ~ k1*r^2 + k2*r^4 about the best-fit centre. Diagnostic, not the model.

    A fit that explains most of the deviation says "this is ordinary radial lens
    distortion". A poor fit says the deviation is NOT radial, which would point at
    the mould or the detector rather than the lens -- worth knowing before anyone
    reaches for cv2.undistort.

    That distinction is the whole reason this diagnostic exists. A distortion
    centred on the optical axis is the lens; an irregular mould has no reason to be
    radial about anything, still less about the point the lens happens to look
    through. Measured on this rig (three runs of 2026-09-02, agreeing to 2%): 96-97%
    radial about (333, 225) px, which is 13 px from the image centre. That is a lens.

    ⚠ SIGN: this rig is PINCUSHION, not barrel. The dome pitch GROWS outward --
    29.4 px per cell near the axis, 34.5 px beyond r = 250 -- so the periphery is
    magnified relative to the centre. Correcting it the wrong way round doubles the
    error instead of removing it, and the two are easy to confuse because both
    produce a radial quiver.
    """
    ok = np.isfinite(node_px[..., 0]) & np.isfinite(dev[..., 0])
    if ok.sum() < 8:
        return None
    P = node_px[ok]
    D = dev[ok]

    def resid(cxy):
        d = P - np.asarray(cxy)
        r2 = (d ** 2).sum(1)
        # dev_radial = (k1 r^2 + k2 r^4) * unit_radial  ->  linear in (k1, k2)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = d / np.linalg.norm(d, axis=1, keepdims=True)
        M = np.column_stack([r2, r2 ** 2])
        # project the observed deviation onto the radial direction
        proj = (D * u).sum(1)
        k, *_ = np.linalg.lstsq(M, proj, rcond=None)
        pred = M @ k
        return proj - pred, k, np.linalg.norm(D, axis=1), proj

    best = None
    c0 = P.mean(0)
    # a coarse search then a local refine; scipy is not a dependency of this repo
    for scale in (64.0, 16.0, 4.0, 1.0):
        grid = [(c0[0] + dx * scale, c0[1] + dy * scale)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        vals = [(float(np.sqrt(np.mean(resid(c)[0] ** 2))), c) for c in grid]
        vals.sort()
        c0 = np.array(vals[0][1])
        best = vals[0]
    res, k, mag, proj = resid(c0)
    explained = 1.0 - float(np.var(res) / max(np.var(proj), 1e-12))
    radial_frac = float(np.mean(np.abs(proj)) / max(np.mean(mag), 1e-12))
    return {"centre_px": [float(c0[0]), float(c0[1])],
            "k1": float(k[0]), "k2": float(k[1]),
            "radial_r2": explained,
            "radial_fraction_of_deviation": radial_frac}


# =============================================================================
#  2. the raster measures the scale
# =============================================================================

def contact_cells(grid_dir, seqs, n_max=1):
    """Per poke, the divergence peak in continuous (row_f, col_f) lattice coords.

    ml/contact_detect.py verbatim -- the same function the two-cylinder analysis
    uses -- so a contact means here exactly what it means there, including its
    refusal to return a peak it cannot support.
    """
    dx = np.load(grid_dir / "disp_dx.npy")
    dy = np.load(grid_dir / "disp_dy.npy")
    if dx.shape[0] != len(seqs):
        sys.exit("[ERROR] disp_dx.npy has %d pokes but poke_windows.csv lists %d."
                 % (dx.shape[0], len(seqs)))
    out, dropped = {}, {"no_peak": 0, "empty": 0}
    for i, seq in enumerate(seqs):
        a, b = dx[i].astype(float), dy[i].astype(float)
        ok = (a != 0) | (b != 0)
        if ok.sum() < 20:
            dropped["empty"] += 1
            continue
        div = cd.divergence(a, b, ok=ok)
        pk = cd.find_peaks(div, n=n_max, verbose=False)
        if not pk:
            dropped["no_peak"] += 1
            continue
        val, rf, cf = pk[0]
        out[seq] = (float(rf), float(cf), float(val))
    return out, dropped


def cells_to_px(node_px, rf, cf):
    """Continuous lattice index -> sensor pixel, through the MEASURED node grid.

    This is the step that makes the coordinate real. Interpolating through the
    ideal origin+pitch lattice instead would silently re-impose the assumption
    that the image is undistorted -- the very thing being measured.

    Returns None when the enclosing cell has an unmeasured corner, rather than
    falling back to the ideal lattice: a fallback would mix two coordinate systems
    in one dataset and the fit could not tell them apart.
    """
    rows, cols = node_px.shape[:2]
    r0 = int(np.clip(np.floor(rf), 0, rows - 2))
    c0 = int(np.clip(np.floor(cf), 0, cols - 2))
    fr, fc = rf - r0, cf - c0
    q = node_px[r0:r0 + 2, c0:c0 + 2]
    if not np.isfinite(q).all():
        return None
    w = np.array([[(1 - fr) * (1 - fc), (1 - fr) * fc],
                  [fr * (1 - fc), fr * fc]])
    return (float((w * q[..., 0]).sum()), float((w * q[..., 1]).sum()))


# =============================================================================
#  the warp models
# =============================================================================

def grouped_cv(uv, xy, groups, model, k=5, seed=0, ctx=None):
    """k-fold CV with all repeats of a raster point in ONE fold.

    Splitting repeats across folds lets a flexible model memorise a point from its
    own duplicates and report a residual that is really just repeat noise. On a
    297-poke raster with 99 points that is a factor-of-three optimism, which is
    exactly the size of the effect being measured.
    """
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    fold_of = {g: i for g, i in zip(uniq, rng.permutation(len(uniq)) % k)}
    err = np.full(len(uv), np.nan)
    for f in range(k):
        te = np.array([fold_of[g] == f for g in groups])
        tr = ~te
        if te.sum() == 0 or tr.sum() < 20:
            continue
        fit = fit_model(uv[tr], xy[tr], model, normaliser(uv[tr]), ctx=ctx)
        err[te] = np.linalg.norm(predict(fit, uv[te]) - xy[te], axis=1)
    e = err[np.isfinite(err)]
    return {"model": model, "n": int(e.size),
            "rms_mm": float(np.sqrt(np.mean(e ** 2))),
            "median_mm": float(np.median(e)),
            "p95_mm": float(np.percentile(e, 95)),
            "max_mm": float(e.max())}


def affine_geometry(fit_affine):
    """Read px/mm, rotation and shear straight off the affine part.

    This is what settles the anisotropy question in src/circle_tracker_config.h:
    the two singular values ARE the millimetres per pixel along the image axes,
    measured rather than assumed from a 36x30 mm sheet filling a 4:3 sensor.
    """
    C = np.array(fit_affine["coeffs"])          # (3, 2): [1, u, v] -> (x, y)
    M = C[1:].T / fit_affine["norm"]["scale"]   # d(mm)/d(px)
    U, S, Vt = np.linalg.svd(M)
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return {"mm_per_px_major": float(S[0]), "mm_per_px_minor": float(S[1]),
            "px_per_mm_major": float(1.0 / S[0]), "px_per_mm_minor": float(1.0 / S[1]),
            "anisotropy": float(S[0] / S[1]),
            "rotation_deg": rot,
            "jacobian_mm_per_px": M.tolist()}


# =============================================================================
#  figure
# =============================================================================

def make_figure(path, node_px, dev, dev_stats, rad, pairs, cvs, best_fit, lat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # nodes-only has nothing to put in the residual panel, and an empty axis reads
    # as a failed plot rather than as a mode.
    n_panels = 2 if pairs is None else 3
    fig, ax = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.2))

    a = ax[0]
    ok = np.isfinite(node_px[..., 0])
    # Scale the arrows so the LARGEST lands at ~60 px on the image: a fixed gain
    # either buries a small deviation or throws a large one off the axes, and this
    # figure has to stay readable for both.
    gain = 60.0 / max(dev_stats["max_px"], 1e-6)
    a.quiver(node_px[ok][:, 0], node_px[ok][:, 1],
             dev[ok][:, 0] * gain, dev[ok][:, 1] * gain,
             np.hypot(dev[ok][:, 0], dev[ok][:, 1]),
             angles="xy", scale_units="xy", scale=1, cmap="viridis", width=0.004)
    a.set_title("dome deviation from the best affine lattice (x%.1f)\n"
                "RMS %.2f px, max %.2f px, %d nodes"
                % (gain, dev_stats["rms_px"], dev_stats["max_px"],
                   dev_stats["n_nodes"]), fontsize=10)
    a.set_xlim(0, 640); a.set_ylim(480, 0); a.set_aspect("equal")
    a.set_xlabel("u (px)"); a.set_ylabel("v (px)")

    if pairs is not None:
        a = ax[1]
        uv, xy = pairs["uv"], pairs["xy"]
        pred = predict(best_fit, uv)
        r = np.linalg.norm(pred - xy, axis=1)
        s = a.scatter(uv[:, 0], uv[:, 1], c=r, s=18, cmap="magma")
        plt.colorbar(s, ax=a, label="residual (mm)")
        a.set_title("raster residual after %s\nRMS %.3f mm (cross-validated)"
                    % (best_fit["model"],
                       [c for c in cvs if c["model"] == best_fit["model"]][0]["rms_mm"]),
                    fontsize=10)
        a.set_xlim(0, 640); a.set_ylim(480, 0); a.set_aspect("equal")
        a.set_xlabel("u (px)"); a.set_ylabel("v (px)")

        a = ax[2]
        names = [c["model"] for c in cvs]
        vals = [c["rms_mm"] for c in cvs]
        a.bar(names, vals, color=["#888"] + ["#2a7"] * (len(names) - 1))
        for i, v in enumerate(vals):
            a.text(i, v, " %.3f" % v, ha="center", va="bottom", fontsize=9)
        a.set_ylabel("cross-validated RMS residual (mm)")
        a.tick_params(axis="x", labelrotation=30, labelsize=8)

        # ⚠ THIS CHART LIES WHEN THE ESTIMATOR IS THE LIMIT, so say so on the chart.
        # The contact position comes from a divergence peak on the 13x17 lattice; when
        # its repeat-to-repeat scatter is comparable to the residual, every model piles
        # up at that floor and the bars become indistinguishable. Read naively that
        # says "distortion does not matter", which is the opposite of true -- the domes
        # measure it at sub-pixel precision and four independent runs agree. What the
        # flat bars actually mean is that THIS validator cannot see the effect.
        floor = pairs.get("repeat_mm")
        spread = max(vals) - min(vals)
        if floor and floor > 0.5 * spread:
            a.axhline(floor, color="#c33", ls="--", lw=1.4)
            a.text(len(names) - 0.5, floor, " contact-estimator\n floor %.2f mm" % floor,
                   color="#c33", fontsize=8, va="bottom", ha="right")
            a.set_title("MODELS ARE NOT SEPARABLE HERE\n"
                        "the estimator floor exceeds the spread between them",
                        fontsize=10, color="#c33")
        else:
            a.set_title("what de-fisheyeing actually buys\n"
                        "(affine is the do-nothing baseline)", fontsize=10)
    else:
        a = ax[1]
        txt = ("lattice pitch %.2f x %.2f px\norigin (%.1f, %.1f)\n\n"
               "deviation from best affine\n RMS %.2f px\n p95 %.2f px\n max %.2f px"
               % (lat["pitch_x"], lat["pitch_y"], lat["origin_x"], lat["origin_y"],
                  dev_stats["rms_px"], dev_stats["p95_px"], dev_stats["max_px"]))
        if rad:
            txt += ("\n\nradial fit\n centre (%.0f, %.0f) px\n k1 %.3e\n k2 %.3e\n"
                    " explains %.0f%% of the radial part\n deviation is %.0f%% radial"
                    % (rad["centre_px"][0], rad["centre_px"][1],
                       rad["k1"], rad["k2"], 100 * rad["radial_r2"],
                       100 * rad["radial_fraction_of_deviation"]))
        txt += "\n\nscale is NOT measured here --\nthe robot raster provides it."
        a.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10,
               transform=a.transAxes)
        a.set_axis_off()

    fig.tight_layout()
    fig.savefig(str(path), dpi=140)
    plt.close(fig)


# =============================================================================
#  main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Fit sensor-pixel -> millimetre, distortion included.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None,
                    help="run name or path (default: newest calib_* run)")
    ap.add_argument("--grid-dir", default=None,
                    help="blue_circle_grid output (default ml/grid_out/<run>)")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic ground truth end to end; no run, no robot")
    ap.add_argument("--nodes-only", action="store_true",
                    help="measure the dome lattice's distortion and stop. Needs no "
                         "robot raster, so it works on runs that already exist.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS,
                    help="warp fitted for the OUTPUT (default %s). Every model is "
                         "cross-validated and reported either way." % DEFAULT_MODEL)
    ap.add_argument("--radius-px", type=float, default=DEFAULT_RADIUS_PX,
                    help="dome radius for the detector; must match the value "
                         "blue_circle_grid ran with (default %.1f)" % DEFAULT_RADIUS_PX)
    ap.add_argument("--max-frames", type=int, default=120,
                    help="tare frames pooled for the node measurement (default 120; "
                         "<=0 for all). The median converges long before 300.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(CALIB_OUT))
    ap.add_argument("--figure", default=None,
                    help="default: output/runs/<run>/pixel_mm_warp.png")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    run = resolve_run(args.run)
    grid_dir = Path(args.grid_dir) if args.grid_dir else GRID / run.name
    if not grid_dir.is_dir():
        sys.exit("[ERROR] %s not found -- run the post-processing pipeline over %s "
                 "first." % (grid_dir, run.name))
    lat = load_lattice(grid_dir)

    print("\n" + "=" * 74)
    print("  PIXEL -> MILLIMETRE WARP")
    print("=" * 74)
    print("  run        : %s" % run.name)
    print("  grid       : %s" % grid_dir)
    print("  lattice    : %d cols x %d rows, pitch %.2f x %.2f px, origin (%.1f, %.1f)"
          % (lat["cols"], lat["rows"], lat["pitch_x"], lat["pitch_y"],
             lat["origin_x"], lat["origin_y"]))

    # ---- 1. the lattice measures the lens ----------------------------------
    print("\n  [1/2] measuring the dome lattice ...")
    node_px, seen, nstats = measure_nodes(run, lat, args.radius_px, args.max_frames)
    sol, pred, dev, dev_stats = affine_from_index(node_px)
    rad = radial_fit(node_px, dev)

    print("        %d/%d nodes measured over %d tare frames (%.0f domes/frame)"
          % (nstats["nodes_measured"], nstats["nodes_total"], nstats["n_frames"],
             nstats["markers_per_frame"]))
    print("        deviation from the best AFFINE lattice:")
    print("          RMS %.3f px   p95 %.3f px   max %.3f px"
          % (dev_stats["rms_px"], dev_stats["p95_px"], dev_stats["max_px"]))
    if rad:
        print("        radial model: centre (%.0f, %.0f), k1 %.3e, k2 %.3e"
              % (rad["centre_px"][0], rad["centre_px"][1], rad["k1"], rad["k2"]))
        print("          explains %.0f%% of the radial component; the deviation is "
              "%.0f%% radial" % (100 * rad["radial_r2"],
                                 100 * rad["radial_fraction_of_deviation"]))
    print("\n        %s" % verdict(dev_stats["rms_px"], dev_stats["max_px"]))

    pairs, cvs, best_fit, geom = None, [], None, None

    if not args.nodes_only:
        # ---- 2. the raster measures the scale -------------------------------
        print("\n  [2/2] pairing raster pokes with contacts ...")
        plan = load_plan(run)
        seqs = load_windows(grid_dir)
        cells, dropped = contact_cells(grid_dir, seqs)

        uv, xy, grp, dep, lost = [], [], [], [], 0
        for seq, (rf, cf, _v) in sorted(cells.items()):
            if seq not in plan:
                continue
            p = cells_to_px(node_px, rf, cf)
            if p is None:
                lost += 1
                continue
            uv.append(p)
            xy.append((plan[seq]["x_mm"], plan[seq]["y_mm"]))
            grp.append(plan[seq]["point_index"])
            dep.append(plan[seq]["depth_mm"])
        uv = np.array(uv, float)
        xy = np.array(xy, float)
        grp = np.array(grp)

        print("        %d pokes -> %d contacts -> %d usable pairs"
              % (len(seqs), len(cells), len(uv)))
        print("        dropped: %d with no supportable peak, %d with too few tracked "
              "markers, %d over an unmeasured node" % (dropped["no_peak"],
                                                       dropped["empty"], lost))
        if len(uv) < 40:
            sys.exit("\n[ERROR] only %d usable pairs -- not enough to fit and validate "
                     "a warp.\n        Check the pipeline ran fully over this run, and "
                     "that the\n        indenter was small enough to make a localisable "
                     "contact." % len(uv))
        print("        raster points represented: %d, repeats %.1f on average"
              % (len(np.unique(grp)), len(uv) / max(len(np.unique(grp)), 1)))

        rep = repeat_scatter(uv, grp)
        if rep is not None:
            print("        contact-centroid repeatability: %.2f px RMS between repeats "
                  "of the same point" % rep)

        norm = normaliser(uv)
        ctx = {"node_px": node_px, "lattice": lat}
        for m in MODELS:
            cvs.append(grouped_cv(uv, xy, grp, m, k=args.folds, ctx=ctx))
        best_fit = fit_model(uv, xy, args.model, norm, ctx=ctx)
        geom = affine_geometry(fit_model(uv, xy, "affine", norm))
        # repeatability in MILLIMETRES, so it is directly comparable with the
        # residuals rather than needing the reader to convert it in their head
        rep_mm = (rep * 0.5 * (1.0 / geom["px_per_mm_major"] +
                               1.0 / geom["px_per_mm_minor"])) if rep else None
        pairs = {"uv": uv, "xy": xy, "groups": grp, "depths": dep,
                 "repeat_px": rep, "repeat_mm": rep_mm}

        print("\n  CROSS-VALIDATED RESIDUALS (grouped by raster point)")
        print("    %-8s %8s %8s %8s %8s" % ("model", "RMS", "median", "p95", "max"))
        for c in cvs:
            print("    %-8s %7.3f  %7.3f  %7.3f  %7.3f"
                  % (c["model"], c["rms_mm"], c["median_mm"], c["p95_mm"], c["max_mm"]))
        aff = [c for c in cvs if c["model"] == "affine"][0]["rms_mm"]
        bst = min(c["rms_mm"] for c in cvs)
        print("\n    affine (do nothing) %.3f mm  ->  best %.3f mm   "
              "= %.0f%% of the error removed" % (aff, bst, 100 * (1 - bst / aff)))
        print("\n  MEASURED SCALE (from the affine part -- not assumed)")
        print("    %.3f px/mm and %.3f px/mm along the image axes  (anisotropy %.3f)"
              % (geom["px_per_mm_major"], geom["px_per_mm_minor"], geom["anisotropy"]))
        print("    image rotated %.2f deg relative to the robot XY frame"
              % geom["rotation_deg"])
        print("    src/circle_tracker_config.h currently implies 17.78 and 16.00 "
              "px/mm (anisotropy 1.111)")

    # ---- outputs ------------------------------------------------------------
    figp = Path(args.figure) if args.figure else \
        REPO / "output" / "runs" / run.name / "pixel_mm_warp.png"
    figp.parent.mkdir(parents=True, exist_ok=True)
    make_figure(figp, node_px, dev, dev_stats, rad, pairs, cvs, best_fit, lat)
    print("\n  figure -> %s" % figp)

    if args.nodes_only:
        print("  --nodes-only: no calibration written (the lattice gives shape, not "
              "scale).")
        print("=" * 74 + "\n")
        return 0

    out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run": run.name,
        "image_size": [640, 480],
        "frame": "robot base XY, millimetres",
        "pixel_to_mm": best_fit,
        # The reverse direction is a convenience for drawing predictions back onto
        # a frame, not a measurement path. lattice_affine has no reverse of the
        # same kind (it reads a table of pixels, so inverting it means inverting
        # the table), so the reverse is always fitted as a plain polynomial and
        # said so here rather than quietly being a different model than it claims.
        "mm_to_pixel": fit_model(pairs["xy"], pairs["uv"],
                                 "poly3" if args.model == "lattice_affine"
                                 else args.model, normaliser(pairs["xy"])),
        "mm_to_pixel_note": ("fitted as poly3; the forward model is lattice_affine"
                             if args.model == "lattice_affine" else "same model as "
                             "the forward direction"),
        "affine_geometry": geom,
        "cross_validated_residuals_mm": cvs,
        "lattice": lat,
        "lattice_deviation_px": dev_stats,
        "radial_fit": rad,
        "node_px": np.where(np.isfinite(node_px), node_px, None).tolist(),
        "node_seen": seen.tolist(),
        # Recorded because it is the number that decides whether the residual table
        # means anything: when it is comparable to the residuals, the models are not
        # separable and the table says nothing about the lens either way.
        "contact_repeatability_px": pairs.get("repeat_px"),
        "contact_repeatability_mm": pairs.get("repeat_mm"),
        "n_pairs": int(len(pairs["uv"])),
        "n_raster_points": int(len(np.unique(pairs["groups"]))),
        "detector_radius_px": args.radius_px,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("  calibration -> %s" % outp)
    print("  consume it with ml/undistort.py, never by re-reading this JSON by hand.")
    print("=" * 74 + "\n")
    return 0


def repeat_scatter(uv, grp):
    """RMS spread of the contact pixel between repeats of the same raster point.

    The estimator's own noise floor. A fit residual near this number means the lens
    model is as good as the data allows; a residual far above it means the model is
    missing something real.
    """
    rs = []
    for g in np.unique(grp):
        m = grp == g
        if m.sum() < 2:
            continue
        rs.append(((uv[m] - uv[m].mean(0)) ** 2).sum(1))
    if not rs:
        return None
    return float(np.sqrt(np.mean(np.concatenate(rs))))


# =============================================================================
#  self-test -- synthetic ground truth, no run, no robot
# =============================================================================
#
# SCOPE, DELIBERATELY. This exercises everything this file adds: the lattice
# deviation measurement, the radial diagnostic, the contact peak -> pixel path,
# every warp model, the grouped cross-validation and the round trip through
# undistort.py. It does NOT synthesise camera frames to re-test
# marker_overlay.detect, which is validated code that this file only calls, and
# which a hand-rolled renderer would mostly test the renderer of. The detector is
# exercised against REAL tare frames every time --nodes-only runs.
#
# The synthetic physics is the real physics: millimetres are an AFFINE function of
# the lattice index (a regular mould viewed by a robot with a rigid frame), and
# the lens then bends index -> pixel radially. So `lattice_affine` is the model
# the data was generated by, and it should win.

SELF_TEST_LATTICE = {"cols": 17, "rows": 13, "pitch_x": 31.0, "pitch_y": 30.0,
                     "origin_x": 76.0, "origin_y": 43.0}
SELF_TEST_CENTRE = (333.0, 225.0)
# Chosen so the AFFINE-REMOVED deviation is ~18 px peak, matching what the real
# EVK1 lens measures on the runs of 2026-09-02 (17.8 px max, 3.9 px RMS, three
# runs agreeing to 2%). The raw radial offset is larger than that -- an affine fit
# absorbs most of a smooth radial term -- so injecting "17.8 px of distortion"
# directly would have made the test about a third as hard as reality.
SELF_TEST_K1 = -6.0e-4


def _synth_nodes(lat=SELF_TEST_LATTICE, centre=SELF_TEST_CENTRE, k1=SELF_TEST_K1):
    """The dome grid as the lens would show it: ideal index, bent radially.

    The offset is k1 * r^2 ALONG the radius, matching radial_fit()'s model exactly
    -- so a recovered k1 is comparable with the injected one. Writing this as
    `d * (1 + k1 * |d|^2)` instead makes the offset k1 * r^3, which at r = 350 px
    is 8600 px rather than 25: the grid folds inside out and every downstream check
    fails for reasons that have nothing to do with the code under test.
    """
    C = np.array(centre, float)
    out = np.zeros((lat["rows"], lat["cols"], 2))
    for r in range(lat["rows"]):
        for c in range(lat["cols"]):
            p = np.array([lat["origin_x"] + c * lat["pitch_x"],
                          lat["origin_y"] + r * lat["pitch_y"]], float)
            d = p - C
            rr = float(np.hypot(d[0], d[1]))
            out[r, c] = C + d * (1.0 + k1 * rr)          # offset = k1 * r^2
    return out


def _synth_contact(rows, cols, rf, cf, sigma=1.6, amp=4.0):
    """A radial source centred on (rf, cf): markers pushed away from the contact.

    Divergence of `d * exp(-|d|^2 / 2s^2)` peaks at the centre, which is exactly
    the signal ml/contact_detect.py looks for -- so this drives the real estimator
    rather than a stand-in for it.
    """
    rr, cc = np.mgrid[0:rows, 0:cols].astype(float)
    dr, dc = rr - rf, cc - cf
    g = np.exp(-(dr ** 2 + dc ** 2) / (2.0 * sigma ** 2))
    return amp * dc * g, amp * dr * g


def self_test():
    import tempfile
    ok_all = True

    def check(label, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s%s" % ("ok " if cond else "FAIL", label,
                               ("   " + detail) if detail else ""))

    lat = dict(SELF_TEST_LATTICE)
    rows, cols = lat["rows"], lat["cols"]
    node_px = _synth_nodes()

    # ---- 1. the lattice deviation measurement -------------------------------
    sol, pred, dev, ds = affine_from_index(node_px)
    check("affine fit uses every node", ds["n_nodes"] == rows * cols)
    check("injected distortion is detected", ds["max_px"] > 3.0,
          "max %.2f px" % ds["max_px"])
    rad = radial_fit(node_px, dev)
    check("deviation is recognised as radial", rad["radial_fraction_of_deviation"] > 0.9,
          "%.0f%% radial" % (100 * rad["radial_fraction_of_deviation"]))
    check("radial centre recovered",
          np.hypot(rad["centre_px"][0] - SELF_TEST_CENTRE[0],
                   rad["centre_px"][1] - SELF_TEST_CENTRE[1]) < 25.0,
          "(%.0f, %.0f) vs (%.0f, %.0f)" % (rad["centre_px"][0], rad["centre_px"][1],
                                            SELF_TEST_CENTRE[0], SELF_TEST_CENTRE[1]))
    check("a perfectly affine lattice shows NO distortion",
          affine_from_index(_synth_nodes(k1=0.0))[3]["max_px"] < 1e-9)

    # ---- 2. the contact peak -> pixel path ----------------------------------
    # ground truth: mm is affine in lattice index (regular mould, rigid frame)
    A = np.array([[12.0, 0.0], [1.90, 0.28], [-0.26, 1.83]])   # [1, cf, rf] -> (x, y)

    def true_mm(cf, rf):
        return np.array([1.0, cf, rf]) @ A

    targets = [(float(rf), float(cf))
               for rf in np.linspace(1.5, rows - 2.5, 7)
               for cf in np.linspace(1.5, cols - 2.5, 9)]
    repeats = 3
    seqs, dxs, dys, plan_rows = [], [], [], []
    rng = np.random.RandomState(7)
    for rep in range(repeats):
        for pi, (rf, cf) in enumerate(targets):
            a, b = _synth_contact(rows, cols, rf, cf)
            # a little noise, so repeats are not identical and the CV is not trivial
            a = a + rng.normal(0, 0.02, a.shape)
            b = b + rng.normal(0, 0.02, b.shape)
            seq = len(seqs)
            seqs.append(seq)
            dxs.append(a)
            dys.append(b)
            xy = true_mm(cf, rf)
            plan_rows.append({"seq": seq, "point_index": pi,
                              "x_mm": xy[0], "y_mm": xy[1]})

    with tempfile.TemporaryDirectory() as td:
        g = Path(td)
        np.save(g / "disp_dx.npy", np.array(dxs, np.float32))
        np.save(g / "disp_dy.npy", np.array(dys, np.float32))
        import csv as _csv
        with open(g / "poke_windows.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["seq"])
            w.writeheader()
            for s in seqs:
                w.writerow({"seq": s})
        cells, dropped = contact_cells(g, load_windows(g))

    check("every synthetic contact yields a peak", len(cells) == len(seqs),
          "%d/%d, dropped %s" % (len(cells), len(seqs), dropped))
    cell_err = [np.hypot(cells[r["seq"]][0] - targets[r["point_index"]][0],
                         cells[r["seq"]][1] - targets[r["point_index"]][1])
                for r in plan_rows if r["seq"] in cells]
    check("peak lands on the right cell", np.median(cell_err) < 0.6,
          "median %.3f cell, p95 %.3f" % (np.median(cell_err),
                                          np.percentile(cell_err, 95)))

    # cells_to_px must refuse rather than fall back when a corner is unmeasured
    holed = node_px.copy()
    holed[5, 5] = np.nan
    check("an unmeasured node refuses the pixel, not guesses it",
          cells_to_px(holed, 5.4, 5.4) is None and
          cells_to_px(holed, 1.2, 1.2) is not None)
    rt = cells_to_px(node_px, 4.0, 6.0)
    check("cells_to_px reproduces a node exactly",
          np.hypot(rt[0] - node_px[4, 6, 0], rt[1] - node_px[4, 6, 1]) < 1e-9)

    # ---- 3. the models ------------------------------------------------------
    uv, xy, grp = [], [], []
    for r in plan_rows:
        if r["seq"] not in cells:
            continue
        rf, cf, _ = cells[r["seq"]]
        p = cells_to_px(node_px, rf, cf)
        if p is None:
            continue
        uv.append(p)
        xy.append((r["x_mm"], r["y_mm"]))
        grp.append(r["point_index"])
    uv, xy, grp = np.array(uv), np.array(xy), np.array(grp)
    check("pairs survive the whole chain", len(uv) == len(seqs),
          "%d pairs" % len(uv))

    ctx = {"node_px": node_px, "lattice": lat}
    cv = {m: grouped_cv(uv, xy, grp, m, k=5, ctx=ctx)["rms_mm"] for m in MODELS}

    # THE PER-POKE RESIDUAL IS NOT THE CALIBRATION'S ACCURACY, and confusing the
    # two would be the easiest mistake to make with this output. A contact is
    # localised on the 13x17 lattice, so its position is quantised to a fraction of
    # a cell -- measured at 0.25 cell here, about 0.48 mm. Every model is pinned at
    # that floor and they are indistinguishable there, which says nothing about the
    # lens. What the calibration is FOR is the fitted surface, and averaging ~190
    # independent quantisation errors into 6-10 coefficients beats the floor by
    # roughly the square root of the number of pokes per coefficient.
    cell_mm = float(np.hypot(*A[1]) + np.hypot(*A[2])) / 2.0
    floor_mm = float(np.median(cell_err)) * cell_mm
    print("        estimator quantisation floor ~ %.3f mm (%.2f cell x %.2f mm/cell)"
          % (floor_mm, np.median(cell_err), cell_mm))
    print("        %-15s %10s %10s" % ("model", "per-poke", "surface"))

    truth_rc = np.array(targets, float)
    truth_uv = lattice_forward(node_px, truth_rc)
    truth_xy = np.array([true_mm(cf, rf) for rf, cf in targets])
    surf = {}
    for m in MODELS:
        fit_m = fit_model(uv, xy, m, normaliser(uv), ctx=ctx)
        e = np.linalg.norm(predict(fit_m, truth_uv) - truth_xy, axis=1)
        surf[m] = float(np.sqrt(np.mean(e ** 2)))
        print("        %-15s %8.4f mm %8.4f mm" % (m, cv[m], surf[m]))

    check("per-poke residuals sit at the quantisation floor, not below it",
          all(0.5 * floor_mm < cv[m] < 2.5 * floor_mm for m in MODELS))
    check("the fitted surface beats the per-poke floor",
          surf["lattice_affine"] < 0.5 * floor_mm,
          "%.4f mm vs floor %.3f mm" % (surf["lattice_affine"], floor_mm))
    check("lattice_affine is the most accurate surface",
          surf["lattice_affine"] == min(surf.values()),
          "best is %s" % min(surf, key=surf.get))
    check("correcting distortion beats ignoring it",
          surf["lattice_affine"] < surf["affine"],
          "lattice_affine %.4f vs affine %.4f mm"
          % (surf["lattice_affine"], surf["affine"]))

    # ---- 4. the round trip through undistort.py -----------------------------
    fit = fit_model(uv, xy, "lattice_affine", normaliser(uv), ctx=ctx)
    inv = lattice_inverse(node_px, uv, lat)
    fwd = lattice_forward(node_px, inv)
    check("lattice_inverse inverts lattice_forward",
          np.nanmax(np.linalg.norm(fwd - uv, axis=1)) < 0.05,
          "max %.4f px" % np.nanmax(np.linalg.norm(fwd - uv, axis=1)))
    outside = lattice_inverse(node_px, [(5.0, 5.0), (635.0, 475.0)], lat)
    check("a pixel outside the dome grid returns NaN, not a saturated guess",
          bool(np.isnan(outside).all()))

    blob = json.loads(json.dumps({"pixel_to_mm": fit,
                                  "mm_to_pixel": fit_model(xy, uv, "poly3",
                                                           normaliser(xy)),
                                  "image_size": [640, 480]}))
    from undistort import Calibration
    cal = Calibration(blob)
    # tested at a TRUE (pixel, mm) pair, so this measures the calibration rather
    # than one poke's quantisation error
    x0, y0 = cal.pixel_to_mm(truth_uv[0][0], truth_uv[0][1])
    check("Calibration survives a JSON round trip",
          np.hypot(x0 - truth_xy[0][0], y0 - truth_xy[0][1]) < 0.25,
          "%.4f mm off" % np.hypot(x0 - truth_xy[0][0], y0 - truth_xy[0][1]))
    u1, v1 = cal.mm_to_pixel(x0, y0)
    check("mm_to_pixel returns to the pixel it came from",
          np.hypot(u1 - truth_uv[0][0], v1 - truth_uv[0][1]) < 6.0,
          "%.2f px" % np.hypot(u1 - truth_uv[0][0], v1 - truth_uv[0][1]))
    check("outside the calibrated field, pixel_to_mm gives NaN",
          not np.isfinite(cal.pixel_to_mm(5.0, 5.0)[0]))

    # both probes inside the dome span (x 89..561, y 50..397)
    sc_c = cal.local_scale_px_per_mm(325, 225)
    sc_e = cal.local_scale_px_per_mm(120, 90)
    check("local scale varies across the field (that IS the fisheye)",
          abs(sc_c[0] - sc_e[0]) / sc_c[0] > 0.01,
          "centre %.2f px/mm, corner %.2f px/mm" % (sc_c[0], sc_e[0]))

    print("\n  %s" % ("✅ all checks passed" if ok_all else "❌ FAILURES above"))
    return 0 if ok_all else 1


def verdict(rms_px, max_px):
    """Say plainly whether this lens needs correcting at all."""
    if max_px < 0.5:
        return ("VERDICT: the lattice is straight to under half a pixel. This lens "
                "does not\n        need de-fisheyeing; look elsewhere for the error.")
    if max_px < 2.0:
        return ("VERDICT: sub-2 px distortion. Worth correcting only if the "
                "measurement needs\n        better than ~%.1f%% of the field; a "
                "centre crop is the cheaper fix." % (100 * max_px / 640))
    return ("VERDICT: %.1f px of distortion at the field edge -- real, and worth "
            "correcting.\n        Proceed with the raster fit to pin it to "
            "millimetres." % max_px)


if __name__ == "__main__":
    sys.exit(main())
