#!/usr/bin/env python3
"""
undistort.py -- the ONE place that turns sensor pixels into millimetres.

    from ml.undistort import load
    cal = load()                       # calibration/pixel_to_mm.json
    x_mm, y_mm = cal.pixel_to_mm(u, v)         # scalars or arrays
    u_px, v_px = cal.mm_to_pixel(x_mm, y_mm)

    python3 ml/undistort.py             # print what the current calibration says
    python3 ml/undistort.py --demo      # corner-by-corner, correction vs not

WHY THE MODEL MATHS LIVES HERE AND NOT IN THE FITTER
------------------------------------------------------
ml/fit_pixel_mm_warp.py imports `fit_model` and `predict` FROM this file. That is
the wrong way round only if you expect the fitter to be the important half; it is
not. The half that must never be wrong is the one every downstream measurement
calls, and a calibration that is evaluated by a slightly different polynomial than
the one that was fitted is wrong in a way nothing will ever print. One
implementation, imported by both, and that failure cannot happen.

WHAT THE NUMBERS MEAN
---------------------
`pixel_to_mm` returns millimetres in the ROBOT BASE FRAME, because that is the
frame the calibration raster measured. It is not the sensor's own frame and its
origin is not the corner of the pad. For a sensor-local frame, subtract the
position of whatever you want to call the origin -- but do it in millimetres,
AFTER this call, never by shifting pixels first.

APPLY IT TO MARKER CENTRES, NOT TO IMAGES
------------------------------------------
The cheap and correct place to use this is on the few hundred detected marker
centres per frame, not on the frame itself. Resampling a 640x480 event-count image
through cv2.remap blurs markers that are only ~19 px across and manufactures
intensity that no event produced. `lut()` exists for the cases that genuinely need
a dense map (rendering a corrected preview, for instance) and is deliberately not
the path of least resistance.

⚠ DISPLACEMENTS DO NOT TRANSFORM LIKE POSITIONS.
A marker's displacement is the DIFFERENCE of two positions, and the warp is
nonlinear, so mapping the displacement vector directly is wrong. Convert both
endpoints and subtract:

    dx_mm, dy_mm = cal.displacement_to_mm(u0, v0, u1, v1)      # correct
    # NOT: cal.pixel_to_mm(du, dv)

Over a 3 px displacement near the centre the difference is negligible; near the
corners, where the local scale differs from the centre's by a few percent, it is
not, and "negligible in the middle" is how a systematic error gets shipped.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO / "calibration" / "pixel_to_mm.json"

MODELS = ("affine", "poly2", "poly3", "poly4", "tps", "lattice_affine")

# How many fixed-point steps to invert the measured node grid. The map is very
# nearly affine, so this converges in two; four is the belt.
_INVERT_ITERS = 4
# How far past the outermost dome the inverse will still answer. The pad extends a
# little beyond the last dome; a whole cell beyond it is extrapolation.
_EDGE_MARGIN_CELLS = 0.75
# Reprojection tolerance that decides whether an inverse actually converged.
_INVERT_TOL_PX = 0.5


# =============================================================================
#  model primitives -- fitted by fit_pixel_mm_warp.py, evaluated by everyone
# =============================================================================

def normaliser(uv):
    """Centre and scale inputs to ~[-1, 1].

    A degree-4 fit on raw 0..640 coordinates builds a Vandermonde matrix with a
    condition number in the billions; the coefficients that come back are noise
    that happens to sum to the right answer on the training points.
    """
    uv = np.atleast_2d(np.asarray(uv, float))
    c = uv.mean(0)
    s = float(np.abs(uv - c).max()) or 1.0
    return {"centre": [float(c[0]), float(c[1])], "scale": s}


def design(uv, norm, model):
    """The polynomial design matrix. Term order is part of the file format:
    changing it silently reinterprets every calibration ever written."""
    uv = np.atleast_2d(np.asarray(uv, float))
    u = (uv[:, 0] - norm["centre"][0]) / norm["scale"]
    v = (uv[:, 1] - norm["centre"][1]) / norm["scale"]
    one = np.ones_like(u)
    if model == "affine":
        return np.column_stack([one, u, v])
    if model == "poly2":
        return np.column_stack([one, u, v, u * u, u * v, v * v])
    if model == "poly3":
        return np.column_stack([one, u, v, u * u, u * v, v * v,
                                u ** 3, u ** 2 * v, u * v ** 2, v ** 3])
    if model == "poly4":
        return np.column_stack([one, u, v, u * u, u * v, v * v,
                                u ** 3, u ** 2 * v, u * v ** 2, v ** 3,
                                u ** 4, u ** 3 * v, u ** 2 * v ** 2, u * v ** 3, v ** 4])
    raise ValueError("unknown model: %s" % model)


def _tps_kernel(a, b):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = d2 * np.log(np.sqrt(d2))
    return np.nan_to_num(k)


# =============================================================================
#  the lattice as a coordinate system
# =============================================================================
#
# The dome grid is a regular mould, so its INDEX space (which dome, counting from
# a corner) is undistorted by construction. The measured pixel position of each
# dome is that same grid seen through the lens. So the measured node grid IS the
# distortion, tabulated, and going pixel -> index removes it without ever fitting
# a polynomial to it.
#
# That matters because the two halves of this calibration have very different
# precisions. Dome centres are measured to a fraction of a pixel from hundreds of
# domes. The contact position from a divergence peak is quantised by the lattice
# it is computed on -- tens of pixels per cell -- so it is a far noisier
# observation. Fitting a 10-coefficient polynomial to the noisy quantity when the
# precise quantity already describes the same distortion spends the good data to
# re-derive what was free. `lattice_affine` instead takes the SHAPE from the domes
# and asks the raster only for the six numbers it is actually needed for: scale,
# rotation, shear and origin.

def lattice_forward(node_px, rc):
    """Continuous (row, col) index -> pixel, bilinear over the MEASURED nodes."""
    node_px = np.asarray(node_px, float)
    rows, cols = node_px.shape[:2]
    rc = np.atleast_2d(np.asarray(rc, float))
    out = np.full((len(rc), 2), np.nan)
    for i, (rf, cf) in enumerate(rc):
        r0 = int(np.clip(np.floor(rf), 0, rows - 2))
        c0 = int(np.clip(np.floor(cf), 0, cols - 2))
        fr, fc = rf - r0, cf - c0
        q = node_px[r0:r0 + 2, c0:c0 + 2]
        if not np.isfinite(q).all():
            continue
        w = np.array([[(1 - fr) * (1 - fc), (1 - fr) * fc],
                      [fr * (1 - fc), fr * fc]])
        out[i] = ((w * q[..., 0]).sum(), (w * q[..., 1]).sum())
    return out


def lattice_inverse(node_px, uv, lat, iters=_INVERT_ITERS, margin=_EDGE_MARGIN_CELLS,
                    tol_px=_INVERT_TOL_PX):
    """Pixel -> continuous (row, col) index. Fixed point from the ideal lattice.

    Seeded with the ideal origin+pitch guess and corrected by the local Jacobian of
    the measured grid. Near-affine maps converge in two steps; there is no closed
    form because the grid is a table, not a formula.

    ⚠ RETURNS NaN OUTSIDE THE CALIBRATED FIELD, and that is the point. The domes
    are the calibration; beyond the outermost one there is no measurement, only a
    polynomial's opinion. An earlier version clipped the index to the lattice
    instead, which SATURATED: a pixel past the edge silently returned the edge
    dome's index, its local scale came out as 464 px/mm rather than ~16, and
    nothing anywhere said so. A NaN propagates into the caller's result and gets
    noticed; a plausible wrong number does not.

    `margin` allows a little honest extrapolation (the pad extends slightly past
    the outermost dome), and convergence is then checked by reprojection: a point
    that cannot be reprojected to within `tol_px` was never really solved.
    """
    node_px = np.asarray(node_px, float)
    rows, cols = node_px.shape[:2]
    uv = np.atleast_2d(np.asarray(uv, float))
    rc = np.column_stack([(uv[:, 1] - lat["origin_y"]) / lat["pitch_y"],
                          (uv[:, 0] - lat["origin_x"]) / lat["pitch_x"]])
    for _ in range(iters):
        p = lattice_forward(node_px, rc)
        err = uv - p                                     # pixels still to cover
        # local Jacobian d(pixel)/d(index) from a half-cell step, measured grid
        j_r = lattice_forward(node_px, rc + [0.5, 0.0]) - \
            lattice_forward(node_px, rc - [0.5, 0.0])
        j_c = lattice_forward(node_px, rc + [0.0, 0.5]) - \
            lattice_forward(node_px, rc - [0.0, 0.5])
        for i in range(len(rc)):
            J = np.array([[j_c[i, 0], j_r[i, 0]], [j_c[i, 1], j_r[i, 1]]])
            if not np.isfinite(J).all() or abs(np.linalg.det(J)) < 1e-9:
                continue
            d = np.linalg.solve(J, err[i])               # (d_col, d_row)
            rc[i] += [d[1], d[0]]
        rc[:, 0] = np.clip(rc[:, 0], -margin, rows - 1 + margin)
        rc[:, 1] = np.clip(rc[:, 1], -margin, cols - 1 + margin)

    bad = ~np.isfinite(np.linalg.norm(lattice_forward(node_px, rc) - uv, axis=1)) | \
        (np.linalg.norm(np.nan_to_num(lattice_forward(node_px, rc), nan=1e9) - uv,
                        axis=1) > tol_px)
    rc[bad] = np.nan
    return rc


def fit_model(uv, xy, model, norm, tps_lambda=1e-3, ctx=None):
    """Least squares (TPS regularised) -> a dict `predict` understands.

    `ctx` carries {"node_px", "lattice"} and is REQUIRED by lattice_affine, which
    is not a function of the pixel alone -- it reads the measured dome grid.
    """
    uv = np.atleast_2d(np.asarray(uv, float))
    xy = np.atleast_2d(np.asarray(xy, float))
    if model == "lattice_affine":
        if not ctx or "node_px" not in ctx or "lattice" not in ctx:
            raise ValueError("lattice_affine needs ctx={'node_px':..., 'lattice':...}")
        node_px = np.asarray(ctx["node_px"], float)
        rc = lattice_inverse(node_px, uv, ctx["lattice"])
        cr = rc[:, ::-1]                                  # (col, row), so it reads x,y
        n2 = normaliser(cr)
        A = design(cr, n2, "affine")
        coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
        return {"model": "lattice_affine", "norm": n2, "coeffs": coef.tolist(),
                "node_px": np.where(np.isfinite(node_px), node_px, None).tolist(),
                "lattice": ctx["lattice"]}
    if model == "tps":
        p = (uv - np.array(norm["centre"])) / norm["scale"]
        n = len(p)
        K = _tps_kernel(p, p) + tps_lambda * np.eye(n)
        P = np.column_stack([np.ones(n), p])
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T
        rhs = np.zeros((n + 3, 2))
        rhs[:n] = xy
        w = np.linalg.lstsq(L, rhs, rcond=None)[0]
        return {"model": "tps", "norm": norm, "control": p.tolist(),
                "weights": w.tolist(), "lambda": tps_lambda}
    A = design(uv, norm, model)
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return {"model": model, "norm": norm, "coeffs": coef.tolist()}


def _node_px_array(raw):
    """JSON stores unmeasured nodes as null; numpy wants NaN."""
    a = np.array([[(np.nan, np.nan) if p is None or p[0] is None else p for p in row]
                  for row in raw], float)
    return a


def predict(fit, uv):
    """Evaluate a fitted warp. -> (n, 2)."""
    uv = np.atleast_2d(np.asarray(uv, float))
    norm = fit["norm"]
    if fit["model"] == "lattice_affine":
        node_px = _node_px_array(fit["node_px"])
        rc = lattice_inverse(node_px, uv, fit["lattice"])
        return design(rc[:, ::-1], norm, "affine") @ np.array(fit["coeffs"])
    if fit["model"] == "tps":
        p = (uv - np.array(norm["centre"])) / norm["scale"]
        ctl = np.array(fit["control"])
        w = np.array(fit["weights"])
        n = len(ctl)
        return _tps_kernel(p, ctl) @ w[:n] + \
            np.column_stack([np.ones(len(p)), p]) @ w[n:]
    return design(uv, norm, fit["model"]) @ np.array(fit["coeffs"])


# =============================================================================
#  the calibration
# =============================================================================

class Calibration(object):
    def __init__(self, data, path=None):
        self.d = data
        self.path = path
        for k in ("pixel_to_mm", "mm_to_pixel"):
            if k not in data:
                raise ValueError("%s is missing '%s' -- was it written by "
                                 "fit_pixel_mm_warp.py?" % (path, k))

    # ---- the two directions -------------------------------------------------
    def pixel_to_mm(self, u, v):
        """Sensor pixel -> millimetres in the robot base frame."""
        uv = np.column_stack([np.ravel(u).astype(float), np.ravel(v).astype(float)])
        xy = predict(self.d["pixel_to_mm"], uv)
        if np.isscalar(u) or np.ndim(u) == 0:
            return float(xy[0, 0]), float(xy[0, 1])
        return xy[:, 0].reshape(np.shape(u)), xy[:, 1].reshape(np.shape(u))

    def mm_to_pixel(self, x, y):
        xy = np.column_stack([np.ravel(x).astype(float), np.ravel(y).astype(float)])
        uv = predict(self.d["mm_to_pixel"], xy)
        if np.isscalar(x) or np.ndim(x) == 0:
            return float(uv[0, 0]), float(uv[0, 1])
        return uv[:, 0].reshape(np.shape(x)), uv[:, 1].reshape(np.shape(x))

    def displacement_to_mm(self, u0, v0, u1, v1):
        """Both endpoints through the warp, THEN subtract. See the module note."""
        x0, y0 = self.pixel_to_mm(u0, v0)
        x1, y1 = self.pixel_to_mm(u1, v1)
        return (np.asarray(x1) - np.asarray(x0), np.asarray(y1) - np.asarray(y0))

    # ---- bulk ---------------------------------------------------------------
    def lut(self):
        """(H, W, 2) float32 of millimetres per pixel centre.

        For rendering a corrected preview or for a per-event lookup. NOT the way
        to correct marker measurements -- see the module docstring.
        """
        w, h = self.d.get("image_size", [640, 480])
        vv, uu = np.mgrid[0:h, 0:w]
        xy = predict(self.d["pixel_to_mm"],
                     np.column_stack([uu.ravel(), vv.ravel()]).astype(float))
        return xy.reshape(h, w, 2).astype(np.float32)

    def local_scale_px_per_mm(self, u, v, eps=1.0):
        """Numerical Jacobian at (u, v) -- how many pixels one millimetre spans HERE.

        The whole point of the correction is that this is not constant. Comparing
        it at the centre and at a corner is the fastest way to see the size of the
        effect on a quantity you actually care about.
        """
        x0, y0 = self.pixel_to_mm(u - eps, v)
        x1, y1 = self.pixel_to_mm(u + eps, v)
        x2, y2 = self.pixel_to_mm(u, v - eps)
        x3, y3 = self.pixel_to_mm(u, v + eps)
        du = np.hypot(x1 - x0, y1 - y0) / (2 * eps)       # mm per px along u
        dv = np.hypot(x3 - x2, y3 - y2) / (2 * eps)
        return (1.0 / du if du else float("nan"),
                1.0 / dv if dv else float("nan"))

    # ---- provenance ---------------------------------------------------------
    def summary(self):
        d = self.d
        g = d.get("affine_geometry", {})
        cvs = d.get("cross_validated_residuals_mm", [])
        best = min(cvs, key=lambda c: c["rms_mm"]) if cvs else None
        aff = next((c for c in cvs if c["model"] == "affine"), None)
        lines = [
            "calibration : %s" % (self.path or "<in memory>"),
            "from run    : %s" % d.get("source_run", "?"),
            "created     : %s" % d.get("created_utc", "?"),
            "model       : %s   (%d pairs over %d raster points)"
            % (d["pixel_to_mm"]["model"], d.get("n_pairs", 0),
               d.get("n_raster_points", 0)),
            "frame       : %s" % d.get("frame", "?"),
        ]
        if g:
            lines += ["scale       : %.3f and %.3f px/mm along the image axes "
                      "(anisotropy %.3f)" % (g["px_per_mm_major"],
                                             g["px_per_mm_minor"], g["anisotropy"]),
                      "rotation    : %.2f deg vs the robot XY frame"
                      % g["rotation_deg"]]
        dv = d.get("lattice_deviation_px")
        if dv:
            lines.append("distortion  : %.2f px RMS, %.2f px max, from the dome lattice"
                         % (dv["rms_px"], dv["max_px"]))
        if best and aff:
            lines.append("residual    : %.3f mm (%s), vs %.3f mm doing nothing (affine)"
                         % (best["rms_mm"], best["model"], aff["rms_mm"]))
        return "\n".join(lines)


def load(path=None):
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            "%s not found. Fit one first:\n"
            "    python3 ml/fit_pixel_mm_warp.py <calib_run>\n"
            "and see franka/CALIB_RASTER.md for how to record the run." % p)
    return Calibration(json.loads(p.read_text()), path=p)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Inspect the pixel -> mm calibration.")
    ap.add_argument("--path", default=None)
    ap.add_argument("--demo", action="store_true",
                    help="show the local scale at the centre and the four corners")
    a = ap.parse_args()
    try:
        cal = load(a.path)
    except FileNotFoundError as e:
        print(e)
        return 1
    print(cal.summary())
    if a.demo:
        w, h = cal.d.get("image_size", [640, 480])
        print("\n  local scale (px/mm), by where you are in the field:")
        print("    %-14s %-9s %-9s" % ("position", "along u", "along v"))
        for label, (u, v) in [("centre", (w / 2, h / 2)),
                              ("top-left", (0.1 * w, 0.1 * h)),
                              ("top-right", (0.9 * w, 0.1 * h)),
                              ("bottom-left", (0.1 * w, 0.9 * h)),
                              ("bottom-right", (0.9 * w, 0.9 * h))]:
            su, sv = cal.local_scale_px_per_mm(u, v)
            print("    %-14s %8.3f  %8.3f" % (label, su, sv))
        print("\n  If those rows differ, that difference IS the fisheye, expressed")
        print("  in the units your measurements are actually in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
