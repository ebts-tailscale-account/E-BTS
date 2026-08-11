#!/usr/bin/env python3
"""
campaign_planb.py -- indentation campaign, block-randomised, with crash-resume.

⭐ USE --depths-mm.  DEPTH IS THE DESIGN VARIABLE.
--------------------------------------------------
    --depths-mm 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0

Depth is what the robot actually controls: batch planb_20260810_145750 hit its
commanded depth to 0.003 +- 0.061 mm. Force is a CONSEQUENCE, measured by the HEX21
and used as the label. The causal chain is

    depth -> deformation -> (what the camera sees, and the force)

so depth is upstream of BOTH the input and the label, which makes it the natural
thing to design on.

⚠ THE FORCE-TARGETED PATH IS KEPT ONLY FOR REFERENCE.  It inverts the stiffness map
to aim at a target force. In practice it MISSED BY 1.25-2.06x, because the map's
stiffness is a secant fitted over 0-0.6 N and the campaign asked for 2.4 N. Worse,
it adds the impedance sag on top of dip_to_depth()'s closed loop, so the allowance
becomes extra penetration. The result was neither clean depths nor the intended
forces. Do not use it without re-reading HANDOFF section 17.3.

The original argument for it was that a fixed depth ladder confounds force with
position (stiffness varies 2.82x here). That is real but much weaker than it looks:
under a depth ladder location explains ~23% of force variance, versus ~5.5% under
force targeting -- and a model predicting force MUST learn the stiffness field
anyway, since it is genuine physics of this sensor, not leakage. The pilot's actual
failure was different: depth was CONSTANT at 2 mm, so force was entirely determined
by location and the model could ignore deformation completely. A ladder does not do
that.

Three further design choices, each for a stated reason:

  * RANDOMISED COMPLETE BLOCK.  N passes over the grid; each location receives
    each force level exactly once across the passes, in a per-location random
    order.  Force level is therefore decorrelated from position within a pass,
    and every pass spans the full force range -- so thermal drift and strap creep
    hit all levels roughly equally instead of aliasing onto one.  A consequence
    worth knowing: STOPPING EARLY STILL LEAVES A BALANCED DATASET.

  * SERPENTINE WITHIN A PASS, ALTERNATING DIRECTION.  Traverse efficiency without
    confounding row parity with sweep direction (that artifact cost us three wrong
    conclusions during surface mapping -- HANDOFF section 16.4b).

  * PRE-CONDITIONING FIRST.  Filled elastomers show the Mullins effect: the first
    large deformation permanently softens the material.  Our own ladder measured
    stiffness falling 0.632 -> 0.368 N/mm (r = -0.826).  Because the block design
    randomises level order, an early deep press at a location would soften the
    material under a later shallow press at the SAME location -- unmodelled
    variance eating into a narrow force range.  --precondition cycles each
    location to its own campaign maximum first, so everything is measured on a
    settled material.

RESUME
------
A ~3 h run that cannot recover from an interruption is a run you have to babysit.
Progress is appended to state.jsonl and fsync'd after every poke, so a hard kill
(Ctrl-C, power, ROS death, reflex) loses at most the poke in flight.  --resume
rebuilds the identical plan, skips what is already done, and continues.

  * The plan is FINGERPRINTED.  If any parameter that changes the plan differs on
    resume, the script aborts rather than silently stitching two different
    campaigns together.
  * Each resume opens a NEW RECORDING SEGMENT (franka_segNN.csv) and records the
    segment index on every poke, because the event-camera recording is a separate
    file per session.  Postprocessing joins on (segment, seq).
  * Nothing is ever overwritten: --fresh archives the previous state first.

⚠ WHERE THE HEX21 LIVES, AND WHAT THAT MEANS HERE
-------------------------------------------------
The HEX21's HOME IS THE WORKSTATION (HANDOFF section 3.2 -- it goes to tactile only
for surface mapping/probing).  So this script, running on tactile, normally CANNOT
read force: --run and --precondition command depth open-loop, protected by the
depth cap and the arm's ~20 N reflex.  Force is recorded on the workstation by the
GUI and joined in post.

That is fine, because nothing here needs force in the loop.  It does mean the
depth->force inversion must come from somewhere:

  * BEST, no replug -- fit it from a RECORDED --precondition run.  Every location
    is pressed to a known depth with the workstation recording force, which is
    exactly the (depth, force, stiffness) data the model needs.
  * --calibrate fits it directly, but reads the HEX21 LOCALLY, so it only works
    with the sensor temporarily moved to tactile.
  * Uncalibrated (linear) is SAFE and the LABELS STAY CORRECT -- force is measured,
    never inferred.  You simply land ~19% short at the top of the range.

ORDER OF OPERATIONS  (launched from the WORKSTATION via master_campaign.py)
---------------------------------------------------------------------------
    1.  --precondition   Cycles every location to its own campaign max.  Settles
                         the Mullins softening, doubles as the STRAP TEST at full
                         depth, and records the data step 2 fits.
    2.  (fit force_model.json from that recording -- offline, no robot)
    3.  --run            The campaign.  --resume after any interruption.

    python3 master_campaign.py planb --remote-script campaign_planb.py \
        --remote-args --ack-deep --force-model force_model.json

    See PLAN_B_RUNBOOK.md.  --remote-args must come LAST.

Everything for one campaign lives in ONE directory:

    ~/E-BTS/recordings/planb_<stamp>/
        plan.csv           the enumerated pokes, in execution order
        plan.json          parameters + fingerprint
        state.jsonl        append-only progress (the resume ledger)
        force_model.json   the fitted depth->force inversion
        franka_segNN.csv   one franka_states log per segment
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

MAP_CSV = Path(__file__).with_name("surface_offset_map.csv")
RUNS_DIR = Path.home() / "E-BTS" / "recordings"

# --- geometry / motion -------------------------------------------------------
HOVER_MM = 5.0          # travel / retract / tare at this height above the surface
APPROACH_MM = 15.0      # from home, first go this far above the taught centre
DWELL_S = 2.0           # the measurement window
TARE_S = 1.0            # no-load window before every dip (its own F/T zero)

# --- depth safety ------------------------------------------------------------
# DEPTH now demonstrated to 5.9 mm, not 4.0. Superseded 2026-08-11 by our own runs:
#   precond_20260810_142103   282 presses, max achieved 5.93 mm
#   planb_3mm_20260810_145750 1008 pokes, max achieved 5.94 mm
#   planb_3mm_20260811_150324  896 pokes, max achieved 4.06 mm
# 337 of those went deeper than 4.5 mm with zero incidents. The old 4.0 figure came
# from pilot_20260807_134855 and was simply out of date, which made the gate refuse
# depths the rig had already proven.
DEMONSTRATED_DEPTH_MM = 5.9
# ⚠ DEPTH IS NOT THE ONLY LIMIT. Force is now the binding one: the straps were only
# validated to 5.67 N (precondition), and batch 2 already measured 7.16 N at 3.88 mm
# because it pressed the STIFFEST locations deep -- which the precondition run never
# did (it pressed each point only to its own campaign max). Hence FORCE_WARN_N.
FORCE_WARN_N = 8.0            # depth_limit_probe's abort ceiling; arm reflex is ~20 N
DEFAULT_MAX_DEPTH_MM = 6.0
HARD_DEPTH_CEILING_MM = 8.0   # typo guard. The straps are the real limit (~10 mm).

# --- impedance sag -----------------------------------------------------------
# Measured over the 99-point map: achieved depth falls short of the command by
# 0.39 mm mean, correlating with contact force at r = +0.79 ->
#     shortfall = SAG_CONST_MM + F / SAG_STIFFNESS_N_PER_MM
# dip_to_depth() closes the loop on measured EE height and mostly removes this,
# but the PLAN still needs it to choose a starting command and, more importantly,
# to decide what force is reachable inside the depth cap.
SAG_CONST_MM = 0.10
SAG_STIFFNESS_N_PER_MM = 3.8

# --- force model -------------------------------------------------------------
# The map's stiffness column is a SMALL-SIGNAL fit: teach_surface collects only up
# to FIT_FORCE_MAX_N = 0.60 N.  Extrapolating a 0.6 N secant to 2 N against a
# material that measurably softens is unsupported, so --calibrate fits a power law
#     F = a * d^b        (b < 1 == softening)
# per calibration location and reports the mean exponent.  Each grid location's
# scale `a` is then pinned to ITS OWN measured small-signal stiffness, so the map's
# spatial information is kept and only the SHAPE comes from the ladder:
#     k = F_FIT / d(F_FIT)  =>  a = F_FIT * (k / F_FIT) ** b
# With b = 1 this reduces exactly to the linear a = k.  Verified in self_test().
MAP_FIT_FORCE_N = 0.60
DEFAULT_EXPONENT = 1.0          # linear until --calibrate says otherwise

SEED = 20260810


# =============================================================================
#  force model
# =============================================================================

def scale_from_stiffness(k, b):
    """Power-law scale `a` for a location whose small-signal stiffness is k."""
    return MAP_FIT_FORCE_N * (k / MAP_FIT_FORCE_N) ** b


def depth_achieved_for_force(force_n, k, b):
    """Elastomer depth (mm) needed to develop `force_n` at stiffness k."""
    if force_n <= 0:
        return 0.0
    a = scale_from_stiffness(k, b)
    return (force_n / a) ** (1.0 / b)


def depth_command_for_force(force_n, k, b):
    """Commanded depth (mm) -- elastomer depth plus the impedance shortfall."""
    return (depth_achieved_for_force(force_n, k, b)
            + SAG_CONST_MM + force_n / SAG_STIFFNESS_N_PER_MM)


def max_reachable_force(k, b, max_depth_mm):
    """Largest force obtainable at this location inside the depth cap."""
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if depth_command_for_force(mid, k, b) <= max_depth_mm:
            lo = mid
        else:
            hi = mid
    return lo


def load_force_model(path):
    """Return the softening exponent b, or the linear default if absent."""
    if path is None or not Path(path).exists():
        return DEFAULT_EXPONENT, None
    data = json.loads(Path(path).read_text())
    return float(data["exponent_b"]), data


# =============================================================================
#  the surface + stiffness map
# =============================================================================

def load_map(path, use_poor=False):
    """Locations from surface_offset_map.csv.

    Only quality_ok rows are used by default: the others have no trustworthy
    surface height OR stiffness, and both are needed to aim a force-targeted
    press.  Skipping 5 of 99 is cheaper than pressing blind at 5.
    """
    if not path.exists():
        sys.exit("[ERROR] %s not found -- run map_offset.py first." % path)
    locs, skipped = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            ok = str(r.get("quality_ok", "")).strip().lower() == "true"
            rec = {
                "point_index": int(r["point_index"]),
                "row": int(r["row"]),
                "col": int(r["col"]),
                "x": float(r["x_m"]),
                "y": float(r["y_m"]),
                "surface_z": float(r["surface_z_m"]),
                "k": float(r["stiffness_n_per_mm"]),
                "fit_r2": float(r.get("fit_r2", "nan") or "nan"),
                "quality_ok": ok,
            }
            if ok or use_poor:
                locs.append(rec)
            else:
                skipped.append(rec)
    if not locs:
        sys.exit("[ERROR] no usable points in %s." % path)
    return locs, skipped


def serpentine(locs, reverse_first=False):
    """Row-major walk, alternating direction each row; optionally start flipped.

    Alternating the START direction between passes keeps any residual
    sweep-direction effect from correlating with row parity.
    """
    by_row = {}
    for l in locs:
        by_row.setdefault(l["row"], []).append(l)
    out = []
    for i, row in enumerate(sorted(by_row)):
        band = sorted(by_row[row], key=lambda l: l["col"])
        flip = (i % 2 == 1)
        if reverse_first:
            flip = not flip
        out.extend(reversed(band) if flip else band)
    return out


# =============================================================================
#  plan
# =============================================================================

def derive_levels(locs, b, max_depth_mm, n_levels, force_min_n, force_max_n=None):
    """Force levels that EVERY location can reach inside the depth cap.

    The most compliant point governs: asking for a force it cannot reach would
    silently clip that location's deepest presses and reintroduce the very
    force/position confound this campaign exists to remove.
    """
    ceilings = [max_reachable_force(l["k"], b, max_depth_mm) for l in locs]
    ceiling = min(ceilings)
    limiter = locs[int(np.argmin(ceilings))]
    if force_max_n is not None:
        if force_max_n > ceiling + 1e-9:
            sys.exit("[ERROR] --force-max %.2f N is unreachable at r%d c%d "
                     "(k = %.3f N/mm), whose ceiling inside %.1f mm is %.2f N.\n"
                     "        Raise --max-depth-mm, lower --force-max, or accept "
                     "the derived ceiling by omitting --force-max."
                     % (force_max_n, limiter["row"], limiter["col"], limiter["k"],
                        max_depth_mm, ceiling))
        top = force_max_n
    else:
        top = math.floor(ceiling * 50.0) / 50.0     # round DOWN to 0.02 N
    if top <= force_min_n:
        sys.exit("[ERROR] reachable ceiling %.2f N is not above --force-min %.2f N."
                 % (top, force_min_n))
    levels = [force_min_n + (top - force_min_n) * i / (n_levels - 1)
              for i in range(n_levels)]
    return levels, ceiling, limiter


def build_plan(locs, levels, b, seed, replicate_n, replicate_extra, depths=None,
               sequential=False):
    """Randomised complete block + an optional replication tail.

    Passes == number of levels, so each location gets each level exactly once.

    TWO WAYS TO DEFINE A LEVEL
    --------------------------
    depths is None  -> FORCE-targeted. `levels` are newtons; the depth is derived by
                       inverting the stiffness map. Kept for reference, but see the
                       warning below.
    depths given    -> DEPTH-targeted (recommended). `depths` are millimetres and are
                       commanded VERBATIM at every location.

    ⚠ WHY DEPTH-TARGETED IS THE DEFAULT NOW.  Depth is the variable the robot actually
    controls -- batch planb_20260810_145750 hit commanded depth to 0.003 +- 0.061 mm.
    Force is a CONSEQUENCE we measure. Force-targeting in that same batch missed its
    targets by 1.25-2.06x, because it inverts a stiffness map fitted over 0-0.6 N and
    extrapolates it to 2.4 N. So it delivered neither clean depths nor the intended
    forces. Command the thing you can hit; measure the thing you cannot.

    The causal chain is depth -> deformation -> (image, force): depth is upstream of
    BOTH the thing the camera sees and the label, so it is the natural design variable.

    ⚠ NO SAG TERM in depth mode. dip_to_depth() already closes the loop on measured EE
    height. Adding the impedance shortfall on top -- as the force path does -- makes the
    allowance become extra penetration, which is exactly the 1.25-2.06x overshoot.
    """
    rng = random.Random(seed)
    n_pass = len(depths) if depths is not None else len(levels)

    # per-location permutation of level indices, one per pass
    if sequential:
        # Every location gets level p on pass p: a clean ascending sweep.
        # ⚠ COST: level becomes perfectly correlated with time. All 94 pokes at a
        # given depth happen inside one contiguous ~14 min window, so any drift
        # (thermal, strap creep, illumination) or one-off disturbance lands on ONE
        # depth level and is indistinguishable from a real depth effect. The
        # randomised design exists precisely to spread that across all levels.
        # Batch planb_3mm_20260811_150324 measured corr(force, time) = +0.079 and a
        # drift of order 0.1 N over 115 min, so the distortion is small but real.
        # MITIGATION: the replication tail repeats early conditions at the END of the
        # run, which measures the drift directly -- keep --replicate-locations > 0.
        perm = {l["point_index"]: list(range(n_pass)) for l in locs}
    else:
        perm = {l["point_index"]: rng.sample(range(n_pass), n_pass) for l in locs}

    plan = []

    def emit(loc, li, pass_idx, block, rep=0):
        if depths is not None:
            d_cmd = float(depths[li])                     # commanded verbatim
            d_tgt = d_cmd
            # informational only: what this depth is PREDICTED to develop here
            f = scale_from_stiffness(loc["k"], b) * d_cmd ** b
        else:
            f = levels[li]
            d_tgt = depth_achieved_for_force(f, loc["k"], b)
            d_cmd = depth_command_for_force(f, loc["k"], b)
        plan.append({
            "seq": len(plan),
            "block": block,
            "pass": pass_idx,
            "repeat": rep,
            "point_index": loc["point_index"],
            "row": loc["row"], "col": loc["col"],
            "x": loc["x"], "y": loc["y"],
            "surface_z": loc["surface_z"],
            "stiffness_n_per_mm": loc["k"],
            "level_idx": li,
            "target_depth_mm": d_tgt,
            "target_force_n": f,
            "depth_cmd_mm": d_cmd,
        })

    for p in range(n_pass):
        for loc in serpentine(locs, reverse_first=(p % 2 == 1)):
            emit(loc, perm[loc["point_index"]][p], p, "main")

    # --- replication tail: repeated identical conditions -> the noise floor ---
    # Without repeated (location, force) pairs there is no way to separate model
    # error from sensor noise, and the whole error budget becomes unattributable.
    if replicate_n > 0 and replicate_extra > 0:
        chosen = pick_spread(locs, replicate_n)
        for rep in range(replicate_extra):
            for loc in serpentine(chosen, reverse_first=(rep % 2 == 1)):
                for li in range(n_pass):        # NOT len(levels) -- empty in depth mode
                    emit(loc, li, n_pass + rep, "replicate", rep=rep + 1)
    return plan


def pick_spread(locs, n, include_extremes=False):
    """n locations spread over the grid AND over the stiffness range.

    Greedy farthest-point in (row, col, stiffness) so the selection does not
    accidentally sample one corner or one stiffness regime.

    include_extremes pins the softest and stiffest points into the selection. Use
    it for CALIBRATION: the softest point sets the campaign's reachable force
    ceiling, so it is precisely where the force model must not be extrapolated.
    Leave it off for the replication block, where even spread is what matters.
    """
    if n >= len(locs):
        return list(locs)
    rr = np.array([l["row"] for l in locs], float)
    cc = np.array([l["col"] for l in locs], float)
    kk = np.array([l["k"] for l in locs], float)
    def norm(v):
        s = v.max() - v.min()
        return (v - v.min()) / s if s > 0 else v * 0.0
    P = np.stack([norm(rr), norm(cc), norm(kk)], axis=1)
    if include_extremes:
        picked = list({int(np.argmin(kk)), int(np.argmax(kk))})
    else:
        picked = [int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))]
    while len(picked) < n:
        d = np.min(np.linalg.norm(P[:, None, :] - P[picked][None, :, :], axis=2), axis=1)
        d[picked] = -1.0
        picked.append(int(np.argmax(d)))
    return [locs[i] for i in sorted(picked)]


def file_sha256(path):
    """Content hash. The plan depends on what the map SAYS, not where it lives."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def plan_fingerprint(plan, params):
    """Identity of a plan. Resume refuses to continue across a change."""
    h = hashlib.sha256()
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    for p in plan:
        h.update(("%d|%d|%.6f|%.6f|%.5f|%.5f"
                  % (p["seq"], p["point_index"], p["x"], p["y"],
                     p["target_force_n"], p["depth_cmd_mm"])).encode())
    return h.hexdigest()[:16]


PLAN_FIELDS = ["seq", "block", "pass", "repeat", "point_index", "row", "col",
               "x", "y", "surface_z", "stiffness_n_per_mm", "level_idx",
               "target_depth_mm", "target_force_n", "depth_cmd_mm"]


def save_plan(run_dir, plan, params, fp, meta_only=None):
    with open(run_dir / "plan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        w.writeheader()
        for p in plan:
            w.writerow({k: p[k] for k in PLAN_FIELDS})
    meta = dict(params)
    meta.update(meta_only or {})
    meta["fingerprint"] = fp
    meta["n_pokes"] = len(plan)
    meta["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run_dir / "plan.json").write_text(json.dumps(meta, indent=2, default=str))


# =============================================================================
#  resume ledger
# =============================================================================

class Ledger(object):
    """Append-only progress file. fsync'd per record so a kill loses <= 1 poke."""

    def __init__(self, path, fingerprint):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.done = set()
        self.segment = 0
        self._fh = None

    def load(self):
        """Read completed seqs. Returns (n_records, mismatch_reason_or_None)."""
        if not self.path.exists():
            return 0, None
        n = 0
        max_seg = -1
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue                     # a torn final line: ignore it
                if rec.get("_header"):
                    if rec.get("fingerprint") != self.fingerprint:
                        return n, ("plan fingerprint %s in the ledger does not match "
                                   "the plan this invocation builds (%s)"
                                   % (rec.get("fingerprint"), self.fingerprint))
                    continue
                if "seq" not in rec:
                    continue
                self.done.add(int(rec["seq"]))
                max_seg = max(max_seg, int(rec.get("segment", 0)))
                n += 1
        self.segment = max_seg + 1
        return n, None

    def archive(self):
        if not self.path.exists():
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dst = self.path.with_name("%s.superseded_%s" % (self.path.name, stamp))
        shutil.copy2(str(self.path), str(dst))
        self.path.unlink()
        self.done.clear()
        self.segment = 0
        return dst

    def open(self, params):
        new = not self.path.exists()
        self._fh = open(self.path, "a")
        if new:
            self._write({"_header": True, "fingerprint": self.fingerprint,
                         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()),
                         "params": params})
        return self

    def _write(self, rec):
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def record(self, **rec):
        rec["segment"] = self.segment
        rec["t_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write(rec)
        self.done.add(int(rec["seq"]))

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# =============================================================================
#  reporting
# =============================================================================

def report(plan, locs, skipped, levels, ceiling, limiter, b, model, args, run_dir,
           ledger, remaining, depths=None):
    n_lvl = len(depths) if depths else len(levels)
    bar = "=" * 78
    print(bar)
    print(" PLAN B CAMPAIGN -- force-targeted, block-randomised, resumable")
    print(bar)
    print("  run directory        : %s" % run_dir)
    print("  map                  : %s" % args.map)
    print("  locations            : %d usable%s"
          % (len(locs), "" if not skipped else
             "  (%d skipped: quality_ok = False)" % len(skipped)))
    if skipped:
        print("                         skipped r/c: %s"
              % ", ".join("r%dc%d" % (s["row"], s["col"]) for s in skipped))
    ks = [l["k"] for l in locs]
    print("  stiffness            : %.3f - %.3f N/mm  (%.2fx spread)"
          % (min(ks), max(ks), max(ks) / min(ks)))
    if depths:
        # In depth mode the force model is used ONLY to print a predicted force for
        # information. It cannot bias the experiment, because nothing is aimed at
        # force -- the depths are commanded verbatim. So no warning is warranted.
        print("  force model          : b = %.4f, used only to PREDICT force for the"
              % b)
        print("                         report. It does not affect what is commanded.")
    elif model is None:
        print("  force model          : ⚠ LINEAR (b = 1.000) -- NOT CALIBRATED.")
        print("                         The map's stiffness is a %.2f N secant; using"
              % MAP_FIT_FORCE_N)
        print("                         it up to %.2f N assumes no softening, and the"
              % levels[-1])
        print("                         ladder measured softening (0.632 -> 0.368 N/mm)."
              )
        print("                         Fit one from a recorded --precondition run,")
        print("                         or --calibrate with the HEX21 moved to tactile.")
    else:
        print("  force model          : power law F = a*d^%.4f  (from %s)"
              % (b, model.get("source", "force_model.json")))
        print("                         fitted at %d locations, R2 %.4f - %.4f"
              % (model.get("n_locations", 0), model.get("r2_min", float("nan")),
                 model.get("r2_max", float("nan"))))
    _deep = max(p["depth_cmd_mm"] for p in plan)      # what is actually commanded
    print("  deepest commanded    : %.2f mm  (cap %.2f)%s"
          % (_deep, args.max_depth_mm,
             "" if _deep <= DEMONSTRATED_DEPTH_MM
             else "   ⚠ beyond the %.1f mm demonstrated limit" % DEMONSTRATED_DEPTH_MM))
    if depths:
        pf = [p["target_force_n"] for p in plan]
        print("  MODE                 : ⭐ DEPTH-TARGETED -- these depths are commanded")
        print("                         verbatim; force is MEASURED, not aimed at.")
        print("  level order          : %s"
              % ("⚠ SEQUENTIAL -- level is confounded with time"
                 if args.sequential else "randomised per location (recommended)"))
        print("  depth levels (%d)     : %s mm"
              % (n_lvl, "  ".join("%.2f" % d for d in depths)))
        print("  predicted force      : %.2f - %.2f N (from the stiffness map; "
              "informational only)" % (min(pf), max(pf)))
        # ⚠ The map's stiffness is a 0-0.6 N secant and UNDER-predicts at depth: batch 2
        # measured effective stiffness up to 1.85 N/mm against the map's 1.32 max. Scale
        # the warning by that measured ratio so it does not lull anyone.
        worst = max(pf) * (1.85 / max(l["k"] for l in locs))
        if worst > FORCE_WARN_N:
            print("  ⚠ ⚠  scaling by the stiffness MEASURED at depth (1.85 N/mm vs the")
            print("        map's %.2f), the stiffest location may reach ~%.1f N."
                  % (max(l["k"] for l in locs), worst))
            print("        That is past the %.0f N probe ceiling and past the %.2f N the"
                  % (FORCE_WARN_N, 5.67))
            print("        straps were validated at. The arm reflex (~20 N) still")
            print("        protects the robot; the MOUNTING is what is untested.")
    else:
        print("  reachable ceiling    : %.3f N, set by r%d c%d (k = %.3f N/mm, the most "
              "compliant point)" % (ceiling, limiter["row"], limiter["col"], limiter["k"]))
        print("  force levels (%d)     : %s N"
              % (n_lvl, "  ".join("%.2f" % f for f in levels)))
    dmin = min(p["depth_cmd_mm"] for p in plan)
    dmax = max(p["depth_cmd_mm"] for p in plan)
    print("  commanded depths     : %.2f - %.2f mm" % (dmin, dmax))
    main = [p for p in plan if p["block"] == "main"]
    reps = [p for p in plan if p["block"] == "replicate"]
    print("  main block           : %d passes x %d locations = %d pokes"
          % (n_lvl, len(locs), len(main)))
    if reps:
        rl = sorted(set(p["point_index"] for p in reps))
        print("  replication block    : %d locations x %d levels x %d extra = %d pokes"
              % (len(rl), n_lvl, args.replicate_extra, len(reps)))
        print("                         -> %d conditions observed %d times each"
              % (len(rl) * n_lvl, args.replicate_extra + 1))
    print("  TOTAL                : %d pokes" % len(plan))
    per = TARE_S + args.dwell_s + 4.5          # measured 7.55 s/poke on the pilot
    print("  estimated time       : %.1f h at %.1f s/poke  (pilot measured 7.55)"
          % (len(plan) * per / 3600.0, per))
    if ledger.done:
        print("  ─" * 38)
        print("  RESUME               : %d of %d already done, %d remaining (%.1f h)"
              % (len(ledger.done), len(plan), len(remaining),
                 len(remaining) * per / 3600.0))
        print("  recording segment    : %02d  (franka_seg%02d.csv)"
              % (ledger.segment, ledger.segment))
    print(bar)


# =============================================================================
#  modes that need the robot
# =============================================================================

def _ros_setup(args, note):
    """Shared bring-up. Imported lazily so --dry-run needs no ROS."""
    import rospy
    from franka_msgs.msg import FrankaState
    import panda_fk as pfk
    import map_surface as m
    from home_and_level import flat_down_quat, tilt_deg
    from servo_client import CartesianServo
    from franka_grid_logger import (Ctx, FrankaLogger, clear_reflex, warn_if_loaded,
                                    hold, dip_to_depth)

    rospy.init_node("campaign_planb", disable_signals=True)
    if not args.no_recovery:
        clear_reflex()
    warn_if_loaded()
    servo = CartesianServo()
    points = m.load_points()

    st = rospy.wait_for_message("/franka_state_controller/franka_states",
                                FrankaState, timeout=5.0)
    pfk.self_check(st.q, st.O_T_EE, tol_mm=1.0, label="live")

    ns = dict(rospy=rospy, m=m, pfk=pfk, servo=servo, points=points,
              flat_down_quat=flat_down_quat, tilt_deg=tilt_deg, Ctx=Ctx,
              FrankaLogger=FrankaLogger, hold=hold, dip_to_depth=dip_to_depth)
    print("[SETUP] %s" % note)
    return ns


def _preflight(ns, args, grid_like, max_depth_mm):
    m, servo, points = ns["m"], ns["servo"], ns["points"]
    saved_depth, saved_approach = m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM
    m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = max_depth_mm, APPROACH_MM
    try:
        m.preflight_reach(servo, points,
                          [{"x": g["x"], "y": g["y"], "z_plane": g["surface_z"]}
                           for g in grid_like], HOVER_MM / 1000.0)
    finally:
        m.MAX_DEPTH_MM, m.CENTER_APPROACH_MM = saved_depth, saved_approach


def run_campaign(args, plan, locs, run_dir, ledger, remaining):
    """The main loop. Every completed poke is committed to the ledger."""
    ns = _ros_setup(args, "campaign -- HEX21 is NOT required on tactile "
                          "(it records on the workstation)")
    m, servo, points = ns["m"], ns["servo"], ns["points"]
    Ctx, FrankaLogger, hold = ns["Ctx"], ns["FrankaLogger"], ns["hold"]
    dip_to_depth = ns["dip_to_depth"]

    _preflight(ns, args, locs, max(p["depth_cmd_mm"] for p in plan))

    out_csv = run_dir / ("franka_seg%02d.csv" % ledger.segment)
    ctx = Ctx()
    logger = FrankaLogger(str(out_csv), ctx, log_hz=args.log_hz)
    print("[RUN] segment %02d -> %s | %d pokes to do"
          % (ledger.segment, out_csv.name, len(remaining)))

    hover = HOVER_MM / 1000.0
    t0 = time.time()
    done_here = 0
    try:
        ctx.set("home")
        _, q0 = servo.current_pose()
        quat = ns["flat_down_quat"](q0) if not args.no_level else list(q0)
        print("[RUN] flange tilt from vertical: %.2f deg" % ns["tilt_deg"](q0))
        cx, cy, cz = points["center"]["xyz"]
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat,
                      name="home -> hover above centre")

        for p in remaining:
            x, y, sz = p["x"], p["y"], p["surface_z"]
            d_mm = p["depth_cmd_mm"]
            hover_z = sz + hover
            el = time.time() - t0
            eta = el / done_here * (len(remaining) - done_here) if done_here else 0.0
            print("[%4d/%4d] seq %-5d r%-2d c%-2d  %.2f N -> %.2f mm   "
                  "elapsed %.0f min, ETA %.0f min"
                  % (done_here + 1, len(remaining), p["seq"], p["row"], p["col"],
                     p["target_force_n"], d_mm, el / 60, eta / 60), flush=True)

            ctx.set("travel", idx=p["seq"], col=p["col"], row=p["row"],
                    target=(x, y, hover_z), surface_z=sz)
            m._gross_move(servo, x, y, hover_z, quat, name="travel")

            # TARE: still, OUT OF CONTACT. This poke's own F/T zero and the
            # undeformed reference frame the model differences against.
            ctx.set("tare", target=(x, y, hover_z))
            hold(servo, x, y, hover_z, quat, args.tare_s)

            ctx.set("dip", target=(x, y, sz - d_mm / 1000.0))
            got, cmd_z, iters = dip_to_depth(servo, x, y, sz, d_mm / 1000.0, quat,
                                             correct=not args.no_depth_correction)

            ctx.set("dwell", target=(x, y, cmd_z))
            hold(servo, x, y, cmd_z, quat, args.dwell_s)

            ctx.set("retract", target=(x, y, hover_z))
            m._gross_move(servo, x, y, hover_z, quat, name="retract")

            # commit AFTER the retract, so an interrupted poke is simply redone
            ledger.record(seq=p["seq"], point_index=p["point_index"],
                          row=p["row"], col=p["col"], block=p["block"],
                          pass_idx=p["pass"], level_idx=p["level_idx"],
                          target_force_n=round(p["target_force_n"], 4),
                          depth_cmd_mm=round(d_mm, 4),
                          achieved_mm=round(got * 1e3, 4),
                          commanded_z=round(cmd_z, 6), iters=iters,
                          csv=out_csv.name)
            done_here += 1

        ctx.set("park")
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat,
                      name="park above centre")
        ctx.set("done")
    finally:
        logger.close()
        ledger.close()
        n_done = len(ledger.done)
        print("\n[RUN] segment %02d: %d pokes this session, %d/%d total."
              % (ledger.segment, done_here, n_done, len(plan)))
        print("[RUN] franka_states -> %s (%d samples)" % (out_csv, logger.count))
        print("[RUN] ledger        -> %s" % ledger.path)
        if n_done < len(plan):
            print("\n  %d pokes remain. Resume with:\n    python3 %s --resume "
                  "--run-dir %s" % (len(plan) - n_done, Path(__file__).name, run_dir))
        else:
            print("\n  ✅ CAMPAIGN COMPLETE -- all %d pokes done." % len(plan))
        print("[RUN] wall time this session: %.1f min" % ((time.time() - t0) / 60))


def run_precondition(args, plan, locs, run_dir):
    """Cycle every location to ITS OWN campaign maximum, N times.

    Not to a global depth: at a stiff location the campaign never goes deeper than
    ~2.5 mm, so conditioning it at 6 mm would impose strain the experiment never
    repeats. Each location is conditioned to exactly the strain it will see.
    """
    deepest = {}
    for p in plan:
        pi = p["point_index"]
        deepest[pi] = max(deepest.get(pi, 0.0), p["depth_cmd_mm"])

    ns = _ros_setup(args, "PRE-CONDITIONING -- HEX21 on tactile is recommended so "
                          "you can watch the force")
    m, servo, points = ns["m"], ns["servo"], ns["points"]
    Ctx, FrankaLogger, hold = ns["Ctx"], ns["FrankaLogger"], ns["hold"]
    dip_to_depth = ns["dip_to_depth"]

    _preflight(ns, args, locs, max(deepest.values()))

    out_csv = run_dir / "franka_precondition.csv"
    ctx = Ctx()
    logger = FrankaLogger(str(out_csv), ctx, log_hz=args.log_hz)
    hover = HOVER_MM / 1000.0
    t0 = time.time()
    order = serpentine(locs)
    total = len(order) * args.precondition_cycles
    n = 0
    try:
        ctx.set("home")
        _, q0 = servo.current_pose()
        quat = ns["flat_down_quat"](q0) if not args.no_level else list(q0)
        cx, cy, cz = points["center"]["xyz"]
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat, name="home")

        for cyc in range(args.precondition_cycles):
            for loc in serpentine(locs, reverse_first=(cyc % 2 == 1)):
                d_mm = deepest[loc["point_index"]]
                x, y, sz = loc["x"], loc["y"], loc["surface_z"]
                hover_z = sz + hover
                n += 1
                print("[precond %4d/%4d] cycle %d  r%-2d c%-2d  -> %.2f mm"
                      % (n, total, cyc + 1, loc["row"], loc["col"], d_mm), flush=True)
                ctx.set("travel", idx=loc["point_index"], col=loc["col"],
                        row=loc["row"], target=(x, y, hover_z), surface_z=sz)
                m._gross_move(servo, x, y, hover_z, quat, name="travel")
                ctx.set("dip", target=(x, y, sz - d_mm / 1000.0))
                dip_to_depth(servo, x, y, sz, d_mm / 1000.0, quat,
                             correct=not args.no_depth_correction)
                ctx.set("dwell", target=(x, y, sz - d_mm / 1000.0))
                hold(servo, x, y, sz - d_mm / 1000.0, quat, args.precondition_hold_s)
                ctx.set("retract", target=(x, y, hover_z))
                m._gross_move(servo, x, y, hover_z, quat, name="retract")

        ctx.set("park")
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat, name="park")
        ctx.set("done")
    finally:
        logger.close()
        print("\n[PRECOND] %d cycles over %d locations in %.1f min -> %s"
              % (args.precondition_cycles, len(order), (time.time() - t0) / 60, out_csv))
        print("[PRECOND] The material is now conditioned. Re-run --calibrate if you "
              "have not already,\n          then start the camera recording and --run.")


def run_calibrate(args, locs, run_dir, b_guess):
    """Force-depth ladder at a few locations, twice, with the HEX21 reading.

    Two passes back to back are the point: if pass 2 matches pass 1 the material
    is already conditioned and the softening is geometric; if pass 2 is softer the
    softening is Mullins and pre-conditioning is mandatory. Either way we learn it
    in one sitting, before committing hours to the campaign.
    """
    chosen = pick_spread(locs, args.calibrate_locations, include_extremes=True)
    depths = [round(x, 3) for x in np.arange(args.calibrate_step_mm,
                                             args.max_depth_mm + 1e-9,
                                             args.calibrate_step_mm)]
    print("=" * 78)
    print(" CALIBRATION LADDER -- %d locations x %d depths x %d passes = %d presses"
          % (len(chosen), len(depths), args.calibrate_passes,
             len(chosen) * len(depths) * args.calibrate_passes))
    print("=" * 78)
    for c in chosen:
        print("   r%-2d c%-2d  k = %.3f N/mm" % (c["row"], c["col"], c["k"]))
    print("   depths: %s mm" % ", ".join("%.2f" % d for d in depths))
    print("   ⚠ HEX21 MUST BE ON TACTILE for this mode.")
    print("=" * 78)
    if args.dry_run:
        print("[DRY RUN] no ROS, no motion.")
        return

    # imported here, not at function top, so --calibrate --dry-run can preview the
    # ladder on a machine without the serial/ROS stack
    from franka_surface_map import WittensteinFT

    ns = _ros_setup(args, "CALIBRATION -- HEX21 must be on tactile")
    m, servo, points = ns["m"], ns["servo"], ns["points"]
    Ctx, FrankaLogger, hold = ns["Ctx"], ns["FrankaLogger"], ns["hold"]
    dip_to_depth = ns["dip_to_depth"]
    _preflight(ns, args, chosen, args.max_depth_mm)

    out_csv = run_dir / "franka_calibrate.csv"
    ctx = Ctx()
    logger = FrankaLogger(str(out_csv), ctx, log_hz=args.log_hz)
    hover = HOVER_MM / 1000.0
    rows = []
    try:
        _, q0 = servo.current_pose()
        quat = ns["flat_down_quat"](q0) if not args.no_level else list(q0)
        cx, cy, cz = points["center"]["xyz"]
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat, name="home")

        with WittensteinFT(port=m.SERIAL_PORT) as ft:
            for ps in range(args.calibrate_passes):
                for loc in chosen:
                    x, y, sz = loc["x"], loc["y"], loc["surface_z"]
                    hover_z = sz + hover
                    ctx.set("travel", idx=loc["point_index"],
                            col=loc["col"], row=loc["row"],
                            target=(x, y, hover_z), surface_z=sz)
                    m._gross_move(servo, x, y, hover_z, quat, name="travel")
                    ctx.set("tare", target=(x, y, hover_z))
                    hold(servo, x, y, hover_z, quat, 0.2)
                    base, base_sd, _ = ft.read_window(args.tare_s)
                    for d_mm in depths:
                        ctx.set("dip", target=(x, y, sz - d_mm / 1000.0))
                        got, cmd_z, _ = dip_to_depth(
                            servo, x, y, sz, d_mm / 1000.0, quat,
                            correct=not args.no_depth_correction)
                        ctx.set("dwell", target=(x, y, cmd_z))
                        fz, sd, nn = ft.read_window(args.dwell_s)
                        force = abs(fz - base)
                        rows.append({"pass": ps, "point_index": loc["point_index"],
                                     "row": loc["row"], "col": loc["col"],
                                     "k_map": loc["k"], "depth_cmd_mm": d_mm,
                                     "depth_achieved_mm": got * 1e3,
                                     "force_n": force, "force_sd_n": sd,
                                     "tare_n": base, "tare_sd_n": base_sd})
                        print("  pass %d r%-2d c%-2d  cmd %.2f  ach %.2f mm  "
                              "F = %.3f N (sd %.3f)"
                              % (ps, loc["row"], loc["col"], d_mm, got * 1e3,
                                 force, sd), flush=True)
                        if force > args.force_ceiling_n:
                            print("  ⚠ force ceiling %.1f N hit -- stopping this "
                                  "ladder." % args.force_ceiling_n)
                            break
                    ctx.set("retract", target=(x, y, hover_z))
                    m._gross_move(servo, x, y, hover_z, quat, name="retract")
        ctx.set("park")
        m._gross_move(servo, cx, cy, cz + APPROACH_MM / 1000.0, quat, name="park")
        ctx.set("done")
    finally:
        logger.close()

    with open(run_dir / "calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fit_force_model(rows, run_dir, args)


def fit_force_model(rows, run_dir, args):
    """Fit F = a*d^b per (location, pass); report the exponent and the Mullins test."""
    import collections
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["point_index"], r["pass"])].append(r)

    fits = []
    for (pi, ps), rs in sorted(groups.items()):
        d = np.array([x["depth_achieved_mm"] for x in rs], float)
        F = np.array([x["force_n"] for x in rs], float)
        good = (d > 0.05) & (F > 0.05)
        if good.sum() < 3:
            continue
        # log-log linear fit == power law
        A = np.vstack([np.log(d[good]), np.ones(good.sum())]).T
        sol, *_ = np.linalg.lstsq(A, np.log(F[good]), rcond=None)
        b, loga = float(sol[0]), float(sol[1])
        pred = np.exp(loga) * d[good] ** b
        ss_res = float(((F[good] - pred) ** 2).sum())
        ss_tot = float(((F[good] - F[good].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fits.append({"point_index": pi, "pass": ps, "b": b, "a": math.exp(loga),
                     "r2": r2, "n": int(good.sum()),
                     "k_map": rs[0]["k_map"]})

    if not fits:
        print("[CALIB] ⚠ not enough valid points to fit a model.")
        return

    b_all = np.array([f["b"] for f in fits])
    r2_all = np.array([f["r2"] for f in fits])
    b_mean = float(np.mean(b_all))

    print("\n" + "=" * 78)
    print(" FORCE MODEL  F = a * d^b")
    print("=" * 78)
    for f in sorted(fits, key=lambda x: (x["point_index"], x["pass"])):
        print("   pid %-3d pass %d  b = %.4f  a = %.4f  R2 = %.4f  (k_map %.3f)"
              % (f["point_index"], f["pass"], f["b"], f["a"], f["r2"], f["k_map"]))
    print("   mean exponent b   : %.4f  (sd %.4f)   1.0 = linear, <1 = softening"
          % (b_mean, float(np.std(b_all))))
    print("   R2                : %.4f - %.4f" % (r2_all.min(), r2_all.max()))

    # --- the Mullins test: does pass 2 differ from pass 1 at the same location? --
    by_pass = {}
    for f in fits:
        by_pass.setdefault(f["pass"], {})[f["point_index"]] = f
    verdict = "not tested (single pass)"
    if 0 in by_pass and 1 in by_pass:
        shared = sorted(set(by_pass[0]) & set(by_pass[1]))
        if shared:
            ratios = [by_pass[1][p]["a"] / by_pass[0][p]["a"] for p in shared]
            mr = float(np.mean(ratios))
            print("\n   MULLINS TEST over %d shared locations:" % len(shared))
            print("   a(pass2)/a(pass1) = %.4f  (1.00 = no change)" % mr)
            if mr < 0.95:
                verdict = ("PERMANENT softening (%.1f%% drop) -- pre-conditioning is "
                           "MANDATORY" % ((1 - mr) * 100))
            elif mr > 1.05:
                verdict = "material STIFFENED between passes -- investigate before running"
            else:
                verdict = ("no significant change (%.1f%%) -- already conditioned; "
                           "the softening is geometric" % ((mr - 1) * 100))
            print("   -> %s" % verdict)

    model = {"exponent_b": b_mean, "exponent_sd": float(np.std(b_all)),
             "r2_min": float(r2_all.min()), "r2_max": float(r2_all.max()),
             "n_locations": len(set(f["point_index"] for f in fits)),
             "n_fits": len(fits), "mullins_verdict": verdict,
             "map_fit_force_n": MAP_FIT_FORCE_N,
             "source": "calibrate %s" % time.strftime("%Y-%m-%d %H:%M"),
             "fits": fits}
    out = run_dir / "force_model.json"
    out.write_text(json.dumps(model, indent=2))
    print("\n   written -> %s" % out)
    print("   Use it with:  --force-model %s" % out)
    print("=" * 78)


# =============================================================================
#  self test
# =============================================================================

def self_test():
    """Arithmetic checks that do not need ROS, the map, or the robot."""
    fails = []

    def check(name, cond, detail=""):
        print("  %-52s %s%s" % (name, "PASS" if cond else "FAIL",
                                "" if cond else "   " + detail))
        if not cond:
            fails.append(name)

    # b = 1 must reduce exactly to the linear model
    k = 0.75
    d = depth_achieved_for_force(1.5, k, 1.0)
    check("b=1 reduces to linear d = F/k", abs(d - 1.5 / k) < 1e-9,
          "got %.9f want %.9f" % (d, 1.5 / k))

    # the scale must reproduce the map's own secant at the fit force
    for kk in (0.467, 0.776, 1.319):
        for b in (0.80, 0.90, 1.0):
            dfit = depth_achieved_for_force(MAP_FIT_FORCE_N, kk, b)
            check("secant at F=%.2f reproduces k=%.3f (b=%.2f)"
                  % (MAP_FIT_FORCE_N, kk, b),
                  abs(MAP_FIT_FORCE_N / dfit - kk) < 1e-9,
                  "got %.9f" % (MAP_FIT_FORCE_N / dfit))

    # inversion round-trips
    for F in (0.2, 1.0, 2.5):
        for b in (0.8, 1.0):
            dd = depth_achieved_for_force(F, 0.6, b)
            back = scale_from_stiffness(0.6, b) * dd ** b
            check("round trip F=%.1f b=%.1f" % (F, b), abs(back - F) < 1e-9)

    # softening must require MORE depth than linear for the same force
    d_lin = depth_achieved_for_force(2.0, 0.467, 1.0)
    d_soft = depth_achieved_for_force(2.0, 0.467, 0.85)
    check("softening needs more depth than linear", d_soft > d_lin,
          "%.3f vs %.3f" % (d_soft, d_lin))

    # reachability must be monotone in the cap and consistent with the command
    f4 = max_reachable_force(0.467, 1.0, 4.0)
    f6 = max_reachable_force(0.467, 1.0, 6.0)
    check("deeper cap reaches more force", f6 > f4, "%.3f vs %.3f" % (f4, f6))
    check("reachable force sits exactly on the cap",
          abs(depth_command_for_force(f6, 0.467, 1.0) - 6.0) < 1e-6)

    # the sag term must be present
    check("command exceeds elastomer depth by the sag",
          abs(depth_command_for_force(2.0, 0.6, 1.0)
              - depth_achieved_for_force(2.0, 0.6, 1.0)
              - (SAG_CONST_MM + 2.0 / SAG_STIFFNESS_N_PER_MM)) < 1e-12)

    # --- plan structure ------------------------------------------------------
    locs = [{"point_index": i, "row": i // 9, "col": i % 9,
             "x": 0.6 + 0.003 * (i % 9), "y": 0.01 + 0.003 * (i // 9),
             "surface_z": 0.268, "k": 0.5 + 0.008 * i, "fit_r2": 0.99,
             "quality_ok": True} for i in range(99)]
    levels = [0.2, 0.5, 0.8, 1.1]
    plan = build_plan(locs, levels, 1.0, 42, 0, 0)
    check("plan size = locations x levels", len(plan) == 99 * 4,
          "got %d" % len(plan))
    from collections import Counter
    per_loc = {}
    for p in plan:
        per_loc.setdefault(p["point_index"], []).append(p["level_idx"])
    check("every location gets every level exactly once",
          all(sorted(v) == list(range(4)) for v in per_loc.values()))
    for ps in range(4):
        c = Counter(p["level_idx"] for p in plan if p["pass"] == ps)
        if len(c) < 2:
            check("pass %d spans multiple levels" % ps, False)
    check("each pass spans the full force range",
          all(len(set(p["level_idx"] for p in plan if p["pass"] == ps)) == 4
              for ps in range(4)))
    check("seq is dense and ordered",
          [p["seq"] for p in plan] == list(range(len(plan))))

    # determinism: the same seed must give a byte-identical plan
    plan2 = build_plan(locs, levels, 1.0, 42, 0, 0)
    fp1 = plan_fingerprint(plan, {"a": 1})
    fp2 = plan_fingerprint(plan2, {"a": 1})
    check("plan is deterministic for a fixed seed", fp1 == fp2)
    plan3 = build_plan(locs, levels, 1.0, 43, 0, 0)
    check("a different seed changes the fingerprint",
          plan_fingerprint(plan3, {"a": 1}) != fp1)
    check("a changed parameter changes the fingerprint",
          plan_fingerprint(plan, {"a": 2}) != fp1)

    # --- depth-targeted mode -------------------------------------------------
    dl = [0.5, 1.0, 1.5, 2.0]
    pd = build_plan(locs, [], 1.0, 42, 0, 0, depths=dl)
    check("depth mode: plan size = locations x depths", len(pd) == 99 * 4,
          "got %d" % len(pd))
    check("depth mode: commanded depth is the requested depth VERBATIM (no sag)",
          all(abs(q["depth_cmd_mm"] - dl[q["level_idx"]]) < 1e-12 for q in pd))
    check("depth mode: identical depth at every location",
          all(len({round(q["depth_cmd_mm"], 9) for q in pd if q["level_idx"] == i}) == 1
              for i in range(4)))
    per = {}
    for q in pd:
        per.setdefault(q["point_index"], []).append(q["level_idx"])
    check("depth mode: every location gets every depth exactly once",
          all(sorted(v) == list(range(4)) for v in per.values()))
    check("depth mode: predicted force rises with stiffness at fixed depth",
          len({round(q["target_force_n"], 6) for q in pd if q["level_idx"] == 3}) > 1)
    check("depth mode fingerprint differs from force mode",
          plan_fingerprint(pd, {"m": "d"}) != plan_fingerprint(plan, {"m": "d"}))
    # a force-mode plan DOES carry the sag; a depth-mode plan must not
    pf = build_plan(locs, [1.0], 1.0, 42, 0, 0)
    check("force mode still adds the sag term",
          all(q["depth_cmd_mm"] > q["target_depth_mm"] + 1e-9 for q in pf))

    # --- sequential level order ----------------------------------------------
    ps = build_plan(locs, [], 1.0, 42, 0, 0, depths=dl, sequential=True)
    check("sequential: pass p uses level p at EVERY location",
          all(q["level_idx"] == q["pass"] for q in ps))
    check("sequential: each pass is one single depth",
          all(len({q["depth_cmd_mm"] for q in ps if q["pass"] == i}) == 1
              for i in range(4)))
    check("sequential: depths ascend across passes",
          [sorted({q["depth_cmd_mm"] for q in ps if q["pass"] == i})[0]
           for i in range(4)] == sorted(dl))
    check("sequential still covers every location x level once",
          sorted((q["point_index"], q["level_idx"]) for q in ps)
          == sorted((q["point_index"], q["level_idx"]) for q in pd))
    check("sequential differs from randomised", [q["level_idx"] for q in ps]
          != [q["level_idx"] for q in pd])
    check("sequential is seed-independent",
          [q["level_idx"] for q in ps]
          == [q["level_idx"] for q in build_plan(locs, [], 1.0, 999, 0, 0,
                                                 depths=dl, sequential=True)])
    psr = build_plan(locs, [], 1.0, 42, 9, 2, depths=dl, sequential=True)
    check("sequential keeps the replication tail (drift measurement)",
          len([q for q in psr if q["block"] == "replicate"]) == 9 * 4 * 2)

    # replication tail
    planr = build_plan(locs, levels, 1.0, 42, 9, 2)
    reps = [p for p in planr if p["block"] == "replicate"]
    check("replication tail size", len(reps) == 9 * 4 * 2, "got %d" % len(reps))
    check("main block unchanged by the tail",
          [p["seq"] for p in planr[:99 * 4]] == [p["seq"] for p in plan])

    # serpentine really alternates
    s0 = [l["point_index"] for l in serpentine(locs)]
    s1 = [l["point_index"] for l in serpentine(locs, reverse_first=True)]
    check("serpentine flips with reverse_first", s0 != s1)
    check("serpentine visits every location once",
          sorted(s0) == sorted(l["point_index"] for l in locs))

    # pick_spread
    sp = pick_spread(locs, 9)
    check("pick_spread returns the requested count", len(sp) == 9)
    check("pick_spread returns distinct locations",
          len(set(p["point_index"] for p in sp)) == 9)
    ks = [l["k"] for l in locs]
    spx = pick_spread(locs, 5, include_extremes=True)
    kx = [l["k"] for l in spx]
    check("include_extremes pins the softest and stiffest",
          min(ks) in kx and max(ks) in kx,
          "got %s" % sorted(kx))
    check("include_extremes still returns the requested count", len(spx) == 5)

    # --- ledger --------------------------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        lp = Path(td) / "state.jsonl"
        led = Ledger(lp, "abc123")
        led.open({"x": 1})
        for s in (0, 1, 2):
            led.record(seq=s, row=0, col=s)
        led.close()

        led2 = Ledger(lp, "abc123")
        n, err = led2.load()
        check("ledger reloads the completed set", led2.done == {0, 1, 2} and err is None,
              "done=%s err=%s" % (led2.done, err))
        check("ledger advances the segment on resume", led2.segment == 1,
              "got %d" % led2.segment)

        led3 = Ledger(lp, "DIFFERENT")
        _, err3 = led3.load()
        check("ledger rejects a fingerprint mismatch", err3 is not None)

        # a torn final line must not lose the earlier records
        with open(lp, "a") as f:
            f.write('{"seq": 3, "par')
        led4 = Ledger(lp, "abc123")
        led4.load()
        check("torn trailing line is ignored, prior records survive",
              led4.done == {0, 1, 2}, "got %s" % led4.done)

        # archive must preserve the old ledger rather than delete it
        led5 = Ledger(lp, "abc123")
        led5.load()
        arch = led5.archive()
        check("archive preserves the superseded ledger",
              arch is not None and arch.exists() and not lp.exists())
        check("archive resets progress", led5.done == set() and led5.segment == 0)

    print("\n  %d checks, %d failed" % (len(fails) + 0, len(fails)) if fails
          else "\n  all checks passed")
    return 1 if fails else 0


# =============================================================================
#  main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Plan B force-targeted campaign with crash-resume.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    mode = ap.add_argument_group("mode (pick one; default is the campaign)")
    mode.add_argument("--calibrate", action="store_true",
                      help="ladder at a few locations with the HEX21 -> force_model.json")
    mode.add_argument("--precondition", action="store_true",
                      help="cycle every location to its own campaign max (Mullins)")
    mode.add_argument("--self-test", action="store_true",
                      help="arithmetic + plan + ledger checks; no ROS, no map, no robot")
    mode.add_argument("--dry-run", action="store_true",
                      help="build and print the plan, write the sidecar; no motion")

    res = ap.add_argument_group("resume")
    res.add_argument("--resume", action="store_true",
                     help="continue the run in --run-dir, skipping completed pokes")
    res.add_argument("--run-dir", default=None,
                     help="campaign directory (default: newest planb_* under "
                          "~/E-BTS/recordings, or a new one)")
    res.add_argument("--fresh", action="store_true",
                     help="archive any existing ledger and start from poke 0")

    des = ap.add_argument_group("design")
    des.add_argument("--map", default=str(MAP_CSV), help="surface_offset_map.csv")
    des.add_argument("--force-model", default=None,
                     help="force_model.json from --calibrate (else LINEAR + a warning)")
    des.add_argument("--depths-mm", type=float, nargs="+", default=None,
                     help="⭐ DEPTH-TARGETED mode (recommended): command these depths "
                          "verbatim at every location, e.g. --depths-mm 0.5 1.0 1.5 "
                          "2.0 2.5 3.0 3.5 4.0. Depth is what the robot actually "
                          "controls (0.003 mm achieved); force is measured, not aimed "
                          "at. Overrides --n-levels/--force-min/--force-max.")
    des.add_argument("--n-levels", type=int, default=9)
    des.add_argument("--force-min", type=float, default=0.20)
    des.add_argument("--force-max", type=float, default=None,
                     help="default: the largest force EVERY location can reach")
    des.add_argument("--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM)
    des.add_argument("--replicate-locations", type=int, default=9)
    des.add_argument("--replicate-extra", type=int, default=2,
                     help="extra observations of each replicated condition")
    des.add_argument("--sequential", action="store_true",
                     help="Walk the levels IN ORDER instead of randomising: pass 0 = "
                          "shallowest at every location, pass 1 = next, and so on. "
                          "⚠ This CONFOUNDS LEVEL WITH TIME -- see the note in "
                          "build_plan(). The replication tail is what lets you measure "
                          "and correct the resulting drift, so keep it enabled.")
    des.add_argument("--seed", type=int, default=SEED)
    des.add_argument("--use-poor-points", action="store_true",
                     help="also press the quality_ok=False map points (not advised)")
    des.add_argument("--indenter-mm", type=float, default=3.0,
                     help="indenter diameter, recorded in plan.json for provenance "
                          "(default 3.0). Does not affect the plan.")

    mot = ap.add_argument_group("motion")
    mot.add_argument("--dwell-s", type=float, default=DWELL_S)
    mot.add_argument("--tare-s", type=float, default=TARE_S)
    mot.add_argument("--log-hz", type=float, default=200.0)
    mot.add_argument("--no-level", action="store_true")
    mot.add_argument("--no-recovery", action="store_true")
    mot.add_argument("--no-depth-correction", action="store_true")
    mot.add_argument("--ack-deep", action="store_true",
                     help="acknowledge a depth cap beyond the %.1f mm demonstrated "
                          "limit" % DEMONSTRATED_DEPTH_MM)

    cal = ap.add_argument_group("calibration")
    cal.add_argument("--calibrate-locations", type=int, default=5)
    cal.add_argument("--calibrate-passes", type=int, default=2,
                     help="2 passes back to back = the Mullins test")
    cal.add_argument("--calibrate-step-mm", type=float, default=0.5)
    cal.add_argument("--force-ceiling-n", type=float, default=8.0,
                     help="abort a ladder above this force")

    pre = ap.add_argument_group("pre-conditioning")
    pre.add_argument("--precondition-cycles", type=int, default=3)
    pre.add_argument("--precondition-hold-s", type=float, default=0.5)

    args = ap.parse_args()

    if args.self_test:
        return self_test()

    # ---- safety gate on depth ----------------------------------------------
    if args.max_depth_mm > HARD_DEPTH_CEILING_MM:
        sys.exit("[ERROR] --max-depth-mm %.1f exceeds the %.1f mm hard ceiling."
                 % (args.max_depth_mm, HARD_DEPTH_CEILING_MM))
    _deepest_req = (max(args.depths_mm) if args.depths_mm else args.max_depth_mm)
    if _deepest_req > DEMONSTRATED_DEPTH_MM and not args.ack_deep:
        sys.exit(
            "[ERROR] --max-depth-mm %.2f is beyond the %.1f mm DEMONSTRATED limit.\n"
            "        %.1f mm is vetted by 648 presses (pilot_20260807_134855, max\n"
            "        force 5.9 N). Deeper is extrapolation: the straps are the real\n"
            "        limit and they have not been tested there.\n"
            "        Run --precondition first (it is the strap test at full depth),\n"
            "        then pass --ack-deep."
            % (_deepest_req, DEMONSTRATED_DEPTH_MM, DEMONSTRATED_DEPTH_MM))

    # ---- load the map and the force model ----------------------------------
    locs, skipped = load_map(Path(args.map), use_poor=args.use_poor_points)
    b, model = load_force_model(args.force_model)

    depths = sorted(args.depths_mm) if args.depths_mm else None
    if depths:
        if min(depths) <= 0:
            sys.exit("[ERROR] --depths-mm must all be > 0.")
        if max(depths) > args.max_depth_mm + 1e-9:
            sys.exit("[ERROR] deepest requested %.2f mm exceeds --max-depth-mm %.2f."
                     % (max(depths), args.max_depth_mm))
        levels, ceiling, limiter = [], None, None
    else:
        levels, ceiling, limiter = derive_levels(
            locs, b, args.max_depth_mm, args.n_levels, args.force_min, args.force_max)

    plan = build_plan(locs, levels, b, args.seed,
                      args.replicate_locations, args.replicate_extra, depths=depths,
                      sequential=args.sequential)

    deepest = max(p["depth_cmd_mm"] for p in plan)
    if deepest > args.max_depth_mm + 1e-6:
        sys.exit("[ERROR] internal: planned depth %.3f mm exceeds the cap %.3f mm."
                 % (deepest, args.max_depth_mm))

    # Fingerprinted parameters describe the EXPERIMENT, never where its files
    # happen to live. The map enters by CONTENT hash and the force model by its
    # fitted exponent, so resuming from a different working directory -- or after
    # moving the run -- does not falsely refuse. A false refusal is dangerous:
    # it invites --fresh, which is what discards a half-finished campaign.
    params = {
        "map_sha256": file_sha256(Path(args.map)),
        "mode": "depth" if depths else "force",
        "level_order": "sequential" if args.sequential else "randomised",
        "depths_mm": [round(d, 4) for d in depths] if depths else None,
        "n_levels": len(depths) if depths else args.n_levels,
        "force_min_n": None if depths else args.force_min,
        "force_levels_n": None if depths else [round(f, 4) for f in levels],
        "max_depth_mm": args.max_depth_mm, "exponent_b": round(b, 8),
        "seed": args.seed,
        "replicate_locations": args.replicate_locations,
        "replicate_extra": args.replicate_extra,
        "dwell_s": args.dwell_s, "tare_s": args.tare_s,
        "use_poor_points": args.use_poor_points,
        "n_locations": len(locs),
        "sag_const_mm": SAG_CONST_MM,
        "sag_stiffness_n_per_mm": SAG_STIFFNESS_N_PER_MM,
    }
    fp = plan_fingerprint(plan, params)
    # recorded for provenance, deliberately NOT fingerprinted
    meta_only = {"map_path": str(Path(args.map).resolve()),
                 "force_model_path": (str(Path(args.force_model).resolve())
                                      if args.force_model else None),
                 "indenter_diameter_mm": args.indenter_mm}

    # ---- resolve the run directory -----------------------------------------
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.resume:
        # only directories that actually STARTED count. A --dry-run preview writes
        # plan.csv but no ledger, and must never be mistaken for the live campaign.
        cands = sorted(d for d in RUNS_DIR.glob("planb_*")
                       if (d / "state.jsonl").exists())
        if not cands:
            previews = sorted(RUNS_DIR.glob("planb_*"))
            sys.exit("[ERROR] --resume: no campaign with a ledger under %s.%s"
                     % (RUNS_DIR,
                        "" if not previews else
                        "\n        %d planb_* director%s but hold no "
                        "state.jsonl,\n        so they are dry-run previews, not "
                        "interrupted runs:\n          %s"
                        % (len(previews),
                           "y exists" if len(previews) == 1 else "ies exist",
                           "\n          ".join(d.name for d in previews[-5:]))))
        run_dir = cands[-1]
        print("[RESUME] newest campaign with a ledger: %s" % run_dir)
    else:
        run_dir = RUNS_DIR / ("planb_" + time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(run_dir / "state.jsonl", fp)
    n_rec, err = ledger.load()

    if err and not args.fresh:
        sys.exit(
            "[ERROR] cannot resume: %s\n\n"
            "        The ledger in %s was written by a campaign built from\n"
            "        DIFFERENT parameters. Continuing would stitch two different\n"
            "        experiments into one dataset.\n\n"
            "        Either pass the original parameters, or --fresh to archive\n"
            "        that ledger and start over (nothing is deleted)."
            % (err, ledger.path))
    if args.fresh:
        arch = ledger.archive()
        if arch:
            print("[FRESH] previous ledger archived -> %s" % arch.name)
        n_rec = 0

    remaining = [p for p in plan if p["seq"] not in ledger.done]

    save_plan(run_dir, plan, params, fp, meta_only)
    report(plan, locs, skipped, levels, ceiling, limiter, b, model, args, run_dir,
           ledger, remaining, depths=depths)

    if args.calibrate:
        run_calibrate(args, locs, run_dir, b)
        return 0

    if args.dry_run:
        print("\n[DRY RUN] plan written to %s -- no ROS, no motion." % (run_dir / "plan.csv"))
        if model is None and not depths:
            print("[DRY RUN] ⚠ force model is LINEAR -- the top levels will land ~19% "
              "short.\n          Labels stay correct either way (force is MEASURED, "
              "not assumed).")
        return 0

    if args.precondition:
        run_precondition(args, plan, locs, run_dir)
        return 0

    if not remaining:
        print("\n  ✅ nothing to do -- all %d pokes are already in the ledger." % len(plan))
        return 0

    if model is None and not depths:
        print("\n⚠ ⚠ ⚠  The force model is LINEAR (uncalibrated). Target forces above")
        print("      ~%.1f N will be systematically UNDER-shot, because the material" % MAP_FIT_FORCE_N)
        print("      softens and the map's stiffness is a %.2f N secant." % MAP_FIT_FORCE_N)
        print("      Labels stay CORRECT either way -- force is measured, not assumed.")
    print("      You simply land near 2.0 N instead of 2.44 N at the top.\n")

    ledger.open(dict(params, **meta_only))
    run_campaign(args, plan, locs, run_dir, ledger, remaining)
    return 0


if __name__ == "__main__":
    sys.exit(main())
