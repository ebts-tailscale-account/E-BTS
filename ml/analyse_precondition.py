#!/usr/bin/env python3
"""Analyse a recorded --precondition run.

Answers three things the campaign depends on:
  1. Did anything alarming happen at 6 mm (straps, force runaway)?
  2. Is the softening PERMANENT (Mullins) or reversible? -> cycle 1 vs 2 vs 3
  3. What is the force-depth exponent b -> force_model.json for future campaigns

The franka log is on the TACTILE clock and ft.csv on the WORKSTATION clock, so the
measured offset is applied before joining. Only DWELL windows are used: force is
steady there, so the +-25 ms offset uncertainty is irrelevant (a ramp would not be).
"""
import csv, json, sys
from pathlib import Path
import numpy as np

RUN = Path(sys.argv[1])
meta = json.loads((RUN / "metadata.json").read_text())
# tactile_time - workstation_time
off = float(meta.get("tactile_minus_workstation_offset_s", 0.0))
off2 = meta.get("offset_after_s")
print("clock offset (tactile - workstation): %.4f s  (after: %s, drift %.4f s)"
      % (off, "%.4f" % off2 if off2 is not None else "n/a",
         float(meta.get("offset_drift_s", 0.0))))

# ---- F/T on the workstation clock -------------------------------------------
ft = np.loadtxt(RUN / "ft.csv", delimiter=",", skiprows=1, usecols=(0, 3))
ft_t, ft_fz = ft[:, 0], ft[:, 1]
print("ft.csv: %d samples, %.1f s span, Fz %.3f .. %.3f N"
      % (len(ft_t), ft_t[-1] - ft_t[0], ft_fz.min(), ft_fz.max()))

# ---- franka log on the tactile clock ----------------------------------------
rows = []
with open(RUN / "franka_precondition.csv") as f:
    for r in csv.DictReader(f):
        rows.append((float(r["unix_time_s"]), r["phase"], r["point_index"],
                     r["ee_z"], r["surface_z"], r["target_z"]))
print("franka log: %d samples, phases %s"
      % (len(rows), sorted(set(r[1] for r in rows))))

# franka timestamps -> workstation clock
def to_ws(t):
    return t - off

# ---- segment into phase blocks, in order ------------------------------------
# The dip takes ~2.8 s (depth-correction iterations), so a fixed time offset before
# the dwell lands MID-CONTACT. Use the phase tags: the TRAVEL block immediately
# preceding each dwell is at hover, genuinely out of contact.
blocks = []
cur = None
for t, phase, pid, eez, sz, tz in rows:
    if cur is None or phase != cur["phase"]:
        if cur:
            blocks.append(cur)
        cur = {"phase": phase, "t0": t, "t1": t, "ee_z": [], "sz": [], "pid": pid}
    cur["t1"] = t
    if phase == "dwell":
        try:
            cur["ee_z"].append(float(eez)); cur["sz"].append(float(sz))
        except (TypeError, ValueError):
            pass
    if pid not in ("", "None"):
        cur["pid"] = pid
if cur:
    blocks.append(cur)
print("phase blocks: %d" % len(blocks))

seen = {}
out = []
for i, blk in enumerate(blocks):
    if blk["phase"] != "dwell" or not blk["ee_z"]:
        continue
    prev = None
    for j in range(i - 1, -1, -1):
        if blocks[j]["phase"] == "travel":
            prev = blocks[j]; break
        if blocks[j]["phase"] == "dwell":
            break
    if prev is None:
        continue
    pid = int(blk["pid"])
    cyc = seen.get(pid, 0); seen[pid] = cyc + 1

    a, b = to_ws(blk["t0"]), to_ws(blk["t1"])
    ta, tb = to_ws(prev["t0"]), to_ws(prev["t1"])
    m = (ft_t >= a + 0.05) & (ft_t <= b - 0.02)
    tm = (ft_t >= ta + 0.10) & (ft_t <= tb - 0.05)     # settled part of travel
    if m.sum() < 20 or tm.sum() < 20:
        continue
    base = float(np.median(ft_fz[tm]))
    force = abs(float(np.median(ft_fz[m])) - base)
    depth = (float(np.median(blk["sz"])) - float(np.median(blk["ee_z"]))) * 1e3
    out.append({"pid": pid, "cycle": cyc, "depth_mm": depth, "force_n": force,
                "sd_n": float(np.std(ft_fz[m])), "tare_n": base,
                "tare_sd_n": float(np.std(ft_fz[tm])), "n": int(m.sum()),
                "dur_s": b - a})

print("joined: %d (point, cycle) observations\n" % len(out))
if not out:
    sys.exit("[ERROR] nothing joined -- check the clock offset and phase tags.")

# ---- 1. safety: what did 6 mm actually do? ----------------------------------
F = np.array([o["force_n"] for o in out])
D = np.array([o["depth_mm"] for o in out])
print("=" * 74)
print(" 1. WHAT ACTUALLY HAPPENED")
print("=" * 74)
print("  achieved depth : %.2f - %.2f mm  (mean %.2f)" % (D.min(), D.max(), D.mean()))
print("  measured force : %.2f - %.2f N   (mean %.2f)" % (F.min(), F.max(), F.mean()))
hi = np.argsort(F)[-5:][::-1]
print("  top 5 forces   : %s" % ", ".join("%.2f N @ %.2f mm (pid %d cyc %d)"
      % (F[i], D[i], out[i]["pid"], out[i]["cycle"]) for i in hi))
print("  8 N probe ceiling exceeded: %d of %d" % (int((F > 8.0).sum()), len(F)))

# ---- 2. Mullins: cycle over cycle at the SAME location ----------------------
print("\n" + "=" * 74)
print(" 2. MULLINS TEST -- same location, successive cycles")
print("=" * 74)
bycyc = {}
for o in out:
    bycyc.setdefault(o["cycle"], {})[o["pid"]] = o
for c in sorted(bycyc):
    v = np.array([o["force_n"] for o in bycyc[c].values()])
    print("  cycle %d : n=%3d  mean force %.3f N  median %.3f" % (c, len(v), v.mean(),
                                                                  np.median(v)))
if 0 in bycyc and max(bycyc) >= 1:
    last = max(bycyc)
    shared = sorted(set(bycyc[0]) & set(bycyc[last]))
    if shared:
        r = np.array([bycyc[last][p]["force_n"] / bycyc[0][p]["force_n"]
                      for p in shared if bycyc[0][p]["force_n"] > 0.05])
        mr = float(np.median(r))
        print("\n  force(cycle %d)/force(cycle 0) over %d shared locations:" % (last, len(r)))
        print("    median %.4f   mean %.4f   (1.00 = no change)" % (mr, r.mean()))
        if mr < 0.95:
            v = "PERMANENT softening (%.1f%%) -- pre-conditioning WAS necessary" % ((1-mr)*100)
        elif mr > 1.05:
            v = "STIFFENED %.1f%% -- unexpected, investigate" % ((mr-1)*100)
        else:
            v = "no significant change (%.1f%%) -- material was already settled" % ((mr-1)*100)
        print("    -> %s" % v)

# ---- 3. fit the force model -------------------------------------------------
print("\n" + "=" * 74)
print(" 3. FORCE MODEL   F = a(k) * d^b,   a(k) = 0.6 * (k/0.6)^b")
print("=" * 74)
kmap = {}
with open(RUN / "plan.csv") as f:
    for r in csv.DictReader(f):
        kmap[int(r["point_index"])] = float(r["stiffness_n_per_mm"])

# use the LAST cycle only: that is the conditioned state the campaign will see
last = max(bycyc)
pts = [(kmap[o["pid"]], o["depth_mm"], o["force_n"])
       for o in bycyc[last].values()
       if o["pid"] in kmap and o["depth_mm"] > 0.3 and o["force_n"] > 0.08]
print("  fitting %d points from cycle %d (the conditioned state)" % (len(pts), last))

def resid(b):
    e = []
    for k, d, f in pts:
        a = 0.6 * (k / 0.6) ** b
        e.append(np.log(f) - np.log(a * d ** b))
    return np.array(e)

lo, hi = 0.30, 2.00
for _ in range(200):
    m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
    if (resid(m1) ** 2).sum() < (resid(m2) ** 2).sum():
        hi = m2
    else:
        lo = m1
b = 0.5 * (lo + hi)
r = resid(b)
pred = np.array([0.6 * (k / 0.6) ** b * d ** b for k, d, _ in pts])
act = np.array([f for _, _, f in pts])
ss_res = float(((act - pred) ** 2).sum()); ss_tot = float(((act - act.mean()) ** 2).sum())
print("  exponent b     : %.4f     (1.0 = linear, <1 = softening)" % b)
print("  log-resid sd   : %.4f" % float(np.std(r)))
print("  R2 on force    : %.4f" % (1 - ss_res / ss_tot))
print("  median |err|   : %.3f N  (%.1f%% of mean force)"
      % (float(np.median(np.abs(act - pred))),
         100 * float(np.median(np.abs(act - pred))) / act.mean()))

lin = np.array([k * d for k, d, _ in pts])
print("  vs LINEAR (b=1): R2 %.4f, median |err| %.3f N"
      % (1 - float(((act - lin) ** 2).sum()) / ss_tot,
         float(np.median(np.abs(act - lin)))))

# what the running campaign will actually achieve at its top level
print("\n  CONSEQUENCE for the campaign now running (linear plan, b=1):")
for tgt, cmd_at_soft in ((2.44, 5.97),):
    ks = min(kmap.values())
    a = 0.6 * (ks / 0.6) ** b
    d_ach = cmd_at_soft - 0.10 - tgt / 3.8
    print("    top level asks %.2f N at the softest point (%.2f mm commanded)"
          % (tgt, cmd_at_soft))
    print("    -> fitted model says it will actually reach %.2f N" % (a * d_ach ** b))

model = {"exponent_b": float(b), "r2": float(1 - ss_res / ss_tot),
         "n_points": len(pts), "fitted_from": RUN.name, "cycle_used": int(last),
         "map_fit_force_n": 0.6,
         "source": "precondition run %s" % RUN.name}
(RUN / "force_model.json").write_text(json.dumps(model, indent=2))
print("\n  written -> %s" % (RUN / "force_model.json"))
