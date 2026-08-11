#!/usr/bin/env python3
"""Join a planb campaign run and emit per-poke measurements + a 2D report."""
import csv, json, sys
from pathlib import Path
import numpy as np

RUN = Path(sys.argv[1])
meta = json.loads((RUN / "metadata.json").read_text())
off = float(meta.get("tactile_minus_workstation_offset_s", 0.0))

ft = np.loadtxt(RUN / "ft.csv", delimiter=",", skiprows=1, usecols=(0, 3))
ft_t, ft_fz = ft[:, 0], ft[:, 1]

rows = []
with open(RUN / "franka_seg00.csv") as f:
    for r in csv.DictReader(f):
        rows.append((float(r["unix_time_s"]), r["phase"], r["point_index"],
                     r["ee_z"], r["surface_z"]))

# plan + ledger, keyed by seq (point_index in the log is the SEQ we set in ctx.set)
plan = {int(r["seq"]): r for r in csv.DictReader(open(RUN / "plan.csv"))}
led = {}
for l in open(RUN / "state.jsonl"):
    if not l.strip():
        continue
    r = json.loads(l)
    if not r.get("_header"):
        led[int(r["seq"])] = r

# contiguous phase blocks
blocks, cur = [], None
for t, ph, pid, eez, sz in rows:
    if cur is None or ph != cur["phase"]:
        if cur:
            blocks.append(cur)
        cur = {"phase": ph, "t0": t, "t1": t, "ee_z": [], "sz": [], "pid": pid}
    cur["t1"] = t
    if ph == "dwell":
        try:
            cur["ee_z"].append(float(eez)); cur["sz"].append(float(sz))
        except (TypeError, ValueError):
            pass
    if pid not in ("", "None"):
        cur["pid"] = pid
if cur:
    blocks.append(cur)

out = []
for i, b in enumerate(blocks):
    if b["phase"] != "dwell" or not b["ee_z"]:
        continue
    tare = None
    for j in range(i - 1, -1, -1):
        if blocks[j]["phase"] == "tare":
            tare = blocks[j]; break
        if blocks[j]["phase"] == "dwell":
            break
    if tare is None:
        continue
    seq = int(b["pid"])
    if seq not in plan:
        continue
    a, e = b["t0"] - off, b["t1"] - off
    ta, te = tare["t0"] - off, tare["t1"] - off
    m = (ft_t >= a + 0.05) & (ft_t <= e - 0.02)
    tm = (ft_t >= ta + 0.10) & (ft_t <= te - 0.05)
    if m.sum() < 50 or tm.sum() < 50:
        continue
    base = float(np.median(ft_fz[tm]))
    seg = ft_fz[m]
    p = plan[seq]
    lg = led.get(seq, {})
    # creep: slope of force across the dwell
    tt = ft_t[m]; slope = float(np.polyfit(tt - tt[0], seg, 1)[0]) if len(tt) > 50 else np.nan
    out.append({
        "seq": seq, "block": p["block"], "pass": int(p["pass"]),
        "point_index": int(p["point_index"]), "row": int(p["row"]), "col": int(p["col"]),
        "level_idx": int(p["level_idx"]),
        "target_force_n": float(p["target_force_n"]),
        "stiffness_map": float(p["stiffness_n_per_mm"]),
        "depth_cmd_mm": float(p["depth_cmd_mm"]),
        "depth_achieved_mm": float(lg.get("achieved_mm", np.nan)),
        "force_n": abs(float(np.median(seg)) - base),
        "force_sd_n": float(np.std(seg)),
        "creep_n_per_s": slope,
        "tare_n": base, "tare_sd_n": float(np.std(ft_fz[tm])),
        "n_ft": int(m.sum()), "dwell_s": e - a,
        "t_rel_s": a - ft_t[0],
    })

print("joined %d of %d pokes" % (len(out), len(plan)))
fields = list(out[0].keys())
with open(RUN / "pokes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
print("-> %s" % (RUN / "pokes.csv"))
