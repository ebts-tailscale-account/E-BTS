#!/usr/bin/env python3
import csv, sys, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(sys.argv[1])
d = list(csv.DictReader(open(RUN / "pokes.csv")))
F = np.array([float(r["force_n"]) for r in d])
T = np.array([float(r["target_force_n"]) for r in d])
K = np.array([float(r["stiffness_map"]) for r in d])
DC = np.array([float(r["depth_cmd_mm"]) for r in d])
DA = np.array([float(r["depth_achieved_mm"]) for r in d])
RW = np.array([int(r["row"]) for r in d])
CL = np.array([int(r["col"]) for r in d])
TS = np.array([float(r["t_rel_s"]) for r in d]) / 60.0
levels = sorted(set(T))

INK = "#1b1b1f"; MUTED = "#6c6a74"; ACC = "#2f6f8f"; ACC2 = "#c2703d"; GRID = "#dcdae2"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.4))

# --- 1. violin: achieved force per target level ------------------------------
a = ax[0, 0]
data = [F[T == t] for t in levels]
vp = a.violinplot(data, positions=range(len(levels)), widths=0.82,
                  showmedians=True, showextrema=False)
for b in vp["bodies"]:
    b.set_facecolor(ACC); b.set_alpha(0.55); b.set_edgecolor(ACC); b.set_linewidth(0.8)
vp["cmedians"].set_color(INK); vp["cmedians"].set_linewidth(1.4)
a.plot(range(len(levels)), levels, "o--", color=ACC2, ms=4, lw=1.2, label="target (planned)")
a.set_xticks(range(len(levels)))
a.set_xticklabels(["%.2f" % t for t in levels], rotation=45, ha="right")
a.set_xlabel("planned force level (N)"); a.set_ylabel("measured force (N)")
a.set_title("Achieved force per level  —  n=112 each", loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=7.5, loc="upper left")
a.grid(axis="y", color=GRID, lw=0.6)
a.set_axisbelow(True)

# --- 2. overall coverage histogram -------------------------------------------
a = ax[0, 1]
a.hist(F, bins=44, color=ACC, alpha=0.8, edgecolor="white", linewidth=0.4)
a.axvline(np.median(F), color=ACC2, lw=1.4, label="median %.2f N" % np.median(F))
a.set_xlabel("measured force (N)"); a.set_ylabel("pokes")
a.set_title("Force coverage  —  %.2f to %.2f N" % (F.min(), F.max()),
            loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=7.5)
a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)

# --- 3. depth tracking --------------------------------------------------------
a = ax[0, 2]
a.scatter(DC, DA, s=7, c=ACC, alpha=0.4, linewidths=0)
lim = [0, max(DC.max(), DA.max()) * 1.05]
a.plot(lim, lim, color=ACC2, lw=1.2, ls="--", label="perfect tracking")
a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("commanded depth (mm)"); a.set_ylabel("achieved depth (mm)")
err = DA - DC
a.set_title("Depth tracking  —  err %.3f ± %.3f mm" % (err.mean(), err.std()),
            loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=7.5, loc="upper left")
a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)

# --- 4. the confound check ----------------------------------------------------
a = ax[1, 0]
sc = a.scatter(K, F, c=T, s=8, cmap="viridis", alpha=0.75, linewidths=0)
a.set_xlabel("mapped stiffness (N/mm)"); a.set_ylabel("measured force (N)")
r = np.corrcoef(K, F)[0, 1]
a.set_title("Force vs position-stiffness  —  r = %+.3f" % r, loc="left",
            fontweight="bold")
cb = fig.colorbar(sc, ax=a, pad=0.02); cb.set_label("target level (N)", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)

# --- 5. spatial coverage map --------------------------------------------------
a = ax[1, 1]
nr, nc = RW.max() + 1, CL.max() + 1
grid = np.full((nr, nc), np.nan)
cnt = collections.defaultdict(list)
for i in range(len(F)):
    cnt[(RW[i], CL[i])].append(F[i])
for (rr, cc), v in cnt.items():
    grid[rr, cc] = np.mean(v)
im = a.imshow(grid, cmap="magma", aspect="auto", origin="lower")
a.set_xlabel("col"); a.set_ylabel("row")
a.set_title("Mean force per location  —  %d sites" % len(cnt), loc="left",
            fontweight="bold")
cb = fig.colorbar(im, ax=a, pad=0.02); cb.set_label("mean force (N)", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
for rr in range(nr):
    for cc in range(nc):
        if np.isnan(grid[rr, cc]):
            a.plot(cc, rr, "x", color="#999", ms=5, mew=1.2)

# --- 6. drift over the run ----------------------------------------------------
a = ax[1, 2]
a.scatter(TS, F, s=6, c=ACC, alpha=0.35, linewidths=0)
z = np.polyfit(TS, F, 1)
a.plot([TS.min(), TS.max()], np.polyval(z, [TS.min(), TS.max()]), color=ACC2, lw=1.5,
       label="slope %+.4f N/min" % z[0])
a.set_xlabel("time into run (min)"); a.set_ylabel("measured force (N)")
a.set_title("Drift over the run  —  r = %+.3f" % np.corrcoef(TS, F)[0, 1],
            loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=7.5)
a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)

fig.suptitle("E-BTS Plan B campaign  ·  %s  ·  1008 pokes, 94 locations, 9 force levels"
             % RUN.name, x=0.008, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.965])
outp = RUN / "campaign_report.png"
fig.savefig(outp, dpi=135)
print("wrote %s" % outp)
