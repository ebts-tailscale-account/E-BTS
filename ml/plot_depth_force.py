#!/usr/bin/env python3
"""Two plots: force distribution per commanded depth, and every poke as depth vs force.

The second panel is the physically important one. Force is NOT a function of depth
alone -- it is stiffness(location) x depth -- so the 94 per-location curves fan out.
The width of that fan IS the spatial stiffness variation, and it is exactly what a
camera-only model has to resolve from the image without being told where it pressed.
"""
import csv, sys, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUN = Path(sys.argv[1])
d = list(csv.DictReader(open(RUN / "pokes.csv")))
F = np.array([float(r["force_n"]) for r in d])
DA = np.array([float(r["depth_achieved_mm"]) for r in d])
DC = np.array([float(r["depth_cmd_mm"]) for r in d])
K = np.array([float(r["stiffness_map"]) for r in d])
PI = np.array([int(r["point_index"]) for r in d])
levels = sorted(set(DC))

INK = "#1b1b1f"; MUTED = "#6c6a74"; ACC = "#2f6f8f"; ACC2 = "#c2703d"; GRID = "#dcdae2"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 6.2))

# ---- 1. violin: measured force at each commanded depth -----------------------
data = [F[DC == t] for t in levels]
vp = a1.violinplot(data, positions=range(len(levels)), widths=0.8,
                   showmedians=True, showextrema=False)
for b in vp["bodies"]:
    b.set_facecolor(ACC); b.set_alpha(0.6); b.set_edgecolor(ACC); b.set_linewidth(0.8)
vp["cmedians"].set_color(INK); vp["cmedians"].set_linewidth(1.5)
for i, t in enumerate(levels):
    v = F[DC == t]
    a1.scatter(np.full(len(v), i) + np.linspace(-0.14, 0.14, len(v)), v,
               s=2.5, c=INK, alpha=0.20, linewidths=0, zorder=3)
a1.set_xticks(range(len(levels)))
a1.set_xticklabels(["%.1f" % t for t in levels])
a1.set_xlabel("commanded indentation depth (mm)")
a1.set_ylabel("measured force (N)")
a1.set_title("Force distribution per depth  —  n=%d each" % len(data[0]),
             loc="left", fontweight="bold")
a1.grid(axis="y", color=GRID, lw=0.6); a1.set_axisbelow(True)

# ---- 2. every poke: achieved depth vs force, one line per location -----------
order = np.argsort(K)
sm = plt.cm.ScalarMappable(cmap="viridis",
                           norm=plt.Normalize(vmin=K.min(), vmax=K.max()))
for pid in np.unique(PI):
    m = PI == pid
    o = np.argsort(DA[m])
    a2.plot(DA[m][o], F[m][o], "-", color=sm.to_rgba(K[m][0]), lw=0.7, alpha=0.45,
            zorder=1)
a2.scatter(DA, F, c=K, cmap="viridis", s=9, alpha=0.85, linewidths=0, zorder=2)
a2.set_xlabel("achieved indentation depth (mm)")
a2.set_ylabel("measured force (N)")
a2.set_title("Every poke (n=%d)  —  %d locations, one line each"
             % (len(F), len(np.unique(PI))), loc="left", fontweight="bold")
cb = fig.colorbar(sm, ax=a2, pad=0.015)
cb.set_label("mapped stiffness (N/mm)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
a2.grid(color=GRID, lw=0.6); a2.set_axisbelow(True)

# the fan, quantified: force spread at the deepest level
deep = DC == max(levels)
a2.annotate("at %.1f mm the SAME depth gives\n%.2f - %.2f N depending on where"
            % (max(levels), F[deep].min(), F[deep].max()),
            xy=(np.median(DA[deep]), np.median(F[deep])),
            xytext=(0.06, 0.88), textcoords="axes fraction",
            fontsize=8.5, color=ACC2,
            arrowprops=dict(arrowstyle="->", color=ACC2, lw=1.1,
                            connectionstyle="arc3,rad=-0.2"))

fig.suptitle("%s  ·  indentation depth vs measured force" % RUN.name,
             x=0.006, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = RUN / "depth_vs_force.png"
fig.savefig(out, dpi=140)
print("wrote %s" % out)

# ---- the numbers behind the picture -----------------------------------------
print("\n%-11s %-6s %-22s %-20s" % ("cmd depth", "n", "achieved", "force"))
for t in levels:
    m = DC == t
    print("%-11.2f %-6d %-22s %.2f - %.2f N (med %.2f)"
          % (t, m.sum(), "%.3f +- %.3f" % (DA[m].mean(), DA[m].std()),
             F[m].min(), F[m].max(), np.median(F[m])))
print("\nspread at the deepest level: %.2fx  (%.2f -> %.2f N at the same commanded depth)"
      % (F[deep].max() / F[deep].min(), F[deep].min(), F[deep].max()))
print("that fan is the spatial stiffness field -- what the camera must infer unaided.")
