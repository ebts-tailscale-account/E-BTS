#!/usr/bin/env python3
"""Interactive 3D view of the per-indentation peak force vectors.

Writes a self-contained `output/<batch>/force_vectors_3d.html` -- open it in any
browser to rotate / zoom / hover (each arrow shows poke #, Fx, Fy, Fz, |F|).
Same data as the report's 3D quiver page, but interactive. Rotate to look down
the Fz axis to inspect the lateral (Fx-Fy) direction spread.

Usage:  python3 vector3d_interactive.py [batch]     # default: most recent
"""

import os
import sys

import numpy as np
import matplotlib.cm as cm
import plotly.graph_objects as go

REPO = os.path.dirname(os.path.abspath(__file__))


def resolve_batch(arg):
    if arg and os.path.isdir(arg):
        return arg
    if arg:
        p = os.path.join(REPO, "output", arg)
        if os.path.isdir(p):
            return p
    out = os.path.join(REPO, "output")
    dirs = [os.path.join(out, d) for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))] \
        if os.path.isdir(out) else []
    return max(dirs, key=os.path.getmtime) if dirs else None


def main():
    batch = resolve_batch(sys.argv[1] if len(sys.argv) > 1 else None)
    if not batch:
        sys.exit("No batch found under output/.")
    name = os.path.basename(batch.rstrip("/"))
    pdir = os.path.join(batch, "pokes")
    files = sorted(f for f in os.listdir(pdir) if f.endswith("_ft.csv"))
    if not files:
        sys.exit("No per-poke slices in %s." % pdir)

    colors = ["rgb(%d,%d,%d)" % tuple(int(255 * c) for c in cm.viridis(x)[:3])
              for x in np.linspace(0, 1, len(files))]

    fig = go.Figure()
    for k, pf in enumerate(files, 1):
        d = np.atleast_1d(np.genfromtxt(os.path.join(pdir, pf), delimiter=",", names=True))
        fx, fy, fz = d["Fx_N"], d["Fy_N"], d["Fz_N"]
        i = int(np.argmax(np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)))
        vx, vy, vz = float(fx[i]), float(fy[i]), float(fz[i])
        mag = (vx ** 2 + vy ** 2 + vz ** 2) ** 0.5
        hover = "poke %d<br>Fx %.2f  Fy %.2f  Fz %.2f<br>|F| %.2f N" % (k, vx, vy, vz, mag)
        fig.add_trace(go.Scatter3d(
            x=[0, vx], y=[0, vy], z=[0, vz], mode="lines+markers", name="poke %d" % k,
            line=dict(color=colors[k - 1], width=6),
            marker=dict(color=colors[k - 1], size=[2, 6]),
            text=["", hover], hoverinfo="text"))

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                               marker=dict(color="black", size=4), name="origin", hoverinfo="skip"))

    fig.update_layout(
        title="E-BTS '%s' -- peak contact-force vector per indentation (N)" % name,
        scene=dict(xaxis_title="Fx (N)", yaxis_title="Fy (N)", zaxis_title="Fz (N)",
                   aspectmode="data"),  # true directions (equal scale); rotate to see lateral spread
        legend=dict(itemsizing="constant"))

    out = os.path.join(batch, "force_vectors_3d.html")
    fig.write_html(out, include_plotlyjs=True)  # self-contained, works offline
    print("Wrote %s  (open in a browser: rotate/zoom/hover)" % out)


if __name__ == "__main__":
    main()
