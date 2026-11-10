# Silicone cover molds (v3)

Molds for casting the E-BTS silicone cover onto the seat part (`inputs/seat_actual_2026-11-10.f3d`).
The cover follows the four sloped faces of the seat, covers the 36 × 30 window with a 3 mm pad
carrying the marker-dome grid, and ends where the sloped faces end (18.6 mm above the window plane).

## Print

| File | Orientation |
|---|---|
| `output/print_outer.stl` | as exported: bottom on the bed, cavity up |
| `output/print_core_flange_down.stl` | as exported: flange on the bed, dome pits up |

No supports. PLA, 3 walls, 0.08–0.12 mm layers (the sloped cavity walls form the outer silicone surface).
Ironing on top surfaces smooths the window pad floor and the parting plane.
Keep X-Y hole/contour compensation at 0: the slicer treats the cavity as a hole and would grow it.

Hardware: 4 × M3×30 bolts, 4 × M3 nuts (nuts go into the hex pockets under the corner posts).

## Casting

1. Fill the dome pits on the core and the cavity of the outer mold.
2. Lower the core in; the two tapered pins centre it.
3. Tighten the 4 bolts. Excess silicone and air leave through the 8 grooves in the parting plane.
4. After curing remove the bolts, break off the outer mold, peel the cover off the core, trim the groove flash.
5. Pierce the thin membrane in the 4 round Ø3.2 recesses: they sit exactly over the seat's screw holes.

## Geometry

| Parameter | Value |
|---|---|
| Silicone wall, 22°/28° faces | 0.68 mm |
| Silicone wall, 45.7° faces | 1.15 mm |
| Interference (cover smaller than the seat) | 0.1 mm |
| Pad over the window | 3 mm |
| Cover top edge | 18.6 mm above the window plane |
| Hole markers | Ø3.2, 0.3 mm membrane |
| Outer shell wall / flanges | 2 mm / 3 mm |

The seat is not symmetric across its short axis (faces at 22° and 28°); the molds follow it as is.
The dome-pit pad is taken unchanged from the previous core (`inputs/inner30x36.STEP`).
The molds are disposable: the outer flange stands on a 1.6 mm fence and 4 corner posts so the part
prints upright without supports, and the shell is broken off after curing.
`output/build_report.json` lists the parameters and checks (core and outer do not intersect).

## Not yet verified by casting

- Silicone wall and pad thickness are carried over from the previous molds.
- Whether the 0.3 mm membrane survives demolding.
- Whether 8 grooves are enough venting for the displaced silicone.
- How well FDM reproduces the ~0.8 mm dome pits.
