# Calibration raster — runbook

Measure the sensor-pixel → millimetre map of the EVK1 lens, distortion included,
and write it to `calibration/pixel_to_mm.json`.

This is option **A3** from the de-fisheye options list: a regular robot raster of
shallow pokes at known XY, paired with where the camera says each contact was.

---

## What is already known, before any robot moves

The dome lattice is itself a calibration target, and three independently recorded
runs of 2026-09-02 already measured the lens from it:

| | two_cyl_8mm | three_cyl_16mm | four_cyl_24mm |
|---|---|---|---|
| nodes measured | 211 / 221 | 211 / 221 | 210 / 221 |
| deviation from best affine, RMS | 3.94 px | 4.00 px | 3.80 px |
| deviation, max | 17.8 px | 17.8 px | 16.2 px |
| radial centre | (333, 225) | (333, 226) | (332, 225) |
| fraction radial | 97 % | 97 % | 96 % |
| dome pitch, r < 100 px | 29.42 px | 29.41 px | 29.40 px |
| dome pitch, r > 250 px | 34.49 px | 34.51 px | 34.29 px |

Reproduce any row with:

```bash
python3 ml/fit_pixel_mm_warp.py two_cyl_8mm_20260902_203821 --nodes-only
```

**The distortion is real, radial, and centred 13 px from the image centre — i.e.
it is the lens, not the mould.** An irregular mould has no reason to be radial
about the point the lens looks through.

**It is PINCUSHION, not barrel.** The dome pitch *grows* outward: 29.4 px per cell
near the axis, 34.5 px beyond r = 250. So one millimetre at the edge of the field
covers **17 % more pixels** than one millimetre at the centre (22 % by the fitted
quadratic at r = 300). Any single global px/mm figure is wrong by that much at the
field edge. Correcting it the wrong way round doubles the error.

What the lattice **cannot** give you is absolute scale — the true dome pitch in
millimetres is a construction value, never measured — or the tie to the robot
frame. That is what the raster below is for.

---

## Prerequisites

Same as any campaign (see `PLAN_B_RUNBOOK.md`), plus one that is specific here.

* **Fit the 3 mm pin.** Calibration localises a *contact*. A 16 mm face does not
  have a location, it has an area, and its divergence lobe is broad enough that
  the peak wanders by whole lattice cells for reasons that have nothing to do with
  the lens. `calib_raster.py` warns above 6 mm and refuses above 10 mm.
* HEX21 on the **workstation**, GUI running with both the **Force/Torque** source
  and the **Sequence Recording** pane open.
* Corners taught (`record_corners.py` on tactile) and a current
  `surface_offset_map.csv`.
* On tactile, the Cartesian pose **servo reach** controller up — one controller,
  not both.
* Passwordless ssh to tactile; tactile quiesced.

---

## 0. Copy the script to tactile

`master_campaign.py` does **not** ship `--remote-script` across — it expects it to
already live in `~/E-BTS/` on tactile, and fails the `--dry-run` preflight with a
confusing "are the corners taught?" message if it does not. Copy it, and re-copy
after any edit:

```bash
scp franka/calib_raster.py tactile@100.93.60.35:~/E-BTS/calib_raster.py
```

`campaign_planb.py` must also be on tactile (it normally is — `calib_raster.py`
imports its motion loop, ledger and safety gates rather than duplicating them).

---

## 1. Preview the plan (no motion, no ROS)

```bash
python3 master_campaign.py calib --remote-script calib_raster.py \
    --remote-args --indenter-mm 3.0 --dry-run
```

Defaults are an 11 × 9 raster, inset half a lattice cell from the map border,
one depth of 2.0 mm, three repeats: **297 pokes, ≈ 38 min, ≈ 9.5 GB** of `.raw`.

Worth knowing before you commit the time:

* **Repeats are what make the result interpretable.** They measure the
  contact-centroid estimator's own noise, which is the only way to tell a residual
  caused by the lens from one caused by the estimator. `--repeats 1` runs, but
  prints a warning saying the fit cannot be believed.
* **Two depths is a free bias test.** `--depths-mm 1.5 2.5` costs one extra pass
  and answers "does the apparent contact position move when the indenter goes
  deeper?" It should not — the indenter axis has not moved.
* Refused targets are reported. A raster point in a map cell with an untrusted
  corner is skipped rather than pressed at an unknown depth, which leaves a hole in
  coverage — and extrapolating a distortion fit into a hole is how a calibration
  goes quietly wrong.

Check the plan, the estimated time, and the refusal list before continuing.

---

## 2. Run it

```bash
python3 master_campaign.py calib --remote-script calib_raster.py \
    --remote-args --indenter-mm 3.0
```

`--remote-args` must come **last**.

Interrupted? The ledger is `fsync`'d after every poke, so at most the poke in
flight is lost:

```bash
python3 franka/calib_raster.py --resume
```

The plan is fingerprinted; resuming with different parameters aborts rather than
stitching two geometries into one calibration.

---

## 3. Post-process

The run is phase-tagged exactly like a campaign, so the standard chain works
unchanged:

```bash
python3 run_two_cyl_pipeline.py calib_<stamp>
```

You need stages 1–5 (`join_campaign` → `build_frames` → `export_frames` →
`make_poke_windows` → `blue_circle_grid`). Stage 5 streams the whole `.raw` and is
the slow one, ≈ 30 s per GB.

---

## 4. Fit the warp

```bash
python3 ml/fit_pixel_mm_warp.py calib_<stamp>
```

Writes `calibration/pixel_to_mm.json` and
`output/runs/calib_<stamp>/pixel_mm_warp.png`.

It prints two tables. **Read them in the right order, because the first one is
easy to misread as the accuracy of the calibration and it is not.**

* **Per-poke cross-validated residual.** Grouped k-fold: all repeats of a raster
  point sit in one fold, so a model cannot see a point's own answer while
  predicting it. This number is floored by the estimator, not by the lens — a
  contact is localised on the 13 × 17 lattice, so its position is quantised to a
  fraction of a cell (~0.25 cell ≈ 0.5 mm in the self-test). Every model piles up
  at that floor and they are largely indistinguishable there.
* **What the calibration is actually for** is the fitted surface, where ~300
  independent quantisation errors average into 6–10 coefficients. In the self-test
  with the rig's real distortion magnitude, the surface error is 0.12 mm against
  0.30 mm for doing nothing — so correcting removes about 60 % of the position
  error.

Models are compared automatically:

| model | what it is |
|---|---|
| `affine` | the do-nothing baseline: rotation, shear, per-axis scale, no distortion |
| `poly2` … `poly4` | polynomial warps in pixel space |
| `tps` | thin-plate spline |
| `lattice_affine` | **shape from the domes, scale from the raster** |

`lattice_affine` is usually the right answer and is the one to prefer unless the
table says otherwise. The reasoning: the two halves of this calibration have very
different precisions. Dome centres are measured to a fraction of a pixel from
hundreds of domes; the contact position is quantised to tens of pixels. Fitting a
10-coefficient polynomial to the *noisy* quantity spends the good data re-deriving
what the domes already gave you for free. `lattice_affine` takes the distortion
shape from the dome grid and asks the raster only for the six numbers it is
genuinely needed for — scale, rotation, shear and origin.

Select it explicitly with `--model lattice_affine`.

The run also prints the **measured** px/mm along each image axis. Compare it with
what `src/circle_tracker_config.h` currently implies (17.78 and 16.00 px/mm,
anisotropy 1.111) — that config maps a 36 × 30 mm sheet onto a 4:3 sensor with
square 15 µm pixels, which is not geometrically possible, so one of those numbers
is wrong today.

---

## 5. Use it

```python
from ml.undistort import load
cal = load()
x_mm, y_mm = cal.pixel_to_mm(u, v)
```

```bash
python3 ml/undistort.py --demo     # local px/mm at the centre and the corners
```

Three things to get right:

* **Apply it to marker centres, not to images.** A few hundred detected centres
  per frame is cheap and exact. Resampling a 640 × 480 event-count image through
  `cv2.remap` blurs markers that are only ~19 px across and manufactures intensity
  no event produced. `cal.lut()` exists for rendering a corrected preview and is
  deliberately not the easy path.
* **Displacements do not transform like positions.** The warp is nonlinear, so map
  both endpoints and subtract: `cal.displacement_to_mm(u0, v0, u1, v1)`. Never
  `cal.pixel_to_mm(du, dv)`.
* **Outside the dome grid you get NaN, on purpose.** The domes are the
  calibration; past the outermost one there is only a polynomial's opinion. An
  earlier version clipped to the lattice edge instead, and a pixel just past the
  edge came back with a local scale of 464 px/mm rather than ~16, silently.

---

## Checks that run without a robot

```bash
python3 franka/calib_raster.py --self-test        # raster geometry, plan, ledger
python3 ml/fit_pixel_mm_warp.py --self-test       # the whole fit, synthetic truth
python3 ml/fit_pixel_mm_warp.py <any_run> --nodes-only   # real distortion, real data
```

The fit self-test injects a known radial distortion sized to match this rig
(≈ 18 px peak after affine removal) and checks that it is recovered, that the
contact-peak path works, that `lattice_affine` wins, that the inverse refuses
points outside the field, and that the JSON round trip holds.
