# ml/ — the learning side of E-BTS

Everything that turns a recording into a trained force model. **Nothing here touches
the robot or the camera**; it only ever reads files under `recordings/` and writes to
`output/`. The `master_*.py` orchestrators and the robot scripts stayed in the repo
root deliberately — this folder is safe to run at any time.

All paths resolve against the **repo root**, not the working directory, so these run
correctly from anywhere.

## The pipeline, in order

| script | in | out |
|---|---|---|
| `evt2_frames.py` | `camera.raw` | library: random-access EVT2 decoder (binary-seek, 422× faster than metavision's `start_ts`) |
| `build_dataset.py` | a run folder | `output/<run>_frames.h5` — 40 ms frames + force/depth labels |
| `marker_features.py` | frames | library: fast marker detect + track (33× faster than `marker_overlay.py`, validated to 0.004 px) |
| `extract_features.py` | frames | `output/<run>_features.csv` — 37 columns of magnitude + shape features |
| `train_cnn.py` | frames | **a cross-validated score**, no model. Answers "does this work?" |
| `train_final.py` | frames + features | `output/force_model.pt` — the deployable model. Reports no score, by design. |
| `evaluate_holdout.py` | a NEW run + the model | honest accuracy on data the model never saw |

## Which script when

- **"How good is this approach?"** → `train_cnn.py`. Trains 4 models on 3/4 of the data
  each, scores each on the quarter it never saw, discards them.
- **"Give me something I can use."** → `train_final.py`. Trains on all 648 and saves it.
  ⚠ It deliberately prints no accuracy: any score would be on its own training data.
- **"Does it really work?"** → `evaluate_holdout.py` on a fresh recording. The only
  fully honest test.

## Things that will bite you

⚠ **Split by grid row-band, never randomly.** The deformation field has a measured
~8 mm half-width while the grid pitch is 3 mm, so neighbouring presses are near-duplicate
measurements. A random split trains and tests on effectively the same data. This is not
theoretical — the `^1.5` feature set scored *best* under a leaky split and *worst* under
the honest one.

⚠ **Never feed `depth_mm` or `achieved_mm` to a model.** They are robot-side variables;
depth alone explains R² ≈ 0.51 with no camera at all, so including it hides whether the
sensor works.

⚠ **Ensemble seeds.** Per-seed R² measured 0.52–0.77 on 648 samples. A single run is not
a measurement.

⚠ **Frames are raw event counts (uint8, max 27). Never rescale them.** Counts are the
data; the circles and arrows in any figure are visualisation only and never reach a model.

## Current numbers (strict row-band CV, camera-only, no depth input)

| model | RMSE | R² |
|---|---|---|
| predict the mean | 0.989 N | 0.000 |
| depth only — robot, the bar | 0.692 N | 0.510 |
| marker magnitude + shape | 0.669 N | 0.542 |
| CNN, 5-seed ensemble | 0.579 N | 0.657 |
| **CNN + marker, averaged** | **0.509 N** | **0.734** |

⚠ 69% of the remaining error is one fold — rows 9–11, the strapped edge, where every
model fails including the robot-only one. On rows 0–8 the blend reaches ~0.33 N.
