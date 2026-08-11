# Plan B campaign — operator runbook

**Everything launches from the WORKSTATION.** `master_campaign.py` starts the GUI
recording (camera + HEX21), drives `campaign_planb.py` on tactile over ssh,
measures the clock offset either side, and pulls the whole run directory back.

⚠ **The HEX21 stays plugged into the WORKSTATION throughout.** There is no replug
at any step below. (It goes to tactile only for surface mapping / probing, which
is already done.)

---

## Prereqs — same as any campaign

- **Workstation:** GUI running with **both** the Force/Torque source and the
  Sequence Recording pane open — `cd ~/E-BTS && ./build/E_BTS_GUI`
- **Tactile:** arm at HOME, servo *reach* launch up (not MoveIt, not
  franka_control):
  ```bash
  source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
  roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
      robot_ip:=10.1.196.5 load_gripper:=false
  ```
- **Tactile quiesced** — load average < 4.

⚠ `--remote-args` forwards everything after it verbatim, so it **must come last**
on the command line.

---

## Step 1 — Pre-condition  (~20 min)  ⭐ START HERE

```bash
cd ~/E-BTS
python3 master_campaign.py precond --remote-script campaign_planb.py \
    --remote-args --precondition --ack-deep
```

Cycles every location **3× to its own campaign maximum** — not to a global 6 mm,
so no location sees strain the experiment never repeats. This settles the Mullins
softening so the campaign is measured on stable material.

⭐ **This is also the strap test at full depth.** If the straps are going to
complain at 6 mm, they complain here — 20 minutes in, before you have committed
two hours. Watch the force trace in the GUI.

It also *records* force at 94 known depths, which is what step 2 fits.

## Step 2 — Fit the force model  (offline, no robot)

The recorded pre-conditioning run is the calibration data: each location was
pressed to a known depth with force recorded. Fitting it gives the softening
exponent, which is what lets the campaign hit its upper force levels.

**Without this the campaign still runs and the labels are still correct** — force
is *measured* by the HEX21, never inferred. You would just land near **2.0 N**
instead of 2.44 N at the top, with the levels compressed at the upper end.

## Step 3 — Run  (~2.1 h)

```bash
python3 master_campaign.py planb --remote-script campaign_planb.py \
    --remote-args --ack-deep --force-model force_model.json
```

**1008 pokes.** Prints a running ETA. Drop `--force-model force_model.json` to run
uncalibrated (see step 2 for what that costs).

---

## If it stops — for any reason

Re-run the same command with `--resume` added to the remote args:

```bash
python3 master_campaign.py planb --remote-script campaign_planb.py \
    --remote-args --resume --ack-deep --force-model force_model.json
```

It finds the newest campaign that actually started, skips every completed poke,
opens a new recording segment, and continues.

- Progress is `fsync`'d after **every** poke → a hard kill loses at most the poke
  in flight, and that one is simply redone.
- Each resume writes `franka_segNN.csv`, and re-running through
  `master_campaign.py` starts a fresh camera recording to match. Postprocessing
  joins on `(segment, seq)`.
- If any parameter that changes the plan differs, resume **refuses** rather than
  stitching two experiments together. Pass the original arguments, or `--fresh`
  (which archives the old ledger — nothing is ever deleted).

Check progress any time without touching the robot:

```bash
ssh tactile@100.93.60.35 \
  "cd ~/E-BTS && python3 campaign_planb.py --resume --dry-run --ack-deep"
```

---

## What lands on disk

On tactile, `~/E-BTS/recordings/<run>/`, pulled back to the workstation run folder:

```
plan.csv           1008 pokes in execution order
plan.json          parameters + fingerprint + indenter diameter
state.jsonl        the resume ledger, one line per completed poke
franka_segNN.csv   one franka_states log per segment
camera.raw         }
ft.csv             } collected by master_campaign from the GUI
metadata.json      clock offsets, taught geometry, remote args
```

No data in filenames. Everything joins on `point_index` / `seq`.

---

## The design, in one paragraph

Stiffness varies **2.82×** across this block (0.467–1.319 N/mm over 94 good map
points), so a fixed depth ladder confounds force with position — the defect that
made the pilot unusable. This inverts the map instead: each location gets the
depth that lands on a **target force**. Nine levels, **0.20–2.44 N**, where 2.44 N
is the most any location can reach inside 6 mm (set by r0c8, the softest point).
Nine passes, each location seeing each level exactly once in random order, with
serpentine traversal alternating direction per pass. Consequence worth knowing:
**stopping early still leaves a balanced dataset** — every level stays roughly
equally represented.

## Two numbers to keep in mind when results come back

- **Effective N is 1008 pokes, not ~50,000 frames.** A 2 s dwell yields ~50 frames
  whose within-dwell force sd (0.1168 N) equals the Wittenstein noise floor
  (0.116 N) — they are replicates and add no degrees of freedom.
- **~12 spatially independent sites exist on the whole block** (8 mm deformation
  half-width vs 26×31 mm). Dense sampling is good for *fitting*; it does not buy
  independent evidence for *validation*. Group CV by row band, with a buffer.
