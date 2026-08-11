# E-BTS — Project & Session Handoff

**Last updated:** 2026-08-06 (§12 = the single-point repeated-indent workflow; note §0/§10 still describe the pre-2026-08-05 state)
**Purpose:** Everything a fresh assistant (or the user on a different account) needs to continue the E-BTS data-collection rig work without re-deriving context. Read this top-to-bottom before touching the robot.

---

## 0. TL;DR — where we are right now

- **Goal of current phase:** re-map the elastomer surface with the *servo* controller, then run a synchronized trial data-collection sweep recording "almost everything."
- **DONE & deployed/verified:** servo-based surface mapper (`map_surface.py`), the `cartesian_pose_servo_reach.launch` (raises reach cap so the arm can go home→surface), the servo-based experiment logger (`franka_grid_logger.py`) with 92-column logging, `master.py`/`postprocess.py` updates.
- **DONE (camera, 2026-08-04):** event-stream filtering is wired end-to-end — `bias_tuner` now has live **hardware ROI** controls + **Load biases**, and `E_BTS_GUI` **applies the tuned biases + ROI at camera open** from `calibration/camera.bias`/`.roi` (see §3.4). Biases **and** ROI are tuned, saved and installed — the camera side of the filtering work is complete.
- **IMMEDIATE NEXT STEP (user-triggered, physical):** run `python3 map_surface.py --max-points 5` on tactile (servo reach launch up, HEX21 plugged into tactile), eyeball the per-step probe output, then run the full 99-point map.
- **PENDING (Part 2, not yet coded):** make `franka_grid_logger.py` auto-home from **home** to the taught **bottom-left** reference using the same reach-aware approach as `map_surface.py` (it currently reads the *current* pose as the reference = manual pre-position). Optionally feed `surface_map.csv` per-point touch heights into the sweep.

---

## 1. Project context

**E-BTS (Event-Based Tactile Sensor)** — Tactile Lab, Nazarbayev University. Building a synchronized multi-modal rig to collect a dataset for training a **CNN that estimates contact force/location from event-camera marker displacement**.

Three synchronized systems:
1. **Prophesee EVK1 event camera** (Gen3.1, 640×480) looking at the elastomer's internal markers.
2. **Wittenstein/Resense HEX21 6-axis F/T sensor** (ground-truth force).
3. **Franka Emika Panda** robot that indents the elastomer in a grid.

The rig indents a soft silicone elastomer over a grid, capturing camera events + F/T + robot state per indentation, all UNIX-timestamped for alignment.

---

## 2. Machines, network, access

| Host | Address | Role | Key paths |
|---|---|---|---|
| **workstation** | `skymario-ubuntu` (local, this device) | Camera + HEX21 F/T + all scripts/GUI | `/home/skymario/E-BTS/` |
| **tactile** | `tactile@100.93.60.35` (Tailscale SSH) | Franka + ROS Noetic + MoveIt/servo | `~/E-BTS/`, `~/ws_franka/` |

- **Tailscale SSH re-auth:** the tunnel to tactile expires periodically. When `ssh`/`scp` hangs, it's waiting on a Tailscale auth prompt — the user must visit the printed `https://login.tailscale.com/a/...` URL. After that, ssh/scp work again.
- Passwordless ssh assumed (`ssh-copy-id` done).
- **Git:** the E-BTS repo must **NEVER** be committed or pushed without the user's **explicit, per-time** permission. Do not `git push`/`git commit` on your own.

---

## 3. Hardware facts

### 3.1 Franka Panda
- `robot_ip = 10.1.196.5`, FCI enabled.
- **Clean ROS workspace = `~/ws_franka`** (source it LAST so it wins name resolution). There is a second, *broken/contaminated* `panda_moveit_config` in an allegro `catkin_ws` — do NOT use it.
- Standard ready/home joints: `[0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398]`. Home EE ≈ `[0.308, 0.000, 0.591] m` (measured this session).
- **Collision reflex** ~20 N (franka_control thresholds). If the arm latches in REFLEX or you press the user-stop, `franka_states` stops publishing → any client `wait_for_message` times out. Recover: release user-stop (twist button), `error_recovery`, relaunch.
- Franka's own external wrench `O_F_ext_hat_K` is **biased (~±2 N, pose-dependent)** — use the **Wittenstein as ground truth**, or subtract a baseline.

### 3.2 Wittenstein/Resense HEX21 F/T
- USB CDC VCP **VID:PID 0483:5740**, `/dev/ttyACM*`, **2,000,000 baud**, **DIP6 = ON** (electronics-calibrated), DTR/RTS asserted, **single-owner port**.
- 28-byte packets = 7 little-endian float32: `Fx Fy Fz (N), Mx My Mz (mNm), Temp (°C)`, 1 kHz.
- **Normally plugged into the workstation** (for data collection). **Must be moved to tactile for surface mapping** (the mapper needs force feedback locally on tactile). Auto-detected by VID:PID.
- Calibration matrix: `Matrix_HEX21.txt`.

### 3.3 Prophesee EVK1 event camera
- Metavision/OpenEB SDK 3.1.2. Device µs timestamps.
- **Single-owner:** close `E_BTS_GUI` and any vendor GUI before running another camera tool (e.g. `bias_tuner`). **Quit camera tools via the window ✕, NOT Ctrl-C** — a Ctrl-C'd run leaves a stuck FX3 (`04b4:00f4`) and the next open hangs; fix by **replugging the EVK1**.
- HAL facilities: `I_LL_Biases` (biases: `bias_diff, diff_on, diff_off, fo`=LPF, `hpf`=HPF, `pr, refr`), `I_ROI`, software `PolarityFilterAlgorithm`. `.bias` file format: `<value> % <bias_name>`.
- Biases and ROI are tuned/applied per §3.4 — read that before recording.

### 3.4 Camera calibration — biases + hardware ROI (READ BEFORE RECORDING)

**The EVK1 does NOT retain biases or ROI between sessions.** Every camera open comes up at sensor defaults. Tuning in `bias_tuner` therefore has **zero effect** on a later `E_BTS_GUI` recording unless the values are re-applied — which is why the two files below exist. (Proof: all 13 `.bias` sidecars from the July recordings are byte-identical defaults, and the tuner came up at those same defaults a week after a tuning session.)

**Active calibration = two files, both written by `bias_tuner`'s "Save biases…":**
```
calibration/camera.bias   # Metavision format, "<value> % <bias_name>" per line
calibration/camera.roi    # E-BTS companion: "x y width height", or "disabled"
```
`E_BTS_GUI` loads them in `CameraSessionWorker::connectToCamera()` and programs the sensor **before** `camera_->start()` (`src/camera_calibration.h`). Values are set-then-read-back, so the console names anything the sensor clamped or refused:
```
[E_BTS_GUI] [calib] biases applied: 7/7 from calibration/camera.bias
[E_BTS_GUI] [calib]   bias_diff = 299
[E_BTS_GUI] [calib] ROI applied: x=… y=… w=… h=… (N px, M% of frame)
```
Override the path with `$E_BTS_CAMERA_BIAS`. Missing file / missing facility / rejected ROI ⇒ warning + continue at defaults; it never costs you the camera session.

**Tuned biases (2026-08-04, installed):** `bias_diff 299, bias_diff_off 0, bias_diff_on 377, bias_fo 1550, bias_hpf 1411, bias_pr 1250, bias_refr 1500`.
- **`bias_diff_off = 0` is the OFF-polarity kill** — done in *hardware* via the bias, NOT the software `PolarityFilterAlgorithm` (which was never written and is not needed).

**Tuned ROI (2026-08-04, installed):** `0 7 640 450` — full width, rows 7–456, i.e. 7 rows trimmed off the top and 23 off the bottom. 288,000 of 307,200 px = **93.75% of the frame retained**, so this is an edge trim, not a bandwidth-reduction crop; tighten it if event rate becomes the constraint.

**Hardware ROI (`I_ROI`) — what it does and doesn't do:**
- Masked rows/columns are **never read out**, so events outside the box are never generated. This costs **no** host computation — it *reduces* USB bandwidth, decode cost, `.raw` size and CNN preprocessing. (Contrast the software polarity filter, which pays per-event on the host.)
- The `.raw` is **sparse, not cropped**: event coordinates stay **absolute** (an ROI at x=192 yields x∈192…447, not 0…255) and the header still reports `% geometry 640x480`. You still crop in software when building tensors.
- Nothing in the `.raw` or `.bias` records the ROI, so `E_BTS_GUI` writes a `<run>.roi` sidecar next to each `.raw`; `postprocess.py` carries it into `output/<run>/camera.roi`. Without that a run has an undocumented spatial crop.
- HAL 3.1.2 API note: use `set_ROIs(cols_mask, rows_mask, enable)` (pure-virtual ⇒ no direct HAL link needed). `set_ROI(DeviceRoi)`/`create_ROI` are non-virtual HAL symbols and may fail to link. The 4.0.0-style `Mode`/`Window`/`set_lines` API is **not** in 3.1.2. Pass `enable` in the same call — `I_ROI::enable()` warns an ROI must already be set.
- Two disjoint boxes produce their **grid intersection** (2 boxes → 4 regions), so the tuner deliberately exposes one rectangle only.

### Run procedure (camera calibration)
```
# close E_BTS_GUI first (single-owner camera!), then:
cd ~/E-BTS && ./calibration/bias_tuner/build/bias_tuner
#   Load biases…  -> calibration/camera.bias      (restores a previous session)
#   drag a rectangle on the live view             (sets + enables the ROI)
#   Save biases…  -> calibration/camera.bias      (writes BOTH .bias and .roi)
# quit via the window X, NOT Ctrl-C. Then relaunch E_BTS_GUI and check the
# "[calib]" console lines actually list your values.
```
⚠ **Saving is the only thing that persists.** Dialling a ROI in the tuner and closing it leaves nothing behind — sensor state dies with the process. Watch the status line for `Saved camera.bias + camera.roi (ROI …)`.
⚠ **`*.bias` is gitignored** (`.gitignore:37`), so `calibration/camera.bias` (and its `.roi`) are **not version-controlled** — the active calibration lives only on this workstation.

---

## 4. The Cartesian pose SERVO controller (CRITICAL — read this)

We drive the Franka with a **topic-driven Cartesian pose servo** (`CartesianPoseServoController`) instead of MoveIt, because it reaches the **exact commanded depth** and is **force-blind** (won't stop/undershoot against the soft silicone like MoveIt's trajectory controller did).

**Source/launch on tactile:** `~/ws_franka/src/franka_ros/franka_example_controllers/{src,launch,include}/…cartesian_pose_servo*`.

### 4.1 How to use it
```
source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
# stock (conservative 0.30 m reach cap):
roslaunch franka_example_controllers cartesian_pose_servo.launch robot_ip:=10.1.196.5 load_gripper:=false
# OR start-from-home version (0.55 m cap — see 4.3):
roslaunch franka_example_controllers cartesian_pose_servo_reach.launch robot_ip:=10.1.196.5 load_gripper:=false
```
- Only **one** controller at a time (servo OR MoveIt — they claim the same Cartesian interface).
- Stream `geometry_msgs/PoseStamped` targets (frame `panda_link0`) to `/cartesian_pose_servo_controller/target_pose`.
- Python client: **`servo_client.py`** → class `CartesianServo` with `current_pose()`, `send_target(x,y,z,quat)`, `move_to(x,y,z,quat,tol,timeout,poll)`, `step(dx,dy,dz)`.

### 4.2 Controller limits (from the yaml)
```
max_linear_velocity: 0.03 m/s   # VERY slow — a 0.4 m leg takes ~25 s. This is NOT a hang.
max_linear_acceleration: 0.25
max_linear_jerk: 2.0
max_angular_velocity: 0.3 rad/s
min_duration: 0.5 s             # every move (even 0.1 mm) takes >= 0.5 s
max_translation: 0.30 m         # <-- the reach clamp, see below
internal_controller: joint_impedance   # tracks a commanded pose to only ~0.5 mm
```

### 4.3 GOTCHA #1 — the 0.30 m reach clamp is anchored at the LAUNCH pose
In `starting()` the controller captures `start_pos_` = the EE pose **when the controller is launched**, and in `targetPoseCallback()` it **clamps every target to a sphere of radius `max_translation` (0.30 m) around that fixed start pose**. Walking in hops does NOT help — the sphere is fixed at the launch pose.

⇒ If you launch the servo **at home**, the elastomer (~0.43 m forward) is **out of reach**: the arm creeps at 0.03 m/s into the clamp boundary and never descends (you end up hitting the user-stop, which kills `franka_states`).

**Fix (chosen):** `cartesian_pose_servo_reach.launch` — identical to the stock launch but overrides `max_translation` to **0.55 m** (leaving the stock 0.30 m default untouched). Backstops unchanged: 0.03 m/s creep + 20 N reflex + supervision. Confirm it's active with:
```
rosparam get /cartesian_pose_servo_controller/max_translation   # should print 0.55
```
Scripts also print `servo cap 0.xxx m` in their pre-flight — if it says 0.300 you launched the wrong one.

### 4.4 GOTCHA #2 — do NOT use tight `move_to` tolerance while probing
The internal joint-impedance controller tracks a commanded pose to only ~**0.5 mm**, so `move_to` can never converge to a sub-mm tolerance — it just times out every step (`move_to timeout: residual 0.0005 m`), and a tol ≥ the step returns *without moving*. For force-probing, **pace** the descent instead: `send_target(depth)` → `sleep(PROBE_SETTLE_S ≥ min_duration)` → read force, and record the **true EE pose** at contact (the 0.5 mm command lag is then irrelevant). Gross positioning moves use a loose `move_to` and settle (see 4.5).

### 4.5 GOTCHA #3 — settle between moves or you get a REFLEX
The controller re-plans a fresh min-jerk segment whenever a **new** target arrives, and that segment assumes **zero initial velocity**. If you send the next target while the arm is still moving (e.g. `move_to` returned early on a loose tol), the commanded velocity jumps → a joint-acceleration discontinuity → the robot reflexes (`cartesian_motion_generator_joint_acceleration_discontinuity`) and **freezes**. Fix: every gross move must **wait until the arm has arrived AND stopped** (poll pose; require within ~3 mm and inter-poll motion < ~0.6 mm for a few polls), then hold briefly, before issuing the next target (`_gross_move` in `map_surface.py`). Probe steps settle by using `PROBE_SETTLE_S > min_duration`. **After a reflex the controller is latched** — relaunch the (reach) launch (or send `/franka_control/error_recovery`) before it will move again.

---

## 5. Taught reference points — `corner_joints.json`

On tactile: `~/E-BTS/corner_joints.json`. Recorded by `record_corners.py` (hand-guide the tool to *kiss* each point, no indent, press Enter). Each of the **5 points** (center + 4 corners) stores **both** `joints[7]` and `xyz[3]`, captured from the **same `FrankaState` snapshot** — so the `xyz` **is** the forward kinematics of the taught joint state. That means you can reproduce a taught pose as a Cartesian servo target without a separate FK solver.

Elastomer ≈ **30 × 36 mm**. Taught corner xyz (base frame, m):
```
center        [0.65444,  -0.00171, 0.26879]
bottom-left   [0.63738,   0.01290, 0.26881]
top-left      [0.67344,   0.01239, 0.26874]
top-right     [0.67340,  -0.01730, 0.26970]
bottom-right  [0.63803,  -0.01663, 0.26967]
```
- **X** 0.637→0.673 (≈36 mm) = the "vertical"/row edge; **+X = row up**.
- **Y** +0.013→−0.017 (≈30 mm) = the "horizontal"/col edge; **−Y = col right**.
- **Z** ~0.2687→0.2697 (~**1 mm surface tilt** across the block) — this tilt is why a single flat touch-Z caused the far column to miss/over-press ("col-3 out of bounds"). The surface map fixes this.

---

## 6. Files — what/where/status

### 6.1 Workstation `/home/skymario/E-BTS/` (source of truth for scripts)
| File | Purpose | Status |
|---|---|---|
| `master.py` | Orchestrates ONE synchronized run: writes `control/start.cmd` for the GUI (camera+F/T), ssh-runs the Franka logger on tactile, measures workstation↔tactile clock offset, pulls CSV back, writes metadata. | Updated (bringup → servo, pre-position note) |
| `franka/franka_grid_logger.py` | **Experiment sweep** + 92-col `franka_states` logger. Servo motion. | Servo-wired; **Part 2 pending** (still reads current pose as start) |
| `franka/map_surface.py` | **Surface mapper**: 99-pt servo-probed height map (INSET_MM=0 → covers edges). | DONE, deployed, verified |
| `tidy_recordings.py` | Retro-groups loose `recordings/` files into one folder per run with canonical names. `--dry-run` first. | DONE |
| `franka/retry_points.py` | Re-probe NO_CONTACT map points with a deeper floor (`--max-depth-mm`, default 4) and merge back into `surface_map.csv` + heatmap. Reuses all of `map_surface.py`. Auto-detects non-'ok' rows or `--points 8,98`. | DONE, deployed |
| `franka/cartesian_pose_servo_reach.launch` | Servo launch with `max_translation=0.55`. | DONE, deployed to the ROS package |
| `postprocess.py` | recordings/ → output/<run>/: aligns clocks, zeros baselines, segments pokes from `ee_z` dips, per-poke F/T + event slices, summary. | Updated: labels (col,row) from logged columns for any grid size; also carries the `.roi` sidecar → `output/<run>/camera.roi` |
| `visualize.py` | `report.pdf` (force/torque/event-rate overview, resultant page, 3D quiver, per-poke). `--zoom FIRST LAST --margin`. | Existing |
| `vector3d_interactive.py` | Plotly HTML of per-poke peak force vectors. | Existing |
| `calibration/bias_tuner/` | Standalone Qt5 camera calibration GUI: slider per auto-discovered bias, live event view, accumulation-time control, ON/OFF event-rate stats, **live hardware-ROI controls (drag on the view, or x/y/w/h + Full frame)**, **Load** and Save → `.bias` + `.roi`. Build: its own CMake. | **Updated 2026-08-04** (ROI + Load added), built cleanly |
| `src/camera_calibration.h` | Parses `.bias`/`.roi`, applies biases + ROI to an open camera (set-then-read-back, never throws), writes the per-run `.roi` sidecar. | **New 2026-08-04** |
| `calibration/camera.bias`, `calibration/camera.roi` | **Active** camera calibration consumed by `E_BTS_GUI` at open (§3.4). Gitignored — workstation-local. | Installed 2026-08-04: 7 biases + ROI `0 7 640 450` |
| `src/…`, `qml/…` | The `E_BTS_GUI` C++/Qt app (camera session, Wittenstein worker, F/T pane + graph, sequence recording controller). | Updated: `camera_session_worker` applies calibration at open + writes the `.roi` sidecar per recording |
| `SOFTWARE_README.md`, `arduino/two_phase_driver/…` | Software readme + Arduino complementary 25 Hz driver (D2/D4). | Existing |

### 6.2 Tactile `~/E-BTS/` (deployed copies + ROS helpers)
| File | Notes |
|---|---|
| `franka_grid_logger.py` | Deployed servo logger. Backup: `franka_grid_logger.moveit.bak` |
| `map_surface.py` | Deployed servo mapper. Backup: `map_surface.moveit.bak` |
| `servo_client.py` | `CartesianServo` client (imported by the two above) |
| `corner_joints.json` | The 5 taught points (§5) |
| `home_and_level.py` | `flat_down_quat()` (numpy-only leveling), `HOME_JOINTS_READY`, quat helpers. (Its `main()` uses MoveIt; we only import the helpers.) |
| `franka_surface_map.py` | Contains `WittensteinFT` (serial FT reader, reused by the mapper) and the old MoveIt `FrankaArm`. |
| `record_corners.py`, `corner_probe.py`, `read_pose.py` | Teach/verify utilities |
| `surface_map.csv`, `surface_map.png`, `surface_grid_preview.png` | Map outputs |
- **Redeploy after editing the workstation copy:** `scp franka/<file> tactile@100.93.60.35:~/E-BTS/<file>` (and launches → `~/ws_franka/src/franka_ros/franka_example_controllers/launch/`).

---

## 7. `map_surface.py` (servo surface mapper) — how it works

Builds `z_touch(x,y)` over the elastomer by force-probing a **9×11 = 99-point** bilinear grid over the 4 taught corners (2 mm inset), serpentine order.

- **Motion:** `CartesianServo` only, via `_gross_move` (settles between moves — §4.5). Orientation held **flat-down** (`flat_down_quat`, approach axis → global −Z), computed from the current pose → auto-leveled, no manual jog.
- **Homing (no joint-home under servo):** from **home → straight to a hover `CENTER_APPROACH_MM` (15 mm) above the taught center in ONE settled move**, then the raster starts at **bottom-left**. `preflight_reach()` aborts **before any motion** if any target exceeds the servo clamp (prints `farthest target X m … servo cap Y m`).
- **Probing:** paced `send_target` descent (§4.4), records the **true EE z** at first force rise (Wittenstein threshold = `max(2σ, 0.05 N)`); retract on failure; 2 mm hard depth floor.
- **Config:** `INSET_MM=0` (grid spans corner-to-corner = covers the edges; `--inset-mm` to pull in), `HOVER_MM=3, MAX_DEPTH_MM=2, CENTER_APPROACH_MM=15, PROBE_STEP_MM=0.4, PROBE_SETTLE_S=0.6`; settle: `REACH_TOL_M=0.003, STOP_EPS_M=0.0006, SETTLE_HOLD_S=0.4, PROGRESS_EPS_M=0.001, NO_PROGRESS_S=5`.
- **Stuck detection:** `_gross_move` aborts only if the error stops *decreasing* for `NO_PROGRESS_S` (a min-jerk move starts gently, so low instantaneous speed early on is NOT a freeze).
- **Output:** `surface_map.csv` (point_id,row,col,x,y,z_plane,z_touch,depth_from_plane_mm,contact_fz,status) + `surface_map.png` heatmap.
- **CLI:** `--dry-run` (grid + preview PNG, no ROS/motion), `--max-points N` (cautious), `--no-level`.

### Run procedure (surface map)
```
# tactile, arm at HOME, HEX21 plugged into tactile:
source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
roslaunch franka_example_controllers cartesian_pose_servo_reach.launch robot_ip:=10.1.196.5 load_gripper:=false
# new terminal:
cd ~/E-BTS
python3 map_surface.py --dry-run        # check surface_grid_preview.png
python3 map_surface.py --max-points 5   # cautious first run; watch per-step "z=… Fz=… <-- CONTACT"
python3 map_surface.py                  # full 99-point map (~15-20 min; slow servo, not a hang)
```

---

## 8. `franka_grid_logger.py` (experiment sweep) — how it works

Runs the indentation sweep + logs `franka_states`, invoked on tactile by `master.py`.
- **Grid:** 3×3 = **9 points**, `V_SPACING=0.018`, `H_SPACING=0.015`, `INDENT=0.002 (2 mm)`, `HOVER=0.005`, `DWELL=2.0 s`, boustrophedon.
- **Motion:** rewritten to REUSE `map_surface.py` (`import map_surface as m`) — `m.preflight_reach`, `m.approach_start` (home→center), `m._gross_move` (settled). Reads **`surface_map.csv`** and indents `INDENT_MM=2` **below each point's mapped `z_touch`** (tilt/sag-corrected), `HOVER_MM=5`, `DWELL_S=2`. Per point: travel→dip→dwell(hold)→retract; phases tagged `home/travel/dip/dwell/retract/park`.
- **Does NOT read the F/T** — the HEX21 is on the WORKSTATION for the experiment (camera+F/T recorded by the GUI). It only servos to mapped heights + logs franka_states.
- **Logging ("record almost everything"):** **93** columns/sample — full `O_T_EE`, `q/q_d/dq`, `tau_J`, `tau_ext_hat_filtered`, both external wrenches, elbow, `m_total`, success-rate, contact/reflex flags, **plus** commanded `target_xyz`, the point's `surface_z`, and `phase/point_index/col/row`. Logger starts BEFORE homing so its timeline matches the camera/F/T window. `postprocess.py` now segments pokes via `surface_z` (tilt-robust).
- Prereq: **reach launch from HOME**, `surface_map.csv` present on tactile. `--max-points N` for a cautious run (also `master.py --max-points N`).

---

## 9. Synchronized run + post-processing pipeline

- **Sync model:** everything UNIX-timestamped. Camera + F/T share the workstation clock (sub-ms, both fire from `begin_recording`). Franka is on the tactile clock; `master.py` measures the offset NTP-style over ssh (~−10.9 ms typical) and records it in `<run>_metadata.json` so the Franka timeline shifts onto the workstation timeline in post.
- **One run (workstation):**
  1. Launch GUI from repo root: `cd ~/E-BTS && ./build/E_BTS_GUI`; open the **Force/Torque** source and **Sequence Recording** pane.
  2. Bring up the servo on tactile; stage the arm.
  3. `python3 master.py <run_name>` → writes `<run>_<stamp>.raw`, `<run>_<stamp>_ft.csv`, `<run>_franka.csv`, `<run>_metadata.json` under `recordings/`.
- **Post:** `python3 postprocess.py` → `output/<run>/` (aligned, baseline-zeroed, per-poke slices) → `python3 visualize.py` (`report.pdf`) → `python3 vector3d_interactive.py` (HTML).

---

## 10. Open / pending tasks

1. **Run the 99-point surface map** (§7) — user-triggered. Review `--max-points 5` output first.
2. **Part 2:** reach-aware auto-homing (HOME → taught bottom-left) for `franka_grid_logger.py` (mirror `map_surface.py`'s `preflight_reach` + up/over/down + reach launch). Optionally consume `surface_map.csv` for per-point depths.
3. **Trial experiment:** the 9-point sweep recording everything, once Part 2 is in.
4. **Camera / event-stream filtering: DONE (2026-08-04).** Biases tuned + auto-applied, OFF polarity killed in hardware (`bias_diff_off=0`, so the software `PolarityFilterAlgorithm` is unnecessary), hardware ROI `0 7 640 450` installed, ROI controls + Load built into `bias_tuner`, per-run `.roi` provenance wired through `postprocess.py`. Defisheye: **dropped** (not needed). Remaining option if event rate becomes a problem: tighten the ROI (currently retains 93.75% of the frame).
5. **Deferred (from earlier):** comment out marker-tracking *calls* (do **not** delete the tracker source files); model silicone surface curvature.
6. **Existing recordings are throwaway test data** — captured at default biases, full frame, before any of the above. Do not treat `recordings/` or `output/sweep/` as dataset.

---

### 4.6 GOTCHA #4 — a loaded control PC causes `communication_constraints_violation`
`libfranka: Move command aborted: motion aborted by reflex! ["communication_constraints_violation"]` with `control_command_success_rate` < 1.0 means the **1 kHz FCI loop dropped packets** — a *timing/network* fault, NOT a motion or geometry problem (where the arm was standing is irrelevant). Verified-good baseline on tactile: PREEMPT_RT kernel, user in `realtime` group, `franka_control_node` at **SCHED_FIFO prio 99**, FCI wired on `eno1` (10.1.196.242 → robot 10.1.196.5), `performance` governor, 0 TX errors.
The cause was **system load**: Chrome **62.5% CPU** (one renderer 47%), VS Code + desktop 15%, **load average 11.5/16 cores**, and `eno1` **RX dropped ≈ 2.9 M** packets. Fix: **quiesce the control PC** — close Chrome/VS Code before a run (`pkill chrome`), want load average < ~4.
Mitigations added: `franka_grid_logger.py` now (a) throttles logging with `--log-hz` (**default 200 Hz**; rospy deserializing FrankaState at the full 1 kHz is a real CPU cost and buys nothing because `franka_control` already filters state at `cutoff_frequency: 100` Hz), (b) sends **error_recovery at startup** to clear a latched reflex (no relaunch needed; `--no-recovery` to skip), (c) prints a loud warning if load average > 4.

### 4.7 GOTCHA #5 — an aborted map run used to DESTROY surface_map.csv
`map_surface.py`'s `finally: save_csv(rows)` ran even when the run aborted before probing anything → `rows == []` → it overwrote a complete 99-point map with a header-only file (and a blank all-NaN heatmap). A `--max-points N` run likewise overwrote the full map with N rows. **This happened once for real** (triggered by the reflex above). Now fixed three ways: (1) `finally` **skips saving entirely if no points were probed**, (2) `--max-points` runs write to `surface_map_partial.{csv,png}` so the real map is never touched, (3) `save_csv` **backs up to `surface_map.prev.csv`** and warns loudly when writing fewer `ok` points than the file it replaces. A durable copy also lives at `franka/surface_maps/surface_map_97ok_backup.csv`.

### 4.8 GOTCHA #6 — the GUI's control/ dir is RELATIVE to its working directory
`SequenceRecordingController` is constructed with relative paths `"control"` / `"recordings"`, so they resolve against the **GUI's cwd**. Launching `./E_BTS_GUI` from `build/` makes it watch `build/control` and write `build/recordings`, while `master.py` wrote `<repo>/control/start.cmd` → **the GUI never saw it, and a full 99-point run recorded NO camera/F/T data** while master.py still printed "Done". (Symptom: stale `start.cmd`/`stop.cmd` left sitting in `<repo>/control/`.)
Fixed in `master.py`: it now (a) **auto-detects the running GUI's dirs** via `/proc/<pid>/cwd` (`find_gui_dirs()`), so either launch dir works, (b) **aborts** if the GUI isn't running or doesn't consume `start.cmd` within 10 s (the Sequence Recording pane runs the watcher — without it nothing records), (c) **collects** the new `.raw`/`_ft.csv`/`.bias` into `<repo>/recordings/`, and (d) prints an OK/MISSING **summary per stream** so a silent miss is impossible. Also `--no-gui` to run the Franka alone.

### 4.10 Per-run folders + per-poke TARE (Wittenstein Fz drift)
**Run folders:** `master.py` writes every run to `recordings/<run>_<YYYYMMDD_HHMMSS>/` with **canonical filenames** inside — `camera.raw`, `ft.csv`, `camera.bias`, `camera.roi`, `franka.csv`, `metadata.json` — mirrored on tactile at `~/E-BTS/recordings/<run>_<stamp>/`. `postprocess.py` accepts a run-folder name or path (and still handles the legacy flat layout). `tidy_recordings.py [--dry-run]` retro-groups loose files into this layout.

**Tare:** the Wittenstein Fz drifts over a long run, so one baseline at the start goes stale (measured: a −0.02 N/s drift turns identical 3 N presses into −3.39 N and −3.99 N). Fix — the sweep now **holds still, out of contact, at hover height for `TARE_S`=1 s immediately before every dip** (logged as `phase="tare"`), and `postprocess.py` zeroes each indentation on **its own** pre-contact window (`TARE_WINDOW_S`=0.6 s ending `TARE_GUARD_S`=0.2 s before contact), falling back to the global baseline if too sparse. Verified: both synthetic presses recover to −3.05 N. It prints `Per-poke tare: N/M pokes zeroed on their own window`. This is a numerical tare of recorded raw data — the raw F/T values are preserved, nothing is destructively re-zeroed at the sensor.

### 4.11 GOTCHA #7 — the servo does NOT give a true depth under contact (impedance sag)
"Force-blind" means the servo never *aborts or stops* on contact — it does **not** mean infinitely stiff. `franka_control` runs `internal_controller: joint_impedance`, a **compliant** controller that holds position via torques proportional to error, so a steady contact force leaves a steady position error. Measured over the 99-point run: achieved depth fell short of the command by **0.39 mm mean (0.07–0.71 mm)**, correlating with contact force at **r = +0.79** → shortfall ≈ **0.10 mm** (free-space tracking) **+ F / 3.8 N/mm** (effective Cartesian Z stiffness). A commanded 2.00 mm gave **1.29–1.93 mm, mean 1.61 mm**.
Fix: `dip_to_depth()` in `franka_grid_logger.py` **closes the loop on the measured `ee_z`** — command nominal depth, measure, push deeper by the shortfall, repeat to `DEPTH_TOL_MM`=0.05 mm (`DEPTH_ITERS`=6, `DEPTH_SETTLE_S`=0.5), with a hard `DEPTH_EXTRA_MAX_MM`=1.5 safety cap. Converges in ~2 iterations (simulated 1.73 → 1.96 mm). The dwell holds the **corrected** command. `--no-depth-correction` restores the old behaviour. The run prints achieved mean/min/max depth at the end.

### 4.12 GOTCHA #8 — segment pokes by the logged `phase`, never by a depth threshold
`postprocess.py` used to infer contact from `ee_z < surface_z - press_margin` (1.5 mm). Because of §4.11 the real depths were 1.29–1.93 mm, so only 71 points cleared the threshold and the `MIN_DWELL_S` contiguity check trimmed it to **69 of 99** — a silent 30% data loss. The sweep **tags every sample** with `phase` and `point_index`, so segmentation now uses one window per `dwell` point: exact and threshold-free. Note `np.genfromtxt(names=True)` turns the text `phase` column into NaN — read it with `load_phase_columns()`. The depth-threshold path remains only as a fallback for logs without `phase`.

### 4.13 The two clocks DRIFT
Measured **−7.7 ms** during a run and **+33 ms** 4.4 h later (15 samples, spread only 5.8 ms → real drift, not noise ≈ 9 ms/h); Tailscale rtt ~50 ms also bounds any single estimate to ±25 ms. `master.py` now measures the offset **before AND after** every run, stores both plus `offset_drift_s`, and warns above 20 ms. Fine for 2 s dwells/segmentation; do **not** trust sub-10 ms event-onset alignment.

### 4.14 Never `read_bytes()` a recording
`master.py`'s collection step did `dst.write_bytes(src.read_bytes())` — on a **2.38 GB** `.raw` that loads the whole file into RAM and killed the collection (run folder left empty, no metadata.json, though all data survived in `build/recordings/`). Now uses `shutil.move` (instant rename on the same filesystem, no duplicate) with a streaming `copy2` fallback.

### 4.9 No more /tmp
The Franka CSV is written straight to **`~/E-BTS/recordings/<run>_franka.csv` on tactile** (`REMOTE_RECORDINGS` in master.py; the logger's default `out_csv` is also under `~/E-BTS/recordings/`), then copied to `<repo>/recordings/` on the workstation. Nothing lands in `/tmp` any more.

## 11. Lessons / gotchas checklist (don't relearn these the hard way)

- [ ] Servo 0.30 m clamp is anchored at the **launch pose** → from home use the **reach launch** (0.55 m); confirm with `rosparam get …/max_translation`.
- [ ] Servo tracks only ~0.5 mm → **pace probes with `send_target`**, never tight `move_to` tol; record the true EE pose at contact.
- [ ] Servo is **slow** (0.03 m/s, `min_duration` 0.5 s) — long moves aren't hangs.
- [ ] **One** controller at a time (servo XOR MoveIt).
- [ ] Move the **HEX21 to tactile** for surface mapping; it's normally on the workstation.
- [ ] Camera is **single-owner**; quit tools via ✕ not Ctrl-C; replug EVK1 if the FX3 sticks.
- [ ] Camera **biases and ROI do not survive a session** — they live in `calibration/camera.bias`/`.roi` and are re-applied by `E_BTS_GUI` at open (§3.4). Check the `[calib]` console lines before trusting a recording.
- [ ] In `bias_tuner`, **dialling a value is not saving it** — hit Save (writes `.bias` + `.roi`), or it dies with the process.
- [ ] The `.bias` format has **no ROI field** and the `.raw` header always says `640x480`; ROI provenance is the separate `.roi` sidecar.
- [ ] Hardware ROI ⇒ events outside are never read out (**free**, reduces load); `.raw` coords stay **absolute**, so still crop in software.
- [ ] `*.bias` is **gitignored** — the active camera calibration is workstation-local, not in the repo.
- [ ] **Never** git commit/push E-BTS without explicit per-time permission — this holds **even when the harness is in an auto/bypass permission mode**.
- [ ] Don't delete marker-tracking source; only comment out its function calls.
- [ ] Wittenstein is ground truth; Franka's external wrench is biased.
- [ ] Tailscale SSH needs periodic re-auth (visit the printed login URL).
- [ ] Keep a hand near the **user-stop** for the first dip of any new motion (servo is force-blind; only the 20 N reflex backs it up).
- [ ] **Close Chrome/VS Code on tactile before any run** — load average must be low or the FCI drops packets (§4.6). Check with `uptime`.
- [ ] `communication_constraints_violation` = load/network, not motion. Don't debug geometry for it.
- [ ] A map/experiment abort must never overwrite `surface_map.csv` — guards are in place (§4.7), and `surface_map.prev.csv` + `franka/surface_maps/` hold backups.
- [ ] **Open BOTH the Force/Torque source AND the Sequence Recording pane** in the GUI before `master.py`, or nothing records (§4.8). master.py now aborts instead of failing silently.
- [ ] Check master.py's end-of-run **OK/MISSING summary** — never assume "Done" means all three streams landed.

---

## 12. Single-point repeated indentation (2026-08-06) — the `mid5` workflow

A second, independent workflow alongside the grid sweep: teach **one** point by hand, indent it **5×**, record all three streams, and post-process for **repeatability at one location**. It needs **no `surface_map.csv`** — the taught (kissed) height *is* the surface reference.

⚠ **`master.py` and `postprocess.py` are NOT touched by any of this.** Every file below is new; the sweep pipeline still works exactly as documented in §8–§9. This was deliberate: a variant gets a **copy**, never a flag bolted onto the validated original.

### 12.1 Files (all new)
| File | Where | Purpose |
|---|---|---|
| `franka/panda_fk.py` | workstation + tactile | Numpy-only Panda FK, `q[7] → O_T_EE`. **Validated to 0.001 mm** against the 5 (joints, xyz) pairs in `corner_joints.json`; recovered flange→EE = identity, i.e. with `load_gripper:=false` and no EE transform, `O_T_EE` *is* the flange pose. `self_check()` re-verifies on live data. |
| `franka/record_one_point.py` | tactile | Teach ONE point: hand-guide, press Enter once → `one_point.json`. Records `joints/O_T_EE/xyz/quat/dq`. |
| `franka/indent_midpoint.py` | tactile | The motion+logging executor: N× (travel→tare→dip→dwell→retract) at the point, 93-col `franka_states` log. Reuses `map_surface._gross_move`/`preflight_reach` and `franka_grid_logger`'s `Ctx`/`FrankaLogger`/`hold`/`clear_reflex`. |
| `master_midpoint.py` | workstation | **A `cp -p` of `master.py`.** Resolves the target itself from `one_point.json`, then orchestrates camera + F/T + the remote indents. |
| `postprocess_indent.py` | workstation | Separate post-processing (see §12.4). |

**Why the target comes from the joints:** the servo controller takes Cartesian targets only, so XY + orientation are computed as `FK(taught joints)` rather than read from the logged `O_T_EE` — the geometry comes from the encoders. The two are cross-checked and printed (agreed to ~1e-7 mm on `mid5`). `master_midpoint.py` resolves the target on the workstation and passes it to tactile via `--x/--y/--surface-z/--quat`; the remote script re-derives it from its own `one_point.json` and **warns if the two disagree by >0.05 mm**, so a stale file on either machine cannot pass unnoticed.

### 12.2 Run procedure
```
# 1. TEACH — franka_control ONLY. The servo controller holds position and fights
#    hand-guiding, so it must NOT be running yet.
ssh tactile
source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
roslaunch franka_control franka_control.launch robot_ip:=10.1.196.5 load_gripper:=false
python3 ~/E-BTS/record_one_point.py      # kiss the spot (no indent), Enter once

# 2. Stop franka_control. Release the user-stop. Bring up the SERVO REACH launch, arm HOME.
roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
    robot_ip:=10.1.196.5 load_gripper:=false

# 3. RECORD — workstation. GUI up with BOTH panes open (§4.8).
cd ~/E-BTS && ./build/E_BTS_GUI          # check the [calib] lines list the tuned biases + ROI
python3 master_midpoint.py mid5 --dry-run   # prints the resolved target, records nothing
python3 master_midpoint.py mid5

# 4. POST
python3 postprocess_indent.py            # defaults to the newest mid*/ run folder
```
The HEX21 stays on the **WORKSTATION** throughout — teaching is hand-guided and needs no force, so unlike surface mapping there is **no replug**.

### 12.3 `mid5_20260805_235328` results (first good run)
Target `[0.65572, -0.00968, 0.26934]`, taught in `robot_mode 2 (MOVE)`, `max|dq| 0.00095 rad/s`, FK vs measured `~1e-7 mm`. Clock offset +25.3 ms before → +30.1 ms after (4.8 ms drift).

| repeat | indent (mm) | peak Fz (N) | mean Fz (N) | tare Fz (N) | events in dwell |
|---|---|---|---|---|---|
| 1 | 1.890 | −1.337 | −1.015 | −0.7215 | 2,852,372 |
| 2 | 1.909 | −1.361 | −1.027 | −0.7298 | 2,864,826 |
| 3 | 1.907 | −1.354 | −1.026 | −0.7398 | 2,839,830 |
| 4 | 1.910 | −1.342 | −1.028 | −0.7427 | 2,823,197 |
| 5 | 1.911 | −1.389 | −1.019 | −0.7443 | 2,813,466 |

**indent 1.905 ± 0.008 mm (spread 0.021 mm); |peak Fz| 1.356 ± 0.018 N (CV 1.3%).** No reflexes, no collisions, `robot_mode 2` throughout, `control_command_success_rate` min 0.910 / mean 0.986.

Two things this shows:
- **A 2.0 mm command lands at ~1.905 mm** and is repeatable to ~0.01 mm. Far tighter than the sweep's 0.18–1.50 mm scatter, because the depth reference is a hand-kissed height rather than the map's `z_touch` (which already sits ~1.2–1.6 mm into the elastomer — see §12.5).
- **The tare drifted −0.7215 → −0.7443 N over 32 s** (~0.7 mN/s). Per-repeat taring absorbs it; a single global baseline would have skewed repeat 5 by ~23 mN.

### 12.4 `postprocess_indent.py` — why it is separate
`postprocess.py` finds pokes by thresholding `ee_z` against `surface_z` with `--press-margin` (default **1.5 mm**). At ~1.905 mm achieved that leaves only 0.4 mm of slack — the same razor-thin margin that silently dropped **28 of 97** pokes in the `run1` sweep.

`postprocess_indent.py` **does not threshold anything.** The logger already tags every sample with `phase` (`home/travel/tare/dip/dwell/retract/park`) and `point_index`, so each indentation is segmented **exactly**:
- contact window = `phase == "dwell"` for that `point_index`
- that repeat's F/T zero = `phase == "tare"` for the same `point_index`

Other differences: **drift-aware clock alignment** (interpolates linearly between the before/after offsets in `metadata.json` instead of using one constant), and it is **non-destructive** — it only *reads* `recordings/<run>/` and *copies* provenance, whereas `postprocess.py` `shutil.move()`s the source files out of `recordings/` (see the ⚠ in §12.5).

Outputs to `output/<run>_indent/`: `repeats_summary.csv`, `repeats/repeatNN_{ft,events,franka}.csv`, `event_rate.csv`, `report_indent.pdf` (whole-run overview → the 5 repeats overlaid on contact onset → one accumulated event image per repeat), and a `metadata.json` carrying what was derived. Flags: `--no-events` (skip the `.raw` pass, fast), `--no-report`.
⚠ The per-repeat event CSVs are ~110 MB each (~560 MB per run) at this event rate. Use `--no-events` when you only need the force/robot numbers.

### 12.5 Gotchas learned here
- ⚠ **`surface_map.csv`'s `z_touch` is NOT the surface** — it sits ~1.2–1.6 mm *below* the taught plane (mapping recorded `contact_fz ≈ −3 N` as "first contact"). So a hand-kissed z and a mapped `z_touch` differ by >1 mm and **must not be mixed** as depth references. This workflow uses the kissed height only.
- ⚠ **The Wittenstein timestamps are bursty**: consecutive samples land ~11–14 µs apart inside a USB packet, then a gap, so `median(diff(t))` is ~1e-5 s even though the true rate is ~1 kHz. Anything sizing a window in *samples* must use the **mean** interval (`span / n`), not the median diff. (This bit a smoothing window here and inflated it to ~2000 samples.)
- **The event rate carries almost no contact signal** — it sits at ~1.4 M events/s whether pressed or hovering, because the illumination is strobed by the Arduino complementary **25 Hz** driver (`arduino/two_phase_driver/`). Measured: the 1 ms event-rate autocorrelation is **r = 0.98 at a 40 ms lag**, with two bursts per cycle. So **40 000 µs accumulation = exactly one illumination cycle**, which makes frames directly comparable. The contact signal is **spatial** (where marker events land), not temporal — use a pre-contact reference frame and difference it.
- **Teach before servo, not after.** Hand-guiding needs `franka_control` alone; the servo controller holds position and fights you. Also **release the user-stop** before step 2 — a point taught in `robot_mode 5 (USER_STOPPED)` is still geometrically valid, but the servo will not move until the stop is released and a reflex is cleared.
- `record_one_point.py` has two guards worth keeping: it **rejects a snapshot while the arm is still drifting** (`max|dq| < 0.004 rad/s`) and **aborts if `fk(joints)` disagrees with the measured `O_T_EE` by >1 mm** — the latter is what would catch a changed tool/EE transform before it silently biases every target.

---

## 13. Marker displacement vs force — the event-video → optical-flow pipeline (IN PROGRESS, 2026-08-06)

**Goal:** graph **inter-marker displacement against measured force**, first for one indentation, then as a mean over the 5 repeats with a lighter ±STD band on the same axes. Working from `mid5_20260805_235328`, repeat 1.

### 13.1 Step 1 — event video (DONE): `event_video.py`
Renders one indentation's events to video. Run:
```
python3 event_video.py --repeat 1 --fps 60 --accum-us 40000            # full frame
python3 event_video.py --repeat 1 --fps 60 --accum-us 40000 --roi 306 91 71 37
```
Outputs to `output/<run>_indent/video/`: `<name>.mkv` (FFV1 gray8, **LOSSLESS, gray value == event count**), `<name>_preview.mp4`, `<name>_annotated.mp4` (time/phase/Fz burned in), `<name>.json` (per-frame window centre, phase, tared Fz — the join key to force).

**FPS vs accumulation — these are independent.** Non-overlapping 40 ms frames give only **25 fps**. `--fps 60` slides the 40 ms window at a 16.67 ms stride, so frames overlap 58.3%. This is safe because the illumination is strobed at 25 Hz (§12.5), and a 40 ms window spans exactly one cycle *whatever its phase* — so brightness content is identical frame to frame. ⚠ But the **effective temporal resolution stays 40 ms**; consecutive 60 fps frames are not independent samples. `--fps 25` gives non-overlapping frames. *(The user dislikes the striding — prefer offering 25 fps non-overlapping unless high fps is needed for frame-to-frame marker correspondence, which is exactly why it is used here.)*

⚠ **Gray value IS the event count — never rescale the analysis video.** Counts per 40 ms frame are tiny (median 1, global max 25 full-frame / 10 in the ROI), so they fit in 8 bits directly and the mapping is exactly invertible. Scaling to a 99.5th-percentile `vmax` clipped 0.8% of nonzero pixels — **precisely the marker cores, which is where sub-pixel centroid precision lives**. The `_preview`/`_annotated` mp4s ARE percentile-stretched and 8× nearest-neighbour upscaled, for viewing only — never measure from those. Losslessness was *verified*, not assumed: ffmpeg-decoded frames are byte-identical to the source counts.
⚠ **Odd ROI dimensions break the mp4s.** x264 `yuv420p` needs even width/height; `71x37` is odd on both. The `.mkv` keeps native odd geometry (FFV1 gray has no chroma subsampling); the viewing copies are integer-upscaled, which fixes parity as a side effect.

### 13.2 Geometry facts measured so far
- Marker grid: roughly **15 columns × 11 rows**, in two bands with an empty horizontal gap around **y ≈ 180–240** (sensor coords).
- **Column pitch ≈ 37–38 px, row pitch ≈ 41 px, marker diameter ≈ 20–25 px** (radius ~10–12 px).
- ROI→sensor mapping for a crop: `x_sensor = x_roi + ROI_X`, `y_sensor = y_roi + ROI_Y`.
- ⚠ **ROI `306 91 71 37` isolates exactly two markers** (blobs at ROI x≈19 and x≈56, hard zero gap at x≈31–43) — verified — **but they are NOT the markers of interest.** The right pair still has to be found; that is what the overlay video in §13.3 is for.
- Peak event density in that ROI was 10 events/px vs 25 full-frame, i.e. those two markers are not the brightest. Centroid noise scales with count, so prefer brighter markers if the choice is free.

### 13.3 Step 2 — marker detection + blue-circle overlay (DONE): `marker_overlay.py`
```
python3 marker_overlay.py --probe                                  # radius + detection stats
python3 marker_overlay.py --mkv <...>_60fps.mkv --view-scale 2     # the circles video
```
Outputs `<stem>_circles.mp4` (blue circles on black, **no event pixels**) and
`<stem>_detections.json`.

**FIXED circle size — the markers are physically 0.75 mm in RADIUS, so image size cannot vary.** Letting the radius follow each blob's thresholded area made identical markers appear to jiggle in size; that was threshold noise. One radius is used for all. It is *measured*, not guessed: on the mean of the 60 `tare` frames blob geometry is threshold-insensitive (170 blobs, median radius 9.6–9.8 px, bbox 19×19 px, across `thresh_rel` 1.25→1.50). **Measured R = 9.52 px ⇒ 12.70 px/mm**, making marker diameter 1.5 mm and the measured 31 px column pitch ≈ 2.44 mm. `--radius-px` overrides.

Three things that had to be right:
1. ⚠ **LOCAL-CONTRAST NORMALISATION is mandatory** (`blur(3) / blur(bg_sigma=40)`). Marker brightness varies ~9× across the frame (row-band energy 14051 → 1542), so a single global threshold found only **84–89 of ~165** markers and missed y 80–160 and 240–280 *entirely*. `bg_sigma` must exceed the ~31 px pitch or it normalises the markers away.
2. **Size gate**, as fractions of the expected area π·R² = 285 px²: keep 100–627 px² (0.35×–2.20×), bbox aspect 0.59–1.70. Blobs too small to *be* a marker are dropped, so markers that are deliberately not visible stay uncircled — intended. The aspect gate also killed the fast-motion artefact where a smeared marker fragments (`dip`/`retract` frames had spiked 161 → 191 detections; max is now 170).
3. ⚠ **PERSISTENT ROSTER — this is the important one.** Per-frame detection alone let markers flicker in and out, which breaks the arrow stage: nearest-blue-circle matching can bind two arrows to the *same* circle. Fix: seed a roster from the baseline mean positions, propagate it **sequentially** (match each marker to the nearest detection to *where it was in the previous frame*, not to its baseline — anchoring on baseline loses markers once displacement accumulates), and **delete any marker missing even once**. Result: **152 seeded → 141 kept, present in all 313 frames**, `tracks` shape `(313, 141, 2)`, **no NaNs, constant cardinality**. Match radius 12 px, deliberately under half the 31 px pitch so a neighbour mismatch is geometrically impossible (measured max frame-to-frame step: 3.96 px).

Centres are **intensity-weighted centroids** (weights = event counts) → sub-pixel. Baseline per-frame centroid jitter: **sx 0.26 px, sy 0.27 px** — that is the displacement noise floor.

Already measured from the tracks: **max displacement 8.64 px = 0.680 mm**, mean across markers at peak **3.08 px = 0.242 mm** ⇒ SNR ≈ 12:1 at peak.

`<stem>_detections.json` keys: `tracks[frame][marker][x,y]` (gapless, stable indices — **use the index, never a nearest-circle search**), `raw_frames` (pre-roster detections), `roster{n_seeded,n_kept,kept_indices,match_r_px,max_missing,missing_per_seeded}`, `marker_radius_px`, `scale_px_per_mm`, `baseline_tare`, `peak_dwell`, `tare_frames`, `peak_frames`, `frame_phase`, `frame_Fz_N_tared`.

### 13.4 Step 3 — three-frame arrow scheme (DONE): `marker_arrows.py`
The goal is the **net displacement end result**, not the full trajectory — but the intermediate frames are still needed to keep marker *identity* stable, which is why 60 fps was used. Process **three frames at a time**: `baseline`, `current`, `next`.
- arrow from the blue circle in **baseline** → the blue circle in **current** (net displacement so far),
- a *separate* arrow from the blue circle in **current** → the blue circle in **next** (the incremental step).

Then iterate: `current` becomes `next`, `next` advances one frame.

**Correspondence is already solved and must NOT be re-solved by proximity.** Because §13.3 produced a gapless roster with stable indices, arrow *m* at every frame is just `tracks[0][m] → tracks[k][m] → tracks[k+1][m]`. Index the roster; never search for the nearest circle. This is what makes two arrows unable to collide on one circle.

Deliverables for this step:
1. a **video with labelled vector arrows**, and
2. a **static figure showing the first baseline frame and the FINAL displacement location of each marker** (baseline circle → final circle, with the connecting arrow, for all markers).

⚠ **Arrows need magnification to be visible and it MUST be stated on the figure.** Net displacement peaks at 8.6 px and the frame-to-frame step is often sub-pixel, so at 1–2× view scale an unmagnified step arrow is invisible. Use separate gains for the two arrow types, print both gains in the HUD/legend, and never let a reader infer length directly.
⚠ Labelling all 141 arrows is unreadable — label the largest-magnitude ones and give a scale reference instead.

**Measured on `mid5` repeat 1** (141 markers, 313 frames):

| pose | mean | median | max |
|---|---|---|---|
| **peak dwell** (frame 137, Fz = −0.918 N) | 0.246 mm | 0.219 mm | **0.586 mm** |
| final (post-retract) | 0.031 mm | 0.026 mm | 0.122 mm |

The field is physically coherent — arrows **diverge from a contact point near x ≈ 330, y ≈ 180 px**, the upper band pushing up-and-right at 0.4–0.6 mm while the far lower band barely moves. Recovery is ~87% (0.031 mm residual vs 0.246 mm peak), i.e. mostly elastic, so the elastomer is not creeping between repeats.

Outputs: `<stem>_arrows.mp4`, `<stem>_peak_displacement.png/.pdf`, `<stem>_final_displacement.png/.pdf`, `<stem>_displacement.csv` (per-frame per-marker dx/dy/net in px **and** mm).

⚠ Because the 60 fps frames overlap 58%, the STEP arrows are **not independent** frame to frame — they are a smoothed view of motion, not true 16.7 ms increments. They do their job (anchoring identity); do not quote step magnitudes as instantaneous velocity. Net arrows are unaffected.

### 13.5 Step 4 — force vs displacement: DEFERRED (decided 2026-08-06)

**Not being done on the `mid5` dataset, on purpose.** The arrow field revealed that the actual contact location could not be identified after the fact, and the silicone surface is not uniform — so a displacement-vs-force curve from this run would not be interpretable, and any marker pair chosen retrospectively would be arbitrary.

**The plan instead:** re-record a fresh dataset where the indentation **xy on the silicone is known up front from CAD**, so the specific markers involved are known before any processing. Only then do:
- displacement between the two chosen markers vs force for one indentation, then
- the **mean over the 5 repeats with a lighter-coloured ±STD band on the same axes**.

Everything needed for that join already exists and does not need rebuilding: `<stem>_displacement.csv` has per-marker net displacement per frame, and the video sidecar `<name>.json` has `frame_Fz_N_tared` at each frame's window centre (tared on that repeat's own `tare` phase). Convert px → mm with `scale_px_per_mm` (12.70).

⚠ Do **not** resurrect this step against `mid5` — the blocker is the unknown contact location, not the tooling.

### 13.6 Available tooling on the workstation
`cv2 4.13.0`, `scipy 1.10.1`, `matplotlib 3.7.5`, `PIL 7.0.0`, `ffmpeg 4.2.7` (FFV1 + libx264 lossless), `metavision_core` (SDK 3.1.2). **`skimage` is NOT installed.**

### 13.7 The pipeline runner: `run_marker_pipeline.py`

One entry point that **launches the existing scripts in sequence** — it is an orchestrator, not a rewrite; each stage stays a standalone file that can still be run and debugged on its own.

```
python3 run_marker_pipeline.py                      # newest mid*/ run, repeat 1
python3 run_marker_pipeline.py mid5_20260805_235328 --repeat 1
python3 run_marker_pipeline.py --all                # every repeat in the run
python3 run_marker_pipeline.py --fps 25             # non-overlapping frames
python3 run_marker_pipeline.py --skip postprocess   # skip a stage by name
python3 run_marker_pipeline.py --dry-run            # print the commands only
```

Stages, in order:
| # | script | produces |
|---|---|---|
| 1 | `postprocess_indent.py` | `output/<run>_indent/` — per-repeat F/T + robot slices, `repeats_summary.csv`, `report_indent.pdf` |
| 2 | `event_video.py` | `…/video/<stem>.mkv` (lossless, gray == event count) + `<stem>.json` sidecar |
| 3 | `marker_overlay.py` | `<stem>_circles.mp4` + `<stem>_detections.json` (gapless roster `tracks`) |
| 4 | `marker_arrows.py` | `<stem>_arrows.mp4`, peak/final displacement figures, `<stem>_displacement.csv` |

It **verifies the expected artifact exists after each stage** and stops on the first failure rather than cascading, then prints a manifest of everything produced. Stage 1 is per-run (not per-repeat), so with `--all` it runs once and stages 2–4 loop over repeats.

Step 4 (force vs displacement) is deliberately **not** a stage — see §13.5.

---

## 14. Supervised ML — feasibility analysis of `run1` and the data-collection design that follows (2026-08-06)

**Goal:** frames accumulated at 40 ms from the event `.raw` → **force** (supervised regression).

**Verdict: `run1_20260805_180345` is NOT a trainable dataset.** It is a valid *pipeline* test — all 99 pokes present, recorded with the tuned biases + ROI, per-poke tare present — but it cannot support a force model. The reasons are quantitative and are recorded below so they are not re-derived. Every number in §14.1 was measured from `output/run1_20260805_180345/pokes/pokes_summary.csv` and `franka.csv`.

### 14.1 Why `run1` cannot train a force model

**(a) Effective N is 99, not 4,949.** 40 ms frames over a 2 s dwell give 50 frames/poke → 4,949 total. But the force is *flat* during a dwell:

| quantity | value |
|---|---|
| within-dwell Fz sd (pokes 01/25/50/75/99) | **0.1168 N** |
| Wittenstein noise floor (`mid5_center` tare sigma) | **0.116 N** |

Identical. All within-dwell variation is sensor noise, so the 50 frames are 50 replicates of one (image, force) pair. Replicates average label noise; they do **not** add degrees of freedom. **Effective N = 99.**

**(b) Force was never a controlled variable.** All 99 pokes commanded the same 2.0 mm. Force varied only through uncontrolled local stiffness, surface-map error and impedance sag:

| | |
|---|---|
| \|peak Fz\| range | 0.452 – 1.463 N |
| \|peak Fz\| sd | 0.173 N |
| reproducibility at a fixed point (`mid5_center`, 5 repeats) | **0.017 N** |

Note the reproducibility is 10× smaller than the spread, so the spread is **real signal, not noise** — but it is signal about *the elastomer*, not about applied force. A model fit here predicts a nuisance variable over a ~1 N window and would not extrapolate.

**(c) `indent_mm` is not an independent input — it is partly an output.** `corr(indent_mm, |peak_Fz|) = −0.67`, **negative**, i.e. physically backwards. This is §4.11's impedance sag: stiffer spot → more force → more sag → *less* achieved depth. (`run1` predates `dip_to_depth()`; achieved depth 1.29–1.93 mm for a 2.00 mm command — these are exactly the §4.11 numbers.)

**(d) A robot-only model already explains 43% of the force.** Leave-one-out, **zero camera input**:

| predictor | LOO R² | RMSE |
|---|---|---|
| predict the mean (baseline) | — | 0.173 N |
| **depth alone** | **0.426** | **0.131 N** |
| x, y alone (2-NN) | 0.403 | 0.134 N |
| x, y, depth | 0.402 | 0.134 N |

⚠ **0.131 N RMSE is the number any camera-based model must beat.** And ⚠ **do NOT feed `indent_mm` to the model as an input feature** — it hands over 43% of the answer through a channel unrelated to the sensor, and destroys the ability to attribute performance to the camera. Depth is a *validation* variable, not a feature.

**(e) Location is a strong confound.** The force map is smoothly structured — stiff at the clamped edges (col 0 ≈ 1.05–1.28 N, col 7 ≈ 0.91–1.46 N), soft in the middle (≈ 0.60–0.80 N). All 99 pokes sit at 99 *different* locations, so a model that merely localises contact regresses force at R² ≈ 0.40. ⚠ **A random train/test split leaks** — neighbours 4 mm apart are near-duplicates. **Always use grouped CV by location** (leave-one-column-out / leave-one-region-out).

**(f) Event *rate* carries no force information.** `corr(n_events, |peak_Fz|) = −0.184`. Confirms §12.5: the 25 Hz strobe dominates the temporal signal. **The signal is purely spatial.** 40 ms accumulation is nevertheless the right choice — it is exactly one illumination cycle, so frames are comparable regardless of phase.

**(g) Raw pixels + linear regression is not viable.** 640×450 = 288,000 features vs 99 effective samples (p/n ≈ 2900). OLS is rank-deficient and fits the training set to *exactly zero residual* by memorisation. The fix is **features, not a different estimator**.

**(h) A CNN is out on `run1`.** ~10⁴–10⁵ parameters vs 99 independent samples. Augmentation manufactures images, not force labels.

### 14.2 The one thing in `run1` worth mining: the dip ramp

Phase durations, verified from `franka.csv` (99 pokes each):

| phase | duration | 40 ms frames | what it gives |
|---|---|---|---|
| `tare` | 1.00 s | ~25 | out-of-contact **reference frame** for differencing — essential, and already present |
| `dip` | 1.00 s | ~25 | **force ramps 0 → peak at a fixed location** |
| `dwell` | 2.00 s | 50 | constant force (50 redundant frames) |

The dip is the only place in the dataset where **force varies while position is held constant** — ~2,475 frames that break the location/force confound. Caveats: the silicone is viscoelastic so ramp force at a given depth ≠ quasi-static value; and at ~8 mm/s tool speed there is motion blur, though per-frame marker motion is ~1.3 px, comparable to the 3.96 px max step the tracker already handles (§13.3).

### 14.3 Sensor spatial resolution — the number that sizes the whole campaign

Measured from `mid5_center_20260806_191916` `…_detections.json` (**204 markers**, better than the 141 on the older `mid5`):

| quantity | value |
|---|---|
| marker nearest-neighbour pitch | **2.317 mm** (median; 27.7–32.8 px, 5–95%) |
| scale | **12.61 px/mm** |
| marker radius | 9.46 px = 0.75 mm |
| marker field extent | 40.8 × 30.2 mm |
| displacement noise floor | 0.26 px = **0.020 mm** |

⚠ **The marker pitch is the sensor's spatial sampling limit.** Indentable area (taught corners, §5) = 36 × 30 mm = 1080 mm². Independent contact locations ≈ 1080 / 2.317² ≈ **201** — which matches the 204 markers, because it is the same sampling argument.

⇒ **Indenting on a 1 mm grid (1,147 points) oversamples the sensor ~5.7×.** The extra points are strongly correlated, not new information. Grid pitch should be **≈ the marker pitch (2 mm)**; spend the remaining budget on **depth levels and repeats**, which buy genuinely new information.

### 14.4 Campaign sizing (measured rates)

Event data rate **3.97 MB/s** of `.raw` (2.38 GB / 600 s, `run1`). Cycle time **~6.06 s/indent** measured, **~8 s** with `dip_to_depth()` active. Disk free: **747 GB**.

| design | locations | × depths | indents | robot time | `.raw` |
|---|---|---|---|---|---|
| **pilot — 3 mm pitch** | 13 × 11 = 143 | 6 | **858** | **1.9 h** | **27 GB** |
| **main — 2 mm pitch** | 19 × 16 = 304 | 6 | **1,824** | **4.1 h** | **58 GB** |
| 1 mm pitch (rejected) | 37 × 31 = 1,147 | 6 | 6,882 | 15.3 h | 219 GB |

The 1 mm design costs 8× the time and disk of the pilot for ~5.7× redundant locations, and 15 h of continuous operation across which the elastomer creeps and the clocks drift (§4.13). **Rejected.**

⚠ **Do NOT write per-poke event CSVs at this scale.** They are 32 MB each (`run1`: 3.4 GB for 99 pokes) → 58 GB at 1,824 pokes, *on top of* the `.raw`. Store the accumulated 40 ms frames instead: 640×450 uint8 × 50 = 14.4 MB/poke uncompressed, and the frames are extremely sparse (median count 1) so they compress ~10–20×. Use HDF5 (`h5py` is installed) or the existing lossless FFV1 `.mkv` path.

### 14.5 Depth levels

Contact is **nonlinear** (Hertzian, F ∝ d^1.5). Calibrating F = k·d^1.5 on `mid5_center` (1.98 mm → 1.08 N) gives k ≈ 0.388:

| commanded depth | predicted \|Fz\| | step from previous |
|---|---|---|
| 0.5 mm | ~0.14 N | — |
| 1.0 mm | ~0.39 N | 0.25 N |
| 1.5 mm | ~0.71 N | 0.32 N |
| 2.0 mm | ~1.10 N | 0.39 N |
| 2.5 mm | ~1.53 N | 0.43 N |
| 3.0 mm | ~2.02 N | 0.48 N |

**The levels are NOT too close.** The smallest step (0.25 N) is ~15× the 0.017 N reproducibility and ~10× the depth-control tolerance expressed in force. They are cleanly separable.

⚠ **Provenance of that fit, since it gets quoted a lot:** it is **five repeats at ONE location (the centre) over 0.3–2.0 mm only**, from `mid5_center_20260806_191916`'s dip ramps, tared per repeat on its own `tare` phase. Per-repeat (n, k): (1.77, 0.227), (1.77, 0.229), (1.48, 0.275), (1.63, 0.249), (1.42, 0.286) → median n = 1.63, k = 0.249. **Anything quoted past 2 mm is extrapolation**, and k varies with location (`run1` saw 0.45–1.46 N at nominally the same commanded depth).

Design notes:
- **Include 2.0 mm** — it was missing from the first proposal (0.5/1/1.5/2.5/3), and it is the level with the most existing comparison data.
- **Add a 0 mm / no-contact level** — free (the `tare` phase already provides it) and it anchors the zero.
- ⚠ **0.5 mm is at the detection limit.** ~0.14 N is only 1.2× the *per-sample* F/T noise (recoverable by averaging over the dwell), but marker displacement scales to ~0.037 mm = **0.47 px against a 0.26 px noise floor → SNR ≈ 1.8**. Keep it (it defines the floor and the low-force regime matters most) but expect it to be the noisiest class; consider extra repeats there.
- ⚠ **Randomise / interleave the depth order.** Sweeping depth monotonically confounds depth with viscoelastic creep and F/T drift — which over a multi-hour run is exactly the §4.10 drift problem at campaign scale.

### 14.5b HOW THE ELASTOMER IS MOUNTED — corrected 2026-08-06

⚠ **The silicone is ~5 mm thick and STRAPPED AT THE SIDES, with NOTHING UNDERNEATH.** It is an edge-supported sheet, *not* a layer bonded to a rigid backing. An earlier draft of §14/§15 assumed a bonded backing and reasoned about confinement, substrate stiffening and "100% strain" — **all of that was wrong and has been removed.** Consequences:

- **There is no surface to bottom out on.** Indentation depth is not limited by the 5 mm thickness.
- **The practical limit is the STRAPS**, estimated to give way near **~10 mm**. `depth_limit_probe.py` refuses past `STRAP_LIMIT_MM = 8.0`.
- The sheet still stiffens at large deflection, but by **membrane tension** (the sheet stretches) rather than confinement — a different mechanism with a different exponent.
- **Corroboration:** `run1`'s force map is stiff near the edges (col 0 ≈ 1.05–1.28 N) and soft in the middle (≈ 0.60–0.80 N) — the signature of edge support carrying load directly near the straps. The mounting description and the measured map agree.

So 4–5 mm indents are ordinary, with margin. The probe is still worth ten minutes before committing a long campaign to deep indents — not for safety from a backing, but because the ladder would otherwise rest on a 5-point fit extrapolated 2.5× past its range into a regime (membrane stretching) it never sampled.

### 14.6 Marker tracker vs end-to-end — measured, not assumed

The tracker's real-time cost was **measured** on a realistic sparse frame (37.2k events per 40 ms frame, = `run1`'s rate):

| implementation | per frame | % of the 40 ms budget |
|---|---|---|
| current (`bg_sigma=40` blur at full res) | **22.8 ms** | **57%** |
| background estimated at 1/8 scale | **5.8 ms** | **14%** |

So: the *current* implementation genuinely is a real-time liability at 57% of budget — that concern was correct. But it is an **implementation** problem, not an architectural one: the background field is smooth by construction, so estimating it on a 1/8-scale image is mathematically near-identical and gives a 4× speedup to a comfortable 14%.

⚠ **The tracker's real deployment liability is not speed — it is that it is stateful and brittle.** It needs a baseline frame plus sequential propagation, and it *deletes any marker missing even once* (§13.3: 152 seeded → 141 kept). A dropped marker or a fast displacement breaks the roster mid-stream.

**Resolution: this is not a choice.** Both consume the same 40 ms frames.
- **Marker-feature model = the baseline and the oracle.** It says whether the physics is recoverable at all, and sets the ceiling.
- **CNN on frames = the deployment target.** If it matches the marker model, ship it and drop the tracker. If it falls far short, you are data-starved and you know precisely why.

⚠ **Either way, plan for a tare/reference frame at inference.** A single 40 ms frame with no baseline forces the model to infer the undeformed marker positions from the image itself. Differencing against a tare frame is far easier and is standard practice for tactile sensors — the rig already produces one per poke.

### 14.7 Software stack (workstation, verified 2026-08-06)

| | |
|---|---|
| GPU | **NVIDIA RTX 2080 Ti, 11 GB**, driver 570.133.07 |
| CPU / RAM | 20 cores / 62 GB |
| disk free | 747 GB |
| Python | **3.8.10** (EOL Oct 2024) |
| installed | `torch 2.4.1+cu121` (**CUDA verified working**), `numpy 1.24.4`, `scipy 1.10.1`, `pandas 2.0.3`, `cv2 4.13.0`, `matplotlib 3.7.5`, `h5py 3.11.0`, `metavision_core` (SDK 3.1.2) |
| **missing** | **`scikit-learn`**, `xgboost`, `tables`; `skimage` also absent (§13.6) |

- **For ridge / SVR / trees: `scikit-learn` on CPU. No CUDA, no PyTorch.** A few thousand samples × tens of features fits in milliseconds; a GPU is pure overhead. Install `scikit-learn==1.3.2` (last release supporting Python 3.8).
- **For a CNN later: the existing `torch 2.4.1+cu121` on the 2080 Ti.** 11 GB is ample for single-channel 640×450 frames.
- Python 3.8 is EOL — fine for this work, but pin versions, and consider a venv on a newer Python before any long-lived training code.

### 14.8 Model progression (do them in this order, grouped CV throughout)

1. **Ridge on marker-displacement features** — the baseline. Must beat **0.131 N RMSE** (§14.1d) or the camera has demonstrated nothing.
2. **Add physics terms** (d^1.5, ‖displacement‖^1.5) — contact mechanics is mildly nonlinear, and a linear model on the right features usually captures it.
3. **Kernel ridge (RBF) or SVR** — compare. SVR is not obviously better than kernel ridge here and is fussier to tune; do not start here.
4. **Gradient boosting** — good in the few-hundred-to-few-thousand-sample regime, and gives feature importances.
5. **CNN on frames** — only once the sample count justifies it.

⚠ **The honest way to answer "is N enough" is a learning curve**, not a rule of thumb: plot CV R² against training-set size from the pilot and see whether it is still climbing. That decides whether to densify from 3 mm to 2 mm pitch, add repeats, or stop.

### 14.9 Open decisions before the campaign runs

- ~~Elastomer thickness~~ — **5 mm** (2026-08-06). Ladder capped at 3.0 mm (60% strain) until `depth_limit_probe.py` has been run; see §15.2.
- Whether `dip_to_depth()` (§4.11) is enabled — it should be, so commanded depth is the depth achieved; it adds ~2 s/indent. **On by default** in `sweep_campaign.py`.
- Frame storage format (HDF5 vs FFV1 `.mkv`) and where the campaign's ~20 GB lives.
- Session splitting if the ladder or grid grows (re-tare, quiesce the control PC per §4.6).

---

## 15. The campaign tooling (2026-08-06) — `sweep_campaign` / `master_campaign` / `depth_limit_probe`

Three new files implementing §14's design. `master.py`, `postprocess.py`, `master_midpoint.py` and `indent_midpoint.py` are **untouched** — per §12, a variant gets a copy.

| File | Where | Purpose |
|---|---|---|
| `franka/sweep_campaign.py` | → tactile | Grid × depth-ladder executor. 93-col log (schema unchanged, so `postprocess_indent.py` works as-is) + `campaign_plan.csv` sidecar. |
| `master_campaign.py` | workstation | Orchestrator. Copy of `master_midpoint.py`; forwards every campaign knob verbatim to the remote planner. |
| `franka/depth_limit_probe.py` | → tactile | **Force-limited** descent to find where F(depth) blows up. Gate for any depth > 3 mm. |

**Deployed to tactile 2026-08-07**, md5-verified against the workstation copies. Re-deploy after any edit: `scp franka/sweep_campaign.py franka/depth_limit_probe.py tactile@100.93.60.35:~/E-BTS/`

Verified **on tactile** (real ROS imports, real `corner_joints.json`, not stubs): both `--dry-run` paths execute; the campaign planner produces 648 indents over 108 locations with all six depths exactly 108× each; `matplotlib 3.1.2` is present so the map heatmap will render; and the interactive tty-fallback correctly engages over non-interactive ssh. ⚠ **The motion paths have still never run on hardware** — dry-runs exercise planning and I/O only.

### 15.1 `sweep_campaign.py` — three decisions baked in

1. **Surface datum = the bilinear plane over the four HAND-KISSED corners**, not `surface_map.csv`'s `z_touch`. Evidence: fitting `F = k·d^n` to the dip ramps gives **n = 0.72** against `run1`'s `z_touch` reference but **n = 1.63** against `mid5_center`'s kissed reference. A sub-linear exponent is impossible for a cone indenter (Sneddon 2.0, Hertz sphere 1.5), so 0.72 is the signature of a shifted depth origin — confirming §12.5 quantitatively. `--surface-map` exists as an escape hatch and is deliberately **not** the default.
2. **Depth order is shuffled within each location** (seeded, reproducible). Verified on the real taught corners: `corr(plan_index, depth) = +0.0002`, each depth appearing exactly 108×, while travel stays at 0.50 mm mean / 3.01 mm max because locations remain serpentine. `--order random` fully shuffles if you'd rather pay the travel.
3. **The 93-column log schema is unchanged.** `point_index` is a globally unique indent counter — what `postprocess_indent.py` segments on. The commanded depth lives in `campaign_plan.csv`, keyed by `point_index`. ⚠ **Without that sidecar the run is unlabelled**; `master_campaign.py` pulls it back and warns loudly if it doesn't arrive.

**Planned pilot (verified against the real `corner_joints.json` geometry):** taught quad measures **29.55 × 36.06 mm**, so the requested 24 × 32 mm span leaves margins of **2.78 / 2.03 mm** — at least 2 mm off every edge, as intended. Grid **9 × 12 = 108 locations** at 3.00 / 2.91 mm achieved pitch, × 6 depths = **648 indents ≈ 1.4 h ≈ 21 GB**. Surface tilt across the explored area: 0.78 mm.

```
python3 master_campaign.py --dry-run            # forwards to the remote planner
python3 master_campaign.py pilot --max-points 6 # cautious
python3 master_campaign.py pilot                # the full campaign
```

### 15.2 `depth_limit_probe.py` — why 4 and 5 mm need measuring, not extrapolating

Not for safety from a backing — **there isn't one** (§14.5b). Because of extrapolation: `F = 0.25·d^1.63` predicts 2.5 N at 4 mm and 3.6 N at 5 mm from a **5-repeat, one-location, 0.3–2.0 mm** fit, i.e. 2.5× beyond its data and into the membrane-stretching regime it never sampled. We could **not** pin the transition from existing data either: the dip ramp reaches full depth in ~1 s, leaving too few samples in the shallow bands to fit a local exponent (only 1.5–2.0 mm had enough, n = 1.58).

The servo is force-blind and the only backstop is the ~20 N reflex, so the probe descends in **0.25 mm steps, reading force between every step**, and aborts on the first of: force ceiling (8 N), stiffness blow-up (dF/dz > 15 N/mm — membrane stiffening or a strap loading up), or the depth ceiling. It retracts in a `finally`, before any file I/O. Hard refusal past `STRAP_LIMIT_MM = 8.0`.

#### `--map`: the 9-point depth-limit map (2026-08-07)

Because normal force is **not uniform** across the elastomer (edges stiffer than middle — §14.1e), a single-point limit does not generalise. `--map` probes **centre + 4 corners + 4 edge midpoints** and interpolates.

⚠ **Those 9 points are exactly a 3×3 lattice**, so the probe calls `sweep_campaign.build_grid(..., n_u=3, n_v=3)` rather than re-deriving the geometry. Verified: probe x-range `0.63947..0.67141` and y-range `-0.01448..0.01010` match the 108-point campaign grid to 1e-9. Node spacing 12.0 × 16.0 mm, margins 2.77 / 2.03 mm. If the two ever disagreed about the inset rectangle, the map would describe different silicone than the campaign indents and nothing would warn you — hence one implementation, one rectangle.

⚠ **The descent lives in exactly one function** (`probe_point`). `--map` loops it; it does not reimplement it. Duplicating force-limited descent logic is the one refactor that could silently lose an abort path.

**Interpolation is piecewise-bilinear, not biquadratic.** A quadratic through 3 nodes overshoots between them, and overshooting a *depth limit* upward is the error mode that costs hardware. Verified exact at all 9 nodes and no overshoot anywhere on a 41×41 sample.

**The recommended ceiling is the SHALLOWEST node × `--safety-factor` (0.8), not the per-location interpolant.** Between nodes the true limit can dip below the interpolated surface. The per-location map exists for force-targeting (below), not for pushing individual points deeper.

Outputs: `depth_limit_map.csv` (+ `.json` sidecar with the interpolant), `depth_limit_map_steps.csv` (every step at every node), `depth_limit_map.png` (max-depth and stiffness heatmaps). ~3–6 min for all 9 nodes.

```
python3 depth_limit_probe.py --map --dry-run
python3 depth_limit_probe.py --map                 # HEX21 must be on TACTILE
```

#### First real probe run — the measured force law (2026-08-07, centre, 20 steps)

The centre node was probed to the depth ceiling. **It stopped because `--max-depth-mm` was still 5.0** — a leftover from the dead 5 mm-thickness premise — **not because of anything physical**: force there was only 2.24 N, 28% of the 8 N ceiling. Default raised to **7.0 mm commanded** (`STRAP_LIMIT_MM = 8.0` unchanged).

| | |
|---|---|
| achieved at 5.00 mm commanded | **4.548 mm** (sag 0.45 mm at ~2 N) |
| force there | **2.244 N** |
| best fit, 0.4–4.5 mm | **F = 0.496·d + 0.090, R² = 0.994** — essentially LINEAR |
| stiffness trend | **0.632 → 0.368 N/mm, r = −0.826 — SOFTENING** |
| extrapolated at 8 mm | ~3.7 N = **46% of the 8 N ceiling** |

⚠ **This overturned the membrane-stiffening prediction in the earlier draft.** The sheet gets *softer* with depth, not stiffer — an unbacked edge-strapped sheet deflects bodily, so incremental resistance drops rather than climbing. That is now the second boundary-condition guess this data has corrected (after the rigid backing, §14.5b). **Force is not the binding constraint anywhere in the reachable range; the optical limit is.**

Two consequences for the campaign:
- **Because F is linear in d, an evenly spaced DEPTH ladder produces an evenly spaced FORCE ladder** — exactly what a training set wants, and it means no depth-spacing compensation is needed.
- **The ladder can go much deeper than 3 mm.** At ~6 mm achieved the range becomes ~0.34–3.1 N instead of 0.34–1.58 N — roughly double the force span, for free.

⚠ Also revises §14.5: the measured force at 0.5 mm is **0.31 N**, not the 0.14 N predicted by the old Hertzian guess. The low end is ~2× more measurable than feared, so the 0.5 mm level's marker-displacement SNR is nearer 4 than 1.8.

#### Ceiling raised to 15 mm (2026-08-07, operator's call)

The 5.0 mm ceiling stopped the first probe at 2.24 N; raising it to 7.0 was still too shallow. **Default `--max-depth-mm` is now 15.0**, hard cap `--strap-limit-mm 20.0` (a typo guard, not a physical model).

⚠ **This is above the operator's own earlier ~10 mm strap-failure estimate**, raised at their explicit direction. Recorded here so the tension is visible rather than lost: the assistant flagged it once, the operator — who can see the rig, and whose read on the mounting has been right twice where the assistant's was wrong — decided. The probe prints the reminder in its banner every run.

The remaining guards at that depth:

| depth achieved | force (softening curve) | % of the 8 N ceiling |
|---|---|---|
| 6 mm | 2.86 N | 36% |
| 10 mm | 4.56 N | 57% |
| 15 mm | **6.68 N** | **83%** |

so the 8 N force backstop lands near **~18 mm achieved** — real, but it will not fire before 15 mm. The operator remains the primary stop.

⚠ **SAG MATTERS AT THIS DEPTH.** Measured `sag = 0.258·F − 0.148 mm`, so ~1.6 mm at 6.7 N. **Commanding 15 mm ACHIEVES ~13.2 mm**; command ~16.6 to achieve 15. Every `--max-depth-mm`, `STRAP_LIMIT_MM` and ladder value is a *commanded* depth; `campaign_plan.csv` and the probe CSVs record both.

#### `--prompt-from-mm` and coast-N: keeping the interactive map usable

At 0.25 mm steps to **15 mm that is 60 steps, i.e. 48 prompts per node × 9 nodes = 432 keypresses**, which would push anyone toward `--no-interactive` and lose the optical limit entirely. Two mitigations:

1. The probe **auto-descends (backstops live) to `--prompt-from-mm`, default 3.0 mm**, then prompts.
2. ⚠ **At any prompt, typing a NUMBER coasts that many steps before asking again.** Skim the shallow region, fine-step near the limit — the operator picks the granularity live. Verified in simulation: a 48-step descent needed **5 prompts instead of 48**.

Coasting skips the *prompt*, never the checks — the force and stiffness backstops stay live through every coasted step.

The 3.0 mm default is grounded, not arbitrary: the centre probe showed no optical trouble to 4.5 mm, and **the centre is the worst case optically** — it is the most compliant point, so it deforms most for a given depth; edge nodes go out of frame *deeper*, not shallower. `--prompt-from-mm 0` restores prompting from step 1.

#### Centre probed to 10 mm — and the stiffening does arrive (2026-08-07)

Second centre run, 44 steps, operator-stopped at **10.074 mm / 4.892 N** (stopped out of caution about the sensor, not a marker failure — the log now says "called the limit", not "optical limit", because the script cannot know which).

| band | local exponent | dF/dz |
|---|---|---|
| 0.3–2.5 mm | 1.11 | 0.548 N/mm |
| 2.5–5.0 mm | 0.86 | 0.423 N/mm |
| 5.0–7.5 mm | 0.87 | **0.398 N/mm** (minimum) |
| 7.5–10.1 mm | 1.28 | **0.594 N/mm** |

⚠ **The membrane stiffening does happen — it just starts past ~7.5 mm, not at 3 mm.** So the earlier "softening, full stop" conclusion was also incomplete: the sheet softens to ~5–7.5 mm and then stiffens as tension finally takes over. Over the whole 0.37–10.07 mm range it is still near-linear (`F = 0.451·d + 0.099`, R² = 0.9956) because the two effects partly cancel.

Revised sag: **`sag = 0.218·F − 0.053 mm`** (1.01 mm at 4.89 N).

#### `--resume` / `--skip-nodes`: finish a map without repeating nodes

`--map --resume [MAP_CSV]` loads a previous map CSV, **probes only the missing nodes, and writes one merged map**. `--skip-nodes 0,4` skips explicitly.

⚠ **Node geometry always comes from the freshly built grid, never from the prior file** — only the *measurement* (depth/force/k/n/reason) is carried over, so a stale CSV cannot smuggle in stale coordinates. Reused rows are marked `from_previous_run=1`. Prior `*_steps.csv` rows are merged too, with a union of columns so an older schema cannot crash the write.

The centre run was seeded in as **node 4** this way. ⚠ It was measured at the *taught centre* `[0.65444, −0.00171]`, which is **1.21 mm** from the 3×3 lattice's node 4 at `[0.65556, −0.00216]` — under 10% of the 12×16 mm node spacing, so acceptable, but it is recorded in that row's `reason` string rather than silently merged.

Verified end to end in simulation: 8 nodes probed + 1 reused → 9-node map, 116 merged step rows, geometry refilled from the live grid.

#### ⚠ FIRST FULL MAP — and the taught quad is NOT equidistant from the frame (2026-08-07)

All 9 nodes probed, every one operator-set. Result (max indentation, mm):

```
   v=1.0     4.70    3.24    4.46      <- x ~ 0.6714, the PLASTIC WALL side
   v=0.5     7.50   10.07    6.93
   v=0.0     7.69    5.46    8.37
             u=0.0   u=0.5   u=1.0
```

⚠ **The `v_hi` edge (TL/TR side, x ≈ 0.6734) has a plastic wall much closer than the other three.** The operator stopped that whole row early because the indenter was about to strike it. Those three depths are a **geometric** limit, not a material one, and they were dragging the recommended ceiling down to 2.59 mm.

**The proof is node 8: it stopped at 4.46 mm having reached only 1.56 N — the softest force of all nine nodes.** Nothing mechanical stops you at 1.56 N when the centre took 4.89 N to 10 mm.

| | |
|---|---|
| ceiling **with** the wall row | 3.24 mm × 0.8 = **2.59 mm** |
| ceiling **without** it | 5.46 mm × 0.8 = **4.37 mm** |

⚠ **THE DESIGN RULE THIS GIVES US: required clearance scales with depth.** The indenter is a *cone*, so its cross-section at the silicone surface grows as `d·tan(θ)`. Back-solving from the three wall-limited stops assuming the wall sits at the taught corner line: θ = 23°, 32°, 24° — consistent. So **clearance ≈ 0.5 × depth**: a 2 mm margin only supports ~4 mm of indent next to a wall, and reaching 10 mm anywhere needs ~5 mm of clearance. (This also finally puts a number on AGENTS.md's open "measure the cone angle" item, at least approximately.)

#### Per-edge margins: `--margin-mm`

A symmetric inset cannot express "one edge needs more room", so `build_grid` now takes **per-edge margins** and both scripts expose `--margin-mm` taking **1 value** (all edges), **2** (u, v symmetric) or **4** (`u_lo u_hi v_lo v_hi`). It overrides `--span-mm`.

Both scripts now print an **edge table with the physical coordinate of each edge**, because `u_lo`/`v_hi` are meaningless when you are standing at the robot watching a collision:

```
  edge clearances from the taught quad:
     u_lo (BL/TL side)   2.77 mm  at x 0.6374, y +0.0129
     u_hi (BR/TR side)   2.77 mm  at x 0.6380, y -0.0166
     v_lo (BL/BR side)   2.03 mm  at x 0.6374, y +0.0129
     v_hi (TL/TR side)   2.03 mm  at x 0.6734, y +0.0124
     (cone widens ~0.5*depth: 2.0 mm clearance supports ~4.1 mm of indent)
```

⚠ **Changing margins MOVES the grid**, which invalidates previously probed nodes. Two guards: `--reprobe-nodes N,M` discards prior results for moved nodes, and the merge step **computes how far each reused node has moved and warns above 0.5 mm** (recorded as `reused_node_moved_mm`). With `--margin-mm 2.8 2.8 2.0 6.0` the v=0 row moves 0.04 mm (safe to reuse), v=0.5 moves ~2 mm and v=1 moves ~4 mm (must re-probe).

The wall-contaminated map is preserved at `~/E-BTS/depth_limit_map.wallcontaminated.bak.csv`.

⚠ **Still unknown: how far the wall actually is from the taught corner on that edge.** The 0.5×depth rule assumes it sits exactly at the taught line. If the frame is further out or further in, the margin needs adjusting — that measurement has to come from someone looking at the rig.

#### ⚠ THE OPTICAL LIMIT IS THE REAL LIMIT — the operator sets it (2026-08-07)

An indent whose markers cannot be tracked is a **useless training sample no matter how mechanically safe it was**. So the binding constraint is not force, it is the depth at which the silicone leaves frame or the marker pattern breaks down — and only a human watching the live event view can see that. The probe is therefore **interactive by default**.

**Two stopping authorities, not interchangeable:**

| | owns | how |
|---|---|---|
| **Operator** | the **optical** limit | steps down one increment at a time, watching the live view; `s` sets this node's limit |
| **Machine** | the **force** limit | force ceiling + stiffness abort, **active in every mode** — interactive does *not* disable them |

At each step: `[Enter]` deeper · `s` stop here (optical limit) · `b` back off one step · `q` abort run.

- ⚠ `b` (back-off) points are recorded with `step_dir="up"` and **excluded from the k/n fit** — a back-off point is on the unloading branch, and silicone is viscoelastic, so pooling both branches would bias the power law. Verified in simulation: k/n recovered exactly (0.250 / 1.63) with a back-off in the sequence.
- **Interactive defaults ON when stdin is a tty**, and falls back to automatic with a loud warning otherwise — a piped or `ssh`-batch run would otherwise hang forever holding the arm at depth.
- Each node records `stopped_by` ∈ {`operator`, `force`, `stiffness`, `depth`}. `report_map` warns if the map is all-machine (optical limit never captured) or **mixed** (nodes that hit a backstop first have an unknown, possibly shallower optical limit).
- When every node is operator-set, the 0.8 safety factor stacks on top of your own judgement; the report says so and suggests `--safety-factor 0.9`.

**Where the operator stands:** camera live view on the **workstation**, arm + force on **tactile**. No single-owner conflict — the probe needs no camera and the GUI needs no robot.

**Confirm it once, quantitatively.** Your eye sets the limit on the live view, but the *tracker* is what has to cope. Record one indent at the chosen ceiling with `master_midpoint.py`, run `marker_overlay.py --probe`, and check `roster.n_kept` holds up (204 markers at 2 mm as of `mid5_center`). If it collapses, come down and re-map.

#### What the map unlocks: constant-force instead of constant-depth

A constant depth ladder over a non-uniform sheet produces **different force ranges at different locations** — which partially re-introduces the force↔location confound the campaign exists to break (§14.1e). The map's per-node `k` and `n` let you invert `F = k·d^n` and **precompute per-location depths that hit target FORCES** instead. `report_map` prints the k spread across the 9 nodes so the data decides: if k varies by less than ~2×, constant depth is fine; `run1` suggested ~3× (0.45–1.46 N at one commanded depth), in which case force-targeting is worth the extra bookkeeping.

⚠ **`--source hex21` (default) needs the HEX21 MOVED TO TACTILE.** Tactile is the PC wired to the Franka (FCI on `eno1`, 10.1.196.242 → robot 10.1.196.5), and the probe reads force *locally* so the abort logic needs no clock alignment — same replug as surface mapping (§3.2). ⚠ **It goes BACK to the workstation before any campaign recording**, since the GUI records camera + F/T on one clock and that is the whole basis of the sync model (§9). `--source franka` falls back to `O_F_ext_hat_K`, biased ±2 N (§3.1) — a runaway guard, not an instrument.

`sweep_campaign.py` **refuses** depths > 3.0 mm until you pass `--i-have-run-the-probe`.

### 15.2a ⚠ SANITY RUN `pilot_20260807_134217` — the depth ladder must be SHIFTED (2026-08-07)

`--max-points 6` at one location (row0/col0), all six depths. All three streams landed. Everything mechanical worked: depth correction converged (2.5→2.496, 2.0→2.001, 3.0→2.993), every indent got its own `tare` phase, clock drift −1.4 ms over the run.

⚠ **But the two shallowest levels made NO CONTACT AT ALL:**

| commanded | achieved | \|ΔFz\| |
|---|---|---|
| 0.5 mm | 0.584 | **0.007 N** ← nothing |
| 1.0 mm | 1.095 | **0.003 N** ← nothing |
| 1.5 mm | 1.579 | 0.222 N |
| 2.0 mm | 2.001 | 0.566 N |
| 2.5 mm | 2.496 | 0.982 N |
| 3.0 mm | 2.993 | 1.363 N |

**The bilinear plane over the kissed corners sits ABOVE the true contact surface.** Two independent measurements agree:
- the pilot's force ladder extrapolates to zero force at **d ≈ 0.84 mm**;
- re-analysing the `--map` probe's contact onset per node gives **mean +0.863 mm, range +0.051 … +1.310 mm** (spread 1.26 mm).

Internal consistency check: node 4 is the only node whose surface reference was a *directly kissed point* rather than the bilinear plane, and it is the only one with ≈ zero offset (+0.051 mm). Every plane-referenced node shows ~0.8–1.3 mm. So the datum, not the sensor, is what is off.

⇒ **Shift the ladder to `1.5 2.0 2.5 3.0 3.5 4.0`.** With offsets spanning 0.05–1.31 mm that puts every level in contact everywhere (worst case 0.19 mm of real indentation, best case 3.95 mm), and widens the force span from ~1.1 N to ~4.3 N. Deepest expected force ≈ 4.5 N at the stiffest node — well under the 8 N probe ceiling. Needs `--i-have-run-the-probe` since it exceeds `MAX_SAFE_DEPTH_MM = 3.0`, which the completed map now justifies.

⚠ **Revised data rate: 9.49 MB/s, not the 3.97 MB/s implied by `run1`** (measured over the pilot's 47 s recording). **The 648-indent campaign is ~49 GB, not 21 GB.** 746 GB free, so fine — but budget accordingly.

### 15.2d GOTCHA — the plan sidecar filename was derived, and silently missed

`sweep_campaign.py` built the plan path as `stem.replace("_franka","") + "_plan.csv"`. With `out_csv = .../franka.csv` that yields **`franka_plan.csv`**, while `master_campaign.py` scp'd back **`campaign_plan.csv`** — so the depth labels stayed on tactile and the run printed a loud (correct) warning. No data was lost; the file was recovered. **Fixed by pinning the name**: `Path(out_csv).with_name("campaign_plan.csv")`. Lesson: never derive a filename that a *different program* has to guess.

### 15.2e THE PILOT CAMPAIGN RAN: `pilot_20260807_134855` — recording intact, post-processing crashed

**648 indents, 108 locations × 6 depths (1.5–4.0 mm shifted ladder), 81.5 min.** The recording is **complete and verified**; only the post-processing died.

**Integrity checks (all pass):**

| stream | check | result |
|---|---|---|
| `camera.raw` | 45.27 GB, EVT2 header valid, `(size − header) % 4 == 0` | **no torn word** |
| `camera.raw` | tail-decoded TIME_HIGH: covers 4887.7 s vs F/T's 4887.8 s | **complete, 0.1 s** |
| `franka.csv` | 232,893 samples, `point_index` 0–647 contiguous, all 648 have a dwell | **complete** |
| `franka.csv` | last phase logged | **`park`** = clean finish |
| `ft.csv` | 4,887,488 samples, 1000 Hz, max gap 54 ms | **complete** |
| `campaign_plan.csv` | 648 rows, 108 locations × 6 depths ×108 each | **present** (the §15.2d fix worked) |
| `metadata.json` | both clock offsets, drift 10.4 ms | complete |

⚠ **Tail-decoding EVT2 is the cheap way to test a huge `.raw` for truncation** — it is a fixed 4-byte word format, so seek to the last few MB, mask `(w >> 28) == 0x8` for TIME_HIGH words and read the last one (`(w & 0x0FFFFFFF) × 64 µs`). No need to decode 45 GB.

**THE DATASET IS GOOD** — the shifted ladder did its job:

| commanded | n | achieved (mm) | mean \|Fz\| | min–max \|Fz\| | zero-contact |
|---|---|---|---|---|---|
| 1.5 mm | 108 | 1.555 ± 0.068 | 0.563 N | 0.243–1.013 | **0** |
| 2.0 | 108 | 2.007 ± 0.037 | 0.953 | 0.446–1.884 | **0** |
| 2.5 | 108 | 2.505 ± 0.039 | 1.409 | 0.605–2.963 | **0** |
| 3.0 | 108 | 2.994 ± 0.032 | 1.864 | 0.885–4.137 | **0** |
| 3.5 | 108 | 3.494 ± 0.049 | 2.329 | 1.097–5.125 | **0** |
| 4.0 | 108 | 3.974 ± 0.104 | 2.748 | 1.381–5.945 | **0** |

**Force range 0.243–5.945 N, sd 0.989 N — against `run1`'s 0.452–1.463 N, sd 0.173 N. A 5.7× wider spread, and 0 of 648 indents missed contact.** Depth alone now explains only R² = 0.551 (residual sd 0.663 N); the rest is spatial stiffness variation, which is exactly the signal the camera is supposed to explain.

#### ⚠ WHY THE POST-PROCESSING CRASHED — `postprocess.py` does not scale past ~99 pokes

`postprocess.py:309` allocates `buckets = [[] for _ in windows]` and accumulates **every** event sub-array for **every** poke in RAM, writing only after the streaming pass completes (line 320).

- dwell windows hold ~648 × 2 s × 2.3 Mev/s ≈ **3.0 billion events**
- metavision dtype ≈ 16 B/event ⇒ **~48 GB resident** before a single byte is written
- the machine has 62 GB ⇒ **OOM**

(For `run1`: 99 pokes at ~1 Mev/s ≈ 3.2 GB — which is why it worked there and nobody noticed.) The inner loop is also `48,880 chunks × 648 windows = 31.7 M` mask operations, so it would have taken hours even with infinite RAM.

⚠⚠ **DO NOT RE-RUN `postprocess.py` ON THIS RUN.** Line 353 does `shutil.move(src, ...)` — it **moves** `camera.raw`/`ft.csv`/`franka.csv` out of `recordings/`. It crashed before reaching that, which is the only reason the 45 GB source is still where it belongs. `postprocess_indent.py` uses `shutil.copy2` instead (line 485) and is the safe one.

**The right path forward (this is §14.4's advice, now forced by reality):**
1. `postprocess_indent.py --no-events` for the force/robot numbers — fast, non-destructive, drift-aware clock (matters: 10.4 ms drift here).
2. Join depth labels from `campaign_plan.csv` on `point_index` — neither post-processor knows about it yet.
3. **Never slice per-indent event CSVs at this scale.** Build 40 ms accumulated frames straight to HDF5 in the same streaming pass, writing incrementally: 648 × 50 frames × 640×450 uint8 ≈ 9.3 GB raw, and the frames are sparse so they compress ~10–20× to well under 1 GB — versus tens of GB of event CSV.

### 15.2f `evt2_frames.py` + `build_dataset.py` — the campaign-scale extractor (2026-08-07)

Two new files replace `postprocess.py`'s event slicing for campaign runs. Both only ever **read** `recordings/`.

| file | purpose |
|---|---|
| `evt2_frames.py` | `Evt2Reader` — random-access EVT2 decoder: binary-seek to any timestamp, accumulate 40 ms count frames |
| `build_dataset.py` | run folder → one HDF5 of frames + force/depth labels, written incrementally |

#### Random access into a 45 GB `.raw`

⚠ **`metavision_core`'s `EventsIterator` has no true seek** — `start_ts=4000 s` on this file took **198 s** because it streams everything before the target. EVT2 is a fixed 4-byte word format, so a plain binary search over byte offsets is exact and cheap:

```
bits 31..28 type   0x0 CD_OFF | 0x1 CD_ON | 0x8 TIME_HIGH
CD_*  : 27..22 ts_low (6 bits, us) | 21..11 x | 10..0 y
0x8   : 27..0  ts_high (28 bits, units of 64 us)
t_us  = (ts_high << 6) | ts_low
```

TIME_HIGH recurs at least every 64 µs (~368 bytes at 2.3 Mev/s), so any 64 KB probe contains many. **Same seek in 0.47 s — 422× faster**, and the decoder is **bit-exact against metavision** (maxdiff 0 over the first 200 ms).

⚠ **Two time bases.** The raw's own TIME_HIGH starts at sensor uptime (**1377.077 s** here), while `EventsIterator` re-bases to 0 and `postprocess.py` maps unix→device as `(t − t_ft[0])·1e6` on the re-based scale. `Evt2Reader.base_us()` exposes the offset; `frames()` takes re-based times by default.

#### Why frames beat events, measured

A 40 ms window holds ~92,000 events over 288,000 px:

| representation | size |
|---|---|
| events, x/y/t as (u2,u2,u4) | 722 KB |
| **dense 640×480 uint8 count frame** | **281 KB** |

Dense wins 2.6× *before* compression because the event rate is high, and gzip takes it ~9× further (15% occupancy). Lossless here: counts peak at **27** so uint8 is exact (⚠ never rescale, §13.1), and **every event is one polarity** — `bias_diff_off = 0` kills OFF in hardware, verified 923,529 ON / 0 OFF, so the polarity bit carries zero information.

#### ⚠ WHY THE WHOLE RAMP IS KEPT — force does NOT predict displacement

Tested on the two extremes of the pilot's 4 mm level:

| indent | force | max marker displacement | ÷ pitch | direct tare→peak match |
|---|---|---|---|---|
| row11 col2 (stiffest) | 5.945 N | 9.24 px (0.74 mm) | 0.31 | 211/211 ✓ |
| row11 col8 (softest) | **1.381 N** | **21.67 px (1.73 mm)** | **0.74** | **206/211 ✗** |

⚠ **The SOFTEST indent moves markers 2.3× further than the stiffest, despite 4.3× less force.** A stiff spot resists — high force, little deformation. So picking the highest-force indent as "worst case" for tracking is exactly backwards.

On the soft one, 5 of 211 markers exceed the 12 px match radius (largest jump 20.9 px) and displacement reaches 0.74× the 29.4 px marker pitch — deep into the range where a nearest-neighbour match binds to the *wrong* marker (§13.3). **So the tare→peak two-frame shortcut is unsafe and the dip frames are kept**, letting the roster propagate at ~0.3 px/frame. `--no-dip` exists but warns.

Sequential tracking keeps **207 of 211** markers gapless across all frames on both indents.

⚠ **Drop the final 40 ms window of every indent** — it is cut short by the phase boundary, so it holds a fraction of the events and its detections collapse (17 vs ~211 on the pilot). `build_dataset.py` does this automatically.

⚠ **Frames differ from a naive extraction by ~80% of a frame** because the windows are **drift-corrected** (31.9 ms at this indent, interpolated between the two measured clock offsets). That is intended; the drift-aware version is correct.

#### Output

```
python3 build_dataset.py recordings/pilot_20260807_134855          # ~32 min, ~3 GB
python3 build_dataset.py <run> --indents 598,633                   # spot check
```

`/frames` uint8 [N,480,640] gzip (chunk = 1 frame) · `/frame_indent` · `/frame_phase` (0/1/2) · `/frame_t_us` · `/indents` with `point_index, depth_mm, row, col, x, y, achieved_mm, tare_Fz_N, tare_sd_N, dwell_Fz_N, peak_Fz_N, frame_offset, n_frames`. Forces are tared on each indent's **own** tare window. **This is the depth-label join** that neither post-processor did.

#### ✅ BUILT AND VERIFIED: `output/pilot_20260807_134855_frames.h5` (2026-08-07)

**91,342 frames · 648 indents · 3.00 GB · 9.4× compression · 27.4 min.** This is the trainable dataset.

| check | result |
|---|---|
| `point_index` 0–647 | contiguous, none missing |
| `sum(n_frames)` vs `len(frames)` | 91,342 = 91,342 |
| `frame_offset` chain | contiguous and gapless |
| phases | tare 16,167 · dip 43,424 · dwell 31,751; all 3 present per indent |
| force labels | reproduce the independent CSV analysis **exactly** (0.563/0.953/1.409/1.864/2.329/2.748 N) |
| NaNs / zero-contact | **0 / 0** |
| tare σ | 0.1134 N (sensor floor 0.116) ✓ |
| marker detection, 6 random indents across the ladder | **211 tare / ~212 dwell markers, R = 9.39–9.42 px**, occupancy 15.2–15.3% |
| empty frames (200 sampled) | 0 |
| max count (200 sampled) | 25 — uint8 headroom fine |

Marker counts and radius are stable to ±1 marker and ±0.03 px across the whole depth ladder, so detection is not degrading at depth.

**Next: §14.8 step 1** — ridge on marker-displacement features with grouped CV by location, which must beat the robot-only baseline (depth alone now explains R² = 0.551, residual sd 0.663 N).

### 15.2g ✅ FIRST TRAINED MODELS — the camera beats the robot (2026-08-07)

Two routes were built in parallel. **Both beat the robot-only baseline using the camera alone.**

| file | what |
|---|---|
| `marker_features.py` | fast marker detector + tracker — 31–35× faster than `marker_overlay.py`, validated to **0.0042 px** mean disagreement |
| `extract_features.py` | 648 indents → `output/pilot_20260807_134855_features.csv` (37 cols) in **2.6 min** (was ~6.2 h) |
| `train_cnn.py` | small CNN on the difference image, grouped CV |
| `output/pilot_cnn_cache.npz` | 648 × (tare, dwell) at 320×240, 38 MB — built in 20 s |

#### ⚠⚠ THE CV SPLIT CHANGES THE ANSWER MORE THAN THE MODEL DOES

This is the most important methodological result in the project so far.

| model (camera-only unless noted) | leave-one-LOCATION-out | **row-band (strict)** |
|---|---|---|
| depth only — robot, no camera | 0.642 N / 0.578 | **0.692 N / 0.510** |
| marker magnitude only | 0.684 / 0.522 | 0.705 / 0.492 |
| marker magnitude + shape | 0.495 / **0.749** | 0.669 / **0.542** |
| magnitude + shape + `^1.5` terms | 0.465 / **0.779** | 0.733 / **0.451** |

Leave-one-location-out holds out one grid point (6 presses) while its 3 mm neighbours stay in training. **The deformation field has a measured ~8 mm half-width and is still at 10% of peak 18 mm away**, so those neighbours are near-duplicate measurements — the split leaks badly.

⚠ **The `^1.5` feature set is the cautionary tale: best of all under the leaky split (R² 0.779), WORST of all under the strict one (0.451).** Textbook overfitting that only proper CV exposes. **Always use contiguous row-band splits for this dataset.**

#### Final scoreboard — strict row-band CV, 4 folds, no robot-side inputs

| model | RMSE | R² |
|---|---|---|
| predict the mean | 0.989 N | 0.000 |
| depth only — **robot, the bar** | 0.692 N | 0.510 |
| marker magnitude only | 0.705 N | 0.492 |
| marker magnitude + shape | 0.669 N | 0.542 |
| CNN on frames, 5-seed ensemble | 0.579 N | 0.657 |
| **CNN + marker model, averaged** | **0.509 N** | **0.734** |

**The camera beats the robot-only baseline by 26% RMSE with no depth input.** The two routes are complementary — averaging them gains more than either alone, so they are reading partly different things.

#### ⚠ THE HEADLINE IS DOMINATED BY ONE FOLD — the model has learned the interior, not the edge

Per-fold RMSE (N) tells a very different story from the aggregate:

| fold (rows) | depth | CNN | marker | **blend** | \|Fz\| mean / max |
|---|---|---|---|---|---|
| 1 — rows 0–2 | 0.376 | 0.487 | 0.601 | **0.423** | 1.69 / 4.32 |
| 2 — rows 3–5 | 0.583 | 0.168 | 0.378 | **0.213** | 1.41 / 4.37 |
| 3 — rows 6–8 | 0.576 | 0.319 | 0.440 | **0.320** | 1.40 / 4.49 |
| **4 — rows 9–11** | 1.050 | 0.987 | 1.046 | **0.844** | **2.08 / 5.94** |
| overall | 0.692 | 0.579 | 0.669 | **0.509** | |

⚠ **Fold 4 contributes 69% of the total squared error** and is 4× the best fold. It holds rows 9–11 — the strapped edge, where mean force is 2.08 N vs 1.50 N elsewhere and the spread is 1.27 N vs 0.82 N. Under grouped CV the model has *never seen that stiffness regime* when tested on it, so it extrapolates and fails. **Every** model fails there, including depth-only (1.050) — so it is a property of the data, not of any one architecture.

Read the result two ways, both honest:
- **On rows 0–8 (three-quarters of the elastomer): blend RMSE ≈ 0.33 N vs depth-only 0.52 N — 37% better.** This is the well-sampled interior and the model works well there.
- **On the strapped edge it does not generalise.** Fixing that needs either more sampling in the stiff regime or accepting the sensor is characterised for the interior only. Note this is the same `v_hi` edge that caused the indenter/wall clearance problem (§15.2), so it has been the awkward region throughout.

CNN notes: input is the **difference image** (dwell − tare), 120×160, 3 channels (diff + 2 coordinate channels), ~15k params, `AdaptiveAvgPool2d((3,4))` **not** global pooling — force depends on *where* you pressed, so translation invariance is a defect here, and the coordinate channels let the net represent "soft middle vs stiff edge". Per-seed R² varies **0.52–0.77 (sd 0.093)** on 648 samples, so **always ensemble seeds**; a single run is not a measurement. `--channels both` (tare + dwell) was *worse* (0.479) than the difference alone. Training is ~30 s/run on the 2080 Ti.

⚠ **The earlier "shape features FAIL, R² = −0.012" (§15.2f) was an n=48 artifact and is retracted.** Re-run on 200 random 48-indent subsets: magnitude+shape scores 0.427 vs magnitude-only 0.471 — at n=48 a 17-feature model overfits. At n=648 it wins. The negative result was correct for its sample size and simply did not generalise.

Also worth recording: the **indenter is 3 mm in diameter** (was an open TODO in `AGENTS.md`), and `r50_mm`/`r25_mm` are poorly conditioned here — the unbacked membrane deflects globally rather than in a local patch, so "decay length" saturates. `mean_tangential_mm` (r = 0.673) and `disp_max_mm` (0.658) are the strongest single camera features.

## 16. RESET — external review and the plan to start over (2026-08-07)

An ML specialist (Amir) reviewed the work. **The verdict was that the approach was wrong, and most of it is correct.** This section records the critique, an honest assessment of what survives it, and the plan. **Nothing in §14–15 should be treated as a result any more** — the infrastructure and the physical measurements survive, the models and the dataset do not.

### 16.1 The critique, as given

> 0) define task(s) · 1) define metrics · 2) EDA the dataset — *"Thats the steps you start with"*
> Dataset must be **structured, logical, organized**. **No data in names of files.** Names and columns should be **coherent**.
> After EDA → experiment with different models. *"They will fail 97% of the time, then you optimize."* Maybe use a pretrained model. Then hyperparameter optimization.
> On the results figure: *"It was not a diagonal… not good… the graphs are bad themselves… not informative."*
> *"Dataset was also shit to begin with. The trajectory it created was too close to the straps on one side."*

### 16.2 Honest assessment — what is right, and one thing that is worse than he thought

**Right, and not arguable:**

| critique | the evidence in our own data |
|---|---|
| **Wrong order — modelled before defining task/metrics/EDA** | True. EDA happened *incidentally*, scattered through §14–15, never as a deliberate phase. The metric was chosen after seeing results. |
| **Dataset badly structured** | **43 files carry a timestamp in the filename.** Data lives in `recordings/` and `output/` under three different conventions. Column names are incoherent across files (`dwell_Fz_N`, `disp_mean_mm`, `point_index`, `cx_mm`). |
| **Trajectory too close to the straps** | Measured: the strapped edge produced **69% of all model error** and stiffness varies **9.4×** across the block. We worked around this instead of fixing it. |
| **Graphs not informative** | Fair. The 3-panel map is dense and cannot be judged at a glance. |

⚠ **And the "not a diagonal" call was right for a reason he could not have seen from that figure.** He was shown the *map*, not the predicted-vs-measured scatter. Fitting a line to the actual cross-validated scatter:

```
cross-validation, 648 presses : slope 0.576   intercept +0.610   r 0.825
hard holdout,      24 presses : slope 1.080   intercept +0.052   r 0.950
```

**A perfect model gives slope 1.000. On the honest test the slope is 0.576** — the model compresses everything toward the mean, exactly as the per-depth error walk (+0.19 N at 1.5 mm → −0.32 N at 4.0 mm) already implied. The 24-press holdout looked near-diagonal only because it barely reached the high-force regime. **His instinct was correct and the defect is real.**

### 16.3 What survives the reset

These are measurements and infrastructure, independent of the modelling mistakes. **Do not re-derive them** — they cost robot time.

**Physical facts:** marker pitch 2.317 mm · marker radius 0.75 mm → 12.5 px/mm · ~211 markers · displacement noise floor 0.26 px = 0.020 mm · deformation field half-width ~8 mm, 10% of peak at 18 mm · force law at centre `F = 0.451·d + 0.099` (R² 0.996, 0.4–10 mm) · stiffness varies 9.4× across the block · impedance sag `= 0.218·F − 0.053 mm` · 40 ms = exactly one 25 Hz illumination cycle · elastomer 5 mm thick, edge-strapped, unbacked · indenter is a 3 mm plastic cone, half-angle 23–32°.

**Infrastructure:** `evt2_frames.py` (binary-seek EVT2 decoder, bit-exact, 422× faster than metavision's `start_ts`) · `marker_features.py` (33× faster detector/tracker, validated to 0.0042 px) · camera calibration (`camera.bias`, ROI `0 7 640 450`) · the synchronised recording pipeline itself · per-press tare (drift is 0.6 mN/s, thermal, r = −0.67 with temperature).

**Discard:** every trained model · `force_model.pt` · the dataset layout and naming · all §15.2g accuracy numbers.

### 16.4 The plan

**Order is deliberate: the rig and the data come first, then Amir's sequence.** Do not start modelling until phase 4 is written down.

#### Phase 0 — Fix the rig *(blocks everything)*
- ⚠ **Resolve the strap problem at source.** It is the single largest defect in the data. Either re-mount so stiffness is uniform, or formally define a reduced usable area and never leave it. Working around it in software was tried and failed.
- Measure and record the indenter geometry properly (diameter, tip radius, cone angle) rather than back-solving it.
- Decide the indenter set — one tip or several. Several makes the sensor general but costs a full collection per tip.

#### Phase 1 — Define the task
Write it down before touching anything else. At minimum: what the model consumes (one 40 ms frame? a frame plus a tare reference?), what it emits (scalar normal force? force + contact location? shear?), the operating envelope it must cover (force range, contact area, location), the latency budget, and what "good enough" means numerically for the actual application.

#### Phase 2 — Define metrics and the evaluation protocol
Chosen **before** any model exists, and then not changed. Primary metric (RMSE in N is the natural choice — it is in the units of the thing). ⚠ **Also fix the diagonal-slope check as a first-class metric**, not an afterthought: a model can have good RMSE while systematically compressing, and that is what happened. Define the baselines to beat and the split policy in advance. ⚠ **Splits must be grouped by location** — the deformation field is ~8 mm wide against a 3 mm pitch, so a random split leaks badly (demonstrated: the `^1.5` feature set ranked best under a leaky split and worst under an honest one).

#### Phase 3 — Dataset design and collection *(the user's first priority)*
**Structure, per the critique:**
- ⚠ **No data in filenames.** No timestamps, no parameters. A session is a directory with an opaque id; everything descriptive lives in a `metadata.json` and a top-level `manifest.csv`.
- One documented schema, `snake_case`, units as a consistent suffix (`force_z_n`, `depth_mm`, `disp_mean_mm`), stable id columns (`session_id`, `press_id`, `location_id`) used identically everywhere.
- Clean separation: `data/raw/` (never modified) → `data/processed/<version>/` → `data/features/<version>/`, each with a written schema.

**Coverage — design by FORCE, not depth.** We now know `F = k(x,y)·d` with k varying 9.4×, so a fixed depth ladder produces wildly non-uniform force coverage. That is exactly why only 9% of the last dataset exceeded 3 N while the model was expected to work to 5.9 N. Use the measured stiffness map to precompute per-location depths that hit **target force levels**, giving a balanced force histogram. Include repeats for a noise estimate.

#### Phase 4 — EDA, deliberately and in writing
Distributions, coverage gaps, correlations, outliers, sensor dropouts, per-session drift, leakage checks. Produce a written EDA document **before** any model. Much of this exists scattered through §14–15; it needs doing properly and in one place.

#### Phase 5 — Modelling
Baselines first (predict-the-mean, robot-only). Then several model families, expecting most to fail. Look for pretrained models worth transferring. Hyperparameter optimisation last, not first.

### 16.4a DECISION: pretrained models, not another from-scratch CNN (2026-08-08)

⚠ **Do not build another CNN from scratch.** The operator has decided the next modelling attempt uses **pretrained** models. Three were evaluated: **TDNN**, **ResNet-18**, **VGG-16** -> `ml/PRETRAINED_MODELS_ANALYSIS.md`.

⚠ **DECIDED 2026-08-10: ResNet-18 is the agreed direction, and the work is PAUSED** until data collection is done. VGG-16 is out (138 M params vs a few hundred presses, 89.4% of it an FC head that gets deleted). TDNN is not a pretrained option at all — the only downloadable weights are speaker-ID models consuming log-mel audio. The analysis also found a real defect in `ml/train_cnn.py`: it standardises the target to unit sd then uses `smooth_l1_loss`, so **every residual beyond ~0.99 N gets constant gradient** — the tails are down-weighted twice, contributing to the slope-0.576 compression. ⚠ And a statistical ceiling: per-seed R² sd is 0.093, so at n≈648 detecting an architecture difference needs ΔR² ≈ 0.16 — **architecture selection sits below the noise floor while data design sits above it.** One claim in that report is wrong: the `torchvision` ABI "blocker" — it imports and constructs `resnet18` fine. That analysis is **done**; implementation still waits until after data collection.

Context the analysis has to respect: input is a 40 ms event-count frame (640×480 uint8, single polarity, counts ≤ 27) or the difference against a per-press tare; target is scalar normal force; sample count is in the hundreds; splits must be grouped by location; and ⚠ **translation invariance is a defect here** — force depends on where you pressed because `k(x,y)` varies several-fold, so architectures ending in global average pooling need that addressed rather than inherited.

### 16.4b ✅ PHASE 0 COMPLETE — surface + stiffness mapped (2026-08-10)

The depth datum is now **measured, not interpolated**, and its accuracy is known.

| | |
|---|---|
| taught corners | 5/5 force-referenced, fit R² 0.994–0.9997, 5–7 distinct z levels each |
| **surface map** | **99/99 points (94 good), 9 × 11 raster, 3.26 × 3.06 mm pitch, full 26.1 × 30.6 mm** |
| **repeatability** | **0.042 mm** (same 25 points, two runs 29 min apart) |
| surface structure | smooth ramp along u + **one ~9.5 mm undulation, 0.22 mm amplitude** — resolved |
| residual vs that model | 0.095 mm (2.3× repeatability) |
| stiffness | **0.467–1.348 N/mm, 2.9× spread** — enables force-targeted presses |
| retries | 7 of 99 needed one, max 3 attempts, all recovered |

**Files:** `franka/teach_surface.py` (you kiss XY → robot measures Z objectively; a hand kiss was off by up to **1.005 mm**, the probe repeats to 0.042 mm — ~24× better) and `franka/map_offset.py` (grid map, reuses `teach_surface.probe_one` so map and reference cannot disagree). Outputs on tactile: `surface_points.json`, `surface_offset_map.{json,csv,png}`.

#### ⚠ THREE WRONG CONCLUSIONS ALONG THE WAY — all from under-sampling, all corrected

Recorded because each looked convincing and each would have cost real time.

1. **"A 0.26 mm traversal-direction bias."** Serpentine sweeps alternate rows in opposite directions, and the surface alternated with row parity — correlation r = +0.75/+0.81 across two runs. **Wrong.** Switching to raster (every row swept the same way) left the alternation unchanged at +0.269 mm, and rows whose direction changed moved only 0.019 mm, inside repeatability. Serpentine *confounds* row parity with sweep direction; raster separates them.

2. **"A real 0.27 mm ripple — a buckle in the strapped sheet."** **Also wrong.** It was an alias.

3. **"A 4.55 mm ripple, matching the marker lattice aliased by the sampling pitch."** The arithmetic was seductive — a 2.317 mm marker pitch sampled at 3.06 mm folds to 4.50 mm, against 4.55 mm observed. **Also wrong.** A 25-point line at **0.6 mm** spacing (3.9 samples per marker period) found the lattice at **R² 0.006, amplitude 0.016 mm** — absent, below the noise floor. The 4.55 mm component scored R² 0.015.

⚠ **The methodological error behind #3 was mine and is worth not repeating: I searched for periods shorter than the data's Nyquist limit.** The 9×11 map samples at 3.06 mm, so it cannot resolve anything below **6.12 mm**; scanning from 4 mm upward found an alias and reported it as a measurement. Restricting the scan to > Nyquist gives **9.34 mm, R² 0.940** — matching the independent 0.6 mm line's **9.94 mm, R² 0.933**. **Always bound a periodogram by 2 × sampling pitch.**

⇒ **Do not map finer.** A 9.5 mm feature at 3.06 mm pitch is 3.3 samples per period, comfortably resolved. The "measured span grows with density" pattern (0.98 → 1.38 → 1.55 mm) was the alias, not missing structure. A ~2.5 h high-resolution map was recommended on the strength of the artifact and is **cancelled**.

#### ⚠ Geometry cross-check: the taught area is smaller than intended

Intent was 1.5 mm margins on a 36 × 30 mm elastomer → 33 × 27 mm usable. Measured from the taught corners:

| axis | intended | measured | actual margin/side |
|---|---|---|---|
| v (the ~36 axis) | 33.0 mm | **30.61** | **2.70 mm** |
| u (the ~30 axis) | 27.0 mm | **26.06** | **1.97 mm** |

Also: the quad is **not square** — diagonals differ by 1.04 mm, the corner at BL is **91.14°**, and the taught "centre" sits **1.01 mm** off the corner centroid. None of this breaks anything (the grid is bilinear over a general quadrilateral) but the rectangle mental model is off by ~1 mm. Reclaiming the missing 2.4 mm on v would need re-teaching the corners further out, toward the strap.

#### Grid-density reasoning, for the next time this comes up

At **equal pitch** (9 samples in u, 11 in v) the surface is **6.4× rougher along v than along u** — so anisotropic sampling is justified *by measurement*, not assumption. An earlier 7 × 17 proposal was rejected as unjustified at the time because the 5 × 5 map could not establish it; 9 × 11 was the right call and the anisotropy has since been confirmed.

#### Script safety features added while doing this

- **Immediate retry on low confidence**, escalating the fine step ×1.5 and force band ×1.25 per attempt — low confidence means too few *distinct z levels*, caused by the impedance controller holding then slipping (§4.4), so repeating identical parameters just reproduces the clumping. Keeps the **best** attempt, not the last.
- **Archive before first write.** `save()` runs after every point, so a wrong invocation destroys the previous map before anything looks amiss — which happened once, losing a completed 45-point run. Every run now copies the previous map to `surface_offset_map.prev_<shape>_<n>pts.*`.
- **Resume adopts the stored grid** rather than CLI defaults, and a shape mismatch **aborts** instead of overwriting. `--redo` refuses if no stored points were reusable.
- **`--line-u` / `--out-stem`** for diagnostic lines that must not clobber the real map. (`np.linspace(a,b,1)` returns `[a]`, so a single column needed setting explicitly or it would silently land on the u=0 edge.)
- ⚠ **Fine step must stay ≥ ~0.15 mm.** At 0.05 mm the arm does not move at all for many commands and then slips ~0.5 mm; the quality gate counts **distinct z levels**, not samples, because a line through two clusters gives R² 0.99 and means nothing.

#### Tactile filesystem organised (2026-08-10)

72 → 45 top-level entries. Added `archive/superseded_maps/`, `archive/old_indent_runs/`, `vendor/` (ForceTorqueExplorer + JRE + licences), removed `__pycache__`, wrote `~/E-BTS/README.md`.

⚠ **No `.py` file and no live data file was moved, deliberately.** The scripts import each other as siblings and resolve data with `Path(__file__).with_name(...)` — moving either breaks them *silently*. The README states this prominently so the next tidy-up does not cause an outage. All four active scripts re-verified afterwards: import OK, dry-run OK.

### 16.5 Open decisions before collection

1. Re-mount the straps, or restrict the usable area? (Amir and the measured 69%-of-error both point at re-mounting.)
2. How many indenter diameters — and are the tips actually available?
3. Target force levels and how many presses each.
4. Repeats per condition — needed for a noise floor, and we have never measured one properly at the campaign scale.

### 15.2b ⚠ LOAD AVERAGE IS THE WRONG READINESS METRIC ON TACTILE (2026-08-07)

§4.6 says "want load average < 4". **On this machine that number is misleading and will block runs that are perfectly healthy.** Measured with Chrome closed:

| metric | value |
|---|---|
| load average | **10.09** |
| cores | 16 |
| CPU idle | **94–97%** |
| runnable queue (`vmstat r`) | 0–3 |
| D-state (uninterruptible) processes | **none** |
| runnable threads | 2 of 965 |
| **`control_command_success_rate`** | **1.0** (20 samples) |
| `eno1` TX errors | 0 |

The box is idle; the load figure does not reflect contention here (PREEMPT_RT with a 1 kHz SCHED_FIFO control loop, ~14k interrupts and ~54k context switches/s, inflates it).

⚠ **Use `control_command_success_rate` instead — it is the direct measure of whether the FCI is dropping packets**, which is the thing §4.6 actually cares about:
```
rostopic echo -n 20 /franka_state_controller/franka_states/control_command_success_rate   # want 1.0
vmstat 1 4        # want high id%, low r
```
Chrome still matters — it was burning 44% CPU and that *is* real contention — but judge readiness by success rate and idle %, not by the load average.

### 15.2c ⚠ THE WITTENSTEIN IS NEVER HARDWARE-ZEROED — and must not be (2026-08-07)

`src/gui/wittenstein_worker.cpp` is **read-only**: it opens the port, sets baud/DTR/RTS, syncs to packet boundaries and reads. It never writes a command. The HEX21 protocol as documented (§3.2) is pure 1 kHz streaming of 7 float32 — there is no tare command in it. So "zero the sensor" is not a thing this rig can or should do.

**It would also not solve drift — it would cause it.** Measured on `run1` (605 s of continuous F/T):

| | |
|---|---|
| temperature | 30.87 → 31.17 °C (**+0.30 °C**, still climbing at the end) |
| baseline Fz drift | −0.380 N over the run = **−0.63 mN/s** |
| **corr(temperature, baseline)** | **−0.673** — the drift is substantially thermal |

Now compare the two strategies at campaign length:

| approach | baseline age when used | drift it carries |
|---|---|---|
| one zero at t=0 | up to 86 min | **3.27 N** — larger than the entire 0.34–1.58 N depth ladder |
| **per-indent tare** (what we do) | ~5 s (tare → end of dwell) | **3.2 mN** |

⇒ **Per-indent taring is ~1000× better and is the only approach that survives an 86-minute run.** Every indent holds still out of contact at hover for `--tare-s` (default 1.0 s ≈ 1000 samples, averaging the 0.116 N noise to ~0.004 N) immediately before its dip, tagged `phase="tare"`; `postprocess_indent.py` zeroes each indent on its own window. Raw values are preserved throughout — nothing is destructively re-zeroed.

Optional refinement: the sensor was still warming at the end of a 10-minute run, so powering it on ~20–30 min before a campaign lowers the drift *rate*. Not required — per-indent taring makes it moot.

### 15.3 Why linear regression is the right model — MEASURED, not asserted (2026-08-06)

Tested on `mid5_center_20260806_191916` repeat 1 (204 markers, 248 frames over dip+dwell+retract, contact at the elastomer centre so the location is known). Per-frame aggregate marker displacement vs tared |Fz| from the same sidecar:

| aggregate feature | r with \|Fz\| |
|---|---|
| Σ‖displacement‖ | **+0.9588** |
| mean ‖displacement‖ | +0.9588 |
| RMS ‖displacement‖ | +0.9590 |
| mean of top-10 | +0.9583 |
| max ‖displacement‖ | +0.9466 |

Least-squares fits of |Fz| on those features:

| model | R² | RMSE |
|---|---|---|
| **linear in mean displacement alone** | **0.9193** | **0.0923 N** |
| + a d^1.5 term | 0.9193 | 0.0923 N |
| mean + max displacement | 0.9194 | 0.0923 N |
| mean + max + d^1.5 | 0.9194 | 0.0922 N |

**Two conclusions, and they are the whole argument for the model choice:**

1. ⚠ **Force is LINEAR in marker displacement, even though force is strongly nonlinear in DEPTH** (n = 1.63). Adding a `d^1.5` term changes R² by less than 1e-4. This is linear elasticity: the bulk displacement field is proportional to the applied load regardless of how load relates to indenter depth — the cone-geometry nonlinearity lives entirely in F-vs-depth, not in F-vs-displacement. So a linear model is not a crude approximation here; it is the leading-order physics.
2. **The aggregate features are near-perfectly redundant** (r ≈ 0.96 for all of them, and adding a second buys nothing). Do not stack ten collinear magnitude features and expect gains — extra information has to come from the *shape* of the field (contact location, spread, divergence), not more ways to measure its size.

**Hysteresis is small:** fit on the loading ramp (`dip`+`dwell`), test on unloading (`retract`) → RMSE 0.0880 N, bias only −0.0249 N. Viscoelasticity is not the limiting factor.

**Caveats before over-trusting this:** one location, one repeat, force range 0–1.1 N, and the 0.092 N residual is ~5× the per-frame F/T noise floor (0.018 N), so there *is* unmodelled structure — most likely the contact location / field shape, which is exactly what the campaign grid will let us model.

### 15.4 Software stack for training

Verified on the workstation: **RTX 2080 Ti (11 GB), 20 cores, 62 GB RAM, 747 GB free, `torch 2.4.1+cu121` with CUDA working**, Python 3.8.10 (EOL). **`scikit-learn` is NOT installed** — `pip install scikit-learn==1.3.2` (last release supporting 3.8). Ridge/SVR/trees run on CPU in milliseconds at this sample count; **a GPU is pure overhead until there is a CNN.**

---

## 17. ✅ PLAN B CAMPAIGN — the trainable dataset (2026-08-10)

The dataset the reset (§16) was for. **1008 pokes, 94 locations × 9 force levels, one
uninterrupted 2 h 11 min run**, `recordings/planb_3mm_20260810_145750/`.

### 17.1 Why it is built the way it is

`sweep_campaign.py` crosses the grid with a fixed **depth** ladder. Because stiffness
varies **2.82×** across this block (0.467–1.319 N/mm over the 94 good points of
`surface_offset_map.csv`), identical depth gives wildly different force depending on
where you press — so **force is confounded with position**, which is exactly what made
the pilot untrainable (§14, §16.1).

`campaign_planb.py` inverts the measured stiffness map instead: each location gets the
depth that lands on a **target force**. Force becomes the designed axis.

Three further choices, each for a stated reason:

- **Randomised complete block.** 9 passes; each location receives each level exactly
  once, in a per-location random order. Force level is decorrelated from position
  within a pass, and every pass spans the full range, so drift and creep hit all levels
  equally instead of aliasing onto one. ⭐ **Consequence: stopping early still leaves a
  balanced dataset.**
- **Serpentine within a pass, alternating direction per pass.** Traverse efficiency
  without confounding row parity with sweep direction — the artifact that produced
  three wrong conclusions during surface mapping (§16.4b).
- **Pre-conditioning first**, cycling each location to *its own* campaign maximum
  (not a global 6 mm, so no location sees strain the experiment never repeats).

### 17.2 Run integrity — all green

| | |
|---|---|
| pokes committed | **1008 / 1008**, 0 duplicates, seq contiguous |
| segments | **1** — never interrupted, resume never needed |
| depth error | **0.003 ± 0.061 mm** (achieved vs commanded) |
| repeatability | **0.022 N** over 81 conditions observed 3× (1.2% of mean force) |
| creep during dwell | **none** — 0% of pokes exceed 0.05 N/s |
| force coverage | **0.22 – 5.72 N**, median 1.72 |

**The design goal was met.** Correlation of measured force with:

| | r | |
|---|---|---|
| commanded level | **+0.889** | levels work |
| position stiffness | **+0.235** | weak — the confound is broken |
| grid row | +0.053 | negligible |
| time into run | +0.042 | no drift |

### 17.3 ⚠ Forces ran 1.25–2.06× above target — sag is double-counted

The plan adds the impedance shortfall (`SAG_CONST_MM + F/3.8`) **and** `dip_to_depth`
closes the loop to achieve the commanded depth. The sag allowance therefore became
extra penetration. The overshoot is largest at low force, where the fixed 0.10 mm term
dominates a shallow press:

| planned | 0.20 | 0.48 | 0.76 | 1.04 | 1.32 | 1.60 | 1.88 | 2.16 | 2.44 N |
|---|---|---|---|---|---|---|---|---|---|
| achieved (median) | 0.41 | 0.69 | 1.04 | 1.40 | 1.77 | 2.12 | 2.47 | 2.83 | 3.05 N |

**Harmless here** — labels are *measured*, never inferred, and the range came out wider
than designed (which is better; the pilot's error concentrated above 3 N where it had
only 9% of its data). **For the next campaign pick one mechanism, not both.**

### 17.4 Baselines the camera model must beat — fix these before modelling

| model | RMSE | R² |
|---|---|---|
| predict the mean | 1.034 N | 0.000 |
| depth only | 0.747 N | 0.478 |
| **robot-only (depth × stiffness), grouped row-band CV** | **0.401 N** | **0.850** |
| irreducible noise floor (repeat sd) | **0.022 N** | 0.9996 |

⭐ **0.401 N is the bar.** That baseline is a robot that already knows where it pressed
and how deep; the camera gets neither and must infer both from the image. There is
enormous headroom below it — nothing in the data caps performance near the baseline.

### 17.5 Pre-condition run — the strap and Mullins answers (`precond_20260810_142103`)

**282 presses to 5.93 mm, forces to 5.67 N, zero 8 N exceedances.** The straps held at
full depth.

**No Mullins effect**: force ratio cycle 0 → cycle 2 = **1.007**. The material was
already settled, so pre-conditioning was not strictly necessary — worth knowing rather
than assuming.

⚠ **I had the softening backwards.** I predicted forces would land ~19% *short* because
the shallow ladder measured stiffness falling 0.632 → 0.368 N/mm. Measured, they land
~23% **high**: the material slightly *stiffens* with depth as the 3 mm indenter's
contact area grows. The map's stiffness is nonetheless validated — **r = +0.838**
between mapped stiffness and effective stiffness at campaign depth.

### 17.6 One frame per poke — `ml/build_frames.py`

A 2 s dwell at 40 ms gives 50 frames → 50,400. **That number is an illusion.** Measured
on this run: within-dwell force sd **0.1022 N** equals the tare-window sd **0.1022 N**.
All within-dwell variation is sensor noise; the 50 frames are 50 replicates of one
(image, force) pair. **Effective N = 1008.** Keeping all 50 costs 50× the storage and
invites a split that puts frames from one poke on both sides of a train/test boundary.

Extracted per poke, at the **temporal midpoint** of each phase window (furthest from
both the post-dip settling and the retract):

- `tare_frame` — arm still, out of contact: the undeformed reference
- `dwell_frame` — holding the indent
- label `force_n` = median(dwell Fz) − median(tare Fz), the **median over the whole 2 s**,
  which is the lowest-variance estimate of the constant the single frame depicts

Model input is `dwell − tare`, cancelling the static marker lattice. Both frames are
stored so the difference can be recomputed without re-decoding 68 GB.

**Validated on all 1008** (`frames.h5`, 73 MB — a 940× reduction from 68.6 GB, 0 empty
frames). A single crude scalar already carries the signal:

| | r |
|---|---|
| Σ\|dwell − tare\| vs **depth** | **+0.838** |
| max\|dwell − tare\| vs force | +0.510 |
| Σ\|dwell − tare\| vs force | **+0.478** |

| trivial model | RMSE |
|---|---|
| ridge on Σ\|diff\| alone | 0.908 N |
| ridge on Σ\|diff\| × `k_map` | **0.685 N** |
| *(robot-only baseline)* | *0.401 N* |
| *(predict the mean)* | *1.034 N* |

⭐ **The image encodes DEPTH more directly than FORCE** (0.838 vs 0.478) — unsurprising,
since the camera sees deformation and force = deformation × local stiffness. **To
recover force the model must localise the contact and apply the local stiffness.** This
is the empirical argument for position-preserving pooling: global average pooling would
discard exactly the information that converts what the camera sees into what we want.
Two scalars plus the stiffness map already reach **0.685 N**, so the CNN's real job is
to localise better than one number can — the gap from 0.685 down past 0.401 N is
precisely what spatial structure has to buy.

### 17.7 Clocks

`franka_seg00.csv` is on the **tactile** clock; `ft.csv` and `camera.raw` are on the
**workstation** clock. Offset 0.0301 → 0.0328 s (drift 2.7 ms), interpolated across the
run. **Camera device t = 0 is the first F/T sample** (§9) — both are opened by the GUI
on one clock. Phase tags make segmentation exact; nothing is thresholded.

### 17.8 ⚠ Two bugs found, both fixed

1. **`scp -r host:path/.` silently returns nothing.** Modern scp speaks SFTP and rejects
   a bare `.` component (`error: unexpected filename: .`). The pre-condition run
   therefore recorded `franka_ok: True` with no franka CSV fetched. Fixed to a remote
   glob (`path/*`) in `master_campaign.py`.
2. **Sag double-counting** — §17.3.

### 17.9 Standing rules this run added

- ⛔ **Do not ssh/scp to tactile while a campaign is running.** Pulling 44 MB and running
  two ledger queries during the first run coincided with a
  `communication_constraints_violation` reflex at `control_command_success_rate 0.76`.
  Even at 90% idle, scp plus Python parsing competes with a 1 kHz `SCHED_FIFO` control
  thread. Read progress from the operator's own terminal instead.
- **Load average is still the wrong readiness metric on tactile** (§15.2b): 10.79 with
  90% idle, runnable queue 0, zero D-state processes.

### 17.10 What is next

`ml/PRETRAINED_MODELS_ANALYSIS.md` §4 holds the ResNet-18 configuration; the campaign
does not change it, only the numbers it must beat (§17.4). The falsification ladder,
cheapest first — **rung 1 can kill the idea in seconds**:

1. Frozen ResNet-18 + GAP + ridge. ⚠ If this cannot beat predict-the-mean (1.034 N),
   ImageNet features contain nothing about our force — stop.
2. Frozen + `AdaptiveAvgPool2d((3,4))` — does keeping position help, in isolation?
3. + coordinate channels — does *explicit* position add beyond pooled position?
4. Truncate at `layer2` (0.68 M params vs 11.7 M), unfreeze, LP-FT.
5. Only on an upward trend: full fine-tune, 5 seeds.

⚠ **Grouped 3-fold by row band with a 1-row buffer, always.** ~12 spatially independent
sites (8 mm deformation half-width on a 26×31 mm block) against 3.06 mm row pitch —
random CV leaks and flatters. One poke's frames never split across folds. Report RMSE,
R² **and slope** (the pilot compressed to 0.576 where perfect is 1.000), broken out per
band and per level, 5 seeds, against the 0.401 N robot baseline.

---

## 18. ⭐ DESIGN REVERSAL — depth is the design variable, not force (2026-08-11)

`recordings/planb_3mm_20260811_150324/` — **896 pokes, 94 locations × 8 clean depths,
115 min.** Second batch, and a correction to §17's design.

### 18.1 The reversal, and who was right

§17 designed on **target force**, inverting the stiffness map. The user pushed back:

> *"if we indent 1 mm, I want to be as close as possible to 1 mm. Don't do force
> target. This is indentations cause the displacement we see in the event camera.
> The force is the thing we're estimating based on the indentations."*

**That is correct, and the evidence is our own batch 1:**

| | |
|---|---|
| force-targeting hit its targets to | **1.25–2.06×** (i.e. missed badly) |
| depth commands were hit to | **0.003 ± 0.061 mm** |

Force-targeting inverts a stiffness secant fitted over 0–0.6 N and extrapolates it to
2.4 N, *and* adds the impedance sag on top of `dip_to_depth`'s closed loop so the
allowance becomes extra penetration. It delivered **neither** clean depths nor the
intended forces. Command what you can hit; measure what you cannot.

The causal chain is **depth → deformation → (image, force)**. Depth is upstream of
both the model input and the label, which makes it the natural design variable.

### 18.2 ⚠ My confound argument was real but overstated

I justified force-targeting by saying a fixed depth ladder confounds force with
position. Measured both ways:

| design | location explains … of force variance |
|---|---|
| force-targeted (batch 1) | 5.5% (r = +0.235) |
| depth ladder (batch 2) | **25%** (r = +0.505) — predicted +0.479 by simulation |

Real, but **not leakage**. Stiffness variation is genuine physics of this sensor, and
a model predicting force *must* learn the stiffness field. The pilot's actual failure
was different in kind: depth was **constant** at 2 mm, so force was entirely
determined by location and the model could ignore deformation completely. A ladder
does not do that — force = stiffness × depth, so deformation still has to be read.

### 18.3 Results — the depths are clean

| commanded | achieved | measured force (median) |
|---|---|---|
| 0.50 mm | 0.611 ± 0.088 | 0.48 N |
| 1.00 | 1.023 ± 0.063 | 0.80 |
| **1.50** | **1.500 ± 0.040** | 1.22 |
| **2.00** | **1.992 ± 0.034** | 1.62 |
| **2.50** | **2.484 ± 0.029** | 2.04 |
| 3.00 | 2.969 ± 0.061 | 2.44 |
| 3.50 | 3.426 ± 0.141 | 2.70 |
| 4.00 | 3.878 ± 0.233 | 3.03 |

Overall depth error **−0.0145 ± 0.126 mm**; force **0.22 – 7.16 N** (median 1.66);
repeatability **0.033 N**; corr(force, time) +0.079.

⚠ **The closed loop is exact only in the middle.** 0.5 mm *overshoots* by 0.11 mm (it
cannot bite that shallow) and 4.0 mm *undershoots* by 0.12 mm with sd growing to
0.23 mm. 1.5–2.5 mm tracks to ±0.03 mm. A genuinely clean 0.5 mm needs a different
approach than the depth loop.

⚠ Peak force **7.16 N** exceeded the 6.58 N predicted from the batch-1 stiffness fit —
the stiffest location is stiffer under load than the map implies. Still inside the 8 N
ceiling, but **re-check before adding a 4.5 mm level**.

### 18.4 The two batches are complementary — do not naively pool them

| | batch 1 (force) | batch 2 (depth) |
|---|---|---|
| n | 1008 | 896 |
| force | 0.22–5.72 N | 0.22–7.16 N |
| depths | messy 0.34–5.94 | **exact, 8 levels** |
| corr(force, stiffness) | +0.235 | +0.505 |
| repeatability | 0.022 N | 0.033 N |

Batch 1 reaches **high force at compliant locations**, which a fixed depth ladder
physically cannot. Batch 2 has clean, reproducible depths. Different designs, so
⚠ **train on one and test on the other** rather than merging — and because they were
collected on different days, batch 2 is a genuine **session-independent** test set.

### 18.5 ⚠ Three bugs this exposed

1. **The replication block silently vanished in depth mode.** The tail iterated
   `len(levels)`, which is empty when `--depths-mm` is given, so the run would have
   done 752 pokes and quietly dropped the noise-floor measurement. Now iterates
   `n_pass`.
2. **`master_campaign.py` reported `MISSING Franka states` on a complete run.** It
   hardcoded a check for `franka.csv`; run-dir mode writes `franka_segNN.csv`. The
   220 MB log had arrived fine. Now globs the segments.
3. The deep-depth gate fired on `--max-depth-mm` (default 6.0) rather than on the
   deepest depth actually commanded (4.0), demanding `--ack-deep` for a run well
   inside the demonstrated limit.

### 18.6 Use `--depths-mm`

```bash
python3 master_campaign.py planb2 --remote-script campaign_planb.py \
    --remote-args --depths-mm 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0
```

Depths are commanded **verbatim — no sag term**, since `dip_to_depth` already closes
the loop on measured EE height. The force-targeted path is kept in the script but
marked reference-only.
