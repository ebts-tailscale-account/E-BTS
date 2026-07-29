# E-BTS — Software README

Synchronized multi-modal data collection for the event-based tactile sensor (E-BTS),
Tactile Lab, Nazarbayev University. This document covers the **software**: the
sensors, sampling rates, connections, the programs, and how to run a full
experiment end to end. A separate **physical-setup README** covers the hardware,
wiring, illumination, and mechanical assembly.

> For the older C++ GUI feature overview (camera / circle-tracking / export panes)
> see [`README.md`](README.md). This file is the authoritative guide to the
> synchronized recording pipeline and the Force/Torque + Franka integration.

---

## 1. What the system does

For each experiment, three sensor streams are recorded **simultaneously** and
aligned on a common UNIX time base, then segmented per indentation:

1. **Event camera** watching the silicone marker sheet from below.
2. **6-axis force/torque sensor** (ground-truth contact force).
3. **Franka Panda** robot state (pose, joints, its own force estimate).

A single command (`master.py`) triggers all three, and a post-processing step
turns the raw streams into aligned, per-indentation data plus a PDF report.

---

## 2. Machines & topology

| Role | Host | Runs |
|------|------|------|
| **Workstation** | `skymario-ubuntu` | Event camera + F/T sensor (both USB), the GUI, and all orchestration/analysis scripts (`master.py`, `postprocess.py`, `visualize.py`) |
| **Robot control PC** | `tactile` @ `100.93.60.35` (Tailscale) | Franka Panda + ROS Noetic + MoveIt. Reached over **Tailscale SSH** |

The camera and F/T sensor are on the **same machine** (one clock → sub-ms sync).
The Franka is on a second machine; the workstation↔tactile clock offset is
measured at run time and recorded (see §5).

```
  master.py (workstation)
     ├─ control/start.cmd ──► E_BTS_GUI  ──►  camera .raw  +  F/T ft.csv     (workstation clock)
     ├─ ssh tactile ───────► franka_grid_logger.py ──► franka.csv            (tactile clock)
     └─ control/stop.cmd  ──► (recording stops)
                    ▼
             postprocess.py ──►  output/<run>/  (organized, aligned, per-poke)
                    ▼
             visualize.py   ──►  output/<run>/report.pdf
```

---

## 3. The three sensor systems

### 3.1 Event camera (Prophesee EVK1)
- **Interface:** USB, via **Metavision / OpenEB SDK 3.1.2**.
- **Resolution:** 640 × 480.
- **Data:** change-detection (CD) events `(x, y, polarity, t)` — **event-based**, no
  fixed frame rate; `t` is in **microseconds** on the camera's own device clock,
  counted from the start of the recording.
- **Owned by:** `E_BTS_GUI` (one shared `Metavision::Camera` session). Recordings
  are `.raw`; convert to CSV with `E_BTS_raw_to_csv` (see §7).

### 3.2 Force/torque sensor (Resense / Wittenstein HEX21)
- **Interface:** USB CDC virtual COM port — STMicroelectronics `0483:5740`,
  appears as `/dev/ttyACM*`. Opened at **2,000,000 baud**, with **DTR/RTS asserted**
  (some firmware won't stream otherwise). **Single-owner port** — the vendor
  ForceTorqueExplorer GUI and `read_ft.py` must be closed before the E_BTS GUI can read it.
- **Sampling rate:** **1 kHz**.
- **Packet:** 28 bytes = **7 little-endian `float32`**: `Fx Fy Fz` (**N**),
  `Mx My Mz` (**mNm**), `Temp` (**°C**). No framing bytes → the reader syncs by
  sliding until the temperature reads plausible, and self-heals on a byte slip.
- **Calibration:** electronics **DIP switch 6 = ON** → the box applies the sensor's
  factory calibration matrix and streams *calibrated* F/T. **Do not** re-apply the
  matrix (`wittenstein/.../Matrix__HEX21.txt`, copied to `calibration/`) in software —
  that would double-calibrate. Zero the small resting offset with the sensor's
  hardware **Tara** button or in post-processing (baseline subtraction).
- **Owned by:** `E_BTS_GUI`'s Force/Torque source (`src/gui/wittenstein_worker.*`,
  POSIX `termios`, no extra dependencies). Live graph via `src/gui/ft_graph_view.*`.

### 3.3 Franka Panda (via ROS)
- **Interface:** ROS Noetic on `tactile`; state from `/franka_state_controller/franka_states`.
- **Publish rate:** **1000 Hz** (set in
  `~/ws_franka/src/franka_ros/franka_control/config/default_controllers.yaml`,
  `publish_rate: 1000`; default was 30).
- **Logged fields:** `O_T_EE` (end-effector pose → contact location + velocity by
  differencing), `O_F_ext_hat_K` (estimated external wrench), `q` (7 joint angles),
  collision flags, `robot_mode`.
- **Payload:** the mounted tool (F/T sensor + plate + indenter, ~33.2 g, CoM 17.5 mm
  along +z) is configured via Desk / `set_load` so gravity is compensated. **Note:**
  the Franka's external-wrench estimate still carries an inherent, pose-dependent
  bias of ~±1–3 N — it is a *coarse cross-check*, **not** the ground-truth force. The
  **Wittenstein is the calibrated force channel.**

---

## 4. Connections at a glance

| System | Physical link | Rate | Timestamp | Software |
|--------|---------------|------|-----------|----------|
| Event camera | USB (EVK1) | event-based, µs | device clock (µs from record start) | `E_BTS_GUI` (Metavision) |
| F/T sensor | USB CDC `/dev/ttyACM*`, 2 Mbaud, DIP6=ON | 1 kHz | host UNIX (on write) | `E_BTS_GUI` Force/Torque source |
| Franka | ROS over Tailscale SSH to `tactile` | 1 kHz | host UNIX + ROS time | `franka_grid_logger.py` |
| Trigger/illumination MCU | Arduino/Teensy D2 & D4 | 25 Hz two-phase | — | `arduino/two_phase_driver/` |

---

## 5. Time synchronization

Every stream carries a **UNIX wall-clock** timestamp; alignment is done in
post-processing, not by simultaneous start.

- **Camera + F/T** are both on the **workstation clock** → already tightly aligned
  (sub-ms). The camera's device clock (µs from record start) is anchored to the F/T
  recording start (both fire from the same recorder event).
- **Franka** runs on `tactile`'s clock. `master.py` measures the
  **workstation↔tactile offset** with NTP-style SSH round-trips and records it in
  `metadata.json` (`tactile_minus_workstation_offset_s`). `postprocess.py` /
  `visualize.py` shift the Franka timeline by that offset (typically ~10 ms).
- The robot is *slow* (cm/s), so a few ms of cross-machine skew maps to sub-micron
  position error; the fast pair (F/T ↔ events) is same-clock, so it's the tight one.
- For sub-ms hardware sync you'd use the EVK1's external-trigger channel (not
  required for the current ~1 ms target).

---

## 6. The GUI — `E_BTS_GUI`

Qt5 Quick / C++17 app (Metavision SDK 3.1.2, OpenCV, Qt5 Quick/QuickControls2/Gui).

**Build** (from the repo root):
```bash
cmake -S src -B build
cmake --build build -j
```
(or `./docker/build.sh` — see [`DOCKER.md`](DOCKER.md).)

**Run from the repo root** (so `control/` and `recordings/` land where the scripts expect):
```bash
cd ~/E-BTS && ./build/E_BTS_GUI
```

**Sources** (ribbon → **+ Add Source**): Camera, Circle Tracking, Sequence
Recording, **Force/Torque**, and Export. For a synchronized run, open **Force/Torque**
(starts the serial stream + live graph) and **Sequence Recording** (starts the
`control/` watcher).

**Recording** is driven by a file protocol — `control/start.cmd` (its contents = the
base name) and `control/stop.cmd`. A single `start.cmd` **fans out** to the camera
`.raw` **and** the F/T `_ft.csv` in lockstep (same base name + timestamp).

---

## 7. Programs & commands

All paths are relative to the repo root on the workstation unless noted.

| Program | Where | Purpose |
|---------|-------|---------|
| `master.py` | workstation | Orchestrate one synchronized run: measure clock offset, `start.cmd`, run the Franka sweep on `tactile` over SSH, `stop.cmd`, pull the Franka CSV back, write metadata |
| `franka/franka_grid_logger.py` | **tactile** (auto-run by `master.py`) | Snake-grid indentation sweep + `franka_states` logging |
| `franka/record_franka_force.py` | tactile | Quick N-second dump of the Franka external wrench (offset check; state-only, no motion) |
| `postprocess.py` | workstation | Organize a run into `output/<run>/`, align clocks, zero baselines, segment into indentations, slice per-poke event windows |
| `visualize.py` | workstation | Build `report.pdf` (force + torque + event-polarity, shared time axis) and optional zoom PDFs |
| `build/E_BTS_raw_to_csv` | workstation | Convert `camera.raw` → flat `x,y,polarity,timestamp_us` CSV |

**Typical commands**
```bash
# 1. one synchronized run
python3 master.py my_run

# 2. organize + align + segment (+ slice events)
python3 postprocess.py my_run                 # events -> per-poke .npy
python3 postprocess.py my_run --no-events      # numeric only (fast)
python3 postprocess.py my_run --events-npy     # compact .npy slices (default writes CSV per poke)

# 3. PDF report (+ optional zoom to pokes)
python3 visualize.py my_run
python3 visualize.py my_run --zoom 1 2 --margin 5

# 4. (optional) full event CSV — LARGE (~5x the .raw)
./build/E_BTS_raw_to_csv output/my_run/camera.raw output/my_run/camera_events.csv
```

**Prerequisites for `master.py`:**
- GUI running **from the repo root** with **Force/Torque** + **Sequence Recording** panes open.
- On `tactile`: the clean MoveIt bringup up, FCI enabled, e-stop in hand:
  ```bash
  source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
  roslaunch panda_moveit_config franka_control.launch robot_ip:=10.1.196.5 load_gripper:=false
  ```
- **Passwordless SSH** to `tactile` (`ssh-copy-id tactile@100.93.60.35`).
- `franka/franka_grid_logger.py` present on `tactile` at `~/E-BTS/franka_grid_logger.py`.

---

## 8. Full run procedure

1. **Workstation:** launch `E_BTS_GUI` from `~/E-BTS`; open the **Force/Torque** and
   **Sequence Recording** sources. (Vendor F/T GUI must be closed.)
2. **tactile:** bring up the clean MoveIt stack (above); confirm
   `Ready to take commands for planning group panda_arm`.
3. **Workstation:** `python3 master.py <run>` — homes, sweeps the grid with a 2 mm
   indentation at each point while recording all three streams, then stops.
4. `python3 postprocess.py <run>` — writes `output/<run>/` with aligned + segmented data.
5. `python3 visualize.py <run>` — writes `output/<run>/report.pdf`.

---

## 9. Output layout — `output/<run>/`

```
output/<run>/
  camera.raw            # event stream (Metavision RAW)
  camera.bias           # camera bias/config sidecar
  camera_events.csv     # (optional) full x,y,polarity,timestamp_us  — LARGE
  ft.csv                # full Wittenstein stream, 1 kHz
  franka.csv            # full Franka state series
  metadata.json         # clock offset, T0, run info
  event_rate.csv        # +/- polarity counts per 50 ms bin (for the report)
  report.pdf            # force + torque + event-polarity, per-poke curves
  report_zoom_poke*.pdf # optional zoom(s)
  pokes/
    poke01_ft.csv …     # per-indentation force curve (zeroed)
    poke01_events.npy … # per-indentation event slice (structured x,y,p,t)
    pokes_summary.csv   # contact xy, peak/mean force, timing, event counts
```

> **Ignored by git** (regenerable / large): `output/`, `recordings/`, `*.raw`,
> `*.csv`, `*.bias`, `*.npy`. Only code, configs, and this doc are versioned.

---

## 10. Data formats

- **`ft.csv`** — `unix_time_s, Fx_N, Fy_N, Fz_N, Mx_mNm, My_mNm, Mz_mNm, Temp_C`
- **`franka.csv`** — `unix_time_s, robot_time_s, q0..q6, ee_x, ee_y, ee_z, Fx_ext_N, Fy_ext_N, Fz_ext_N, Tx_ext_Nm, Ty_ext_Nm, Tz_ext_Nm, collided, robot_mode`
- **`camera_events.csv`** — `x, y, polarity, timestamp_us` (device µs)
- **`event_rate.csv`** — `t_center_s, n_pos, n_neg`
- **`pokes/poke*_ft.csv`** — `t_rel_s, unix_time_s, Fx_N … Mz_mNm` (baseline-zeroed)
- **`pokes/poke*_events.npy`** — NumPy structured array `('x','y','p','t')`
- **`pokes/pokes_summary.csv`** — `poke, col, row, t_start_ws, t_end_ws, dur_s, ee_x, ee_y, indent_mm, peak_Fz_ft_N, mean_Fz_ft_N, peak_Fz_ext_N, cam_t0_us, cam_t1_us, n_events`
- **`metadata.json`** — `run_name, t0_unix_s, tactile_minus_workstation_offset_s, offset_rtt_s, franka_ok, franka_csv, note`

---

## 11. The indentation sweep (grid) — `franka/franka_grid_logger.py`

A boustrophedon ("snake") sweep over a flat silicone patch. Key constants at the
top of the file:

- `start_joints` — the **level + just-touching** bottom-left reference pose
  (flange leveled so its face is parallel to the ground; captured with the jog tool).
- `home_joints` — the standard Panda "ready" pose.
- `N_ROWS`, `N_COLS`, `V_SPACING`, `H_SPACING` — grid dimensions/spacing
  (base-frame `VERT`/`RIGHT` directions).
- `INDENT` (2 mm dip), `HOVER` (5 mm safe travel height), `DWELL` (hold time).

**Motion pattern per point:** move at hover height → dip straight down by `INDENT`
→ dwell → retract to hover. Moves are slow (5–10 % velocity scaling).

**Notes / calibration cautions:**
- The surface is assumed **flat and level** relative to the robot base — one touch
  reference is reused for the whole grid. If a far edge shows near-zero contact
  force, the grid extends off the patch or the surface is tilted; shrink the span or
  re-level.
- Baseline zeroing uses the **first 5 s** (no-load lead-in) of each recording.

---

## 12. Trigger / illumination MCU — `arduino/two_phase_driver/`

An Arduino/Teensy sketch (`elapsedMillis`) that drives two output pins as a
**complementary two-phase square wave**: pin **D2** and pin **D4** each toggle every
**20 ms**, so each is high for 20 ms out of every 40 ms → **25 Hz, 50 % duty,
180° out of phase**. Serial at **115200** baud (no data sent). Its physical role
(illumination phases / strobe / sync) is documented in the **physical-setup README**.

---

## 13. Dependencies

**Workstation**
- Metavision / OpenEB **SDK 3.1.2**, OpenCV, **Qt5** (Quick, QuickControls2, Gui) — for `E_BTS_GUI`
- Python 3: `numpy`, `matplotlib`, `metavision_core` (event slicing/binning), `pyserial` (standalone F/T reader only)
- `pdftoppm` (poppler-utils) — optional, for PDF→PNG previews

**tactile**
- ROS **Noetic**, `franka_ros` + MoveIt (`~/ws_franka`), `moveit_commander`, `franka_msgs`
- Passwordless SSH from the workstation

Ubuntu 20.04 on both machines.
