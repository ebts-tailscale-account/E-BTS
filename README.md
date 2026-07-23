```ascii
░▒▓████████▓▒░▒▓███████▓▒░▒▓████████▓▒░▒▓███████▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░  ░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░  ░▒▓█▓▒░        
░▒▓██████▓▒░ ░▒▓███████▓▒░  ░▒▓█▓▒░   ░▒▓██████▓▒░  
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░         ░▒▓█▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░         ░▒▓█▓▒░ 
░▒▓████████▓▒░▒▓███████▓▒░  ░▒▓█▓▒░  ░▒▓███████▓▒░  
```

## Overview

E-BTS is a C++17 toolkit for working with an event-based EVK1 camera through the OpenEB / Metavision SDK. It is the experimental foundation for a tactile sensor built from a multilayer silicone sheet with marker domes, developed at Tactile Lab, Nazarbayev University. The current milestone is estimating normal force and contact location from the marker displacement visible in the event stream; tangential force estimation and simultaneous slow/fast adaptive sensing are longer-term goals, not part of this codebase yet.

Data collection is a two-machine setup: a Qt/C++ GUI (`E_BTS_GUI`) drives the EVK1 camera and marker tracking, while a UR5 robotic arm (`UR5/`) pokes the silicone sheet and logs force readings, triggering camera recordings over a small file-based protocol so the two stay in sync without a direct network link.

See [`AGENTS.md`](AGENTS.md) for the full experimental context (sensor construction, camera and illumination setup, and open calibration questions) and [`requirements.txt`](requirements.txt) for the outstanding information-gathering checklist.

## Features

- **E_BTS_GUI**: a Qt Quick (Material style, green accent) desktop app that auto-connects to the EVK1 on launch and lets you open Camera, Circle Tracking, and Sequence Recording as panes from a persistent ribbon, laid out side by side / docked automatically depending on what's open.
- **Live event visualization** from the EVK1, monochrome rendering by default.
- **Marker/circle tracking** over the raw event stream: detects marker-dome circles from accumulated event windows and tracks their displacement from a user-defined baseline.
- **Sequence recording**, manually from the GUI's ribbon or externally-triggered by a robot-control script via a small file-based protocol (see [Manipulator (UR5) Connection](#manipulator-ur5-connection)) -- both drive the same recorder against the one shared camera session.
- **Export**, from the ribbon: batch-convert one or more `.raw` recordings to CSV (CD events) or `.mp4` video, with an optional black & white rendering mode, without leaving the GUI.
- **RAW-to-CSV conversion** (also available as a standalone CLI) for human-readable inspection of CD (and optional trigger) events.
- **UR5 robot control**, in `UR5/`: grid-poke data collection against the silicone sheet, synchronized with camera recording via the same file protocol.
- **Docker support** for reproducible builds and runs on Linux hosts.

## Repository Layout

```text
src/
  CMakeLists.txt                  # Build configuration for all executables
  combined_main.cpp               # E_BTS_GUI's Qt application entry point
  camera.h / camera.cpp           # CameraSource: live-view CDFrameGenerator branch of the shared camera session
  circle_tracking.h / .cpp        # CircleTrackingSource: detector/tracker branch of the shared camera session
  circle_tracker_cli_main.cpp     # Standalone headless-friendly tracker (E_BTS_event_circle_tracker)
  sequence_recording_controller.h # SequenceRecordingController: shared manual + start.cmd/stop.cmd/quit.cmd recorder
  record_sequence.cpp             # CLI wrapper around SequenceRecordingController (E_BTS_record_sequence)
  raw_to_csv.cpp                  # RAW-to-CSV converter (E_BTS_raw_to_csv)
  gui/camera_session_worker.h/.cpp # Owns the one shared Metavision::Camera; QThread-resident orchestrator
  gui/export_worker.h/.cpp        # Runs RAW-to-CSV/video conversions requested from the ribbon's Export dialog
  gui/frame_view.h/.cpp           # QML FrameView item: letterboxes a QImage into its bounding rect
  qml/                            # Main.qml, Ribbon.qml, panes, dialogs, Theme.qml
  qml.qrc                         # Qt resource bundle for qml/ and icon.png
  circle_tracker_config.h    # Tuning/physical constants for the circle tracker
  event_window_buffer.h      # Timestamp-windowed accumulation of CD events into occupied-pixel maps
  circle_detector.h          # Per-window circle detection (contour-based and map-guided)
  circle_lattice.h           # Baseline marker grid (lattice) inference
  temporal_circle_tracker.h  # Baseline capture and cross-window marker tracking
  tracking_render_utils.h    # Tracker frame rendering and standalone-CLI keyboard handling
  console_input_utils.h      # Shared interactive numeric-prompt parsing
  camera_utils.h             # Camera source selection, RAW-to-CSV/video conversion, CD-event/frame-generator wiring
  camera_feed_utils.h        # Live-view color/monochrome rendering mode
  display_utils.h            # Letterboxed window/frame presentation for the standalone tracker CLI
  path_utils.h               # Timestamped/sanitized output path generation
  signal_utils.h             # Graceful shutdown on SIGINT/SIGTERM
  window_icon_utils.h        # Application window icon handling (standalone tracker CLI)
UR5/
  poke_grid.py                # Main data-collection script: grid poke + force logging + camera-recording trigger
  pose_utils.py                # Shared geometry helpers + motion constants
  start_pose.py / home.py      # Safe-pose bookends for a run
  generate_path.py / run_motion.py  # Earlier practice pipeline, no recording/force integration
  pyForceDAQ/                  # Vendored WITTENSTEIN/ATI force-sensor DAQ library
control/                    # start.cmd/stop.cmd/quit.cmd drop folder for externally-triggered recording
docker/
  build.sh
  entrypoint.sh
  run.sh
recordings/                 # Default host-side output folder (generated files, not committed)
DOCKER.md                   # Docker-specific usage
```

## Requirements

For Docker builds:

- Linux host
- Docker
- EVK1 connected over USB for capture
- X11/XWayland display available for live visualization

For native builds:

- Ubuntu 20.04
- CMake 3.16+
- C++17 compiler
- OpenEB / Metavision SDK 3.1.2
- OpenCV (core, imgcodecs, imgproc, videoio)
- Qt5 (Quick, Quick Controls 2, Gui) for `E_BTS_GUI` -- `sudo apt install qtbase5-dev qtdeclarative5-dev qtquickcontrols2-5-dev qml-module-qtquick-controls2 qml-module-qtquick-layouts qml-module-qtquick2 qml-module-qtgraphicaleffects qml-module-qt-labs-platform` (Qt6 is not used since Ubuntu 20.04's apt repos only carry Qt5)
- EVK1 camera for live capture

## Docker

Docker is the recommended way to build and run E-BTS -- it builds OpenEB/Metavision SDK 3.1.2 from source, installs it, then builds the project's executables, so you don't need to match the exact Ubuntu 20.04 + SDK toolchain by hand.

Build the image:

```bash
./docker/build.sh
```

Run the GUI (requires an X11/XWayland display):

```bash
./docker/run.sh viewer
```

Run the sequence recorder standalone (for external, robot-triggered recording without the GUI):

```bash
./docker/run.sh record
```

Convert RAW to CSV:

```bash
./docker/run.sh csv recording.raw
```

`docker/run.sh` mounts the repo at `/data`, runs from `/data/recordings` so relative output paths land in `recordings/`, passes USB devices through for the EVK1, and forwards the X11 display when `DISPLAY` is set. See [DOCKER.md](DOCKER.md) for full usage, command aliases, mounts, and manual `docker run` examples.

## E_BTS_GUI

```bash
./build/E_BTS_GUI          # native build
./docker/run.sh viewer     # Docker (default command)
```

On launch, `E_BTS_GUI` immediately tries to connect to the first available EVK1. If none is found, it shows a banner explaining that and a **Check Connection** button to retry once the camera is plugged in.

Once connected, the ribbon's **+ Add Source** button opens a menu to toggle any of the three sources on/off:

- **Camera**: the live event feed.
- **Circle Tracking**: marker detection/tracking over the same event stream.
- **Sequence Recording**: a console-only pane (no camera feed) showing what the shared recorder does.

The EVK1 only exposes one open camera handle, so Camera and Circle Tracking are two views of one shared live session rather than independent connections -- opening both lays them out side by side (Camera left, Circle Tracking right); opening Sequence Recording too docks its console along the bottom. Any single source alone fills the whole window.

The ribbon also gains contextual buttons depending on what's open:

- Camera open -> **Record Sequence** (manual RAW recording) and **Accumulation Time**.
- Circle Tracking open -> **Detection %** (minimum marker occupied-pixel density) and, in that pane's own header, **Rebuild Baseline**.

RAW recordings are saved with timestamped names such as `manual_20260703_153012.raw`, regardless of whether they were started from the ribbon or by an external script (see [Manipulator (UR5) Connection](#manipulator-ur5-connection)).

### Export

The ribbon's **Export** button is always available, independent of whether a camera source is open -- it works purely off files already on disk. It opens a dialog where you can:

1. Pick one or more `.raw` files from disk.
2. Choose an output format: **CSV** (CD events) or **Video** (`.mp4`).
3. For video, optionally check **Black and White** to render with the same monochrome palette as the live Camera pane's feed-mode toggle, instead of the default polarity colors.
4. Click **Start Export** to convert every selected file, with per-file progress and a log pane.

Each output is written next to its source `.raw`, with the same base name: `recording.raw` -> `recording.csv` or `recording.mp4`. A black & white video export gets a `_BW` suffix (`recording_BW.mp4`) so it never collides with a standard-color export of the same recording. Video export reuses the live Camera pane's `CDFrameGenerator`/accumulation-time rendering pipeline, replayed as fast as possible rather than in real time. This GUI path does not export trigger events to CSV -- use the [RAW to CSV](#raw-to-csv) CLI directly if you need those.

## Manipulator (UR5) Connection

A UR5 robotic arm pokes the silicone sheet and drives `E_BTS`'s camera recording, without any direct network link between the two -- they stay in sync purely through a small file-based command protocol.

**Protocol.** `SequenceRecordingController` (`src/sequence_recording_controller.h`) watches a `control/` directory at the repo root for three command files, each deleted once acted on so the same three names are reused for every recording in a session:

- `control/start.cmd`: created right before a task begins. Its contents (optional) become the output file's base name, e.g. writing `grasp_01` produces `recordings/grasp_01_<timestamp>.raw`.
- `control/stop.cmd`: created once the task finishes; recording stops immediately.
- `control/quit.cmd`: created to end the watch session (closes the Sequence Recording pane in `E_BTS_GUI`; exits `E_BTS_record_sequence` entirely when run standalone).

Either `E_BTS_GUI` (with Sequence Recording open) or the standalone `E_BTS_record_sequence` CLI must already be running and watching `control/` before the robot script starts -- neither the GUI nor the CLI starts that watcher for you, and it isn't started by the robot side either. Manual (ribbon) and externally-triggered recordings share one recording state, so only one can run at a time -- whichever starts first wins.

**Robot side (`UR5/`).** Copied in from a separate `motion_planning` repo (not a submodule) so a robot-side session has everything needed to drive poke tests, independent of the camera-side code -- it knows nothing about the EVK1 beyond triggering recordings via the protocol above.

- `poke_grid.py` is the actual data-collection script: for every point in a sparse grid swept across the sheet, it writes `start.cmd`, sends a URScript program over the UR secondary interface to hover/press/dwell/retract, polls a WITTENSTEIN force sensor (via the vendored `pyForceDAQ/` library) for the peak `Fz` during the dwell, writes `stop.cmd`, and appends a `poke, row, col, x_mm, y_mm, force_N` row to a master results CSV -- all under `~/E-BTS/recordings/` (hardcoded relative to the home directory, independent of where `UR5/` itself is checked out).
- `pose_utils.py` holds the shared geometry helpers and motion constants (tool offset, fixed tool orientation, accel/vel, approach height, contact depth) that the other scripts import.
- `start_pose.py` / `home.py` bookend a run (safe pose before, home position after); `stop.py` is an undocumented no-op (empty file, not a real e-stop) and `end.py` is unused reference code.
- `generate_path.py` / `run_motion.py` are an earlier practice pipeline with no recording or force-sensing integration -- a fire-and-forget demo, superseded by `poke_grid.py`.

Run order:

```bash
python UR5/start_pose.py     # move to safe start pose
python UR5/poke_grid.py      # prompts for 'sim' or 'real', then runs the sweep
python UR5/home.py           # return home when done
```

**Known limitations, by design:** the surface height used by `poke_grid.py` is a manually calibrated constant, pressed to open-loop (no live contact detection); there is no closed-loop motion-complete signal from the UR controller, so the script sleeps fixed placeholder durations between phases; the grid is intentionally sparse (20 points) to validate the pipeline before a longer run; and none of this has been exercised against real hardware yet. Re-check `Z_SURFACE_M` and the other calibration constants at the top of `poke_grid.py` against the physical setup before running -- a stale calibration plus an open-loop press is exactly the kind of thing that could over-press the sheet.

## RAW to CSV

Convert a RAW file:

```bash
./build/E_BTS_raw_to_csv input.raw          # native
./docker/run.sh csv recording.raw           # Docker
```

Custom output paths:

```bash
./build/E_BTS_raw_to_csv input.raw cd.csv
./build/E_BTS_raw_to_csv input.raw cd.csv triggers.csv
```

CD CSV columns:

```text
x,y,polarity,timestamp_us
```

## Marker Circle Tracking (standalone CLI)

For debugging the detector/tracker without building or running the full GUI:

```bash
./build/E_BTS_event_circle_tracker [camera_serial_or_raw_file]
```

Prompts for the accumulation time and the minimum marker circle pixel density on startup. In the tracking window:

- `B`: capture a new baseline from the next detected circles (apply this with no load on the sensor).
- `Esc` or `Q`: quit.

The first non-empty detection window becomes the baseline automatically if one is not captured manually.

## Native Build

Configure from the repository root:

```bash
cmake -S src -B build
cmake --build build -j
```

This produces the following executables under `build/`:

| Executable | Purpose |
|---|---|
| `E_BTS_GUI` | Main application: ribbon-driven GUI over Camera, Circle Tracking, Sequence Recording, and Export |
| `E_BTS_event_circle_tracker` | Standalone headless-friendly marker circle tracker, for debugging without the full GUI |
| `E_BTS_record_sequence` | CLI wrapper for externally-triggered sequence recording (see [Manipulator (UR5) Connection](#manipulator-ur5-connection)) |
| `E_BTS_raw_to_csv` | RAW-to-CSV converter |

If an old `build/` directory was configured before the project moved to `src/`, recreate it or use a new build directory.

## Generated Files

The repository ignores generated data and build outputs, including:

- `build/`
- `cmake-build-*/`
- `*.raw`
- `*.csv`
- `*.avi`
- `*.mp4`
- `*.log`

Keep large camera recordings outside Git unless there is a specific reason to version them.

## People Working on the Project

| Name                  | Social Links              |
|-----------------------|---------------------------|
| Zaki Al-Farabi        | [GitHub]() · [LinkedIn]() |
| Gleb Tarkinskiy       | [GitHub]() · [LinkedIn]() |
| Airis Kairolla        | [GitHub]() · [LinkedIn]() |
| Dinmukhammed Mukashev | [GitHub]() · [LinkedIn]() |
| Yelenov Amir          | [GitHub]() · [LinkedIn]() |
| Zhanat Kappassov      | [GitHub]() · [LinkedIn]() |

## Acknowledgements

Developed at Tactile Lab, Nazarbayev University.

This project uses event-based vision tools from the Metavision / OpenEB ecosystem.
