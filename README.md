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

See [`AGENTS.md`](AGENTS.md) for the full experimental context (sensor construction, camera and illumination setup, and open calibration questions) and [`requirements.txt`](requirements.txt) for the outstanding information-gathering checklist.

## Features

- **E_BTS_GUI**: a Qt Quick (Material style, green accent) desktop app that auto-connects to the EVK1 on launch and lets you open Camera, Circle Tracking, and Sequence Recording as panes from a persistent ribbon, laid out side by side / docked automatically depending on what's open.
- **Live event visualization** from the EVK1, monochrome rendering by default.
- **Marker/circle tracking** over the raw event stream: detects marker-dome circles from accumulated event windows and tracks their displacement from a user-defined baseline.
- **Sequence recording**, manually from the GUI's ribbon or externally-triggered by a robot-control script via a small file-based protocol (see [Sequence Recording](#sequence-recording)) -- both drive the same recorder against the one shared camera session.
- **RAW-to-CSV conversion** for human-readable inspection of CD (and optional trigger) events.
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
  camera_utils.h             # Camera source selection, RAW-to-CSV conversion, CD-event/frame-generator wiring
  camera_feed_utils.h        # Live-view color/monochrome rendering mode
  display_utils.h            # Letterboxed window/frame presentation for the standalone tracker CLI
  path_utils.h               # Timestamped/sanitized output path generation
  signal_utils.h             # Graceful shutdown on SIGINT/SIGTERM
  window_icon_utils.h        # Application window icon handling (standalone tracker CLI)
docker/
  build.sh
  entrypoint.sh
  run.sh
recordings/                 # Default host-side output folder (generated files, not committed)
DOCKER.md                   # Docker-specific usage
```

## Requirements

For native builds:

- Ubuntu 20.04
- CMake 3.16+
- C++17 compiler
- OpenEB / Metavision SDK 3.1.2
- OpenCV
- Qt5 (Quick, Quick Controls 2, Gui) for `E_BTS_GUI` -- `sudo apt install qtbase5-dev qtdeclarative5-dev qtquickcontrols2-5-dev qml-module-qtquick-controls2 qml-module-qtquick-layouts qml-module-qtquick2 qml-module-qtgraphicaleffects` (Qt6 is not used since Ubuntu 20.04's apt repos only carry Qt5)
- EVK1 camera for live capture

For Docker builds:

- Linux host
- Docker
- EVK1 connected over USB for capture
- X11/XWayland display available for live visualization

## Native Build

Configure from the repository root:

```bash
cmake -S src -B build
cmake --build build -j
```

This produces the following executables under `build/`:

| Executable | Purpose |
|---|---|
| `E_BTS_GUI` | Main application: ribbon-driven GUI over Camera, Circle Tracking, and Sequence Recording |
| `E_BTS_event_circle_tracker` | Standalone headless-friendly marker circle tracker, for debugging without the full GUI |
| `E_BTS_record_sequence` | CLI wrapper for externally-triggered sequence recording (see [Sequence Recording](#sequence-recording)) |
| `E_BTS_raw_to_csv` | RAW-to-CSV converter |

If an old `build/` directory was configured before the project moved to `src/`, recreate it or use a new build directory.

## E_BTS_GUI

```bash
./build/E_BTS_GUI
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

RAW recordings are saved with timestamped names such as `manual_20260703_153012.raw`, regardless of whether they were started from the ribbon or by an external script (see below).

## Sequence Recording

Sequence Recording is driven by one shared recorder (`SequenceRecordingController`, see `src/sequence_recording_controller.h`) with two ways to trigger it:

1. **Manually**, from `E_BTS_GUI`'s Camera-ribbon **Record Sequence** button.
2. **Externally**, from a companion Python (or other) script -- typically driving a robotic hand through a task -- that drops small command files into a `control/` directory:

   - `control/start.cmd`: created right before a task begins. Its contents (optional) become the output file's base name, e.g. writing `grasp_01` produces `recordings/grasp_01_<timestamp>.raw`.
   - `control/stop.cmd`: created once the task finishes; recording stops immediately.
   - `control/quit.cmd`: created to end the watch session (closes the pane in `E_BTS_GUI`; exits `E_BTS_record_sequence` entirely when run standalone).

   Each command file is deleted once acted on, so the same three files are reused for every sequence in a session. See the header comment in `src/sequence_recording_controller.h` for a full Python-side example.

Both triggers share one recording state, so a manual recording and an externally-triggered one can never run at the same time -- whichever starts first wins, and the other is logged as a no-op.

To drive recordings from a script without the GUI running, use the standalone CLI:

```bash
./build/E_BTS_record_sequence [camera_serial]
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

## RAW to CSV

Convert a RAW file:

```bash
./build/E_BTS_raw_to_csv input.raw
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

## Docker

Build the image:

```bash
./docker/build.sh
```

Run the GUI (requires an X11/XWayland display, same as any GLFW-based tool the image ran before):

```bash
./docker/run.sh viewer
```

Run the sequence recorder:

```bash
./docker/run.sh record
```

Convert RAW to CSV:

```bash
./docker/run.sh csv recording.raw
```

See [DOCKER.md](DOCKER.md) for full Docker usage, command aliases, mounts, and manual `docker run` examples.

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
