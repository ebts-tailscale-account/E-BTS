# Running E-BTS with Docker

This image builds OpenEB/Metavision SDK 3.1.2 from source, installs it, then builds the project's executables:

- `E_BTS_GUI`: the main Qt Quick application (ribbon, Camera / Circle Tracking / Sequence Recording panes).
- `E_BTS_event_circle_tracker`: standalone headless-friendly marker tracker, for debugging without the GUI.
- `E_BTS_record_sequence`: CLI wrapper for externally-triggered sequence recording.
- `E_BTS_raw_to_csv`: RAW-to-CSV converter.

Generated files are written under `recordings/` on the host by default.

## Project Layout

C++ sources and the CMake project now live in `src/`:

```text
src/CMakeLists.txt
src/combined_main.cpp
src/camera.h / camera.cpp
src/circle_tracking.h / circle_tracking.cpp
src/circle_tracker_cli_main.cpp
src/sequence_recording_controller.h
src/record_sequence.cpp
src/raw_to_csv.cpp
src/gui/
src/qml/
```

Local CMake builds should use `src/` as the source directory:

```bash
cmake -S src -B build
cmake --build build -j
```

If `build/` was configured before this layout change, recreate it first or use a new build directory.

## Build

Use the project helper:

```bash
./docker/build.sh
```

It builds the image and runs a small sanity check that all expected executables exist.

Optional overrides:

```bash
IMAGE=my-e-bts:dev ./docker/build.sh
OPENEB_VERSION=3.1.2 ./docker/build.sh
```

Equivalent manual build:

```bash
docker build --pull --build-arg OPENEB_VERSION=3.1.2 -t e-bts:openeb-3.1.2 .
```

## Run Helper

Use `docker/run.sh` for normal use:

```bash
./docker/run.sh [command] [args...]
```

The helper:

- Mounts the repo at `/data`.
- Sets `E_BTS_RECORDINGS_DIR=/data/recordings`.
- Runs from `/data/recordings`, so relative output paths appear in `recordings/`.
- Passes USB devices through for the EVK1.
- Forwards X11 display when `DISPLAY` is set.

## E_BTS_GUI

Default command:

```bash
./docker/run.sh
```

Equivalent explicit command:

```bash
./docker/run.sh viewer
```

`E_BTS_GUI` tries to connect to the EVK1 automatically on launch; use the on-screen "Check Connection" button if it starts before the camera is plugged in. From there, use the ribbon's "+ Add Source" menu to open Camera / Circle Tracking / Sequence Recording panes -- see the main [README](README.md#e_bts_gui) for the full layout and control rundown.

RAW recordings are saved in `recordings/` with names like:

```text
manual_20260703_153012.raw
```

## Sequence Recorder

`E_BTS_record_sequence` is driven by a companion script (typically controlling a robotic hand) that drops `start.cmd`/`stop.cmd`/`quit.cmd` files into `control/` -- see [README: Sequence Recording](README.md#sequence-recording) for the full protocol. Run it standalone (without the GUI) with:

```bash
./docker/run.sh record
```

With a specific camera serial:

```bash
./docker/run.sh record <camera_serial>
```

Output `.raw` files are written to `recordings/`.

## RAW to CSV

Convert a RAW file in `recordings/`:

```bash
./docker/run.sh csv recording.raw
```

Custom output path:

```bash
./docker/run.sh csv recording.raw /path/to/data.csv
```

Write trigger events too:

```bash
./docker/run.sh csv recording.raw cd.csv triggers.csv
```

CSV columns for CD events:

```text
x,y,polarity,timestamp_us
```

## Entrypoint Commands

`docker/entrypoint.sh` accepts these command aliases:

- `viewer`, `gui`, or `E_BTS_GUI`
- `tracker` or `E_BTS_event_circle_tracker`
- `record`, `record_sequence`, or `E_BTS_record_sequence`
- `csv`, `raw_to_csv`, or `E_BTS_raw_to_csv`

If no command is given, Docker runs `viewer`.

## Compatibility Helpers

These older direct wrappers still exist:

```bash
./docker/record_sequence.sh <camera_serial>
./docker/raw_to_csv.sh input.raw [cd.csv] [triggers.csv]
```

Prefer `docker/run.sh` for new usage because it matches the current entrypoint and output directory layout.

## Manual Docker Run

For the live viewer:

```bash
mkdir -p recordings
xhost +SI:localuser:root
docker run --rm -it \
  --privileged \
  --net=host \
  --ipc=host \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e E_BTS_RECORDINGS_DIR=/data/recordings \
  -v "$PWD:/data:rw" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/bus/usb:/dev/bus/usb \
  -w /data/recordings \
  e-bts:openeb-3.1.2 viewer
xhost -SI:localuser:root
```

For headless CSV conversion, display and USB forwarding are not needed:

```bash
docker run --rm -it \
  -e E_BTS_RECORDINGS_DIR=/data/recordings \
  -v "$PWD:/data:rw" \
  -w /data/recordings \
  e-bts:openeb-3.1.2 csv recording.raw
```

## Notes

- This setup is intended for a Linux host. Docker Desktop on Windows/macOS makes USB camera passthrough and native GUI display more complicated.
- `--privileged` is broad, but it is the simplest way to prove the EVK1 works in a container. After that, permissions can be tightened with explicit USB device mappings.
- The first build can take a while because OpenEB is compiled inside the image.
