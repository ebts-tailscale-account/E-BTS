# AGENTS.md

## Project context

E-BTS is a C++17 toolkit for working with an event-based EVK1 camera through OpenEB / Metavision SDK 3.1.2. At present, the repository supports live event visualization, manual and sequenced RAW recording, saved frames, and RAW-to-CSV conversion. These tools are the experimental foundation of the project, not the finished system.

The long-term goal is to build software that can operate a **slow adaptive sensor and a fast adaptive sensor simultaneously**. The two sensor paths will eventually need to coexist cleanly and remain understandable enough to tune, test, and evolve independently. The exact architecture and behavior should be developed deliberately from experimental results and explicit project decisions; do not invent missing requirements or prematurely build the final system.

## Confirmed normal-force experiment context

The current experimental milestone is to estimate normal force and contact location. Estimating tangential forces and other force components is a preferred later capability, but it is not part of the first implementation unless explicitly requested.

In project discussions, **sensor** means the multilayer silicone structure with its marker domes. It does not mean the EVK1 camera by itself.

### Mechanical construction and loading

- The nominal sensor surface is 36 mm by 30 mm and has two nominally 2 mm layers: a black marker-support/background layer and a yellow outer layer, for 4 mm total nominal sheet thickness. SORTA-Clear 18 silicone is used. Treat these as construction values, not measured dimensions or material parameters.
- From the camera toward the load-facing side, the reported layer order is: visible silicone marker domes, 2 mm black silicone layer, then 2 mm yellow silicone layer. The black layer holds the markers and provides optical contrast for the EVK1. The yellow layer lets human operators see the active sensor area.
- The marker domes are approximately 0.8 mm in diameter. They are bonded to the black layer by the multilayer curing process rather than attached with glue.
- The sheet is mounted above the camera and bolted into the sensor chamber. For the initial mechanical model, approximate the sheet as fixed along every side. Revisit this boundary condition if calibration exposes edge slip or fixture compliance.
- Assume the 4 mm slab is uniformly and ideally cured for initial work. This is an explicit working assumption, not a verified material characterization.
- Loads are applied through a plastic conical indenter by a standard seven-degree-of-freedom robotic-hand manipulator. The cone permits loading from different directions, although the first measurement target is normal force. The immediate purpose is a proof-of-function experiment rather than a production-ready force sensor.
- Visible response has so far required a point load greater than approximately 300 gram-force (about 2.9 N). The sheet has broken or slipped from the fixture near 10 kilogram-force (about 98 N). These observations are not a calibrated detection threshold, resolution, safe working limit, or rated measurement range.
- A normal load may remain applied for approximately 5 to 60 seconds, so experiments and models must consider whether the estimate drifts during a static hold.

### Camera and illumination behavior

- The EVK1 event-camera resolution is 640 by 480. A macro lens views the silicone straight on, with its optical axis nominally perpendicular to the sensor, from approximately 29 to 30 mm away.
- Two LED pairs flicker continuously at a constant frequency with different phases. Their separate illumination regions keep the upper and lower marker regions event-active. Where the illumination overlaps, its intensity is approximately constant, so the middle produces few events and appears as a black horizontal band.
- The black middle band is intentional and is expected to support a future fast-adaptive sensor path. Do not remove or reinterpret it as a camera defect without discussing the slow/fast illumination design.
- The current live visualization renders both event polarities as white for viewing comfort. That rendering is not a measurement input.
- Force estimation must consume and parse the EVK event stream directly. It must not analyze screenshots, saved display frames, or the white visual rendering as if they were camera intensity measurements.
- Illumination timing may be adjusted for experiments, but changes must preserve or explicitly account for the intended off-phase slow/fast adaptive behavior.

### Reference measurements and calibration

- The robot manipulators can report applied force, and a WITTENSTEIN force sensor can be added to the acquisition setup as an external reference.
- Determine detection threshold, usable force range, resolution, contact-location accuracy, hysteresis, and hold-time drift empirically against the reference sensor. Do not assign these performance values from the initial 300 gram-force observation.
- No specimen-specific tension, compression, shear, or indentation measurements are currently available for the cured silicone. Manufacturer hardness or strength data must not be treated as a complete constitutive model.

## Unresolved inputs for normal-force estimation

Resolve these points before selecting the final vision pipeline, elasticity equations, or calibration protocol:

- Measure the conical indenter tip radius and angle and confirm whether it is effectively rigid over the test range.
- Record marker dome height, pitch, exact row/column layout, and whether the domes use the same silicone formulation as the two sheet layers.
- Confirm the reported camera-to-load layer orientation and document any additional rigid, transparent, adhesive, or air layer in the optical or mechanical stack.
- Identify the macro lens if possible and obtain camera intrinsics, lens distortion, and a physical pixel-to-millimetre calibration at the marker plane.
- Record LED frequency, duty cycle, relative phase, brightness, illuminated regions, and whether LED transitions can be hardware-synchronized or timestamped with the EVK event stream.
- Define a conservative maximum calibration load below the observed break/slip region; do not use 10 kilogram-force as a safe test limit.
- Define how EVK timestamps, manipulator feedback, and WITTENSTEIN measurements will share a clock or be aligned during calibration.
- Decide whether to characterize the actual cured silicone specimen mechanically or to use reference-load calibration to identify an effective model, including hold-time relaxation and hysteresis.

The precise information-gathering checklist is maintained in `requirements.txt` at the repository root.

## Working principle: do not rush

- Do not write, generate, refactor, or otherwise change code unless the user explicitly asks for a code change.
- A request to inspect, explain, diagnose, review, discuss, or plan is not permission to implement anything.
- Do not add speculative abstractions or dual-sensor functionality “for later.” Introduce structure only when a requested change needs it.
- Keep changes narrowly scoped to the prompt. If a change depends on an undecided sensor behavior, timing model, data-flow design, or hardware assumption, explain the uncertainty and ask before choosing an architecture.
- Prefer small, verifiable steps. Preserve working EVK1 capture and analysis behavior while the project advances toward simultaneous slow/fast adaptive sensing.

Documentation-only changes are acceptable only when requested. Never treat this file as permission to make unrelated cleanup changes.

## Code quality requirements

Code must be easy for another contributor to read, understand, and modify.

- Use clear, descriptive names and straightforward control flow.
- Keep functions focused on one responsibility. Split a function when doing so makes its behavior or state transitions easier to understand.
- Every complex function must have a concise comment explaining what it does. The comment should describe its purpose, important inputs/outputs, state or timing behavior, and non-obvious constraints. Prefer explaining **why** the logic exists and how it fits the sensor workflow over narrating individual lines.
- Add comments around non-obvious SDK behavior, concurrency, callbacks, ownership, timing, synchronization, file formats, and hardware assumptions.
- Keep comments accurate when behavior changes. Do not add comments that merely repeat obvious code.
- Avoid hidden coupling and unexplained constants. Give sensor, timing, and threshold values meaningful names and document their units.
- Preserve error handling and clean shutdown behavior, especially around camera start/stop, recording, callbacks, and signal handling.
- Favor maintainability over cleverness or premature optimization. Performance work should be prompted and supported by a concrete need or measurement.

## Current repository map

- `src/main.cpp`: live EVK1 viewer and manual RAW recorder.
- `src/record_sequence.cpp`: timed batch RAW capture.
- `src/raw_to_csv.cpp`: RAW event conversion to CSV.
- `src/event_circle_tracker.cpp`: marker circle tracker orchestration (camera setup, main loop); `src/combined_main.cpp` runs it alongside the live viewer.
- `src/circle_tracker_config.h`, `event_window_buffer.h`, `circle_detector.h`, `circle_lattice.h`, `temporal_circle_tracker.h`, `tracking_render_utils.h`, `live_view_controller.h`: circle-tracker building blocks (tuning constants, event-window buffering, per-window detection, baseline lattice inference, cross-window tracking, rendering, and the combined-mode live-view branch, which hands off its latest frame rather than presenting it itself).
- `src/side_by_side_frame_presenter.h`, `settings_overlay.h`, `action_button_bar.h`: `E_BTS_combined`-only UI — one window with the tracker feed and the live view side by side, a custom-drawn "Settings" overlay (text fields, no GUI toolkit dependency) for live accumulation-time/detection-density tuning, and a bottom action bar with "Rebuild Lattice"/"Quit" buttons. This mode opens immediately with default values instead of prompting on startup.
- `src/console_input_utils.h`: shared interactive numeric-prompt parsing used by every entry point.
- `src/*_utils.h`: camera, display, frame, path, signal, and live-view helpers.
- `src/CMakeLists.txt`: C++17 build configuration for Metavision and OpenCV.
- `docker/` and `DOCKER.md`: reproducible Linux build and runtime support.
- `recordings/`: default host-side recording location; generated recordings should not be committed without an explicit reason.

## Build and verification

For native builds, configure from the repository root:

```bash
cmake -S src -B build
cmake --build build -j
```

The project targets Ubuntu 20.04, CMake 3.16+, C++17, OpenEB / Metavision SDK 3.1.2, and OpenCV. Live capture requires compatible EVK1 hardware, so state clearly when a change was only compiled or reviewed and was not tested against a physical camera.

When code changes are explicitly requested:

- Run the smallest relevant build or test available.
- Do not claim hardware behavior was verified without hardware testing.
- Report any verification that could not be performed and why.
- Do not modify generated files, recordings, build output, or large data files unless explicitly asked.

## Direction for future sensor work

When the user explicitly requests work toward the final system, keep the slow and fast adaptive sensor responsibilities distinguishable. Make timing units, synchronization boundaries, data ownership, configuration, and failure behavior explicit. Any shared interface should come from confirmed common behavior rather than assumptions that both sensors work alike.

The priority is a dependable, understandable experimental system that can grow into simultaneous slow/fast adaptive sensing—not reaching that endpoint as quickly as possible.
