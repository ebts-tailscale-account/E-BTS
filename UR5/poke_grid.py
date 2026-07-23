"""
poke_grid.py

Pipeline-test script: sweeps a sparse grid of points across the silicone
sensor sheet, poking each one to a fixed depth, and for every poke:

  1. drops control/start.cmd in the E-BTS repo so E_BTS_record_sequence
     (must already be running, see below) opens a fresh, timestamped .raw
     file for that poke alone,
  2. drives the UR5 through hover -> press -> dwell -> retract for that one
     point,
  3. samples the WITTENSTEIN force sensor (via pyForceDAQ) during the dwell
     and keeps the peak Fz reading,
  4. drops control/stop.cmd to close that poke's .raw file,
  5. appends one row (poke name, grid position, peak force) to a master
     results CSV.

This does NOT do live force-triggered contact detection. "Confirming
contact" here means the same thing it means in generate_path.py: Z_SURFACE_M
is a manually calibrated height at which the tip is known to touch the
sheet, carried over unchanged from generate_path.py's calibration. Contact
depth is applied on top of that as an open-loop offset, not verified in
real time against the force reading. If real contact-triggered depth
control is wanted later, that's a separate, explicitly-scoped change.

Grid size, spacing, and per-poke timing are placeholder values sized for a
quick pipeline check (~20 pokes, a few minutes total), not for measuring
the sheet at 1 mm resolution -- see AGENTS.md / requirements.txt for why
that full-resolution sweep is deferred. Re-measure X_CORNER_M/Y_CORNER_M/
Z_SURFACE_M and the grid constants below against the actual physical setup
before trusting the numbers.

Prerequisites for a real run:
  - E_BTS_record_sequence already running and watching E_BTS_ROOT's
    control/ and recordings/ directories (see ~/E-BTS/record_testing.py's
    docstring for how to start it).
  - The robot already moved to its safe start pose (start_pose.py).
  - The WITTENSTEIN sensor connected and idle (nothing touching it) when
    this script starts, so bias determination is valid.

After this script finishes, return the robot home with home.py.

None of this has been run against real hardware -- it has only been
written and read for consistency with the existing scripts.
"""

import csv
import socket
import sys
import time
from pathlib import Path

import pandas as pd

from pose_utils import (
    A_real, A_sim, V_real, V_sim,
    FIXED_ROTVEC, approach_height,
    contact_to_tcp_position, pose_str,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "pyForceDAQ"))
from forceDAQ.force import DataRecorder, SensorSettings  # noqa: E402

# --- Sensor-sheet calibration -------------------------------------------
# Carried over unchanged from generate_path.py's calibration of this same
# physical setup. Re-measure if the fixture, camera, or robot mount moved.
X_CORNER_M = 0.01042
Y_CORNER_M = -0.49942
Z_SURFACE_M = 0.223  # robot Z at which the indenter tip just touches the sheet

# --- Grid geometry --------------------------------------------------------
# Nominal sheet is 36mm (x) by 30mm (y) per AGENTS.md. MARGIN_MM keeps pokes
# off the clamped edge. ROWS/COLS give 20 points total, spaced out purely to
# validate the recording/logging pipeline -- not a resolution measurement.
SHEET_WIDTH_MM = 36.0
SHEET_HEIGHT_MM = 30.0
MARGIN_MM = 4.0
ROWS = 4  # steps along x
COLS = 5  # steps along y
STEP_X_MM = (SHEET_WIDTH_MM - 2 * MARGIN_MM) / (ROWS - 1)
STEP_Y_MM = (SHEET_HEIGHT_MM - 2 * MARGIN_MM) / (COLS - 1)

PRESS_DEPTH_M = 0.005  # 5mm poke depth, requested explicitly -- overrides
                       # pose_utils.contact_depth (2mm) for this script only

# --- Per-poke timing (placeholder, not measured against real hardware) ----
# UR script itself blocks for DWELL_SECONDS at the press pose (see
# build_poke_script); these host-side sleeps just estimate when each phase
# of that same motion has physically finished, since these scripts have no
# closed-loop "motion complete" signal from the controller.
TRANSIT_AND_DESCEND_SECONDS = 3.0
DWELL_SECONDS = 2.0
RETRACT_AND_MARGIN_SECONDS = 2.0
FORCE_SAMPLE_INTERVAL_SECONDS = 0.02

UR_PORT = 30002  # secondary interface -- see run_motion.py's comment on why
                 # 30003 (realtime) can't be used for multi-line programs

E_BTS_ROOT = Path.home() / "E-BTS"
CONTROL_DIR = E_BTS_ROOT / "control"
RECORDINGS_DIR = E_BTS_ROOT / "recordings"

FORCE_SENSOR_NAME = "FT12877"  # matches pyForceDAQ.settings' current sensor
FORCE_CALIBRATION_DIR = Path(__file__).resolve().parent / "pyForceDAQ" / "calibration"


def generate_grid():
    """Builds the poke list and writes poke_grid_path.csv for reference.

    Each entry has the grid indices, the poke's position on the sheet in mm
    (origin at the margin-inset corner, not the raw fixture corner), and the
    robot-frame hover/press TCP poses used to drive the UR5.
    """
    waypoint_rows = []
    pokes = []
    for i in range(ROWS):
        for j in range(COLS):
            x_mm = MARGIN_MM + i * STEP_X_MM
            y_mm = MARGIN_MM + j * STEP_Y_MM
            contact_x = X_CORNER_M - (x_mm / 1000.0)
            contact_y = Y_CORNER_M - (y_mm / 1000.0)

            z_hover = Z_SURFACE_M + approach_height
            z_press = Z_SURFACE_M - PRESS_DEPTH_M

            hover_tcp = contact_to_tcp_position([contact_x, contact_y, z_hover])
            press_tcp = contact_to_tcp_position([contact_x, contact_y, z_press])
            hover_pose = [*hover_tcp, *FIXED_ROTVEC]
            press_pose = [*press_tcp, *FIXED_ROTVEC]

            name = f"poke_r{i:02d}_c{j:02d}"
            pokes.append({
                "name": name,
                "row": i,
                "col": j,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "hover_pose": hover_pose,
                "press_pose": press_pose,
            })

            for pose, phase in [(hover_pose, "hover"), (press_pose, "press"), (hover_pose, "retract")]:
                waypoint_rows.append({
                    "poke": name, "row": i, "col": j, "phase": phase,
                    "x": pose[0], "y": pose[1], "z": pose[2],
                    "rx": pose[3], "ry": pose[4], "rz": pose[5],
                })

    pd.DataFrame(waypoint_rows).to_csv("poke_grid_path.csv", index=False)
    print(f"Generated {len(pokes)} poke points -> poke_grid_path.csv")
    return pokes


def build_poke_script(hover_pose, press_pose, accel, vel, name):
    """One poke's URScript: hover -> press -> dwell -> retract to hover."""
    hover_line = pose_str(hover_pose)
    press_line = pose_str(press_pose)
    return (
        "def poke_program():\n"
        f'  textmsg("{name}: hover")\n'
        f"  movel(p[{hover_line}], a={accel}, v={vel}, t=0, r=0)\n"
        f'  textmsg("{name}: press")\n'
        f"  movel(p[{press_line}], a={accel}, v={vel}, t=0, r=0)\n"
        f"  sleep({DWELL_SECONDS})\n"
        f'  textmsg("{name}: retract")\n'
        f"  movel(p[{hover_line}], a={accel}, v={vel}, t=0, r=0)\n"
        "end\n"
        "poke_program()\n"
    )


def send_ur_script(host, script_text):
    """Fire-and-forget send of one poke's program, matching run_motion.py."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, UR_PORT))
        s.sendall(script_text.encode("ascii"))
        time.sleep(1)
    finally:
        s.close()


def sample_peak_force(sensor_process, duration_seconds, interval_seconds):
    """Polls live Fz for duration_seconds, returns the peak reading.

    sensor_process.Fz is a shared-memory value the SensorProcess subprocess
    updates continuously once polling is started (see
    forceDAQ/force/sensor_process.py) -- this just samples it repeatedly
    rather than reading DataRecorder's own file output, since we only need
    one summary number per poke, not pyForceDAQ's own recording.
    """
    peak = float("-inf")
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        peak = max(peak, sensor_process.Fz)
        time.sleep(interval_seconds)
    return peak


def wait_for(predicate, timeout, interval=0.2):
    """Mirrors record_testing.py's wait_for, parameterized here since that
    script's own helpers are hardcoded to its own (wrong, for our purposes)
    control/recordings directories."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def newest_raw_file(before):
    fresh = set(RECORDINGS_DIR.glob("*.raw")) - before
    return max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None


def run_single_poke(poke, host, accel, vel, force_sensor_process, results_writer, results_file):
    """Runs the full record-start / move / sample / record-stop sequence
    for one grid point. Returns True on success, False if anything (missing
    watcher, stalled recording) forced an early abort of this poke."""
    name = poke["name"]
    print(f"\n=== {name} (row={poke['row']}, col={poke['col']}) ===")

    before = set(RECORDINGS_DIR.glob("*.raw"))
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    (CONTROL_DIR / "start.cmd").write_text(name)

    if not wait_for(lambda: newest_raw_file(before) is not None, timeout=10):
        print(f"FAIL: no .raw file appeared for '{name}' -- is E_BTS_record_sequence running?")
        return False
    raw_path = newest_raw_file(before)
    print(f"-> recording {raw_path.name}")

    script_text = build_poke_script(poke["hover_pose"], poke["press_pose"], accel, vel, name)
    send_ur_script(host, script_text)

    time.sleep(TRANSIT_AND_DESCEND_SECONDS)
    peak_force = sample_peak_force(
        force_sensor_process, DWELL_SECONDS, FORCE_SAMPLE_INTERVAL_SECONDS)
    time.sleep(RETRACT_AND_MARGIN_SECONDS)

    (CONTROL_DIR / "stop.cmd").write_text("")
    if not wait_for(lambda: not (CONTROL_DIR / "stop.cmd").exists(), timeout=5):
        print(f"FAIL: stop.cmd not consumed for '{name}' -- watcher may have died")
        return False

    print(f"-> peak Fz = {peak_force:.3f} N")
    results_writer.writerow([name, poke["row"], poke["col"], poke["x_mm"], poke["y_mm"], peak_force])
    results_file.flush()
    return True


def main():
    mode = input("Select mode ('sim' or 'real'): ").strip().lower()
    if mode == "sim":
        host, accel, vel = "172.17.0.2", A_sim, V_sim
    elif mode == "real":
        host, accel, vel = "192.168.0.153", A_real, V_real
    else:
        print("Invalid mode. Exiting.")
        return

    print(f"control dir:    {CONTROL_DIR}")
    print(f"recordings dir: {RECORDINGS_DIR}")
    print("(E_BTS_record_sequence must already be running and watching these directories)")

    pokes = generate_grid()

    sensor_settings = SensorSettings(
        device_id="1",
        sensor_name=FORCE_SENSOR_NAME,
        calibration_folder=str(FORCE_CALIBRATION_DIR),
        reverse_parameter_names="Fz",
    )
    recorder = DataRecorder(force_sensor_settings=[sensor_settings],
                             poll_udp_connection=False, polling_priority="normal")
    print("Setting force sensor bias, do not touch the sensor!")
    recorder.determine_biases(n_samples=100)
    recorder.start_recording()
    force_sensor_process = recorder.force_sensor_processes()[0]

    results_path = RECORDINGS_DIR / f"poke_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    completed = 0
    try:
        with open(results_path, "w", newline="") as results_file:
            writer = csv.writer(results_file)
            writer.writerow(["poke", "row", "col", "x_mm", "y_mm", "force_N"])
            results_file.flush()

            for poke in pokes:
                if run_single_poke(poke, host, accel, vel, force_sensor_process, writer, results_file):
                    completed += 1
                else:
                    print(f"Stopping sweep early after {completed}/{len(pokes)} pokes.")
                    break
    finally:
        recorder.quit()
        print("Force sensor recorder shut down.")

    print(f"\n{completed}/{len(pokes)} pokes completed. Results -> {results_path}")
    print("Run home.py to return the robot to its home position.")


if __name__ == "__main__":
    main()
