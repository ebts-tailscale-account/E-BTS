#!/usr/bin/env python3
"""E-BTS master: one synchronized recording per grid sweep.

Fans a single run out to all three systems and pulls everything back:

  camera + Wittenstein F/T  -> the E_BTS_GUI recorder, via the control/start.cmd
                               file protocol (writes <run>_<stamp>.raw and
                               <run>_<stamp>_ft.csv, both on the WORKSTATION clock)
  Franka                    -> snake grid sweep + franka_states logging on tactile
                               over ssh (writes a CSV on the TACTILE clock)

Because the Franka is on a second machine, we measure the workstation<->tactile
clock offset and record it, so the Franka timeline can be shifted onto the
workstation timeline in post. (The fast pair -- F/T and events -- is already on
one clock, so this only matters for the slow robot signal.)

PREREQS (start these by hand first):
  * On the WORKSTATION, launch the GUI FROM THE REPO ROOT so control/ and
    recordings/ land where this script expects them:
        cd ~/E-BTS && ./build/E_BTS_GUI
    then open the **Force/Torque** source (so the port streams) and the
    **Sequence Recording** pane (so the control/ watcher runs).
  * On TACTILE, bring up the clean MoveIt stack with the robot FCI-enabled:
        source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
        roslaunch panda_moveit_config franka_control.launch robot_ip:=10.1.196.5 load_gripper:=false
  * Passwordless ssh to tactile (ssh-copy-id) so the many ssh/scp calls don't prompt.

Usage:  python3 master.py [run_name]
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# --- config ---
TACTILE            = "tactile@100.93.60.35"
REPO               = Path(__file__).resolve().parent
CONTROL            = REPO / "control"
RECORDINGS         = REPO / "recordings"
# The logger lives permanently on tactile (verified at this path). If you edit
# the workstation copy under franka/, re-copy it over:
#   scp franka/franka_grid_logger.py tactile@100.93.60.35:~/E-BTS/franka_grid_logger.py
REMOTE_SCRIPT      = "~/E-BTS/franka_grid_logger.py"
REMOTE_CSV         = "/tmp/ebts_franka_run.csv"
SETTLE_AFTER_START = 1.0   # let the GUI open its files before the arm moves


def _ssh(args, **kw):
    return subprocess.run(["ssh", TACTILE] + args, **kw)


def measure_clock_offset(samples=5):
    """Estimate (tactile_clock - workstation_clock) in seconds, NTP-style.

    Keeps the lowest-round-trip sample, whose error is bounded by rtt/2.
    """
    best = None
    for _ in range(samples):
        t0 = time.time()
        out = _ssh(["date +%s.%N"], capture_output=True, text=True, timeout=10)
        t1 = time.time()
        if out.returncode != 0:
            continue
        try:
            remote = float(out.stdout.strip())
        except ValueError:
            continue
        rtt = t1 - t0
        offset = remote - (t0 + t1) / 2.0
        if best is None or rtt < best[1]:
            best = (offset, rtt)
    return best  # (offset_s, rtt_s) or None


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    CONTROL.mkdir(exist_ok=True)
    RECORDINGS.mkdir(exist_ok=True)

    print("== E-BTS synchronized run: '%s' ==" % run_name)

    print("Measuring workstation<->tactile clock offset...")
    offset = measure_clock_offset()
    if offset:
        print("  tactile clock is %+.1f ms vs workstation (rtt %.1f ms)" % (offset[0] * 1e3, offset[1] * 1e3))
    else:
        print("  WARNING: could not reach tactile to measure offset; Franka alignment will be uncorrected.")

    t0_unix = time.time()

    # 1. Start camera + F/T recording via the GUI's control protocol.
    #    (The Franka logger already lives on tactile at REMOTE_SCRIPT.)
    print("Starting GUI recording (camera + F/T)...")
    (CONTROL / "start.cmd").write_text(run_name)
    time.sleep(SETTLE_AFTER_START)

    # 3. Run the Franka sweep + logging on tactile (blocks until the sweep ends).
    print("Running Franka grid sweep on tactile (blocks until done)...")
    remote_cmd = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/ws_franka/devel/setup.bash && "
        "python3 %s %s" % (REMOTE_SCRIPT, REMOTE_CSV)
    )
    franka = _ssh([remote_cmd])
    franka_ok = franka.returncode == 0
    print("  Franka sweep %s." % ("finished" if franka_ok else "FAILED (rc=%d)" % franka.returncode))

    # 4. Stop the GUI recording.
    print("Stopping GUI recording...")
    (CONTROL / "stop.cmd").write_text("")

    # 5. Pull the Franka CSV back next to the camera/F/T recordings.
    local_csv = RECORDINGS / ("%s_franka.csv" % run_name)
    print("Pulling Franka CSV back...")
    scp_back = subprocess.run(["scp", "%s:%s" % (TACTILE, REMOTE_CSV), str(local_csv)])
    if scp_back.returncode != 0:
        print("  WARNING: could not pull the Franka CSV from tactile.")

    # 6. Metadata for post-hoc alignment.
    meta = {
        "run_name": run_name,
        "t0_unix_s": t0_unix,
        "tactile_minus_workstation_offset_s": offset[0] if offset else None,
        "offset_rtt_s": offset[1] if offset else None,
        "franka_ok": franka_ok,
        "franka_csv": local_csv.name if scp_back.returncode == 0 else None,
        "note": ("camera .raw and *_ft.csv are named <run>_<timestamp> by the GUI and are on the "
                 "workstation clock; the franka CSV is on the tactile clock -- subtract "
                 "tactile_minus_workstation_offset_s from its unix_time_s to align."),
    }
    (RECORDINGS / ("%s_metadata.json" % run_name)).write_text(json.dumps(meta, indent=2))
    print("Wrote %s_metadata.json. Done -- see %s/" % (run_name, RECORDINGS))


if __name__ == "__main__":
    main()
