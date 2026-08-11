#!/usr/bin/env python3
"""E-BTS master (MIDPOINT variant): one synchronized recording of N repeated
indentations at the single taught point.

A COPY of master.py -- master.py itself is untouched and still runs the 99-point
grid sweep. This copy differs in exactly three ways:
  1. it resolves the indent TARGET ITSELF from one_point.json (joint-space, via
     panda_fk) and prints the derivation before anything moves,
  2. it drives ~/E-BTS/indent_midpoint.py on tactile instead of the grid sweep,
     handing it the explicit target it just resolved,
  3. it records the taught point + the target derivation into metadata.json.

Fans a single run out to all three systems and pulls everything back:

  camera + Wittenstein F/T  -> the E_BTS_GUI recorder, via the control/start.cmd
                               file protocol (writes <run>_<stamp>.raw and
                               <run>_<stamp>_ft.csv, both on the WORKSTATION clock)
  Franka                    -> N indentations at the point + franka_states
                               logging on tactile over ssh (CSV on the TACTILE clock)

HOW THE TARGET IS RESOLVED
--------------------------
ONE point is taught (record_one_point.py) by hand-positioning the tool exactly where
it should indent -- that point IS the midpoint, so nothing is interpolated. The
servo controller takes Cartesian targets only, so:
  XY + orientation <- FK(joints) of the taught snapshot, NOT the logged O_T_EE, so
                      the geometry comes from the encoders. panda_fk is validated to
                      0.001 mm against taught snapshots; the two are cross-checked
                      here and a disagreement is reported.
  Z (surface)      <- the taught (kissed) z. The run dips INDENT_MM below it.
All 5 repeats hit this same point, so the run measures repeatability at one location.

Because the Franka is on a second machine, we measure the workstation<->tactile
clock offset and record it, so the Franka timeline can be shifted onto the
workstation timeline in post. (The fast pair -- F/T and events -- is already on
one clock, so this only matters for the slow robot signal.)

PREREQS (start these by hand first):
  * The HEX21 F/T sensor is plugged into the WORKSTATION for the experiment (it is
    on tactile only for surface mapping). Move it back before this run.
  * On the WORKSTATION, launch the GUI and open BOTH the **Force/Torque** source
    (so the port streams and an *_ft.csv is written) and the **Sequence Recording**
    pane (it runs the control/ watcher -- without it nothing records):
        cd ~/E-BTS && ./build/E_BTS_GUI
    The GUI resolves control/ and recordings/ RELATIVE TO ITS WORKING DIRECTORY, so
    launching from build/ makes it use build/control + build/recordings. This script
    now auto-detects the running GUI's dirs and collects its output into
    <repo>/recordings/, so either launch dir works -- and it ABORTS if the GUI is
    not running or never consumes start.cmd.
  * The point must already be taught:  python3 record_one_point.py  on tactile, which
    writes ~/E-BTS/one_point.json. This script pulls that file back and resolves the
    target from it. NO surface_map.csv is needed.
  * On TACTILE, with the arm at HOME, bring up the Cartesian pose SERVO controller
    (REACH launch: max_translation 0.55 so it can travel home->surface). Run ONE
    controller, not both (it claims the same interface as MoveIt):
        source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
        roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
            robot_ip:=10.1.196.5 load_gripper:=false
    Teaching needs hand-guiding, which the servo controller fights -- so teach FIRST
    (franka_control only), then stop it and bring up the servo reach launch.
    The remote script auto-homes from home to a hover above the midpoint; no
    hand-positioning.
  * Passwordless ssh to tactile (ssh-copy-id) so the many ssh/scp calls don't prompt.

Usage:  python3 master_midpoint.py [run_name] [--repeats 5] [--indent-mm 2]
        python3 master_midpoint.py --dry-run     # show the midpoint, record nothing
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --- config ---
TACTILE            = "tactile@100.93.60.35"
REPO               = Path(__file__).resolve().parent
RECORDINGS         = REPO / "recordings"
# The motion+logging script lives permanently on tactile. If you edit the
# workstation copy under franka/, re-copy it over:
#   scp franka/indent_midpoint.py tactile@100.93.60.35:~/E-BTS/indent_midpoint.py
REMOTE_SCRIPT      = "~/E-BTS/indent_midpoint.py"
REMOTE_POINTS      = "~/E-BTS/one_point.json"      # written by record_one_point.py
REMOTE_RECORDINGS  = "~/E-BTS/recordings"          # no more /tmp -- keep runs together

# --- indentation params (this variant) ---
REPEATS   = 5     # how many indentations at the midpoint
INDENT_MM = 2.0   # dip this far BELOW the reference surface height
HOVER_MM  = 5.0   # travel / retract / tare at this height above the surface
DWELL_S   = 2.0   # hold each indent this long (the measurement window)
SETTLE_AFTER_START = 1.0   # let the GUI open its files before the arm moves
START_ACK_TIMEOUT  = 10.0  # how long to wait for the GUI to consume start.cmd


def _ssh(args, **kw):
    return subprocess.run(["ssh", TACTILE] + args, **kw)


# --------------------------------------------------------------- midpoint

def _load_panda_fk():
    """panda_fk lives in franka/ next to the robot scripts; it is numpy-only."""
    sys.path.insert(0, str(REPO / "franka"))
    try:
        import panda_fk
    except ImportError as e:
        sys.exit("[ABORT] cannot import panda_fk from %s (%s).\n"
                 "        It is needed to compute the joint-space midpoint."
                 % (REPO / "franka", e))
    return panda_fk


def fetch_point(dest):
    """Pull one_point.json back from tactile (it is written there by the teach script)."""
    print("Fetching the taught point from tactile (%s)..." % REMOTE_POINTS)
    r = subprocess.run(["scp", "%s:%s" % (TACTILE, REMOTE_POINTS), str(dest)])
    if r.returncode != 0 or not dest.exists():
        sys.exit("[ABORT] could not fetch %s from tactile.\n"
                 "        Teach the point first:\n"
                 "            ssh %s\n"
                 "            python3 ~/E-BTS/record_one_point.py\n"
                 % (REMOTE_POINTS, TACTILE))
    return json.loads(dest.read_text())


ROBOT_MODES = {0: "OTHER", 1: "IDLE", 2: "MOVE", 3: "GUIDING", 4: "REFLEX",
               5: "USER_STOPPED", 6: "AUTOMATIC_ERROR_RECOVERY"}


def resolve_target(data):
    """Resolve the indent target from the single taught point. See the module docstring.

    Returns (x, y, surface_z, quat[xyzw], diagnostics dict).
    """
    pfk = _load_panda_fk()
    if "point" not in data:
        sys.exit("[ABORT] %s has no 'point' key -- it was not written by\n"
                 "        record_one_point.py. Re-teach with that script." % REMOTE_POINTS)
    pt = data["point"]
    label = data.get("label", "indent_point")
    q = pt["joints"]

    # Geometry from the ENCODERS: recompute FK here rather than trusting the stored xyz.
    xyz_fk, quat = pfk.pose_of(q)
    x, y = float(xyz_fk[0]), float(xyz_fk[1])
    surface_z = float(xyz_fk[2])

    # Cross-check against the O_T_EE measured in the same snapshot. These must agree;
    # a gap means the tool/EE transform changed since panda_fk was validated.
    meas = pt.get("xyz_measured_O_T_EE") or pt.get("xyz")
    d_meas_mm = (np.array(xyz_fk) - np.array(meas, float)) * 1e3

    diag = {
        "label": label,
        "taught_at": data.get("recorded_at"),
        "joints": [float(v) for v in q],
        "xyz_from_joints_fk": [float(v) for v in xyz_fk],
        "xyz_measured_O_T_EE": [float(v) for v in meas],
        "fk_vs_measured_delta_mm": [float(v) for v in d_meas_mm],
        "target_xy": [x, y],
        "surface_z": surface_z,
        "quat_xyzw": [float(v) for v in quat],
        "taught_fk_residual_mm": pt.get("fk_residual_mm"),
        "taught_dq_max_abs": pt.get("dq_max_abs"),
        "taught_robot_mode": pt.get("robot_mode"),
    }

    print("=" * 72)
    print(" INDENT TARGET RESOLVED FROM THE TAUGHT POINT")
    print("=" * 72)
    print("  label                  : %s   (taught %s)" % (label, data.get("recorded_at")))
    print("  joints                 : [%s]" % ", ".join("%+.4f" % v for v in q))
    print("  FK(joints)             : [%s]   <- target"
          % ", ".join("%.5f" % v for v in xyz_fk))
    print("  measured O_T_EE        : [%s]" % ", ".join("%.5f" % v for v in meas))
    print("  FK - measured          : [%s] mm"
          % ", ".join("%+.4f" % v for v in d_meas_mm))
    print("  orientation (taught)   : [%s]" % ", ".join("%+.4f" % v for v in quat))
    print("  surface_z (kissed)     : %.5f m" % surface_z)
    if pt.get("robot_mode") is not None:
        print("  taught in robot_mode   : %d (%s)"
              % (pt["robot_mode"], ROBOT_MODES.get(pt["robot_mode"], "?")))
    print("=" * 72)
    if np.abs(d_meas_mm).max() > 0.05:
        print("  [WARN] FK and the measured pose disagree by %.3f mm. panda_fk's tool"
              % np.abs(d_meas_mm).max())
        print("         transform may no longer match this setup -- verify before trusting")
        print("         the target.")
    return x, y, surface_z, quat, diag


def find_gui_dirs():
    """Locate the control/ + recordings/ dirs the RUNNING E_BTS_GUI is actually using.

    The GUI's SequenceRecordingController takes RELATIVE paths ("control",
    "recordings"), so they resolve against the GUI's working directory. Launching
    it from build/ instead of the repo root silently points it at build/control,
    and commands written to <repo>/control are never seen -- which produced a full
    30-minute run with NO camera/F/T data. So we resolve it from the live process
    instead of assuming. Returns (control_dir, recordings_dir) or (None, None).
    """
    try:
        out = subprocess.run(["pgrep", "-x", "E_BTS_GUI"], capture_output=True, text=True)
        pids = [p for p in out.stdout.split() if p.strip()]
        if not pids:
            return None, None
        cwd = Path(f"/proc/{pids[0]}/cwd").resolve()
        return cwd / "control", cwd / "recordings"
    except Exception:
        return None, None


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
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", nargs="?", default="midpoint")
    ap.add_argument("--repeats", type=int, default=REPEATS,
                    help="How many indentations at the midpoint (default %d)." % REPEATS)
    ap.add_argument("--indent-mm", type=float, default=INDENT_MM,
                    help="Dip this far below the reference surface (default %.1f)." % INDENT_MM)
    ap.add_argument("--hover-mm", type=float, default=HOVER_MM)
    ap.add_argument("--dwell-s", type=float, default=DWELL_S)
    ap.add_argument("--no-gui", action="store_true",
                    help="Run the Franka only (skip camera/F/T recording).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch the taught points, print the midpoint, then exit. "
                         "No recording, no robot motion.")
    args = ap.parse_args()

    # Resolve the target FIRST -- before touching the GUI or the arm, so a bad or
    # missing one_point.json costs nothing.
    RECORDINGS.mkdir(exist_ok=True)
    pts_data = fetch_point(RECORDINGS / "one_point.latest.json")
    mx, my, msurf, mquat, mdiag = resolve_target(pts_data)
    print("  dip_z  = surface - %.1f mm : %.5f m"
          % (args.indent_mm, msurf - args.indent_mm / 1e3))
    print("  hover  = surface + %.1f mm : %.5f m"
          % (args.hover_mm, msurf + args.hover_mm / 1e3))
    print("  repeats                : %d  (dwell %.1f s each)"
          % (args.repeats, args.dwell_s))
    if args.dry_run:
        print("\n[DRY RUN] target only -- nothing recorded, nothing moved.")
        return
    run_name = args.run_name
    RECORDINGS.mkdir(exist_ok=True)

    # Every run gets its OWN folder: recordings/<run>_<YYYYMMDD_HHMMSS>/ holding all
    # of its files under canonical names (camera.raw / ft.csv / camera.bias /
    # franka.csv / metadata.json), which is what postprocess.py already reads.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = "%s_%s" % (run_name, stamp)
    run_dir = RECORDINGS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("== E-BTS synchronized run: '%s' ==" % run_id)
    print("Run folder: %s" % run_dir)

    # Resolve the dirs the RUNNING GUI actually watches (see find_gui_dirs).
    control, gui_recordings = (None, None) if args.no_gui else find_gui_dirs()
    if not args.no_gui:
        if control is None:
            sys.exit("[ABORT] E_BTS_GUI is not running. Start it, open the Force/Torque\n"
                     "        source AND the Sequence Recording pane, then re-run.\n"
                     "        (Use --no-gui to run the Franka alone.)")
        print("GUI is watching: %s" % control)
        print("GUI writes recordings to: %s" % gui_recordings)
        if control.resolve() != (REPO / "control").resolve():
            print("  NOTE: that is NOT <repo>/control -- the GUI was launched from %s."
                  % control.parent)
            print("  Using the GUI's actual dirs; files will be collected into %s." % RECORDINGS)
        control.mkdir(parents=True, exist_ok=True)

    print("Measuring workstation<->tactile clock offset (BEFORE the run)...")
    offset = measure_clock_offset()
    if offset:
        print("  tactile clock is %+.1f ms vs workstation (rtt %.1f ms)" % (offset[0] * 1e3, offset[1] * 1e3))
    else:
        print("  WARNING: could not reach tactile to measure offset; Franka alignment will be uncorrected.")

    t0_unix = time.time()

    # 1. Start camera + F/T recording via the GUI's control protocol, and VERIFY
    #    the GUI consumed the command (it deletes start.cmd when it acts on it).
    #    Without this check a silent mismatch records nothing for the whole run.
    if not args.no_gui:
        print("Starting GUI recording (camera + F/T)...")
        start_file = control / "start.cmd"
        before = set(gui_recordings.glob("*.raw")) if gui_recordings.exists() else set()
        start_file.write_text(run_name)
        t_wait = time.time()
        while start_file.exists() and time.time() - t_wait < START_ACK_TIMEOUT:
            time.sleep(0.2)
        if start_file.exists():
            start_file.unlink(missing_ok=True)
            sys.exit("[ABORT] The GUI never consumed start.cmd in %.0fs -- the\n"
                     "        **Sequence Recording pane** is probably not open (it runs the\n"
                     "        control/ watcher). Open it (and the Force/Torque source), then\n"
                     "        re-run. No robot motion started, nothing recorded." % START_ACK_TIMEOUT)
        print("  GUI acknowledged (start.cmd consumed).")
        time.sleep(SETTLE_AFTER_START)

    # 3. Run the indentations + logging on tactile (blocks until it ends).
    #    The CSV is written straight into tactile's recordings dir (not /tmp).
    #    The target measured above is passed EXPLICITLY, so the remote script executes
    #    this midpoint rather than re-deriving one that could differ.
    print("Running %d indentations at the taught point on tactile (blocks until done)..."
          % args.repeats)
    remote_dir = "%s/%s" % (REMOTE_RECORDINGS, run_id)     # mirror the run folder
    remote_csv = "%s/franka.csv" % remote_dir
    extra = (" --repeats %d --indent-mm %.4f --hover-mm %.4f --dwell-s %.3f"
             " --x %.9f --y %.9f --surface-z %.9f --quat %.9f %.9f %.9f %.9f"
             % (args.repeats, args.indent_mm, args.hover_mm, args.dwell_s,
                mx, my, msurf, mquat[0], mquat[1], mquat[2], mquat[3]))
    remote_cmd = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/ws_franka/devel/setup.bash && "
        "mkdir -p %s && "
        "python3 %s %s%s" % (remote_dir, REMOTE_SCRIPT, remote_csv, extra)
    )
    franka = _ssh([remote_cmd])
    franka_ok = franka.returncode == 0
    print("  Midpoint indentations %s."
          % ("finished" if franka_ok else "FAILED (rc=%d)" % franka.returncode))

    # 4. Stop the GUI recording.
    raw_name = ft_name = meta_gui_name = None
    if not args.no_gui:
        print("Stopping GUI recording...")
        (control / "stop.cmd").write_text("")
        t_wait = time.time()
        while (control / "stop.cmd").exists() and time.time() - t_wait < START_ACK_TIMEOUT:
            time.sleep(0.2)
        time.sleep(1.0)   # let the GUI close/flush its files
        # 4b. Collect this run's camera/F/T files into the run folder under
        #     canonical names (postprocess.py reads these directly).
        new_raw = sorted(set(gui_recordings.glob("*.raw")) - before,
                         key=lambda p: p.stat().st_mtime)
        if not new_raw:
            print("  WARNING: no new .raw appeared in %s -- camera did NOT record."
                  % gui_recordings)
        for raw in new_raw:
            gui_basename = raw.name
            for src, canon in ((raw, "camera.raw"),
                               (raw.with_name(raw.stem + "_ft.csv"), "ft.csv"),
                               (raw.with_suffix(".bias"), "camera.bias"),
                               (raw.with_suffix(".roi"), "camera.roi")):
                if not src.exists():
                    continue
                dst = run_dir / canon
                # MOVE (a rename when same filesystem) -- never read_bytes(): a .raw
                # is easily 2+ GB and slurping it into RAM killed a collection once.
                try:
                    shutil.move(str(src), str(dst))
                except Exception as e:                  # cross-device etc.
                    print("  (move failed, streaming copy instead: %s)" % e)
                    shutil.copy2(str(src), str(dst))
                print("  collected %-12s <- %s (%.1f MB)"
                      % (canon, src.name, dst.stat().st_size / 1e6))
                if canon == "camera.raw":
                    raw_name = canon
                elif canon == "ft.csv":
                    ft_name = canon
            if ft_name is None:
                print("  WARNING: no *_ft.csv for %s -- was the Force/Torque source open?"
                      % gui_basename)
            meta_gui_name = gui_basename

    # 5. Pull the Franka CSV back into the run folder.
    local_csv = run_dir / "franka.csv"
    print("Pulling Franka CSV back...")
    scp_back = subprocess.run(["scp", "%s:%s" % (TACTILE, remote_csv), str(local_csv)])
    if scp_back.returncode != 0:
        print("  WARNING: could not pull the Franka CSV from tactile.")

    # 5b. Re-measure the offset AFTER the run: these clocks genuinely drift
    #     (observed ~40 ms over 4 h), so recording both ends bounds the alignment
    #     error and lets post-processing interpolate across the run if needed.
    print("Measuring clock offset (AFTER the run)...")
    offset_after = measure_clock_offset()
    if offset and offset_after:
        drift_ms = (offset_after[0] - offset[0]) * 1e3
        print("  after: %+.1f ms  (drift over the run: %+.1f ms)"
              % (offset_after[0] * 1e3, drift_ms))
        if abs(drift_ms) > 20.0:
            print("  NOTE: >20 ms of drift during this run -- fine for 2 s dwells, but do "
                  "not trust sub-10 ms event-onset alignment.")

    # 6. Metadata for post-hoc alignment (inside the run folder).
    meta = {
        "run_id": run_id,
        "run_name": run_name,
        "t0_unix_s": t0_unix,
        "tactile_minus_workstation_offset_s": offset[0] if offset else None,
        "offset_rtt_s": offset[1] if offset else None,
        "offset_after_s": offset_after[0] if offset_after else None,
        "offset_after_rtt_s": offset_after[1] if offset_after else None,
        "offset_drift_s": ((offset_after[0] - offset[0])
                           if (offset and offset_after) else None),
        "franka_ok": franka_ok,
        "franka_csv": "franka.csv" if scp_back.returncode == 0 else None,
        "camera_raw": raw_name,
        "ft_csv": ft_name,
        "gui_original_basename": meta_gui_name,
        "remote_franka_csv": remote_csv,
        "repeats": args.repeats,
        "indent_mm": args.indent_mm,
        "hover_mm": args.hover_mm,
        "dwell_s": args.dwell_s,
        "remote_script": REMOTE_SCRIPT,
        "target": mdiag,
        "note": ("All files for this run live in this folder under canonical names. "
                 "camera.raw and ft.csv are on the WORKSTATION clock; franka.csv is on "
                 "the TACTILE clock -- subtract tactile_minus_workstation_offset_s from "
                 "its unix_time_s to align."),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 7. Final report -- say plainly what landed and what did not.
    print("\n== run '%s' summary ==\n  folder: %s" % (run_id, run_dir))
    franka_landed = "franka.csv" if scp_back.returncode == 0 else None
    for label, name in (("camera events", raw_name), ("Wittenstein F/T", ft_name),
                        ("Franka states", franka_landed)):
        if name and (run_dir / name).exists():
            print("  OK      %-16s %s (%.1f MB)"
                  % (label, name, (run_dir / name).stat().st_size / 1e6))
        else:
            print("  MISSING %-16s <nothing recorded>" % label)
    complete = all((n and (run_dir / n).exists())
                   for n in (raw_name, ft_name, franka_landed))
    print("  %s" % ("All three streams recorded." if complete else
                    "INCOMPLETE -- see the warnings above before post-processing."))
    print("  Post-process with:  python3 postprocess.py %s" % run_id)


if __name__ == "__main__":
    main()
