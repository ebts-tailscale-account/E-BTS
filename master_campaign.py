#!/usr/bin/env python3
"""E-BTS master (CAMPAIGN variant): one synchronized recording of a full
grid x depth-ladder indentation campaign -- the ML training-data collector.

A COPY of master_midpoint.py, which is itself a copy of master.py. Both originals
are untouched and still run their own workflows (HANDOFF section 12: a variant gets
a copy, never a flag bolted onto the validated original). This copy differs in
exactly three ways:
  1. there is NO single taught target to resolve -- the grid is derived ON TACTILE
     from corner_joints.json, so --dry-run forwards to the remote planner and shows
     you the real plan rather than a local re-derivation that could drift from it,
  2. it drives ~/E-BTS/sweep_campaign.py on tactile instead of indent_midpoint.py,
  3. it pulls back the campaign_plan.csv sidecar (point_index -> depth) alongside
     franka.csv, and records the campaign parameters into metadata.json.

WHY A CAMPAIGN (HANDOFF section 14)
-----------------------------------
The 99-point sweep is not trainable: every poke commanded the same 2.0 mm, so force
was never an independent variable. Crossing the grid with a depth ladder gives force
a real range at every location, decoupled from position. The depth ladder is a
LABEL, not a model input -- see section 14.1d.

Expect ~1.3 h and ~19 GB of .raw for the default 99-location x 6-depth pilot. Run
`sweep_campaign.py --dry-run` (or this script's --dry-run) for the exact numbers.

Fans a single run out to all three systems and pulls everything back:

  camera + Wittenstein F/T  -> the E_BTS_GUI recorder, via the control/start.cmd
                               file protocol (writes <run>_<stamp>.raw and
                               <run>_<stamp>_ft.csv, both on the WORKSTATION clock)
  Franka                    -> the whole grid x depth campaign + franka_states
                               logging on tactile over ssh (CSV on the TACTILE clock)

HOW THE GEOMETRY IS RESOLVED
----------------------------
The campaign grid is a bilinear interpolation over the FOUR HAND-KISSED corners in
corner_joints.json, built ON TACTILE by sweep_campaign.py. Nothing is re-derived
here, so the two machines cannot disagree about where the arm is going.

This script does still pull corner_joints.json back, for two reasons:
  1. to CROSS-CHECK it -- each corner stores joints and O_T_EE from the same
     snapshot, so FK(joints) must reproduce the stored xyz. A gap means the tool/EE
     transform changed since panda_fk was validated and every target in the grid
     would be biased. Better caught now than after ~1.3 h of robot time.
  2. to record the exact taught geometry into metadata.json, so the run stays
     reproducible after the corners are next re-taught.

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
  * The four corners must already be taught:  python3 record_corners.py  on tactile,
    which writes ~/E-BTS/corner_joints.json. The campaign's surface datum is the
    bilinear plane over those HAND-KISSED corners. NO surface_map.csv is used --
    its z_touch sits below the true surface (HANDOFF sections 12.5 and 14).
  * On TACTILE, with the arm at HOME, bring up the Cartesian pose SERVO controller
    (REACH launch: max_translation 0.55 so it can travel home->surface). Run ONE
    controller, not both (it claims the same interface as MoveIt):
        source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
        roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
            robot_ip:=10.1.196.5 load_gripper:=false
    Teaching needs hand-guiding, which the servo controller fights -- so teach FIRST
    (franka_control only), then stop it and bring up the servo reach launch.
    The remote script auto-homes from home to a hover above the elastomer centre;
    no hand-positioning.
  * Passwordless ssh to tactile (ssh-copy-id) so the many ssh/scp calls don't prompt.
  * QUIESCE TACTILE. This run is ~1.3 h at 1 kHz FCI; a loaded control PC drops
    packets and reflexes out (HANDOFF section 4.6). Close Chrome/VS Code, want
    load average < 4.

Usage:  python3 master_campaign.py --dry-run          # show the plan, record nothing
        python3 master_campaign.py pilot --max-points 6   # cautious first run
        python3 master_campaign.py pilot              # the full campaign
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
#   scp franka/sweep_campaign.py tactile@100.93.60.35:~/E-BTS/sweep_campaign.py
REMOTE_SCRIPT      = "~/E-BTS/sweep_campaign.py"
REMOTE_POINTS      = "~/E-BTS/corner_joints.json"  # written by record_corners.py
REMOTE_RECORDINGS  = "~/E-BTS/recordings"          # no more /tmp -- keep runs together

# --- campaign params (this variant). Defaults mirror sweep_campaign.py; anything
# passed here is forwarded verbatim so the two can never silently disagree. ---
SPAN_MM   = [24.0, 32.0]   # explored area, inset from the ~30 x 36 mm taught quad
PITCH_MM  = 3.0            # grid pitch (marker pitch is 2.32 mm, section 14.3)
DEPTHS_MM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
HOVER_MM  = 5.0   # travel / retract / tare at this height above the surface
DWELL_S   = 2.0   # hold each indent this long (the measurement window)
SETTLE_AFTER_START = 3.0   # let the GUI open its files AND the F/T stream stabilise
                           # before the arm moves. Was 1.0 s; raised after
                           # hard_20260807_223744 lost indent 0 to a 1.08 s Wittenstein
                           # dropout 5.5 s into the run. Costs 2 s per run.
START_ACK_TIMEOUT  = 10.0  # how long to wait for the GUI to consume start.cmd


def _ssh(args, **kw):
    return subprocess.run(["ssh", TACTILE] + args, **kw)


# --------------------------------------------------------------- geometry

def _load_panda_fk():
    """panda_fk lives in franka/ next to the robot scripts; it is numpy-only."""
    sys.path.insert(0, str(REPO / "franka"))
    try:
        import panda_fk
    except ImportError as e:
        sys.exit("[ABORT] cannot import panda_fk from %s (%s).\n"
                 "        It is needed to cross-check the taught corners."
                 % (REPO / "franka", e))
    return panda_fk


def fetch_point(dest):
    """Pull corner_joints.json back from tactile (written there by record_corners.py)."""
    print("Fetching the taught corners from tactile (%s)..." % REMOTE_POINTS)
    r = subprocess.run(["scp", "%s:%s" % (TACTILE, REMOTE_POINTS), str(dest)])
    if r.returncode != 0 or not dest.exists():
        sys.exit("[ABORT] could not fetch %s from tactile.\n"
                 "        Teach the corners first:\n"
                 "            ssh %s\n"
                 "            python3 ~/E-BTS/record_corners.py\n"
                 % (REMOTE_POINTS, TACTILE))
    return json.loads(dest.read_text())


ROBOT_MODES = {0: "OTHER", 1: "IDLE", 2: "MOVE", 3: "GUIDING", 4: "REFLEX",
               5: "USER_STOPPED", 6: "AUTOMATIC_ERROR_RECOVERY"}


def verify_corners(data):
    """Cross-check the four taught corners and summarise the quad. See the docstring.

    Every corner stores BOTH joints and the O_T_EE xyz captured in the SAME snapshot,
    so FK(joints) must reproduce the stored xyz. A disagreement means the tool/EE
    transform changed since panda_fk was validated, which would bias every target in
    the campaign -- worth catching before a ~1.3 h run rather than after it.

    Returns a diagnostics dict for metadata.json.
    """
    pfk = _load_panda_fk()
    pts = data.get("points") or data.get("corners")
    if not pts:
        sys.exit("[ABORT] %s has neither 'points' nor 'corners' -- it was not written\n"
                 "        by record_corners.py. Re-teach with that script." % REMOTE_POINTS)

    need = ["bottom-left", "top-left", "top-right", "bottom-right"]
    missing = [k for k in need if k not in pts]
    if missing:
        sys.exit("[ABORT] %s is missing the corner(s): %s.\n"
                 "        Re-teach with record_corners.py." % (REMOTE_POINTS, ", ".join(missing)))

    print("=" * 72)
    print(" TAUGHT CORNERS (the campaign's surface datum)")
    print("=" * 72)
    worst, per_corner = 0.0, {}
    for name in need + (["center"] if "center" in pts else []):
        pt = pts[name]
        xyz = np.array(pt["xyz"], float)
        entry = {"xyz": [float(v) for v in xyz]}
        if "joints" in pt:
            xyz_fk, _ = pfk.pose_of(pt["joints"])
            d_mm = (np.array(xyz_fk) - xyz) * 1e3
            worst = max(worst, float(np.abs(d_mm).max()))
            entry["joints"] = [float(v) for v in pt["joints"]]
            entry["fk_vs_measured_delta_mm"] = [float(v) for v in d_mm]
            print("  %-13s xyz [%s]   FK-meas [%s] mm"
                  % (name, ", ".join("%.5f" % v for v in xyz),
                     ", ".join("%+.3f" % v for v in d_mm)))
        else:
            print("  %-13s xyz [%s]" % (name, ", ".join("%.5f" % v for v in xyz)))
        per_corner[name] = entry

    BL = np.array(pts["bottom-left"]["xyz"], float)
    TL = np.array(pts["top-left"]["xyz"], float)
    BR = np.array(pts["bottom-right"]["xyz"], float)
    u_mm = float(np.linalg.norm(BR - BL) * 1e3)
    v_mm = float(np.linalg.norm(TL - BL) * 1e3)
    z = [pts[k]["xyz"][2] for k in need]
    tilt_mm = float((max(z) - min(z)) * 1e3)
    print("  taught quad   : %.1f x %.1f mm (u x v), surface tilt %.2f mm across it"
          % (u_mm, v_mm, tilt_mm))
    print("=" * 72)
    if worst > 1.0:
        print("  [WARN] FK and the measured pose disagree by up to %.3f mm on a taught"
              % worst)
        print("         corner. panda_fk's tool transform may no longer match this")
        print("         setup -- the whole grid would be biased. Verify before running.")

    return {"taught_at": data.get("recorded_at"), "corners": per_corner,
            "taught_u_len_mm": u_mm, "taught_v_len_mm": v_mm,
            "corner_z_tilt_mm": tilt_mm, "worst_fk_vs_measured_mm": worst}


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
    ap.add_argument("run_name", nargs="?", default="campaign")
    ap.add_argument("--span-mm", type=float, nargs=2, default=SPAN_MM, metavar=("U", "V"),
                    help="Explored span: U across the ~30 mm axis, V across the "
                         "~36 mm axis (default %.0f %.0f)." % tuple(SPAN_MM))
    ap.add_argument("--pitch-mm", type=float, default=PITCH_MM,
                    help="Grid pitch (default %.1f)." % PITCH_MM)
    ap.add_argument("--margin-mm", type=float, nargs="+", default=None,
                    help="Per-edge clearance from the taught quad: 1 value (all "
                         "edges), 2 (u, v), or 4 (u_lo u_hi v_lo v_hi). Overrides "
                         "--span-mm. Needed because the taught corners are NOT "
                         "equidistant from the frame -- the v_hi (TL/TR, x~0.6734) "
                         "side has a plastic wall close by (HANDOFF section 15.2).")
    ap.add_argument("--depths-mm", type=float, nargs="+", default=DEPTHS_MM,
                    help="The depth ladder in mm (default %s)."
                         % " ".join("%.1f" % d for d in DEPTHS_MM))
    ap.add_argument("--repeats", type=int, default=1,
                    help="Repeats of each (location, depth) pair (default 1).")
    ap.add_argument("--order", choices=["serpentine", "random"], default="serpentine")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--hover-mm", type=float, default=HOVER_MM)
    ap.add_argument("--dwell-s", type=float, default=DWELL_S)
    ap.add_argument("--tare-s", type=float, default=1.0,
                    help="Hold out of contact this long before EVERY dip (default "
                         "1.0 s). Each indent is zeroed on its own baseline.")
    ap.add_argument("--max-points", type=int, default=None,
                    help="Run only the first N indents of the plan (cautious run).")
    ap.add_argument("--no-depth-correction", action="store_true")
    ap.add_argument("--i-have-run-the-probe", action="store_true",
                    help="Permit depths beyond 3 mm (only after depth_limit_probe.py).")
    ap.add_argument("--no-gui", action="store_true",
                    help="Run the Franka only (skip camera/F/T recording).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Ask tactile to plan the campaign and print it, then exit. "
                         "No recording, no robot motion.")
    ap.add_argument("--remote-script", default=None,
                    help="Drive a DIFFERENT tactile-side executor instead of "
                         "sweep_campaign.py -- e.g. campaign_planb.py. The sweep "
                         "flags above are then NOT sent (they would not be "
                         "understood); pass that script's own flags through "
                         "--remote-args, and it receives --run-dir instead of a "
                         "positional CSV.")
    ap.add_argument("--remote-args", nargs=argparse.REMAINDER, default=[],
                    help="Everything after this flag is forwarded VERBATIM to "
                         "--remote-script, so it MUST COME LAST. e.g. "
                         "... --remote-args --ack-deep --force-model force_model.json")
    args = ap.parse_args()

    # REMAINDER gives a list; join it back into the command string.
    args.remote_args = " ".join(args.remote_args).strip()

    if args.remote_args and not args.remote_script:
        sys.exit("[ABORT] --remote-args has no effect without --remote-script.\n"
                 "        You passed: %s" % args.remote_args)

    # Which tactile-side executor, and how it is addressed. sweep_campaign.py takes
    # a positional output CSV; campaign_planb.py owns a whole run DIRECTORY (plan,
    # resume ledger, one franka_segNN.csv per segment), so it gets --run-dir.
    if args.remote_script:
        remote_script = "~/E-BTS/%s" % Path(args.remote_script).name
        remote_uses_run_dir = True
    else:
        remote_script = REMOTE_SCRIPT
        remote_uses_run_dir = False

    # Every campaign knob is forwarded verbatim to the remote planner/executor, so
    # the plan this script reports and the plan the arm executes are the same object.
    campaign_args = (
        "%s --span-mm %.4f %.4f --pitch-mm %.4f --depths-mm %s --repeats %d"
        " --order %s --seed %d --hover-mm %.4f --dwell-s %.3f --tare-s %.3f%s%s%s"
        % ((" --margin-mm " + " ".join("%.4f" % v for v in args.margin_mm))
           if args.margin_mm else "",
           args.span_mm[0], args.span_mm[1], args.pitch_mm,
           " ".join("%.4f" % d for d in args.depths_mm), args.repeats,
           args.order, args.seed, args.hover_mm, args.dwell_s, args.tare_s,
           " --max-points %d" % args.max_points if args.max_points else "",
           " --no-depth-correction" if args.no_depth_correction else "",
           " --i-have-run-the-probe" if args.i_have_run_the_probe else ""))

    # A custom executor does not speak the sweep flags. Sending them would abort on
    # an unrecognised argument -- AFTER the GUI had already started recording, which
    # is how hard_20260807_223744 wasted a session. Replace them wholesale.
    if args.remote_script:
        campaign_args = (" " + args.remote_args) if args.remote_args else ""

    # --dry-run forwards to the REMOTE planner rather than re-deriving the grid here.
    # A local copy of the geometry could drift from tactile's corner_joints.json; the
    # machine that will actually move the arm is the one that gets to describe the plan.
    if args.dry_run:
        print("Planning the campaign on tactile (no motion, nothing recorded)...")
        rc = _ssh(["source /opt/ros/noetic/setup.bash && "
                   "source ~/ws_franka/devel/setup.bash && "
                   "python3 %s --dry-run%s" % (remote_script, campaign_args)]).returncode
        if rc != 0:
            sys.exit("[ABORT] the remote planner failed (rc=%d). Is %s deployed on "
                     "tactile, and are the corners taught?" % (rc, remote_script))
        return

    # The grid is derived on tactile from corner_joints.json; there is no single
    # target to resolve here. Pull the taught corners back anyway so the run folder
    # records the exact geometry the campaign was built on -- without it a run is
    # not reproducible after the corners are re-taught.
    RECORDINGS.mkdir(exist_ok=True)
    corners = fetch_point(RECORDINGS / "corner_joints.latest.json")
    corner_diag = verify_corners(corners)
    run_name = args.run_name

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

    # VALIDATE THE REMOTE ARGUMENTS BEFORE TOUCHING THE GUI.
    # Learned the hard way on 2026-08-07: a --depths-mm value the remote script
    # rejects used to fail AFTER start.cmd had been consumed, so the GUI recorded a
    # few seconds of nothing, the run folder was left half-populated, and the Franka
    # CSV never existed. A dry-run costs ~2 s and makes an argument error free.
    print("Validating the plan on tactile before starting anything...")
    _v = _ssh(["source /opt/ros/noetic/setup.bash && "
               "source ~/ws_franka/devel/setup.bash && "
               "python3 %s --dry-run%s > /dev/null" % (remote_script, campaign_args)])
    if _v.returncode != 0:
        sys.exit("[ABORT] the remote planner rejected these arguments (rc=%d).\n"
                 "        Nothing was recorded and the arm never moved. Re-run with\n"
                 "        --dry-run to see the full message." % _v.returncode)
    print("  plan accepted.")

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

    # 3. Run the campaign + logging on tactile (blocks until it ends -- HOURS, not
    #    minutes; the servo creeps at 0.03 m/s and this is hundreds of indents).
    #    The CSV is written straight into tactile's recordings dir (not /tmp).
    #    The remote script builds the grid itself from corner_joints.json; we forward
    #    the campaign knobs verbatim so its plan matches the one --dry-run printed.
    print("Running the campaign on tactile (blocks until done -- expect HOURS)...")
    remote_dir = "%s/%s" % (REMOTE_RECORDINGS, run_id)     # mirror the run folder
    remote_csv = "%s/franka.csv" % remote_dir
    remote_cmd = (
        "source /opt/ros/noetic/setup.bash && "
        "source ~/ws_franka/devel/setup.bash && "
        "mkdir -p %s && "
        "python3 %s %s%s" % (remote_dir, remote_script,
                             ("--run-dir " + remote_dir) if remote_uses_run_dir
                             else remote_csv,
                             campaign_args)
    )
    franka = _ssh([remote_cmd])
    franka_ok = franka.returncode == 0
    print("  Campaign %s."
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

    # 5. Pull the Franka log(s) back into the run folder.
    local_csv = run_dir / "franka.csv"
    if remote_uses_run_dir:
        # campaign_planb owns a directory: plan.csv, plan.json, state.jsonl (the
        # resume ledger) and one franka_segNN.csv per segment. Pull ALL of it --
        # the ledger is what makes the run resumable and the segments are what the
        # postprocessor joins on, so fetching only "franka.csv" would lose both.
        print("Pulling the whole remote run directory back...")
        # NOT "%s/." -- modern scp speaks SFTP and rejects a bare "." component
        # ("error: unexpected filename: ."), which silently returns nothing. A
        # remote glob is expanded by the shell on tactile and works on both.
        scp_back = subprocess.run(["scp", "-r", "%s:%s/*" % (TACTILE, remote_dir),
                                   str(run_dir)])
        if scp_back.returncode != 0:
            print("  WARNING: could not pull the remote run directory from tactile.")
        else:
            segs = sorted(run_dir.glob("franka_seg*.csv"))
            print("  fetched %d franka segment(s): %s"
                  % (len(segs), ", ".join(p.name for p in segs) or "NONE"))
            if not segs:
                print("  WARNING: no franka_seg*.csv -- did the remote run start?")
    else:
        print("Pulling Franka CSV back...")
        scp_back = subprocess.run(["scp", "%s:%s" % (TACTILE, remote_csv),
                                   str(local_csv)])
        if scp_back.returncode != 0:
            print("  WARNING: could not pull the Franka CSV from tactile.")

    # 5a. Pull the campaign plan sidecar. WITHOUT THIS THE RUN HAS NO DEPTH LABELS:
    #     the 93-column log carries phase/point_index but not the commanded depth,
    #     which lives in campaign_plan.csv keyed by point_index. A run missing it is
    #     unlabelled training data, so this is a loud warning, not a silent skip.
    remote_plan = "%s/campaign_plan.csv" % remote_dir
    if remote_uses_run_dir:
        # campaign_planb's equivalent is plan.csv + plan.json, already pulled by the
        # recursive fetch above. Verify rather than re-fetch -- and keep the warning
        # just as loud, because an unlabelled run is an unusable run either way.
        plan_ok = (run_dir / "plan.csv").exists()
        if plan_ok:
            print("  plan.csv present (%d pokes) + plan.json"
                  % max(0, sum(1 for _ in open(run_dir / "plan.csv")) - 1))
        remote_plan = "%s/plan.csv" % remote_dir
    else:
        plan_ok = subprocess.run(
            ["scp", "%s:%s" % (TACTILE, remote_plan),
             str(run_dir / "campaign_plan.csv")]).returncode == 0
        if plan_ok:
            subprocess.run(["scp", "%s:%s.json" % (TACTILE, remote_plan),
                            str(run_dir / "campaign_plan.csv.json")])
    if not plan_ok:
        print("  *** WARNING: campaign_plan.csv did NOT come back. The log records "
              "point_index but NOT the commanded depth, so this run cannot be "
              "labelled for training until you recover %s from tactile." % remote_plan)

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
        "campaign_plan_csv": "campaign_plan.csv" if plan_ok else None,
        "campaign": {
            "span_mm": list(args.span_mm),
            "margin_mm": list(args.margin_mm) if args.margin_mm else None,
            "pitch_mm": args.pitch_mm,
            "depths_mm": list(args.depths_mm),
            "tare_s": args.tare_s,
            "repeats": args.repeats,
            "order": args.order,
            "seed": args.seed,
            "max_points": args.max_points,
            "depth_correction": not args.no_depth_correction,
            "remote_args": campaign_args.strip(),
            "surface_reference": ("bilinear plane over the 4 hand-kissed corners "
                                  "in corner_joints.json (NOT surface_map z_touch)"),
        },
        "hover_mm": args.hover_mm,
        "dwell_s": args.dwell_s,
        "remote_script": remote_script,
        "remote_args": args.remote_args or None,
        "taught_corners": corner_diag,
        "note": ("All files for this run live in this folder under canonical names. "
                 "camera.raw and ft.csv are on the WORKSTATION clock; franka.csv is on "
                 "the TACTILE clock -- subtract tactile_minus_workstation_offset_s from "
                 "its unix_time_s to align. The commanded depth of each indent is NOT "
                 "in franka.csv -- join campaign_plan.csv on point_index."),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 7. Final report -- say plainly what landed and what did not.
    print("\n== run '%s' summary ==\n  folder: %s" % (run_id, run_dir))
    # NOT hardcoded "franka.csv": campaign_planb owns a run directory and writes one
    # franka_segNN.csv per segment, so the old check reported MISSING on a run whose
    # 220 MB log had in fact arrived (planb_3mm_20260811_150324).
    if scp_back.returncode != 0:
        franka_landed = None
    elif remote_uses_run_dir:
        _segs = sorted(run_dir.glob("franka_seg*.csv"))
        franka_landed = _segs[0].name if _segs else None
    else:
        franka_landed = "franka.csv"
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
