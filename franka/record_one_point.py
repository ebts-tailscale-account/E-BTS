#!/usr/bin/env python3
"""Teach ONE point on the elastomer -- the exact spot to indent -- by recording joints.

You hand-position the tool at the point you want (this IS the midpoint; nothing is
interpolated), press Enter once, and it writes one_point.json.

Records, from a SINGLE FrankaState snapshot:
  joints[7]   the encoder reading -- authoritative
  O_T_EE[16]  the measured pose from the same snapshot (column-major)
  xyz[3]      position, i.e. exact FK of joints
  quat[xyzw]  orientation, held during the indent so contact geometry matches
  dq[7]       joint velocity at the instant of the snapshot (stillness proof)

The TARGET is recomputed as FK(joints) rather than read out of O_T_EE, so the
geometry comes from the joint states. Those two agree to ~1e-3 mm when the tool
transform is what we think it is, which is exactly what the FK self-check verifies.

Two safeguards:
  1. STILLNESS GUARD -- a hand-guided arm that is still drifting gives a pose that is
     not the one you meant to teach. The snapshot is rejected and retried until every
     |dq| is below STILL_DQ.
  2. FK SELF-CHECK -- panda_fk.fk(joints) vs the measured O_T_EE from the same
     snapshot. Disagreement > 1 mm aborts, rather than silently biasing the target.

HOW TO USE (on tactile)
-----------------------
1. franka_control only, so franka_states publishes and you can hand-guide. Do NOT run
   the servo controller -- it holds position and fights you:
       source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
       roslaunch franka_control franka_control.launch robot_ip:=10.1.196.5 load_gripper:=false
2. python3 record_one_point.py
3. Hand-guide the tool so it just KISSES the surface (contact, NO indent), let go,
   wait for it to stop, press Enter.

The kissed height IS the surface reference: the run indents INDENT_MM below it.
Then, on the WORKSTATION:  python3 master_midpoint.py <run_name>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from franka_msgs.msg import FrankaState

import panda_fk as pfk

OUT_PATH = Path(__file__).with_name("one_point.json")
STILL_DQ = 0.004        # rad/s -- every joint must be quieter than this to snapshot
STILL_TRIES = 40        # ~8 s of retries before giving up
FK_TOL_MM = 1.0


def read_state(timeout=5.0):
    return rospy.wait_for_message(
        "/franka_state_controller/franka_states", FrankaState, timeout=timeout)


def snapshot(label, require_still=True):
    """One FrankaState -> dict, rejected while the arm is still moving."""
    dq = np.zeros(7)
    for attempt in range(STILL_TRIES):
        st = read_state()
        dq = np.asarray(st.dq, float)
        if not require_still or np.abs(dq).max() < STILL_DQ:
            break
        if attempt == 0:
            print("     arm still moving (max|dq| = %.4f rad/s) -- waiting for it to"
                  " settle..." % np.abs(dq).max())
        rospy.sleep(0.2)
    else:
        raise SystemExit(
            "\n[ABORT] The arm never went still (max|dq| = %.4f rad/s > %.4f).\n"
            "        Let go of it and let it settle before pressing Enter, or re-run\n"
            "        with --allow-motion if you accept a moving snapshot.\n"
            % (np.abs(dq).max(), STILL_DQ))

    joints = [round(float(a), 9) for a in st.q]
    err_mm = pfk.self_check(joints, st.O_T_EE, tol_mm=FK_TOL_MM, label=label)

    # Target geometry from the JOINTS (not read out of O_T_EE).
    xyz_fk, quat_fk = pfk.pose_of(joints)
    T_meas = pfk.colmajor_to_T(st.O_T_EE)

    return {
        "joints": joints,
        "xyz": [round(float(v), 9) for v in xyz_fk],            # FK of joints
        "quat_xyzw": [round(float(v), 9) for v in quat_fk],     # FK of joints
        "xyz_measured_O_T_EE": [round(float(v), 9) for v in T_meas[:3, 3]],
        "O_T_EE": [round(float(v), 9) for v in st.O_T_EE],
        "dq": [round(float(v), 9) for v in st.dq],
        "dq_max_abs": round(float(np.abs(dq).max()), 9),
        "fk_residual_mm": round(err_mm, 6),
        "robot_mode": int(st.robot_mode),
    }


ROBOT_MODES = {0: "OTHER", 1: "IDLE", 2: "MOVE", 3: "GUIDING", 4: "REFLEX",
               5: "USER_STOPPED", 6: "AUTOMATIC_ERROR_RECOVERY"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="indent_point",
                    help="name for the point (default indent_point)")
    ap.add_argument("--allow-motion", action="store_true",
                    help="skip the stillness guard (not recommended)")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    rospy.init_node("record_one_point", anonymous=True, disable_signals=True)

    print("=" * 70)
    print(" TEACH THE INDENT POINT  (one point, joint states + EE pose)")
    print("=" * 70)
    print("Hand-guide the tool so it just KISSES the surface (contact, NO indent),")
    print("let go, wait for it to STOP, then press Enter.")
    print("This point IS the indent location -- nothing is interpolated.")
    print("Its height IS the surface reference the indent depth is measured from.\n")

    input("  >> Kiss the [%s] point (touch, no indent), press Enter... " % args.label)
    rec = snapshot(args.label, require_still=not args.allow_motion)

    data = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "frame": "panda_link0",
        "label": args.label,
        "note": ("single point taught by kissing the elastomer surface (no indent). "
                 "joints are authoritative; xyz/quat are FK of those joints. This point "
                 "is the exact indent target -- no midpoint interpolation."),
        "point": rec,
        "target": {
            "xyz": rec["xyz"],
            "quat_xyzw": rec["quat_xyzw"],
            "surface_z": rec["xyz"][2],
            "surface_z_note": "the kissed height; the run indents INDENT_MM below it",
        },
    }
    Path(args.out).write_text(json.dumps(data, indent=2))

    print("=" * 70)
    print(" Saved 1 point -> %s" % args.out)
    print("=" * 70)
    print("  joints      : [%s]" % ", ".join("%+.4f" % v for v in rec["joints"]))
    print("  xyz (FK)    : [%s]" % ", ".join("%.5f" % v for v in rec["xyz"]))
    print("  xyz (O_T_EE): [%s]" % ", ".join("%.5f" % v for v in rec["xyz_measured_O_T_EE"]))
    print("  quat        : [%s]" % ", ".join("%+.4f" % v for v in rec["quat_xyzw"]))
    print("  surface_z   : %.5f m   <- indent depth is measured from here" % rec["xyz"][2])
    print("  fk residual : %.4f mm" % rec["fk_residual_mm"])
    print("  max|dq|     : %.5f rad/s (still)" % rec["dq_max_abs"])
    print("  robot_mode  : %d (%s)"
          % (rec["robot_mode"], ROBOT_MODES.get(rec["robot_mode"], "?")))
    print("\nNext, on the WORKSTATION (GUI up, both panes open):")
    print("    python3 master_midpoint.py <run_name> --dry-run")
    print("    python3 master_midpoint.py <run_name>")


if __name__ == "__main__":
    main()
