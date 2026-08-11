#!/usr/bin/env python3
"""Teach TWO points on the elastomer by recording joint states (+ EE pose).

Same idea as record_corners.py, but for two arbitrary points, and it records
more per point so the indent target can be built in JOINT space:

  joints[7]   the encoder reading -- the authoritative record
  O_T_EE[16]  the full measured pose from the SAME snapshot (column-major)
  xyz[3]      position, i.e. exact FK of joints
  quat[xyzw]  orientation, from the same snapshot
  dq[7]       joint velocity at the instant of the snapshot (stillness proof)

Two safeguards the corner script did not have:
  1. STILLNESS GUARD -- a hand-guided arm that is still drifting gives a pose that
     is not the one you think you taught. The snapshot is rejected and retried
     until every |dq| is below STILL_DQ.
  2. FK SELF-CHECK -- panda_fk.fk(joints) is compared against the measured O_T_EE
     from the same snapshot. If they disagree by more than 1 mm the DH/tool
     assumption is wrong and the script aborts, rather than letting every
     downstream joint-space target be silently biased.

HOW TO USE (on tactile)
-----------------------
1. Bring up franka_control so franka_states publishes. Put the arm in hand-guiding
   mode. Do NOT run the servo controller -- it would hold position and fight you.
       source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
       roslaunch franka_control franka_control.launch robot_ip:=10.1.196.5 load_gripper:=false
2. python3 record_two_points.py
3. For each point: hand-guide the tool so it just KISSES the surface (contact, no
   indent), let go, wait for it to stop, press Enter.

Writes two_points.json next to this script. The kissed z IS the surface height at
that point, which is what indent_midpoint.py uses as its depth reference.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from franka_msgs.msg import FrankaState

import panda_fk as pfk

OUT_PATH = Path(__file__).with_name("two_points.json")
STILL_DQ = 0.004        # rad/s -- every joint must be quieter than this to snapshot
STILL_TRIES = 40        # ~8 s of retries before giving up
FK_TOL_MM = 1.0


def read_state(timeout=5.0):
    return rospy.wait_for_message(
        "/franka_state_controller/franka_states", FrankaState, timeout=timeout)


def snapshot(label, require_still=True):
    """One FrankaState -> dict, rejected while the arm is still moving."""
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
    T = pfk.colmajor_to_T(st.O_T_EE)
    xyz = [round(float(v), 9) for v in T[:3, 3]]
    quat = [round(float(v), 9) for v in pfk.R_to_quat(T[:3, :3])]
    err_mm = pfk.self_check(joints, st.O_T_EE, tol_mm=FK_TOL_MM, label=label)
    return {
        "joints": joints,
        "xyz": xyz,
        "quat_xyzw": quat,
        "O_T_EE": [round(float(v), 9) for v in st.O_T_EE],
        "dq": [round(float(v), 9) for v in st.dq],
        "fk_residual_mm": round(err_mm, 6),
        "robot_mode": int(st.robot_mode),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, default=["point_a", "point_b"],
                    help="names for the two points (default point_a point_b)")
    ap.add_argument("--allow-motion", action="store_true",
                    help="skip the stillness guard (not recommended)")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    rospy.init_node("record_two_points", anonymous=True, disable_signals=True)

    print("=" * 70)
    print(" TEACH TWO POINTS ON THE ELASTOMER  (joint states + EE pose)")
    print("=" * 70)
    print("Hand-guide the tool so it just KISSES the surface (contact, NO indent),")
    print("let go, wait for it to STOP, then press Enter.")
    print("The kissed height IS the surface height used as the indent reference.\n")

    points = {}
    for label in args.labels:
        input("  >> Kiss the [%s] point (touch, no indent), press Enter... " % label)
        rec = snapshot(label, require_still=not args.allow_motion)
        points[label] = rec
        print("     xyz    = [%s]" % ", ".join("%.5f" % v for v in rec["xyz"]))
        print("     joints = [%s]\n" % ", ".join("%.4f" % v for v in rec["joints"]))

    a, b = (points[l] for l in args.labels)
    qa, qb = a["joints"], b["joints"]
    pa, pb = np.array(a["xyz"]), np.array(b["xyz"])

    q_mid = pfk.joint_midpoint(qa, qb)
    p_jmid, quat_jmid = pfk.pose_of(q_mid)
    p_cmid = 0.5 * (pa + pb)
    sep_mm = float(np.linalg.norm(pa - pb) * 1e3)
    delta_mm = (np.array(p_jmid) - p_cmid) * 1e3

    data = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "frame": "panda_link0",
        "labels": list(args.labels),
        "note": ("two points taught by kissing the elastomer surface (no indent). "
                 "joints are authoritative; xyz/quat are the FK of those joints from "
                 "the same FrankaState snapshot."),
        "points": points,
        "separation_mm": round(sep_mm, 4),
        "midpoint": {
            "joints": [round(float(v), 9) for v in q_mid],
            "xyz_from_joint_midpoint": [round(float(v), 9) for v in p_jmid],
            "quat_xyzw_from_joint_midpoint": [round(float(v), 9) for v in quat_jmid],
            "xyz_cartesian_midpoint": [round(float(v), 9) for v in p_cmid],
            "joint_vs_cartesian_delta_mm": [round(float(v), 4) for v in delta_mm],
            "surface_z_reference": round(float(0.5 * (pa[2] + pb[2])), 9),
            "surface_z_note": ("mean of the two kissed z values -- the physical surface "
                               "between two kissed points is a chord, so the Cartesian "
                               "mean is the right height reference; the joint-space arc "
                               "z is recorded above for comparison only."),
        },
    }
    Path(args.out).write_text(json.dumps(data, indent=2))

    print("=" * 70)
    print(" Saved 2 points -> %s" % args.out)
    print("=" * 70)
    for l in args.labels:
        print("  %-10s xyz=[%s]  fk_res=%.4f mm"
              % (l, ", ".join("%.5f" % v for v in points[l]["xyz"]),
                 points[l]["fk_residual_mm"]))
    print("\n  separation                 : %.2f mm" % sep_mm)
    print("  midpoint (joint-space FK)  : [%s]" % ", ".join("%.5f" % v for v in p_jmid))
    print("  midpoint (Cartesian mean)  : [%s]" % ", ".join("%.5f" % v for v in p_cmid))
    print("  joint - Cartesian delta    : [%s] mm"
          % ", ".join("%+.3f" % v for v in delta_mm))
    print("  surface z reference        : %.5f m (mean of the two kissed z)"
          % data["midpoint"]["surface_z_reference"])
    if np.linalg.norm(delta_mm) > 0.5:
        print("\n  NOTE: joint-space and Cartesian midpoints differ by %.2f mm. The XY"
              % np.linalg.norm(delta_mm[:2]))
        print("        target comes from the JOINT midpoint; the Z reference comes from")
        print("        the kissed heights. Both are logged per sample.")
    print("\nNext:  python3 indent_midpoint.py --dry-run")


if __name__ == "__main__":
    main()
