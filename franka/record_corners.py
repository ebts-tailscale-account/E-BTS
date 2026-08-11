#!/usr/bin/env python3
"""
Teach the 4 corners of the silicon elastomer by recording joint states.

Instead of computing corner positions from a centre + mm offsets (which needs
the block's exact rotation in the base frame and was getting it wrong), this
records the robot's actual joint configuration at each physical corner. The
probe script then just replays those joint states -- no geometry assumptions.

HOW TO USE
----------
1. Bring up the Franka driver (franka_states must be publishing).
2. Run this on `tactile`:  python3 record_corners.py
3. For each corner, move the arm so the tool is just TOUCHING / kissing that
   corner of the elastomer -- contact, but NOT indenting it (hand-guiding /
   Desk / however you jog it), then press Enter.
4. It writes corner_joints.json next to this script. The recorded pose is the
   surface point itself; the probe script travels ABOVE it to avoid dragging.

Order recorded (matches the probe): center -> bottom-left -> top-left ->
top-right -> bottom-right. Reading is decoupled from motion -- it just snapshots
the joints you positioned by hand, exactly like read_pose.py.
"""

import json
from datetime import datetime
from pathlib import Path

import rospy
from franka_msgs.msg import FrankaState

POINT_ORDER = ["center", "bottom-left", "top-left", "top-right", "bottom-right"]
OUT_PATH = Path(__file__).with_name("corner_joints.json")


def read_state(timeout=5.0):
    """Grab one fresh FrankaState; return (joints[7], xyz[3])."""
    st = rospy.wait_for_message(
        "/franka_state_controller/franka_states", FrankaState, timeout=timeout)
    joints = [round(a, 6) for a in st.q]
    T = st.O_T_EE                      # column-major 4x4; position = [12],[13],[14]
    xyz = [round(T[12], 6), round(T[13], 6), round(T[14], 6)]
    return joints, xyz


def main():
    rospy.init_node("record_corners", anonymous=True, disable_signals=True)

    print("=" * 64)
    print(" TEACH ELASTOMER CENTER + CORNERS  (record joint states)")
    print("=" * 64)
    print("Move the tool so it is just TOUCHING each point (kiss, no indent),")
    print("then press Enter.\n")

    points = {}
    for label in POINT_ORDER:
        input(f"  >> Kiss the [{label}] point (touch, no indent), press Enter... ")
        try:
            joints, xyz = read_state()
        except Exception as e:
            print(f"     [ERROR] could not read state: {e}")
            print("     Retrying once...")
            joints, xyz = read_state()
        points[label] = {"joints": joints, "xyz": xyz}
        print(f"     recorded {label}: xyz={xyz}")
        print(f"                joints={joints}\n")

    data = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "frame": "panda_link0",
        "note": "joints taught kissing each point (center + corners, surface point, no indent)",
        "points": points,
    }
    OUT_PATH.write_text(json.dumps(data, indent=2))

    print("=" * 64)
    print(f" Saved {len(points)} points -> {OUT_PATH}")
    print("=" * 64)
    for label in POINT_ORDER:
        c = points[label]
        print(f"  {label:<13} xyz={c['xyz']}")
    print("\nNow run:  python3 corner_probe.py   (double-checks all corners)")


if __name__ == "__main__":
    main()
