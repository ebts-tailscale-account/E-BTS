#!/usr/bin/env python3
"""
Experiment startup: go to the Franka ready pose, then command the flange flat
(facing straight down, parallel to the ground).

WHAT IT DOES
------------
1. Moves the Panda to the standard ready/home joint configuration.
2. Reads the flange orientation, computes how far it is off vertical, then
   commands an orientation-only move so the flange's approach axis points
   straight down (global -Z), keeping the current XYZ position. Yaw (heading
   about the vertical) is preserved so the wrist barely turns.
3. Re-reads and reports the residual tilt -- should be ~0 deg.

"Flat / facing the ground" here means the tool/approach axis (flange local +Z)
is aligned with global -Z; then the flange face is horizontal by construction.

Quaternion math is done with numpy only (no tf dependency).

Run on `tactile` with the clean MoveIt stack up (see franka_surface_map.py).
"""

import numpy as np

from franka_surface_map import FrankaArm

# Standard Panda ready pose (rad): [0, -45, 0, -135, 0, 90, 45] deg
HOME_JOINTS_READY = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]

DOWN = np.array([0.0, 0.0, -1.0])   # global "straight down"
FLAT_TOL_DEG = 0.5                  # consider flat within this tilt


def quat_to_R(q):
    """[x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    """3x3 rotation matrix -> [x,y,z,w]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = (tr + 1.0) ** 0.5 * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return [x, y, z, w]


def tilt_deg(q):
    """Angle (deg) between the flange approach axis and straight-down."""
    approach = quat_to_R(q)[:, 2]
    cosang = float(np.clip(np.dot(approach, DOWN), -1.0, 1.0))
    return np.degrees(np.arccos(cosang))


def flat_down_quat(q_cur):
    """Orientation with approach axis = global -Z, preserving current yaw."""
    R = quat_to_R(q_cur)
    x_cur = R[:, 0]
    # horizontal component of the current X axis defines the preserved heading
    x_proj = x_cur - np.dot(x_cur, DOWN) * DOWN
    if np.linalg.norm(x_proj) < 1e-6:
        x_proj = np.array([1.0, 0.0, 0.0])
    x_new = x_proj / np.linalg.norm(x_proj)
    z_new = DOWN
    y_new = np.cross(z_new, x_new)
    R_new = np.column_stack([x_new, y_new, z_new])
    return R_to_quat(R_new)


def main():
    print("=" * 64)
    print(" STARTUP: home to ready pose, then level flange flat (down)")
    print("=" * 64)

    arm = FrankaArm(vel_scale=0.1, acc_scale=0.1)

    print("1) Homing to Franka ready pose ...", flush=True)
    arm.move_to_joints(HOME_JOINTS_READY)

    x, y, z = arm.ee_xyz()
    q0 = arm.current_quat()
    print(f"   at xyz=({x:.5f}, {y:.5f}, {z:.5f})")
    print(f"   flange tilt from vertical: {tilt_deg(q0):.3f} deg", flush=True)

    print("2) Commanding flange flat (approach axis -> straight down) ...", flush=True)
    q_flat = flat_down_quat(q0)
    arm.move_to_pose(x, y, z, q_flat)

    q1 = arm.current_quat()
    approach = quat_to_R(q1)[:, 2]
    resid = tilt_deg(q1)
    print(f"   approach axis now: [{approach[0]:+.4f}, {approach[1]:+.4f}, "
          f"{approach[2]:+.4f}]")
    print(f"   residual tilt from vertical: {resid:.3f} deg", flush=True)

    print("-" * 64)
    if resid <= FLAT_TOL_DEG:
        print(f"  FLAT (<= {FLAT_TOL_DEG} deg). Flange faces the ground. Ready.")
    else:
        print(f"  STILL TILTED ({resid:.2f} deg). If a tool offset is baked into "
              f"the EE frame, fine-tune by hand in Desk.")
    print("=" * 64)


if __name__ == "__main__":
    main()
